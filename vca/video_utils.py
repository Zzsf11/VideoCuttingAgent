import os
import subprocess
from typing import Optional, Tuple, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image
from decord import VideoReader, cpu
from scenedetect import detect, AdaptiveDetector

from qwen_omni_utils.v2_5.vision_process import fetch_video
from vca.transnetv2_pytorch.transnetv2_pytorch.inference import TransNetV2Torch
from vca.AutoShot.autoshot_wrapper import AutoShotTorch
from vca.build_database.shot_det_VL import process_video
from vca import config


def _resize_frame(frame: Image.Image, target_height: int, target_width: int) -> Image.Image:
    """Resize a PIL Image to target resolution."""
    return frame.resize((target_width, target_height), Image.LANCZOS)


def _extract_frames_batch_ffmpeg(
    video_path: str,
    frames_dir: str,
    target_fps: Optional[float] = None,
    target_resolution: Optional[Tuple[int, int]] = None,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    max_frames: Optional[int] = None,
    num_workers: int = 8,
) -> Tuple[List[str], int, int, float]:
    """
    Extract frames using decord (consistent with single_video.py).
    Uses batch reading and multi-threaded saving for speed.
    Returns: (frame_paths, height, width, actual_fps)
    """
    _ensure_dir(frames_dir)

    # Load video with decord
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=8)
    video_fps = vr.get_avg_fps()
    total_frames = len(vr)
    video_duration = total_frames / video_fps

    # Determine time range
    start_t = start_sec if start_sec is not None else 0.0
    end_t = end_sec if end_sec is not None else video_duration

    # Target FPS
    fps = target_fps if target_fps is not None else 2.0
    interval = 1.0 / fps

    # Calculate frame indices to extract (same logic as single_video.py)
    frame_indices = []
    t = start_t
    while t < end_t:
        frame_idx = int(t * video_fps)
        if frame_idx < total_frames:
            frame_indices.append(frame_idx)
        t += interval
        # Check max_frames limit
        if max_frames is not None and len(frame_indices) >= max_frames:
            break

    print(f"[Batch Processing] Extracting frames with decord: target_fps={fps}, max_frames={max_frames}")
    print(f"[Batch Processing] Video fps={video_fps:.2f}, duration={video_duration:.2f}s, range=[{start_t:.2f}, {end_t:.2f}]")

    if not frame_indices:
        raise RuntimeError("No frames to extract from the video")

    # Determine target resolution
    target_h, target_w = None, None
    if target_resolution is not None:
        if isinstance(target_resolution, (int, float)):
            target_h = int(target_resolution)
        elif isinstance(target_resolution, (list, tuple)) and len(target_resolution) == 1:
            target_h = int(target_resolution[0])
        elif isinstance(target_resolution, (list, tuple)) and len(target_resolution) == 2:
            target_h, target_w = int(target_resolution[0]), int(target_resolution[1])

    # Function to process and save a single frame
    def save_frame(args):
        frame_data, out_path = args
        img = Image.fromarray(frame_data)

        # Resize if needed
        if target_h is not None:
            orig_w, orig_h = img.size
            if target_w is None:
                # Keep aspect ratio
                target_w_calc = int(orig_w * target_h / orig_h)
                # Ensure width is even
                target_w_calc = target_w_calc if target_w_calc % 2 == 0 else target_w_calc + 1
                img = img.resize((target_w_calc, target_h), Image.LANCZOS)
            else:
                img = img.resize((target_w, target_h), Image.LANCZOS)

        img.save(out_path)

    # Prepare output paths
    frame_paths: List[str] = []
    for i in range(len(frame_indices)):
        filename = f"frame_{i:06d}.png"
        out_path = os.path.join(frames_dir, filename)
        frame_paths.append(out_path)

    # Batch read and save frames in chunks to avoid memory issues
    batch_size = 500
    total_batches = (len(frame_indices) + batch_size - 1) // batch_size
    print(f"[Batch Processing] Processing {len(frame_indices)} frames in {total_batches} batches (batch_size={batch_size})...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(frame_indices))
            batch_indices = frame_indices[start_idx:end_idx]
            batch_paths = frame_paths[start_idx:end_idx]

            # Batch read this chunk
            frames_batch = vr.get_batch(batch_indices).asnumpy()

            # Multi-threaded saving for this batch
            save_args = [(frames_batch[i], batch_paths[i]) for i in range(len(batch_indices))]
            list(executor.map(save_frame, save_args))

            print(f"[Batch Processing] Completed batch {batch_idx + 1}/{total_batches} ({end_idx}/{len(frame_indices)} frames)")

    actual_num_frames = len(frame_paths)
    print(f"[Batch Processing] Extracted {actual_num_frames} frames (max_frames limit: {max_frames if max_frames else 'None'})")

    # Warn if we hit the max_frames limit
    if max_frames is not None and actual_num_frames >= max_frames:
        print(f"[Batch Processing] WARNING: Hit max_frames limit ({max_frames}). Video may have more frames available.")

    # Get dimensions from first frame
    first_frame = Image.open(frame_paths[0])
    width, height = first_frame.size

    return frame_paths, height, width, fps


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_tensor_frame_to_file(frame_tensor: torch.Tensor, output_path: str) -> None:
    """Save a single frame tensor (C,H,W) in [0,255] float/uint8 to a PNG file."""
    if frame_tensor.dtype != torch.uint8:
        frame_tensor = frame_tensor.clamp(0, 255).to(torch.uint8)
    image = Image.fromarray(frame_tensor.permute(1, 2, 0).cpu().numpy())
    image.save(output_path)


def _timecode_to_seconds(timecode: str) -> float:
    """Convert timecode HH:MM:SS.mmm to seconds."""
    hours, minutes, seconds_milliseconds = timecode.split(":")
    seconds, milliseconds = seconds_milliseconds.split(".")
    total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000
    return total_seconds


def _adjust_scene_boundaries(scenes: List) -> List:
    """
    Adjust scene boundaries so that each scene starts at the frame after the previous scene ends.
    This avoids frame overlap between consecutive scenes.

    Input:  [[0, 43], [43, 102], [102, 107], ...]
    Output: [[0, 43], [44, 102], [103, 107], ...]
    """
    if not scenes or len(scenes) <= 1:
        return scenes

    adjusted = []
    for i, scene in enumerate(scenes):
        if i == 0:
            adjusted.append([scene[0], scene[1]])
        else:
            # Start frame should be previous scene's end + 1
            start_frame = adjusted[i - 1][1] + 1
            adjusted.append([start_frame, scene[1]])

    return adjusted


def _scenedetect_shot_detection(
    video_path: str,
    threshold: float = 3.0,
    min_scene_len: int = 15,
) -> List[Tuple[float, float]]:
    """
    Shot detection using scenedetect's AdaptiveDetector.

    Args:
        video_path: Path to the video file
        threshold: Adaptive threshold, lower values are more sensitive (default 3.0)
        min_scene_len: Minimum shot length in frames (default 15)

    Returns:
        shot_list: [(start_sec, end_sec), ...] List of shot time intervals
    """
    scene_list = detect(
        video_path,
        AdaptiveDetector(
            adaptive_threshold=threshold,
            min_scene_len=min_scene_len
        )
    )

    shot_list = []
    for scene in scene_list:
        start_time = _timecode_to_seconds(scene[0].get_timecode())
        end_time = _timecode_to_seconds(scene[1].get_timecode())
        shot_list.append((start_time, end_time))

    return shot_list


def _probe_media_duration_seconds(media_path: str) -> Optional[float]:
    """Return media duration in seconds using ffprobe, or None on failure."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nw=1:nk=1",
                media_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        duration_str = result.stdout.strip()
        return float(duration_str) if duration_str else None
    except Exception:
        return None


def decode_video_to_frames(
    video_path: str,
    frames_dir: str,
    target_fps: Optional[float] = None,
    target_resolution: Optional[Tuple[int, int]] = None,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    max_frames: Optional[int] = None,
    batch_size: Optional[int] = None,
    use_batch_processing: bool = True,
    shot_detection: bool = False,
    shot_detection_model: str = "transnetv2",
    shot_detection_fps: float = 2.0,
    shot_detection_threshold: float = 0.5,
    shot_detection_min_scene_len: int = 15,
    shot_predictions_path: Optional[str] = None,
    shot_scenes_path: Optional[str] = None,
) -> dict:
    """
    Read a video and save sampled frames to disk, with optional shot detection.

    Args:
        video_path: Path to the video file.
        frames_dir: Directory to save frames.
        target_fps: Desired sampling fps. If None, uses default (2.0).
        target_resolution: (height, width). If None, uses smart_resize based on defaults.
        start_sec: Optional start time in seconds.
        end_sec: Optional end time in seconds.
        max_frames: Maximum number of frames to extract.
        batch_size: Number of frames to process at once. If None, defaults to 500 frames per batch.
        use_batch_processing: If True, use batch processing to avoid loading entire video into memory at once.
        shot_detection: If True, perform shot/scene detection.
        shot_detection_model: Model to use for shot detection (default: "transnetv2").
            Options: "transnetv2", "autoshot", "qwen3vl", "scenedetect".
            - "transnetv2": Deep learning based shot boundary detection.
            - "autoshot": Another deep learning based approach.
            - "qwen3vl": Vision-language model based detection.
            - "scenedetect": Uses PySceneDetect's AdaptiveDetector (content-aware detection).
        shot_detection_fps: FPS for shot detection (default: 2.0). Not used for scenedetect.
        shot_detection_threshold: Threshold for shot boundary detection (default: 0.5).
            Recommended values: 0.5 for TransNetV2, 0.296 for AutoShot, 3.0 for scenedetect.
        shot_detection_min_scene_len: Minimum shot length in frames (default: 15). Only used for scenedetect.
        shot_predictions_path: Output path for shot predictions. Defaults to frames_dir/shot_predictions.txt.
        shot_scenes_path: Output path for scene boundaries. Defaults to frames_dir/shot_scenes.txt.

    Returns:
        dict with keys: {"num_frames", "sample_fps", "height", "width", "frame_paths",
                        "shot_predictions_path" (optional), "shot_scenes_path" (optional), "scenes" (optional)}
    """
    _ensure_dir(frames_dir)

    # Check if frames already exist
    existing_frames = sorted([f for f in os.listdir(frames_dir) if f.startswith("frame_") and f.endswith(".png")])
    frames_exist = len(existing_frames) > 0
    
    if frames_exist:
        print(f"[Skip] Found {len(existing_frames)} existing frames in {frames_dir}, skipping frame extraction")
        frame_paths = [os.path.join(frames_dir, f) for f in existing_frames]
        num_frames = len(frame_paths)
        
        # Get dimensions and fps from first frame and video metadata
        first_frame = Image.open(frame_paths[0])
        width, height = first_frame.size
        sample_fps = target_fps if target_fps is not None else 2.0
        
        skip_frame_extraction = True
    else:
        skip_frame_extraction = False

    # Determine if we should use batch processing (ffmpeg direct extraction)
    # Batch processing is more memory efficient for large videos
    if not skip_frame_extraction and use_batch_processing:
        print(f"Using batch processing mode (ffmpeg direct extraction) for memory efficiency")
        frame_paths, height, width, sample_fps = _extract_frames_batch_ffmpeg(
            video_path=video_path,
            frames_dir=frames_dir,
            target_fps=target_fps,
            target_resolution=target_resolution,
            start_sec=start_sec,
            end_sec=end_sec,
            max_frames=max_frames,
        )
        num_frames = len(frame_paths)
        print(f"Extracted {num_frames} frames to {frames_dir}")
    elif not skip_frame_extraction:
        # Original method using fetch_video (loads entire video into memory)
        print(f"Using original fetch_video method (may use more memory)")
        ele = {"video": video_path}
        if target_fps is not None:
            ele["fps"] = float(target_fps)
        if target_resolution is not None:
            if isinstance(target_resolution, (int, float)):
                h = w = int(target_resolution)
                target_resolution = (h, w)
            elif isinstance(target_resolution, (list, tuple)) and len(target_resolution) == 1:
                h = w = int(target_resolution[0])
                target_resolution = (h, w)
            elif isinstance(target_resolution, (list, tuple)) and len(target_resolution) == 2:
                h, w = target_resolution
            else:
                raise ValueError("target_resolution should be a single int/float or a tuple/list of two ints")
            ele["resized_height"] = int(h)
            ele["resized_width"] = int(w)
        if start_sec is not None:
            ele["video_start"] = float(start_sec)
        if end_sec is not None:
            ele["video_end"] = float(end_sec)
        if max_frames is not None:
            ele["max_frames"] = int(max_frames)

        print(f"Fetching video from {video_path}")
        video_tensor, sample_fps = fetch_video(ele, return_video_sample_fps=True)

        # video_tensor: (T, C, H, W) in float32 range [0, 255] after resize
        assert isinstance(video_tensor, torch.Tensor), "fetch_video should return a Tensor for file inputs"
        num_frames, channels, height, width = video_tensor.shape

        frame_paths: List[str] = []
        padding = len(str(num_frames))
        print(f"Saving {num_frames} frames to {frames_dir}")
        for idx in range(num_frames):
            frame = video_tensor[idx]
            filename = f"frame_{idx:0{padding}d}.png"
            out_path = os.path.join(frames_dir, filename)
            _save_tensor_frame_to_file(frame, out_path)
            frame_paths.append(out_path)

    result: Dict[str, Any] = {
        "num_frames": num_frames,
        "sample_fps": float(sample_fps),
        "height": int(height),
        "width": int(width),
        "frame_paths": frame_paths,
    }

    # Shot detection using TransNetV2 or AutoShot
    if shot_detection:
        # Normalize model name
        shot_detection_model = shot_detection_model.lower()
        if shot_detection_model not in ["transnetv2", "autoshot", "qwen3vl", "scenedetect"]:
            raise ValueError(f"Invalid shot_detection_model: {shot_detection_model}. Must be 'transnetv2', 'autoshot', 'qwen3vl' or 'scenedetect'")
        
        # Determine output paths
        final_predictions_path = shot_predictions_path or os.path.join(frames_dir, "shot_predictions.txt")
        final_scenes_path = shot_scenes_path or os.path.join(frames_dir, "shot_scenes.txt")
        
        # Check if shot detection files already exist
        if os.path.exists(final_predictions_path) and os.path.exists(final_scenes_path):
            print(f"[Skip] Found existing shot detection files, skipping shot detection")
            print(f"  - Predictions: {final_predictions_path}")
            print(f"  - Scenes: {final_scenes_path}")
            
            # Load existing results
            pred_arr = np.loadtxt(final_predictions_path)
            scenes = np.loadtxt(final_scenes_path, dtype=int)
            if scenes.ndim == 1 and len(scenes) > 0:
                scenes = scenes.reshape(-1, 2)
            
            result["shot_predictions_path"] = final_predictions_path
            result["shot_scenes_path"] = final_scenes_path
            result["scenes"] = scenes.tolist()
            result["shot_detection_fps"] = shot_detection_fps
            result["shot_detection_model"] = shot_detection_model
            print(f"[Shot Detection] Loaded {len(scenes)} existing scenes")
        else:
            print(f"\n[Shot Detection] Starting {shot_detection_model.upper()} shot detection at {shot_detection_fps} FPS")
            
            if shot_detection_model == "qwen3vl":
                # Use Qwen3VL for shot detection
                # Note: process_video expects frames to be present in frame_folder
                # We already extracted frames to frames_dir
                
                # Determine output folder (process_video writes to output_folder/shot_scenes.txt)
                output_folder = os.path.dirname(final_scenes_path)
                
                process_video(
                    frame_folder=frames_dir,
                    output_folder=output_folder,
                    source_fps=sample_fps,
                    target_fps=shot_detection_fps,
                    visualize=False,  # Disable visualization to save time
                )
                
                # process_video writes shot_scenes.txt to output_folder
                # We need to load it into scenes
                generated_scenes_path = os.path.join(output_folder, "shot_scenes.txt")
                
                if os.path.exists(generated_scenes_path):
                    scenes = np.loadtxt(generated_scenes_path, dtype=int)
                    if scenes.ndim == 1 and len(scenes) > 0:
                        scenes = scenes.reshape(-1, 2)

                    # Adjust scene boundaries to avoid frame overlap
                    scenes = _adjust_scene_boundaries(scenes.tolist())

                    # Save adjusted scenes
                    np.savetxt(final_scenes_path, np.array(scenes), fmt="%d")

                    # If final_scenes_path is different from generated_scenes_path, remove the old one
                    if os.path.abspath(generated_scenes_path) != os.path.abspath(final_scenes_path):
                        import shutil
                        if os.path.exists(generated_scenes_path):
                            os.remove(generated_scenes_path)

                    print(f"[Shot Detection] Scene boundaries saved to {final_scenes_path}")
                    print(f"[Shot Detection] Detected {len(scenes)} scenes")
                else:
                    print("Warning: Qwen3VL shot detection did not produce shot_scenes.txt")
                    scenes = []

                # Add to result
                result["shot_scenes_path"] = final_scenes_path
                result["scenes"] = scenes  # Already converted to list
                result["shot_detection_fps"] = shot_detection_fps
                result["shot_detection_model"] = shot_detection_model

            elif shot_detection_model == "scenedetect":
                # Use PySceneDetect's AdaptiveDetector
                print(f"[Shot Detection] Using scenedetect AdaptiveDetector "
                      f"(threshold={shot_detection_threshold}, min_scene_len={shot_detection_min_scene_len})")

                # Run scenedetect shot detection
                shot_list = _scenedetect_shot_detection(
                    video_path=video_path,
                    threshold=shot_detection_threshold,
                    min_scene_len=shot_detection_min_scene_len,
                )

                print(f"[Shot Detection] Detected {len(shot_list)} shots")

                # Convert shot_list [(start_sec, end_sec), ...] to scene boundaries format
                # Scene format is [[start_frame, end_frame], ...]
                # We use the extracted frames' fps for conversion
                scenes = []
                for start_sec, end_sec in shot_list:
                    start_frame = int(start_sec * sample_fps)
                    end_frame = int(end_sec * sample_fps)
                    scenes.append([start_frame, end_frame])

                # Adjust scene boundaries to avoid frame overlap
                scenes = _adjust_scene_boundaries(scenes)

                # Save scenes to file
                if scenes:
                    np.savetxt(final_scenes_path, np.array(scenes), fmt="%d")
                else:
                    np.savetxt(final_scenes_path, np.array([]).reshape(0, 2), fmt="%d")
                print(f"[Shot Detection] Scene boundaries saved to {final_scenes_path}")

                # Add to result
                result["shot_scenes_path"] = final_scenes_path
                result["scenes"] = scenes
                result["shot_detection_fps"] = sample_fps  # Use actual extracted fps
                result["shot_detection_model"] = shot_detection_model
                result["shot_list"] = shot_list  # Also include original time-based shot list

            else:
                # Initialize model based on choice (transnetv2 or autoshot)
                if shot_detection_model == "autoshot":
                    shot_model = AutoShotTorch()
                else:  # transnetv2
                    shot_model = TransNetV2Torch()
                
                # Calculate effective end_sec for shot detection to match frame extraction
                # This ensures shot detection processes the same duration as the extracted frames
                shot_end_sec = end_sec
                if max_frames is not None:
                    # Calculate the duration based on extracted frames and fps
                    effective_duration = num_frames / float(sample_fps)
                    calculated_end_sec = (start_sec or 0.0) + effective_duration
                    
                    # Use the minimum of user-specified end_sec and calculated end_sec
                    if shot_end_sec is None:
                        shot_end_sec = calculated_end_sec
                    else:
                        shot_end_sec = min(shot_end_sec, calculated_end_sec)
                    
                    print(f"[Shot Detection] Limiting shot detection to match frame extraction: "
                          f"{num_frames} frames @ {sample_fps} fps = {effective_duration:.2f}s "
                          f"(end_sec: {shot_end_sec:.2f}s)")
                
                # Run shot detection on the video with time range limits
                # Note: Model will extract frames internally at the specified FPS
                frames, s_pred, a_pred = shot_model.predict_video(
                    video_path, 
                    target_fps=shot_detection_fps,
                    start_sec=start_sec,
                    end_sec=shot_end_sec
                )
                
                # Save predictions (single shot and abrupt predictions)
                pred_arr = np.stack([s_pred, a_pred], axis=1)
                np.savetxt(final_predictions_path, pred_arr, fmt="%.6f")
                print(f"[Shot Detection] Predictions saved to {final_predictions_path}")
                
                # Convert predictions to scene boundaries
                print("shot_detection_threshold: ", shot_detection_threshold)
                scenes = shot_model.predictions_to_scenes(s_pred, threshold=shot_detection_threshold)

                # Adjust scene boundaries to avoid frame overlap
                scenes = _adjust_scene_boundaries(scenes.tolist())

                np.savetxt(final_scenes_path, np.array(scenes), fmt="%d")
                print(f"[Shot Detection] Scene boundaries saved to {final_scenes_path}")
                print(f"[Shot Detection] Detected {len(scenes)} scenes")

                # Add to result
                result["shot_predictions_path"] = final_predictions_path
                result["shot_scenes_path"] = final_scenes_path
                result["scenes"] = scenes  # Already converted to list
                result["shot_detection_fps"] = shot_detection_fps
                result["shot_detection_model"] = shot_detection_model

    return result
