import os
import subprocess
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
from PIL import Image

from qwen_omni_utils.v2_5.vision_process import fetch_video
from vca.transnetv2_pytorch.transnetv2_pytorch.inference import TransNetV2Torch
from vca.AutoShot.autoshot_wrapper import AutoShotTorch


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
) -> Tuple[List[str], int, int, float]:
    """
    Extract frames using ffmpeg directly (memory efficient, no batching needed).
    Returns: (frame_paths, height, width, actual_fps)
    """
    _ensure_dir(frames_dir)
    
    # Build ffmpeg command
    cmd: List[str] = ["ffmpeg", "-y"]
    
    # Input options
    if start_sec is not None:
        cmd += ["-ss", str(float(start_sec))]
    cmd += ["-i", video_path]
    if end_sec is not None:
        cmd += ["-to", str(float(end_sec))]
    
    # FPS filter - must come before max_frames
    fps = target_fps if target_fps is not None else 2.0
    vf_filters = [f"fps={fps}"]
    
    # Resolution filter
    # If target_resolution is a single number, treat it as height and keep aspect ratio
    if target_resolution is not None:
        if isinstance(target_resolution, (int, float)):
            # Single value = height, width auto-scaled to maintain aspect ratio
            h = int(target_resolution)
            vf_filters.append(f"scale=-2:{h}")  # -2 ensures width is even
        elif isinstance(target_resolution, (list, tuple)) and len(target_resolution) == 1:
            # Single element in list/tuple = height
            h = int(target_resolution[0])
            vf_filters.append(f"scale=-2:{h}")
        elif isinstance(target_resolution, (list, tuple)) and len(target_resolution) == 2:
            # Two elements = (height, width)
            h, w = int(target_resolution[0]), int(target_resolution[1])
            vf_filters.append(f"scale={w}:{h}")
    
    # Combine filters
    if vf_filters:
        cmd += ["-vf", ",".join(vf_filters)]
    
    # Limit frames if specified - MUST be after filters
    if max_frames is not None:
        cmd += ["-frames:v", str(int(max_frames))]
    
    # Output options - start frame numbering from 0
    frame_pattern = os.path.join(frames_dir, "frame_%06d.png")
    cmd += ["-start_number", "0", frame_pattern]
    
    # Execute ffmpeg
    print(f"[Batch Processing] Extracting frames with max_frames={max_frames}, fps={fps}")
    print(f"[Batch Processing] FFmpeg command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Get list of generated frames
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith("frame_") and f.endswith(".png")])
    frame_paths = [os.path.join(frames_dir, f) for f in frame_files]
    
    if not frame_paths:
        raise RuntimeError("No frames were extracted from the video")
    
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


def _format_srt_timestamp(milliseconds: int) -> str:
    """Convert milliseconds (int) to SRT timestamp HH:MM:SS,mmm."""
    ms = int(milliseconds)
    tail = ms % 1000
    s = ms // 1000
    mi = s // 60
    s = s % 60
    h = mi // 60
    mi = mi % 60
    h = f"{h:02d}"
    mi = f"{mi:02d}"
    s = f"{s:02d}"
    tail = f"{tail:03d}"
    return f"{h}:{mi}:{s},{tail}"


def _write_srt_from_sentence_info(sentence_info: List[Dict[str, Any]], srt_path: str) -> None:
    """
    Write SRT file from sentence_info structure.
    Each sentence_info item should have 'text' and 'timestamp' fields.
    timestamp is a list of [word, start_ms, end_ms] for each word.
    """
    _ensure_dir(os.path.dirname(srt_path) or ".")
    with open(srt_path, "w", encoding="utf-8") as f:
        for idx, sent in enumerate(sentence_info):
            text = sent.get('text', '')
            timestamp = sent.get('timestamp', [])
            
            if not timestamp:
                continue
            
            # Get start and end time from word-level timestamps
            start_ms = int(timestamp[0][1]) if len(timestamp[0]) >= 2 else 0
            end_ms = int(timestamp[-1][2]) if len(timestamp[-1]) >= 3 else start_ms
            
            # Clean up text (remove trailing punctuation)
            text = text.rstrip("、。，")
            
            # Write SRT entry
            f.write(f"{idx + 1}\n")
            f.write(f"{_format_srt_timestamp(start_ms)} --> {_format_srt_timestamp(end_ms)}\n")
            f.write(f"{text}\n\n")


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


def _extract_audio_wav_16k(video_path: str, audio_path: str, start_sec: Optional[float], end_sec: Optional[float]) -> None:
    """Extract mono 16k PCM WAV from video using ffmpeg."""
    cmd: List[str] = ["ffmpeg", "-y"]
    if start_sec is not None:
        cmd += ["-ss", str(float(start_sec))]
    cmd += ["-i", video_path]
    if end_sec is not None:
        cmd += ["-to", str(float(end_sec))]
    cmd += ["-vn", "-ac", "1", "-ar", "16000", "-f", "wav", audio_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _transcribe_audio_with_whisper_timestamped(
    audio_path: str,
    model_name: str,
    device: str,
    language: Optional[str] = None,
    asr_kwargs: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run whisper-timestamped on a single audio file and return a dict with keys:
      - text: full transcription
      - sentence_info: list of sentence dicts with 'text' and 'timestamp' fields
      - segments: raw segments from whisper-timestamped (with word-level timestamps)
    
    This uses whisper-timestamped for highly accurate word-level timestamps.
    """
    # Lazy import to avoid hard dependency if ASR is not requested
    try:
        import whisper_timestamped as whisper
    except ImportError:
        raise ImportError(
            "whisper-timestamped is not installed. "
            "Install it with: pip install whisper-timestamped"
        )
    
    # Load model
    model = whisper.load_model(model_name, device=device)
    
    # Load audio
    audio = whisper.load_audio(audio_path)
    
    # Base transcription kwargs
    transcribe_kwargs: Dict[str, Any] = {
        "language": language,
        "task": "transcribe",
        "vad": False,  # Enable VAD with default settings (more conservative)
        # Alternative VAD options to try:
        # "vad": False,  # Disable VAD entirely (may help with hallucinations)
        # "vad": "silero:v4.0",  # More aggressive VAD (may cause over-segmentation)
        "detect_disfluencies": False,
        "compute_word_confidence": True,  # Get confidence scores for each word
        # Additional parameters to reduce hallucinations:
        "condition_on_previous_text": False,  # Prevent context carryover between segments
        "no_speech_threshold": 0.6,  # Higher = more conservative (default 0.6)
        "logprob_threshold": -1.0,  # Filter low-confidence outputs (default -1.0)
        "compression_ratio_threshold": 2.4,  # Detect repetitive outputs (default 2.4)
    }
    
    # Merge user-provided kwargs (can override defaults)
    if asr_kwargs:
        transcribe_kwargs.update(asr_kwargs)
    
    # Run transcription with whisper-timestamped
    result = whisper.transcribe(model, audio, **transcribe_kwargs)
    
    # Extract full text
    full_text = result.get("text", "")
    
    # Convert whisper-timestamped segments to sentence_info format
    # whisper-timestamped result structure:
    # {
    #   "text": "...",
    #   "segments": [
    #     {
    #       "start": 0.5,
    #       "end": 3.2,
    #       "text": "Hello world",
    #       "words": [
    #         {"text": "Hello", "start": 0.5, "end": 1.2, "confidence": 0.95},
    #         {"text": "world", "start": 1.3, "end": 3.2, "confidence": 0.98}
    #       ]
    #     }
    #   ]
    # }
    
    sentence_info = []
    segments = result.get("segments", [])
    
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        
        words = seg.get("words", [])
        
        # Convert to timestamp format: [[word, start_ms, end_ms], ...]
        timestamp = []
        for word_info in words:
            word_text = word_info.get("text", "")
            word_start_ms = int(word_info.get("start", 0) * 1000)
            word_end_ms = int(word_info.get("end", 0) * 1000)
            timestamp.append([word_text, word_start_ms, word_end_ms])
        
        # Fallback: if no word-level timestamps, use segment-level
        if not timestamp:
            seg_start_ms = int(seg.get("start", 0) * 1000)
            seg_end_ms = int(seg.get("end", 0) * 1000)
            timestamp = [[text, seg_start_ms, seg_end_ms]]
        
        sentence_info.append({
            "text": text,
            "timestamp": timestamp
        })
    
    return {
        "text": full_text,
        "sentence_info": sentence_info,
        "segments": segments  # Keep raw segments for reference
    }


def decode_video_to_frames(
    video_path: str,
    frames_dir: str,
    target_fps: Optional[float] = None,
    target_resolution: Optional[Tuple[int, int]] = None,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    max_frames: Optional[int] = None,
    asr_to_srt: bool = False,
    srt_path: Optional[str] = None,
    asr_model: str = "base",
    asr_device: str = "cuda:0",
    asr_language: Optional[str] = None,
    asr_kwargs: Optional[Dict[str, Any]] = None,
    keep_extracted_audio: bool = False,
    batch_size: Optional[int] = None,
    use_batch_processing: bool = True,
    shot_detection: bool = False,
    shot_detection_model: str = "transnetv2",
    shot_detection_fps: float = 2.0,
    shot_detection_threshold: float = 0.5,
    shot_predictions_path: Optional[str] = None,
    shot_scenes_path: Optional[str] = None,
) -> dict:
    """
    Read a video and save sampled frames to disk using vision_process.fetch_video.

    Args:
        video_path: Path to the video file.
        frames_dir: Directory to save frames.
        target_fps: Desired sampling fps. If None, uses default from vision_process (2.0).
        target_resolution: (height, width). If None, uses smart_resize based on defaults.
        start_sec: Optional start time in seconds.
        end_sec: Optional end time in seconds.
        max_frames: Maximum number of frames to extract. If None, uses default (no extra limit beyond library default).
        asr_to_srt: If True, also extract audio and transcribe to an SRT file using whisper-timestamped.
        srt_path: Optional output path for the SRT. Defaults to video basename with .srt next to video.
        asr_model: Whisper model name: "tiny", "base", "small", "medium", "large", "large-v2", "large-v3".
        asr_device: Device for ASR model, e.g., "cuda:0" or "cpu".
        asr_language: Language code for ASR (e.g., "zh", "en"). If None, auto-detect.
        asr_kwargs: Extra kwargs forwarded to whisper-timestamped's transcribe() method.
        keep_extracted_audio: If True, keep the temporary extracted WAV file on disk.
        batch_size: Number of frames to process at once. If None, defaults to 500 frames per batch.
        use_batch_processing: If True, use batch processing to avoid loading entire video into memory at once.
        shot_detection: If True, perform shot/scene detection.
        shot_detection_model: Model to use for shot detection: "transnetv2" or "autoshot" (default: "transnetv2").
        shot_detection_fps: FPS for shot detection (default: 2.0).
        shot_detection_threshold: Threshold for shot boundary detection (default: 0.5).
            Recommended values: 0.5 for TransNetV2, 0.296 for AutoShot.
        shot_predictions_path: Output path for shot predictions. Defaults to frames_dir/shot_predictions.txt.
        shot_scenes_path: Output path for scene boundaries. Defaults to frames_dir/shot_scenes.txt.

    Returns:
        dict with keys: {"num_frames", "sample_fps", "height", "width", "frame_paths", 
                        "srt_path" (optional), "segments" (optional),
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
    print(f"ASR to SRT: {srt_path}")
    if asr_to_srt:
        # Determine output SRT path
        final_srt_path = srt_path or (os.path.splitext(video_path)[0] + ".srt")
        
        # Check if SRT file already exists
        if os.path.exists(final_srt_path):
            print(f"[Skip] Found existing SRT file at {final_srt_path}, skipping ASR")
            result["srt_path"] = final_srt_path
            # Try to read existing SRT file for sentence_info (optional)
            # For now, we just skip and don't populate sentence_info/segments
        else:
            # Calculate effective end_sec for ASR based on max_frames and actual extracted frames
            # This ensures ASR processes the same duration as the extracted frames
            asr_end_sec = end_sec
            if max_frames is not None:
                # Calculate the duration based on extracted frames and fps
                effective_duration = num_frames / float(sample_fps)
                calculated_end_sec = (start_sec or 0.0) + effective_duration
                
                # Use the minimum of user-specified end_sec and calculated end_sec
                if asr_end_sec is None:
                    asr_end_sec = calculated_end_sec
                else:
                    asr_end_sec = min(asr_end_sec, calculated_end_sec)
                
                print(f"[ASR] Limiting audio extraction to match frame extraction: "
                      f"{num_frames} frames @ {sample_fps} fps = {effective_duration:.2f}s "
                      f"(end_sec: {asr_end_sec:.2f}s)")
            
            # Extract audio to temporary WAV inside frames_dir for locality
            audio_wav_path = os.path.join(frames_dir, "audio_16k_mono.wav")
            _extract_audio_wav_16k(video_path, audio_wav_path, start_sec, asr_end_sec)

            # Run whisper-timestamped ASR
            asr_output = _transcribe_audio_with_whisper_timestamped(
                audio_wav_path, asr_model, asr_device, asr_language, asr_kwargs
            )
            
            sentence_info = asr_output.get("sentence_info", [])
            
            # Adjust timestamps if we extracted a clip (add start_sec offset)
            if start_sec is not None and start_sec > 0:
                offset_ms = int(float(start_sec) * 1000)
                adjusted_sentence_info = []
                for sent in sentence_info:
                    adjusted_sent = sent.copy()
                    # Adjust word-level timestamps in the timestamp array
                    if 'timestamp' in sent:
                        adjusted_timestamps = []
                        for ts in sent['timestamp']:
                            # Format: [word, start_ms, end_ms]
                            if isinstance(ts, (list, tuple)) and len(ts) >= 3:
                                adjusted_timestamps.append([
                                    ts[0],  # word
                                    ts[1] + offset_ms,  # start_ms + offset
                                    ts[2] + offset_ms   # end_ms + offset
                                ])
                            else:
                                adjusted_timestamps.append(ts)
                        adjusted_sent['timestamp'] = adjusted_timestamps
                    adjusted_sentence_info.append(adjusted_sent)
                sentence_info = adjusted_sentence_info
            
            # Write SRT file using sentence_info
            _write_srt_from_sentence_info(sentence_info, final_srt_path)
            result["srt_path"] = final_srt_path
            result["sentence_info"] = sentence_info  # Return sentence_info for further processing if needed
            
            # Also save raw segments with word-level timestamps
            if "segments" in asr_output:
                result["segments"] = asr_output["segments"]

            if not keep_extracted_audio:
                try:
                    os.remove(audio_wav_path)
                except OSError:
                    pass

    # Shot detection using TransNetV2 or AutoShot
    if shot_detection:
        # Normalize model name
        shot_detection_model = shot_detection_model.lower()
        if shot_detection_model not in ["transnetv2", "autoshot"]:
            raise ValueError(f"Invalid shot_detection_model: {shot_detection_model}. Must be 'transnetv2' or 'autoshot'")
        
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
            
            # Initialize model based on choice
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
            np.savetxt(final_scenes_path, scenes, fmt="%d")
            print(f"[Shot Detection] Scene boundaries saved to {final_scenes_path}")
            print(f"[Shot Detection] Detected {len(scenes)} scenes")
            
            # Add to result
            result["shot_predictions_path"] = final_predictions_path
            result["shot_scenes_path"] = final_scenes_path
            result["scenes"] = scenes.tolist()  # Convert numpy array to list for JSON serialization
            result["shot_detection_fps"] = shot_detection_fps
            result["shot_detection_model"] = shot_detection_model

    return result
