import os
import json
import copy
import re
import numpy as np
from typing import Annotated as A, Optional, Tuple, List
import cv2
from vca.build_database.video_caption import (
    convert_seconds_to_hhmmss,
    call_vllm_model as call_vllm_model_for_images,  # For single image detection
)
from vca import config
from vca.func_call_shema import doc as D
from vca.vllm_calling import call_vllm_model as call_vllm_model_for_video, get_vllm_embeddings

# Import face_recognition (will be imported lazily to avoid dependency issues)
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("Warning: face_recognition library not available. Install with: pip install face_recognition")


class StopException(Exception):
    """
    Stop Execution by raising this exception (Signal that the task is Finished).
    """

def Review_timeline(timeline, ):
    """
    Review Execution by raising this exception (Signal that the task is Finished).
    """

def Review_audio_video_alignment(alignment, ):
    """
    Review Execution by raising this exception (Signal that the task is Finished).
    """


def review_clip(
    time_range: A[str, D("The time range to check (e.g., '00:13:28 to 00:13:40').")],
    used_time_ranges: A[list, D("List of already used time ranges. Auto-injected.")] = None
) -> str:
    """
    Check if the proposed time range overlaps with any previously used clips.
    You MUST call this tool BEFORE calling finish to ensure no duplicate footage.

    Returns:
        str: A message indicating whether the time range is available or overlaps with used clips.
             If overlap is detected, you should select a different time range.
    """
    def hhmmss_to_seconds(time_str: str) -> float:
        """Convert HH:MM:SS, HH:MM:SS.s, HH:MM:SS:FF, or MM:SS to seconds."""
        parts = time_str.strip().split(':')
        if len(parts) == 4:
            # HH:MM:SS:FF format (with frame number)
            h, m, s, f = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            # Import config to get VIDEO_FPS
            from vca import config
            fps = getattr(config, 'VIDEO_FPS', 24) or 24
            return h * 3600 + m * 60 + s + (f / fps)
        elif len(parts) == 3:
            h, m = int(parts[0]), int(parts[1])
            s = float(parts[2])
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m = int(parts[0])
            s = float(parts[1])
            return m * 60 + s
        else:
            return float(parts[0])

    if used_time_ranges is None:
        used_time_ranges = []

    # Parse the time range
    match = re.search(r'([\d:.]+)\s+to\s+([\d:.]+)', time_range, re.IGNORECASE)
    if not match:
        return f"Error: Could not parse time range '{time_range}'. Please use format 'HH:MM:SS to HH:MM:SS'."

    try:
        start_sec = hhmmss_to_seconds(match.group(1))
        end_sec = hhmmss_to_seconds(match.group(2))
    except Exception as e:
        return f"Error parsing time range: {e}"

    if not used_time_ranges:
        return f"✅ OK: Time range {time_range} is available. No previous clips have been used yet. You can proceed with finish."

    # Check for overlaps
    overlapping_clips = []
    for idx, (used_start, used_end) in enumerate(used_time_ranges):
        if start_sec < used_end and end_sec > used_start:
            overlap_start = max(start_sec, used_start)
            overlap_end = min(end_sec, used_end)
            overlapping_clips.append({
                "clip_idx": idx + 1,
                "used_range": f"{convert_seconds_to_hhmmss(used_start)} to {convert_seconds_to_hhmmss(used_end)}",
                "overlap": f"{convert_seconds_to_hhmmss(overlap_start)} to {convert_seconds_to_hhmmss(overlap_end)}"
            })

    if overlapping_clips:
        result = f"❌ OVERLAP DETECTED: Time range {time_range} overlaps with {len(overlapping_clips)} previously used clip(s):\n"
        for clip in overlapping_clips:
            result += f"  - Clip {clip['clip_idx']}: {clip['used_range']} (overlap: {clip['overlap']})\n"
        result += "\n⚠️ Please select a DIFFERENT time range to avoid duplicate footage. Do NOT call finish with this range."
        return result
    else:
        return f"✅ OK: Time range {time_range} does not overlap with any previously used clips. You can proceed with finish."


def review_finish(
    answer: A[str, D("Output the final shot time range. Must be exactly ONE continuous clip.")],
    target_length_sec: A[float, D("Expected total length in seconds")] = 0.0,
) -> str:
    """
    Review and validate the proposed shot selection before finishing.
    Validates that exactly ONE shot is provided and its duration matches the target.
    You MUST call this tool BEFORE calling finish to ensure the shot is valid.

    IMPORTANT: Only accepts ONE continuous time range. Multiple shots will be rejected.
    Example: [shot: 00:10:00 to 00:10:07.3] for a 7.3s target duration.

    Returns:
        str: Success message if validation passes, or error message if validation fails.
             If validation fails, you should adjust your shot selection.
    """
    def hhmmss_to_seconds(time_str: str) -> float:
        """Convert HH:MM:SS, HH:MM:SS.s, HH:MM:SS:FF, or MM:SS to seconds."""
        parts = time_str.strip().split(':')
        if len(parts) == 4:
            # HH:MM:SS:FF format (with frame number)
            h, m, s, f = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            # Import config to get VIDEO_FPS
            from vca import config
            fps = getattr(config, 'VIDEO_FPS', 24) or 24
            return h * 3600 + m * 60 + s + (f / fps)
        elif len(parts) == 3:
            h, m = int(parts[0]), int(parts[1])
            s = float(parts[2])
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m = int(parts[0])
            s = float(parts[1])
            return m * 60 + s
        else:
            return float(parts[0])

    def seconds_to_hhmmss(sec: float) -> str:
        """Convert seconds to HH:MM:SS.s format."""
        hours = int(sec // 3600)
        minutes = int((sec % 3600) // 60)
        seconds = sec % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:04.1f}"

    # Parse the answer to extract shot time ranges
    # Expected formats: "[shot: 00:10:00 to 00:10:05]" or "shot 1: 00:10:00 to 00:10:05"
    shot_pattern = re.compile(r'\[?shot[\s_]*\d*:\s*([0-9:.]+)\s+to\s+([0-9:.]+)\]?', re.IGNORECASE)
    matches = shot_pattern.findall(answer)

    if not matches:
        return "❌ Error: Could not parse shot time ranges from the answer. Please provide time range(s) in the format: [shot: HH:MM:SS to HH:MM:SS]"

    # Allow multiple shots for stitching (with reasonable limit)
    from vca import config
    max_shots_allowed = getattr(config, 'MAX_SHOTS_PER_CLIP', 3)
    if len(matches) > max_shots_allowed:
        return (
            f"❌ Error: You provided {len(matches)} shots, but maximum allowed is {max_shots_allowed}. "
            f"Please reduce the number of stitched shots or combine them into fewer segments."
        )

    # Calculate total duration and collect clips
    clips = []
    total_duration = 0

    for i, (start_time, end_time) in enumerate(matches, 1):
        try:
            start_sec = hhmmss_to_seconds(start_time)
            end_sec = hhmmss_to_seconds(end_time)
            duration = end_sec - start_sec

            if duration <= 0:
                return f"❌ Error: Shot {i} has invalid duration (start: {start_time}, end: {end_time}). End time must be greater than start time."

            clips.append({
                'start': start_time,
                'end': end_time,
                'start_sec': start_sec,
                'end_sec': end_sec,
                'duration': duration
            })
            total_duration += duration
        except Exception as e:
            return f"❌ Error parsing shot {i} time range ({start_time} to {end_time}): {str(e)}"

    # Validate continuity for multi-shot stitching
    if len(clips) > 1:
        max_gap = getattr(config, 'MAX_STITCH_GAP_SEC', 2.0)
        for i in range(len(clips) - 1):
            gap = clips[i+1]['start_sec'] - clips[i]['end_sec']
            if gap < 0:
                return f"❌ Error: Overlapping shots. Shot {i+1} ends at {clips[i]['end']}, but shot {i+2} starts at {clips[i+1]['start']}"
            if gap > max_gap:
                return (
                    f"❌ Error: Time gap ({gap:.2f}s) between shot {i+1} and {i+2} exceeds maximum ({max_gap}s).\n"
                    f"Stitched shots must maintain visual continuity. Please select closer shots or use a single continuous clip."
                )

    # Check if total duration matches target length (allow tolerance)
    duration_diff = total_duration - target_length_sec

    # Prepare duration summary
    if len(clips) == 1:
        duration_line = f"shot: {clips[0]['start']} to {clips[0]['end']} ({clips[0]['duration']:.2f}s)"
    else:
        duration_line = f"{len(clips)} stitched shots (total {total_duration:.2f}s):\n"
        for i, clip in enumerate(clips, 1):
            duration_line += f"  Shot {i}: {clip['start']} to {clip['end']} ({clip['duration']:.2f}s)\n"

    # Check for very short clips
    min_acceptable = getattr(config, 'MIN_ACCEPTABLE_SHOT_DURATION', 2.0)
    short_clips = [c for c in clips if c['duration'] < min_acceptable]
    short_warning = ""
    if short_clips:
        short_warning = f"\n⚠️ Warning: {len(short_clips)} shot(s) shorter than {min_acceptable}s - consider using longer clips if possible."

    # Allow flexible tolerance
    tolerance = getattr(config, 'ALLOW_DURATION_TOLERANCE', 1.0)
    if abs(duration_diff) > tolerance:
        if duration_diff > 0:
            action = "shorten"
            suggestion = f"Try trimming {duration_diff:.2f}s from the end."
        else:
            action = "extend"
            suggestion = f"Try adding {abs(duration_diff):.2f}s more footage."

        return (
            f"❌ Error: Duration mismatch! Your total duration is {total_duration:.2f}s but target is {target_length_sec:.2f}s.\n"
            f"Current selection:\n{duration_line}"
            f"Difference: {abs(duration_diff):.2f}s ({action} needed)\n"
            f"Suggestion: {suggestion}{short_warning}\n"
            f"⚠️ Please adjust your shot selection before calling finish."
        )

    # If duration exceeds target by small amount, provide trimming suggestion
    tolerance = getattr(config, 'ALLOW_DURATION_TOLERANCE', 1.0)
    if 0 < duration_diff <= tolerance:
        new_end_sec = clips[-1]['end_sec'] - duration_diff
        new_end = seconds_to_hhmmss(new_end_sec)
        return (
            f"✅ OK: Shot validation passed (will auto-trim {duration_diff:.2f}s from last clip).\n"
            f"Current selection:\n{duration_line}"
            f"Target duration: {target_length_sec:.2f}s\n"
            f"Auto-adjusted end time: {new_end}\n"
            f"You can proceed with finish.{short_warning}"
        )

    # Validation passed
    status_msg = "✅ OK: Shot validation passed.\n"
    if len(clips) > 1:
        status_msg += f"✓ {len(clips)} shots stitched successfully with proper continuity\n"

    return (
        f"{status_msg}"
        f"Current selection:\n{duration_line}"
        f"Target duration: {target_length_sec:.2f}s\n"
        f"Duration match: ✓{short_warning}\n"
        f"You can proceed with finish."
    )



class ReviewerAgent:
    """
    ReviewerAgent 用于审核 DVDCoreAgent 生成的 shot 选择。
    在 Core 调用 finish 之前，先通过 Reviewer 审核结果。
    """

    def __init__(self, frame_folder_path=None, video_path=None):
        """
        初始化 ReviewerAgent。

        Args:
            frame_folder_path: 视频帧文件夹路径，用于审核时查看帧内容
            video_path: 视频文件路径
        """
        self.frame_folder_path = frame_folder_path
        self.video_path = video_path


    def check_face_quality(
        self,
        video_path: A[str, D("Path to the video file.")],
        time_range: A[str, D("The time range to check (e.g., '00:13:28 to 00:13:40').")],
        max_break_ratio: A[float, D("Maximum allowed break time ratio (0.0-1.0). Default: 0.3 (30%)")] = 0.3,
        min_face_size: A[int, D("Minimum face size in pixels. Faces smaller than this are considered too small. Default: 50")] = 50,
        sample_fps: A[float, D("Sampling frame rate for face detection. Default: 2.0")] = 2.0,
    ) -> str:
        """
        Check face quality in a video clip using face_recognition library.
        Records "break time" when:
        1. Face is too small (smaller than min_face_size pixels)
        2. No face detected

        Calculates break_time/total_time ratio and returns error if it exceeds max_break_ratio.

        Args:
            video_path: Path to the video file
            time_range: Time range in format "HH:MM:SS to HH:MM:SS" or "MM:SS to MM:SS"
            max_break_ratio: Maximum allowed break time ratio (default: 0.3 = 30%)
            min_face_size: Minimum face size in pixels (default: 50)
            sample_fps: Frame sampling rate for detection (default: 2.0 fps)

        Returns:
            str: Success message if face quality is acceptable, or error message if break ratio is too high.

        Example:
            >>> check_face_quality("/path/to/video.mp4", "00:10:00 to 00:10:10", max_break_ratio=0.3, min_face_size=50)
        """
        if not FACE_RECOGNITION_AVAILABLE:
            return "❌ Error: face_recognition library is not installed. Please install with: pip install face_recognition"

        def hhmmss_to_seconds(time_str: str) -> float:
            """Convert HH:MM:SS, HH:MM:SS.s, HH:MM:SS:FF, or MM:SS to seconds."""
            parts = time_str.strip().split(':')
            if len(parts) == 4:
                # HH:MM:SS:FF format (with frame number)
                h, m, s, f = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                fps = getattr(config, 'VIDEO_FPS', 24) or 24
                return h * 3600 + m * 60 + s + (f / fps)
            elif len(parts) == 3:
                h, m = int(parts[0]), int(parts[1])
                s = float(parts[2])
                return h * 3600 + m * 60 + s
            elif len(parts) == 2:
                m = int(parts[0])
                s = float(parts[1])
                return m * 60 + s
            else:
                return float(parts[0])

        # Parse time range
        match = re.search(r'([\d:.]+)\s+to\s+([\d:.]+)', time_range, re.IGNORECASE)
        if not match:
            return f"❌ Error: Could not parse time range '{time_range}'. Please use format 'HH:MM:SS to HH:MM:SS'."

        try:
            start_sec = hhmmss_to_seconds(match.group(1))
            end_sec = hhmmss_to_seconds(match.group(2))
            duration_sec = end_sec - start_sec

            if duration_sec <= 0:
                return f"❌ Error: Invalid time range. End time must be greater than start time."
        except Exception as e:
            return f"❌ Error parsing time range: {e}"

        # Prefer using pre-extracted frames to avoid reopening/decoding the video.
        frames_dir = self.frame_folder_path if self.frame_folder_path and os.path.isdir(self.frame_folder_path) else None

        try:
            # Track break time and total time
            break_frames = 0
            total_sampled_frames = 0
            face_details = []

            if frames_dir:
                frames_fps = float(getattr(config, "VIDEO_FPS", 2) or 2.0)
                effective_sample_fps = float(sample_fps) if float(sample_fps) > 0 else 2.0
                step_sec = 1.0 / effective_sample_fps

                import glob
                import math

                n = int(math.ceil(duration_sec / step_sec))
                print(f"[Face Detection] Using frames_dir={frames_dir}; sampling ~{max(1, n)} frames from {time_range}...")

                processed_frame_indices = set()  # Track processed frames to avoid duplicates

                for i in range(max(1, n)):
                    t = start_sec + i * step_sec
                    # Changed to > to include the end boundary (was >=)
                    if t > end_sec:
                        break

                    frame_idx = int(round(t * frames_fps))

                    # Skip if we already processed this frame
                    if frame_idx in processed_frame_indices:
                        continue
                    processed_frame_indices.add(frame_idx)

                    stem = f"frame_{frame_idx:06d}"

                    frame_path = None
                    for ext in (".png", ".jpg", ".jpeg"):
                        p = os.path.join(frames_dir, stem + ext)
                        if os.path.exists(p):
                            frame_path = p
                            break
                    if frame_path is None:
                        matches = glob.glob(os.path.join(frames_dir, stem + ".*"))
                        if matches:
                            frame_path = matches[0]

                    if frame_path is None:
                        continue

                    frame = cv2.imread(frame_path)
                    if frame is None:
                        continue

                    total_sampled_frames += 1
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_frame, model="hog")

                    is_break_frame = False
                    reason = ""
                    status = "✅ OK"

                    if len(face_locations) == 0:
                        is_break_frame = True
                        reason = "no_face"
                        status = "❌ NO_FACE"
                    else:
                        max_face_size = 0
                        face_count = len(face_locations)
                        for (top, right, bottom, left) in face_locations:
                            face_height = bottom - top
                            face_width = right - left
                            face_size = min(face_height, face_width)
                            max_face_size = max(max_face_size, face_size)

                        if max_face_size < min_face_size:
                            is_break_frame = True
                            reason = f"face_too_small ({max_face_size}px < {min_face_size}px)"
                            status = f"❌ TOO_SMALL"
                        else:
                            reason = f"face_ok ({face_count} face(s), max_size={max_face_size}px)"
                            status = f"✅ OK"

                    # Print frame-by-frame detection results
                    time_at_frame = frame_idx / frames_fps  # Use actual frame time
                    print(f"  Frame {frame_idx:6d} | Time: {time_at_frame:7.2f}s | {status:15s} | {reason}")

                    if is_break_frame:
                        break_frames += 1
                        face_details.append({
                            "frame": frame_idx,
                            "time": time_at_frame,
                            "reason": reason
                        })

            else:
                # Fallback: decode from video
                if not os.path.exists(video_path):
                    return f"❌ Error: Video file not found: {video_path}"

                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    return f"❌ Error: Could not open video file: {video_path}"

                video_fps = cap.get(cv2.CAP_PROP_FPS)

                start_frame = int(start_sec * video_fps)
                end_frame = int(end_sec * video_fps)

                frame_interval = int(video_fps / float(sample_fps or 2.0))
                if frame_interval < 1:
                    frame_interval = 1

                frame_indices = list(range(start_frame, end_frame, frame_interval))
                if not frame_indices:
                    cap.release()
                    return f"❌ Error: No frames to process in the specified time range."

                print(f"[Face Detection] Decoding video; processing {len(frame_indices)} frames from {time_range}...")

                for idx, frame_idx in enumerate(frame_indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()

                    if not ret:
                        continue

                    total_sampled_frames += 1
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_frame, model="hog")

                    is_break_frame = False
                    reason = ""
                    status = "✅ OK"

                    if len(face_locations) == 0:
                        is_break_frame = True
                        reason = "no_face"
                        status = "❌ NO_FACE"
                    else:
                        max_face_size = 0
                        face_count = len(face_locations)
                        for (top, right, bottom, left) in face_locations:
                            face_height = bottom - top
                            face_width = right - left
                            face_size = min(face_height, face_width)
                            max_face_size = max(max_face_size, face_size)

                        if max_face_size < min_face_size:
                            is_break_frame = True
                            reason = f"face_too_small ({max_face_size}px < {min_face_size}px)"
                            status = f"❌ TOO_SMALL"
                        else:
                            reason = f"face_ok ({face_count} face(s), max_size={max_face_size}px)"
                            status = f"✅ OK"

                    # Print frame-by-frame detection results
                    time_at_frame = frame_idx / video_fps
                    print(f"  Frame {frame_idx:6d} | Time: {time_at_frame:7.2f}s | {status:15s} | {reason}")

                    if is_break_frame:
                        break_frames += 1
                        face_details.append({
                            "frame": frame_idx,
                            "time": time_at_frame,
                            "reason": reason
                        })

                cap.release()

            # Calculate break time ratio
            if total_sampled_frames == 0:
                return f"❌ Error: No frames were successfully processed."

            break_ratio = break_frames / total_sampled_frames
            break_time_sec = (break_frames / total_sampled_frames) * duration_sec

            # Prepare result message
            result_msg = f"\n[Face Quality Check Results]\n"
            result_msg += f"Time range: {time_range} ({duration_sec:.2f}s)\n"
            result_msg += f"Sampled frames: {total_sampled_frames} @ {sample_fps} fps\n"
            result_msg += f"Break frames: {break_frames}/{total_sampled_frames}\n"
            result_msg += f"Break time: {break_time_sec:.2f}s / {duration_sec:.2f}s ({break_ratio * 100:.1f}%)\n"
            result_msg += f"Threshold: {max_break_ratio * 100:.1f}%\n"

            # Check if break ratio exceeds threshold
            if break_ratio > max_break_ratio:
                result_msg += f"\n❌ FAILED: Break time ratio ({break_ratio * 100:.1f}%) exceeds maximum allowed ({max_break_ratio * 100:.1f}%)\n"

                # Show some examples of break frames
                if face_details:
                    result_msg += f"\nBreak frame examples (first 5):\n"
                    for detail in face_details[:5]:
                        result_msg += f"  - Frame {detail['frame']} ({detail['time']:.2f}s): {detail['reason']}\n"

                result_msg += f"\n⚠️ This shot has too many frames without proper faces. Please select a different shot."
                print(result_msg)
                return result_msg
            else:
                result_msg += f"\n✅ PASSED: Break time ratio ({break_ratio * 100:.1f}%) is within acceptable range.\n"
                result_msg += f"Face quality is acceptable. You can proceed with this shot."
                print(result_msg)
                return result_msg

        except Exception as e:
            return f"❌ Error during face detection: {str(e)}"


    def check_face_quality_vlm(
        self,
        video_path: A[str, D("Path to the video file.")],
        time_range: A[str, D("The time range to check (e.g., '00:13:28 to 00:13:40').")],
        main_character_name: A[str, D("Name of the main character/protagonist to look for. Default: 'the main character'")] = "the main character",
        min_protagonist_ratio: A[float, D("Minimum required ratio of frames where protagonist is the main focus (0.0-1.0). Default: 0.7 (70%)")] = 0.7,
        sample_fps: A[float, D("Sampling frame rate for analysis. Default: 1.0")] = 1.0,
        min_box_size: A[int, D("Minimum bounding box size in pixels. Default: 50")] = 50,
    ) -> str:
        """
        Check face quality using VLM frame-by-frame detection (same logic as check_face_quality but using VLM).

        This function loops through frames at sample_fps intervals, calls VLM to detect the protagonist
        in each frame and get bounding box coordinates, then calculates break_ratio based on detection results.

        Args:
            video_path: Path to the video file
            time_range: Time range in format "HH:MM:SS to HH:MM:SS" or "MM:SS to MM:SS"
            main_character_name: Name of the main character to detect (default: "the main character")
            min_protagonist_ratio: Minimum required ratio of non-break frames (default: 0.7 = 70%)
            sample_fps: Frame sampling rate for detection (default: 1.0 fps)
            min_box_size: Minimum bounding box size in pixels (default: 50)

        Returns:
            str: Success message if protagonist ratio is acceptable, or error message with details.

        Example:
            >>> check_face_quality_vlm("/path/to/video.mp4", "00:10:00 to 00:10:10", "Bruce Wayne", min_protagonist_ratio=0.7)
        """
        def hhmmss_to_seconds(time_str: str) -> float:
            """Convert HH:MM:SS, HH:MM:SS.s, HH:MM:SS:FF, or MM:SS to seconds."""
            parts = time_str.strip().split(':')
            if len(parts) == 4:
                # HH:MM:SS:FF format (with frame number)
                h, m, s, f = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                fps = getattr(config, 'VIDEO_FPS', 24) or 24
                return h * 3600 + m * 60 + s + (f / fps)
            elif len(parts) == 3:
                h, m = int(parts[0]), int(parts[1])
                s = float(parts[2])
                return h * 3600 + m * 60 + s
            elif len(parts) == 2:
                m = int(parts[0])
                s = float(parts[1])
                return m * 60 + s
            else:
                return float(parts[0])

        # Parse time range
        match = re.search(r'([\d:.]+)\s+to\s+([\d:.]+)', time_range, re.IGNORECASE)
        if not match:
            return f"❌ Error: Could not parse time range '{time_range}'. Please use format 'HH:MM:SS to HH:MM:SS'."

        try:
            start_sec = hhmmss_to_seconds(match.group(1))
            end_sec = hhmmss_to_seconds(match.group(2))
            duration_sec = end_sec - start_sec

            if duration_sec <= 0:
                return f"❌ Error: Invalid time range. End time must be greater than start time."
        except Exception as e:
            return f"❌ Error parsing time range: {e}"

        if not os.path.exists(video_path):
            return f"❌ Error: Video file not found: {video_path}"

        print(f"[VLM Frame-by-Frame Detection] Analyzing {time_range} ({duration_sec:.2f}s)...")
        print(f"[VLM Frame-by-Frame Detection] Looking for: {main_character_name}")
        print(f"[VLM Frame-by-Frame Detection] Minimum protagonist ratio: {min_protagonist_ratio * 100:.1f}%")

        # Prefer using pre-extracted frames
        frames_dir = self.frame_folder_path if self.frame_folder_path and os.path.isdir(self.frame_folder_path) else None

        try:
            import tempfile
            break_frames = 0
            total_sampled_frames = 0
            detection_details = []

            if frames_dir:
                # Use pre-extracted frames
                frames_fps = float(getattr(config, "VIDEO_FPS", 2) or 2.0)
                effective_sample_fps = float(sample_fps) if float(sample_fps) > 0 else 1.0
                step_sec = 1.0 / effective_sample_fps

                import glob
                import math

                n = int(math.ceil(duration_sec / step_sec))
                print(f"[VLM Detection] Using frames_dir={frames_dir}; sampling ~{max(1, n)} frames...")

                processed_frame_indices = set()

                for i in range(max(1, n)):
                    t = start_sec + i * step_sec
                    # Changed to > to include the end boundary (was >=)
                    if t > end_sec:
                        break

                    frame_idx = int(round(t * frames_fps))

                    if frame_idx in processed_frame_indices:
                        continue
                    processed_frame_indices.add(frame_idx)

                    stem = f"frame_{frame_idx:06d}"
                    frame_path = None
                    for ext in (".png", ".jpg", ".jpeg"):
                        p = os.path.join(frames_dir, stem + ext)
                        if os.path.exists(p):
                            frame_path = p
                            break

                    if frame_path is None:
                        matches = glob.glob(os.path.join(frames_dir, stem + ".*"))
                        if matches:
                            frame_path = matches[0]

                    if frame_path is None:
                        continue

                    total_sampled_frames += 1

                    # Call VLM to detect protagonist in current frame
                    detection_result = self._detect_protagonist_in_frame_vlm(
                        frame_path, main_character_name, min_box_size
                    )

                    is_break_frame = detection_result["is_break"]
                    reason = detection_result["reason"]
                    status = "❌" if is_break_frame else "✅"

                    time_at_frame = frame_idx / frames_fps
                    print(f"  Frame {frame_idx:6d} | Time: {time_at_frame:7.2f}s | {status:15s} | {reason}")

                    if is_break_frame:
                        break_frames += 1
                        detection_details.append({
                            "frame": frame_idx,
                            "time": time_at_frame,
                            "reason": reason
                        })

            else:
                # Decode from video
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    return f"❌ Error: Could not open video file: {video_path}"

                video_fps = cap.get(cv2.CAP_PROP_FPS)
                start_frame = int(start_sec * video_fps)
                end_frame = int(end_sec * video_fps)

                frame_interval = int(video_fps / float(sample_fps or 1.0))
                if frame_interval < 1:
                    frame_interval = 1

                frame_indices = list(range(start_frame, end_frame, frame_interval))
                if not frame_indices:
                    cap.release()
                    return f"❌ Error: No frames to process in the specified time range."

                print(f"[VLM Detection] Decoding video; processing {len(frame_indices)} frames...")

                temp_dir = tempfile.mkdtemp()

                try:
                    for frame_idx in frame_indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = cap.read()

                        if not ret:
                            continue

                        total_sampled_frames += 1

                        # Save temporary frame
                        temp_frame_path = os.path.join(temp_dir, f"temp_frame_{frame_idx}.jpg")
                        cv2.imwrite(temp_frame_path, frame)

                        # Call VLM detection
                        detection_result = self._detect_protagonist_in_frame_vlm(
                            temp_frame_path, main_character_name, min_box_size
                        )

                        is_break_frame = detection_result["is_break"]
                        reason = detection_result["reason"]
                        status = "❌" if is_break_frame else "✅"

                        time_at_frame = frame_idx / video_fps
                        print(f"  Frame {frame_idx:6d} | Time: {time_at_frame:7.2f}s | {status:15s} | {reason}")

                        if is_break_frame:
                            break_frames += 1
                            detection_details.append({
                                "frame": frame_idx,
                                "time": time_at_frame,
                                "reason": reason
                            })

                        # Clean up temporary file
                        os.remove(temp_frame_path)

                finally:
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)

                cap.release()

            # Calculate break ratio
            if total_sampled_frames == 0:
                return f"❌ Error: No frames were successfully processed."

            break_ratio = break_frames / total_sampled_frames
            non_break_ratio = 1.0 - break_ratio

            # Prepare result message
            result_msg = f"\n[VLM Face Quality Check Results (Frame-by-Frame)]\n"
            result_msg += f"Time range: {time_range} ({duration_sec:.2f}s)\n"
            result_msg += f"Character: {main_character_name}\n"
            result_msg += f"Sampled frames: {total_sampled_frames} @ {sample_fps} fps\n"
            result_msg += f"Break frames: {break_frames}/{total_sampled_frames}\n"
            result_msg += f"Protagonist ratio: {non_break_ratio * 100:.1f}%\n"
            result_msg += f"Required ratio: {min_protagonist_ratio * 100:.1f}%\n"

            # Check if ratio meets threshold
            if non_break_ratio < min_protagonist_ratio:
                result_msg += f"\n❌ FAILED: Protagonist ratio ({non_break_ratio * 100:.1f}%) is below minimum threshold ({min_protagonist_ratio * 100:.1f}%)\n"

                if detection_details:
                    result_msg += f"\nBreak frame examples (first 5):\n"
                    for detail in detection_details[:5]:
                        result_msg += f"  - Frame {detail['frame']} ({detail['time']:.2f}s): {detail['reason']}\n"

                result_msg += f"\n⚠️ This shot does not maintain sufficient focus on {main_character_name}. Please select a different shot."
                print(result_msg)
                return result_msg
            else:
                result_msg += f"\n✅ PASSED: Protagonist ratio ({non_break_ratio * 100:.1f}%) meets the minimum threshold.\n"
                result_msg += f"Shot maintains good focus on {main_character_name}. You can proceed with this shot."
                print(result_msg)
                return result_msg

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"❌ Error during VLM frame-by-frame detection: {str(e)}"


    def _detect_protagonist_in_frame_vlm(
        self,
        frame_path: str,
        main_character_name: str,
        min_box_size: int
    ) -> dict:
        """
        Call VLM to detect protagonist in a single frame and return bounding box coordinates.

        Args:
            frame_path: Path to the frame image file
            main_character_name: Name of the main character to detect
            min_box_size: Minimum bounding box size in pixels

        Returns:
            dict: {
                "is_break": bool,        # Whether this frame is a "break" (no protagonist or too small)
                "reason": str,           # Explanation of the detection result
                "bounding_box": dict or None  # {"x": int, "y": int, "width": int, "height": int}
            }
        """
        FRAME_PROTAGONIST_DETECTION_PROMPT = f"""You are an expert at detecting MAIN CHARACTERS vs. MINOR CHARACTERS in video frames.

        **Task**: Detect if {main_character_name} (the MAIN protagonist) is present as a PRIMARY VISUAL SUBJECT in this frame.

        **CRITICAL DISTINCTION - Who to ACCEPT**:
        ✅ ACCEPT if {main_character_name} is:
        - Clearly visible and recognizable (face, body, silhouette, costume, or iconic features)
        - The PRIMARY focus of the frame (center, foreground, or dominant subject)
        - Visible in action scenes even with motion blur (if clearly identifiable)
        - Shown in medium/close shots, medium-wide shots, or action sequences where they are the focal point
        - In long shots IF they are visually distinctive and central to the composition

        **CRITICAL DISTINCTION - Who to REJECT**:
        ❌ REJECT if the frame shows:
        - Minor characters, extras, or background crowd (e.g., henchmen, bystanders, soldiers, civilians)
        - {main_character_name} only as a tiny distant figure with no visual prominence
        - Back view where identity cannot be confirmed with reasonable confidence
        - Group scenes where {main_character_name} is NOT the visual focus
        - Pure environment/establishing shots with no clear protagonist presence

        **Instructions**:
        1. First identify: Is there a MAIN CHARACTER in this frame, or only minor/background characters?
        2. If main character present: Is it {main_character_name} specifically?
        3. If yes, provide bounding box coordinates (x, y, width, height) in pixels
        4. Set protagonist_detected=true ONLY if confident it's {main_character_name} AND they are a primary visual subject

        **Output Format** (JSON only):
        {{
        "protagonist_detected": <true/false>,
        "is_minor_character": <true/false>,
        "bounding_box": {{
            "x": <int, left coordinate>,
            "y": <int, top coordinate>,
            "width": <int, box width>,
            "height": <int, box height>
        }},
        "confidence": <float 0.0-1.0>,
        "reason": "<brief explanation: who is in frame and why accepted/rejected>"
        }}

        **Guidelines**:
        - Return null for bounding_box if protagonist_detected is false
        - Set is_minor_character=true if frame shows minor/background characters instead of protagonist
        - Be PERMISSIVE with protagonist size (allow small sizes if they are still the visual focus)
        - Be STRICT about distinguishing main character from extras/minor characters
        - For action scenes: Accept motion blur if character is identifiable
        - For distant shots: Accept if protagonist is visually distinctive (costume, silhouette, context)
        - Output ONLY valid JSON, no additional text
        """

        try:
            messages = [
                {"role": "system", "content": "You are an expert at character detection and localization in video frames."},
                {"role": "user", "content": FRAME_PROTAGONIST_DETECTION_PROMPT}
            ]

            # Use video_caption's call_vllm_model for single image detection
            # This avoids path issues with cv2.imread in vllm_calling
            response = call_vllm_model_for_images(
                messages,
                endpoint=config.VLLM_ENDPOINT,
                model_name=config.VIDEO_ANALYSIS_MODEL,
                return_json=False,
                image_paths=[frame_path],
                max_tokens=2048,
            )

            if response is None or response.get("content") is None:
                return {
                    "is_break": True,
                    "reason": "VLM returned no response",
                    "bounding_box": None
                }

            content = response["content"].strip()

            # Extract JSON from markdown code blocks if present
            json_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
            if json_block_match:
                content = json_block_match.group(1).strip()

            detection = json.loads(content)

            protagonist_detected = detection.get("protagonist_detected", False)
            is_minor_character = detection.get("is_minor_character", False)
            bounding_box = detection.get("bounding_box", None)
            confidence = detection.get("confidence", 0.0)
            reason_text = detection.get("reason", "")

            # Check for minor characters (highest priority rejection)
            if is_minor_character:
                return {
                    "is_break": True,
                    "reason": f"minor_character_detected ({reason_text})",
                    "bounding_box": None
                }

            if not protagonist_detected:
                return {
                    "is_break": True,
                    "reason": f"no_protagonist ({reason_text})",
                    "bounding_box": None
                }

            if bounding_box is None:
                return {
                    "is_break": True,
                    "reason": "no_bounding_box",
                    "bounding_box": None
                }

            # RELAXED size check: Allow smaller protagonists as long as they are visually present
            box_width = bounding_box.get("width", 0)
            box_height = bounding_box.get("height", 0)
            box_size = min(box_width, box_height)

            # Use a more permissive minimum size (allow protagonists down to 30px if they are the focus)
            relaxed_min_size = max(30, min_box_size // 2)  # Use half of configured min_box_size, but at least 30px

            if box_size < relaxed_min_size:
                return {
                    "is_break": True,
                    "reason": f"protagonist_too_small ({box_size}px < {relaxed_min_size}px)",
                    "bounding_box": bounding_box
                }

            return {
                "is_break": False,
                "reason": f"protagonist_ok (size={box_size}px, conf={confidence:.2f})",
                "bounding_box": bounding_box
            }

        except json.JSONDecodeError as e:
            return {
                "is_break": True,
                "reason": f"JSON parse error: {e}",
                "bounding_box": None
            }
        except Exception as e:
            return {
                "is_break": True,
                "reason": f"VLM error: {e}",
                "bounding_box": None
            }


    def get_protagonist_frame_data(
        self,
        video_path: str,
        time_range: str,
        main_character_name: str = "the main character",
        sample_fps: float = 1.0,
        min_box_size: int = 50,
    ) -> list:
        """
        Get frame-by-frame protagonist detection data for a time range.
        Returns structured data instead of a summary string.

        Args:
            video_path: Path to the video file
            time_range: Time range in format "HH:MM:SS to HH:MM:SS" or "MM:SS to MM:SS"
            main_character_name: Name of the main character to detect
            sample_fps: Frame sampling rate for detection (default: 1.0 fps)
            min_box_size: Minimum bounding box size in pixels (default: 50)

        Returns:
            list: List of frame detection results, each containing:
                {
                    "frame_idx": int,
                    "time_sec": float,
                    "protagonist_detected": bool,
                    "bounding_box": dict or None,  # {"x": int, "y": int, "width": int, "height": int}
                    "confidence": float,
                    "reason": str
                }
        """
        def hhmmss_to_seconds(time_str: str) -> float:
            """Convert HH:MM:SS, HH:MM:SS.s, HH:MM:SS:FF, or MM:SS to seconds."""
            parts = time_str.strip().split(':')
            if len(parts) == 4:
                # HH:MM:SS:FF format (with frame number)
                h, m, s, f = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                fps = getattr(config, 'VIDEO_FPS', 24) or 24
                return h * 3600 + m * 60 + s + (f / fps)
            elif len(parts) == 3:
                h, m = int(parts[0]), int(parts[1])
                s = float(parts[2])
                return h * 3600 + m * 60 + s
            elif len(parts) == 2:
                m = int(parts[0])
                s = float(parts[1])
                return m * 60 + s
            else:
                return float(parts[0])

        # Parse time range
        match = re.search(r'([\d:.]+)\s+to\s+([\d:.]+)', time_range, re.IGNORECASE)
        if not match:
            print(f"❌ Error: Could not parse time range '{time_range}'")
            return []

        try:
            start_sec = hhmmss_to_seconds(match.group(1))
            end_sec = hhmmss_to_seconds(match.group(2))
            duration_sec = end_sec - start_sec

            if duration_sec <= 0:
                print(f"❌ Error: Invalid time range")
                return []
        except Exception as e:
            print(f"❌ Error parsing time range: {e}")
            return []

        # Prefer using pre-extracted frames
        frames_dir = self.frame_folder_path if self.frame_folder_path and os.path.isdir(self.frame_folder_path) else None

        frame_results = []

        try:
            if frames_dir:
                # Use pre-extracted frames
                frames_fps = float(getattr(config, "VIDEO_FPS", 2) or 2.0)
                effective_sample_fps = float(sample_fps) if float(sample_fps) > 0 else 1.0
                step_sec = 1.0 / effective_sample_fps

                import glob
                import math

                n = int(math.ceil(duration_sec / step_sec))
                processed_frame_indices = set()

                for i in range(max(1, n)):
                    t = start_sec + i * step_sec
                    # Changed to > to include the end boundary (was >=)
                    if t > end_sec:
                        break

                    frame_idx = int(round(t * frames_fps))

                    if frame_idx in processed_frame_indices:
                        continue
                    processed_frame_indices.add(frame_idx)

                    stem = f"frame_{frame_idx:06d}"
                    frame_path = None
                    for ext in (".png", ".jpg", ".jpeg"):
                        p = os.path.join(frames_dir, stem + ext)
                        if os.path.exists(p):
                            frame_path = p
                            break

                    if frame_path is None:
                        matches = glob.glob(os.path.join(frames_dir, stem + ".*"))
                        if matches:
                            frame_path = matches[0]

                    if frame_path is None:
                        continue

                    # Call VLM to detect protagonist in current frame
                    detection_result = self._detect_protagonist_in_frame_vlm(
                        frame_path, main_character_name, min_box_size
                    )

                    time_at_frame = frame_idx / frames_fps

                    frame_data = {
                        "frame_idx": frame_idx,
                        "time_sec": time_at_frame,
                        "protagonist_detected": not detection_result["is_break"],
                        "bounding_box": detection_result.get("bounding_box"),
                        "confidence": 0.0,  # Will be extracted from VLM response if available
                        "reason": detection_result["reason"]
                    }

                    frame_results.append(frame_data)

            else:
                # Decode from video (fallback)
                if not os.path.exists(video_path):
                    print(f"❌ Error: Video file not found: {video_path}")
                    return []

                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"❌ Error: Could not open video file: {video_path}")
                    return []

                video_fps = cap.get(cv2.CAP_PROP_FPS)
                start_frame = int(start_sec * video_fps)
                end_frame = int(end_sec * video_fps)

                frame_interval = int(video_fps / float(sample_fps or 1.0))
                if frame_interval < 1:
                    frame_interval = 1

                # Use closed interval to include the end frame
                frame_indices = list(range(start_frame, end_frame + 1, frame_interval))
                if not frame_indices:
                    cap.release()
                    return []

                import tempfile
                temp_dir = tempfile.mkdtemp()

                try:
                    for frame_idx in frame_indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = cap.read()

                        if not ret:
                            continue

                        # Save temporary frame
                        temp_frame_path = os.path.join(temp_dir, f"temp_frame_{frame_idx}.jpg")
                        cv2.imwrite(temp_frame_path, frame)

                        # Call VLM detection
                        detection_result = self._detect_protagonist_in_frame_vlm(
                            temp_frame_path, main_character_name, min_box_size
                        )

                        time_at_frame = frame_idx / video_fps

                        frame_data = {
                            "frame_idx": frame_idx,
                            "time_sec": time_at_frame,
                            "protagonist_detected": not detection_result["is_break"],
                            "bounding_box": detection_result.get("bounding_box"),
                            "confidence": 0.0,
                            "reason": detection_result["reason"]
                        }

                        frame_results.append(frame_data)

                        # Clean up temporary file
                        os.remove(temp_frame_path)

                finally:
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)

                cap.release()

        except Exception as e:
            print(f"❌ Error during protagonist detection: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

        return frame_results


    def check_aesthetic_quality(
        self,
        video_path: A[str, D("Path to the video file.")],
        time_range: A[str, D("The time range to check (e.g., '00:13:28 to 00:13:40').")],
        min_aesthetic_score: A[float, D("Minimum required aesthetic score (1-5 scale). Default: 3.0")] = 3.0,
        sample_fps: A[float, D("Sampling frame rate for analysis. Default: 2.0")] = 2.0,
    ) -> str:
        """
        Check aesthetic quality of a video clip using VLM analysis.
        Analyzes visual appeal, lighting, composition, colors, and cinematography.

        Args:
            video_path: Path to the video file
            time_range: Time range in format "HH:MM:SS to HH:MM:SS" or "MM:SS to MM:SS"
            min_aesthetic_score: Minimum required aesthetic score (default: 3.0 on 1-5 scale)
            sample_fps: Frame sampling rate for analysis (default: 2.0 fps)

        Returns:
            str: Success message if aesthetic quality meets requirements, or error message with details.

        Example:
            >>> check_aesthetic_quality("/path/to/video.mp4", "00:10:00 to 00:10:10", min_aesthetic_score=3.5)
        """
        def hhmmss_to_seconds(time_str: str) -> float:
            """Convert HH:MM:SS, HH:MM:SS.s, HH:MM:SS:FF, or MM:SS to seconds."""
            parts = time_str.strip().split(':')
            if len(parts) == 4:
                # HH:MM:SS:FF format (with frame number)
                h, m, s, f = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                fps = getattr(config, 'VIDEO_FPS', 24) or 24
                return h * 3600 + m * 60 + s + (f / fps)
            elif len(parts) == 3:
                h, m = int(parts[0]), int(parts[1])
                s = float(parts[2])
                return h * 3600 + m * 60 + s
            elif len(parts) == 2:
                m = int(parts[0])
                s = float(parts[1])
                return m * 60 + s
            else:
                return float(parts[0])

        # Parse time range
        match = re.search(r'([\d:.]+)\s+to\s+([\d:.]+)', time_range, re.IGNORECASE)
        if not match:
            return f"❌ Error: Could not parse time range '{time_range}'. Please use format 'HH:MM:SS to HH:MM:SS'."

        try:
            start_sec = hhmmss_to_seconds(match.group(1))
            end_sec = hhmmss_to_seconds(match.group(2))
            duration_sec = end_sec - start_sec

            if duration_sec <= 0:
                return f"❌ Error: Invalid time range. End time must be greater than start time."
        except Exception as e:
            return f"❌ Error parsing time range: {e}"

        if not os.path.exists(video_path):
            return f"❌ Error: Video file not found: {video_path}"

        print(f"[Aesthetic Quality Check] Analyzing {time_range} ({duration_sec:.2f}s)...")
        print(f"[Aesthetic Quality Check] Minimum aesthetic score: {min_aesthetic_score}/5.0")

        # Use VLM to analyze aesthetic quality
        AESTHETIC_ANALYSIS_PROMPT = """You are an expert cinematographer and visual aesthetics analyst specializing in vlog content.

**Task**: Analyze the aesthetic quality and visual appeal of this video clip.

**Analysis Criteria**:
1. **Lighting Quality**: Natural light, artificial light, lighting consistency, shadows, highlights
2. **Color Grading**: Color palette, saturation, contrast, color harmony, mood
3. **Composition**: Framing, rule of thirds, visual balance, depth, leading lines
4. **Camera Work**: Stability, smooth movements, focus, exposure
5. **Visual Interest**: Engaging subjects, dynamic elements, visual variety
6. **Cinematic Feel**: Overall production quality, professional look, artistic appeal

**Output Format** (JSON only):
{
  "overall_aesthetic_score": <float 1.0-5.0>,
  "lighting_score": <float 1.0-5.0>,
  "color_score": <float 1.0-5.0>,
  "composition_score": <float 1.0-5.0>,
  "camera_work_score": <float 1.0-5.0>,
  "visual_interest_score": <float 1.0-5.0>,
  "strengths": ["<strength 1>", "<strength 2>", ...],
  "weaknesses": ["<weakness 1>", "<weakness 2>", ...],
  "recommendation": "<EXCELLENT | VERY_GOOD | GOOD | ACCEPTABLE | POOR>",
  "detailed_analysis": "<Brief 2-3 sentence analysis of the visual aesthetics>"
}

**Scoring Guide**:
- 5.0: Stunning - Professional cinematography, beautiful lighting, exceptional composition
- 4.0: Very Good - High quality visuals, well-composed, attractive aesthetics
- 3.0: Good - Acceptable quality, decent composition, suitable for use
- 2.0: Fair - Some quality issues, basic composition, marginal aesthetics
- 1.0: Poor - Significant quality problems, poor composition, unappealing

**Guidelines**:
- Be objective and specific in your assessment
- Focus on visual appeal and production quality
- Consider the context of vlog content (not cinema film standards)
- Provide constructive feedback
- Output ONLY valid JSON, no additional text
"""

        try:
            messages = [
                {"role": "system", "content": "You are an expert cinematographer and visual aesthetics analyst."},
                {"role": "user", "content": AESTHETIC_ANALYSIS_PROMPT}
            ]

            # Call VLM with video clip
            response = call_vllm_model_for_video(
                messages,
                endpoint=config.VLLM_ENDPOINT,
                model_name=config.VIDEO_ANALYSIS_MODEL,
                return_json=False,
                video_path=video_path,
                video_fps=config.VIDEO_FPS,
                do_sample_frames=False,
                max_tokens=2048,
                use_local_clipping=True,
                video_start_time=start_sec,
                video_end_time=end_sec,
            )

            if response is None or response.get("content") is None:
                return f"⚠️ WARNING: VLM returned no response for aesthetic analysis. Proceeding without validation."

            content = response["content"].strip()

            # Extract JSON from markdown code blocks if present
            json_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
            if json_block_match:
                content = json_block_match.group(1).strip()

            analysis = json.loads(content)

            overall_score = analysis.get("overall_aesthetic_score", 0.0)
            lighting_score = analysis.get("lighting_score", 0.0)
            color_score = analysis.get("color_score", 0.0)
            composition_score = analysis.get("composition_score", 0.0)
            camera_work_score = analysis.get("camera_work_score", 0.0)
            visual_interest_score = analysis.get("visual_interest_score", 0.0)
            strengths = analysis.get("strengths", [])
            weaknesses = analysis.get("weaknesses", [])
            recommendation = analysis.get("recommendation", "UNKNOWN")
            detailed_analysis = analysis.get("detailed_analysis", "")

            # Prepare result message
            result_msg = f"\n[Aesthetic Quality Check Results]\n"
            result_msg += f"Time range: {time_range} ({duration_sec:.2f}s)\n"
            result_msg += f"Overall Aesthetic Score: {overall_score:.2f}/5.0\n"
            result_msg += f"  • Lighting: {lighting_score:.2f}/5.0\n"
            result_msg += f"  • Color: {color_score:.2f}/5.0\n"
            result_msg += f"  • Composition: {composition_score:.2f}/5.0\n"
            result_msg += f"  • Camera Work: {camera_work_score:.2f}/5.0\n"
            result_msg += f"  • Visual Interest: {visual_interest_score:.2f}/5.0\n"
            result_msg += f"Recommendation: {recommendation}\n"
            result_msg += f"Minimum Required: {min_aesthetic_score:.2f}/5.0\n"

            if strengths:
                result_msg += f"\nStrengths:\n"
                for strength in strengths[:3]:  # Show top 3
                    result_msg += f"  ✓ {strength}\n"

            if weaknesses:
                result_msg += f"\nWeaknesses:\n"
                for weakness in weaknesses[:3]:  # Show top 3
                    result_msg += f"  ✗ {weakness}\n"

            if detailed_analysis:
                result_msg += f"\nAnalysis: {detailed_analysis}\n"

            # Check if aesthetic score meets threshold
            if overall_score < min_aesthetic_score:
                result_msg += f"\n❌ FAILED: Aesthetic score ({overall_score:.2f}/5.0) is below minimum threshold ({min_aesthetic_score:.2f}/5.0)\n"
                result_msg += f"\n⚠️ This shot does not meet the aesthetic quality requirements. Please select a shot with:\n"
                result_msg += f"  • Better lighting (natural light preferred)\n"
                result_msg += f"  • Improved composition (well-framed, balanced)\n"
                result_msg += f"  • More vibrant colors and good contrast\n"
                result_msg += f"  • Stable camera work\n"
                result_msg += f"  • More visually interesting content\n"
                print(result_msg)
                return result_msg
            else:
                result_msg += f"\n✅ PASSED: Aesthetic score ({overall_score:.2f}/5.0) meets the minimum threshold.\n"
                if overall_score >= 4.0:
                    result_msg += f"⭐ Excellent visual quality! This shot is highly recommended for the final edit.\n"
                result_msg += f"You can proceed with this shot."
                print(result_msg)
                return result_msg

        except json.JSONDecodeError as e:
            return f"⚠️ WARNING: Could not parse VLM response for aesthetic analysis: {e}\n\nProceeding without validation."
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"⚠️ WARNING: Error during aesthetic analysis: {str(e)}\n\nProceeding without validation."


    def review(self, shot_proposal: dict, context: dict, used_time_ranges: list = None) -> dict:
        """
        审核 shot 选择是否符合要求。

        Args:
            shot_proposal: 包含 shot 选择的信息
                - answer: str, 选择的时间范围 (e.g., "[shot: 00:10:00 to 00:10:07]")
                - target_length_sec: float, 目标时长
            context: 当前 shot 的上下文信息
                - content: str, 目标内容描述
                - emotion: str, 目标情感
                - section_idx: int, 当前 section 索引
                - shot_idx: int, 当前 shot 索引
            used_time_ranges: 已使用的时间范围列表 [(start_sec, end_sec), ...]

        Returns:
            dict: {
                "approved": bool,  # 是否通过审核
                "feedback": str,   # 反馈信息
                "issues": list,    # 发现的问题列表
                "suggestions": list  # 改进建议
            }
        """
        if used_time_ranges is None:
            used_time_ranges = []

        answer = shot_proposal.get("answer", "")
        target_length_sec = shot_proposal.get("target_length_sec", 0.0)

        issues = []
        suggestions = []

        # 1. 检查时间范围格式和时长
        finish_review = review_finish(answer, target_length_sec)
        if "❌" in finish_review:
            issues.append(finish_review)

        # 2. 检查是否与已使用的片段重叠
        # 从 answer 中提取时间范围
        match = re.search(r'\[?shot[\s_]*\d*:\s*([0-9:.]+)\s+to\s+([0-9:.]+)\]?', answer, re.IGNORECASE)
        if match:
            time_range = f"{match.group(1)} to {match.group(2)}"
            overlap_review = review_clip(time_range, used_time_ranges)
            if "❌" in overlap_review:
                issues.append(overlap_review)

        # 3. 检查内容匹配度 (可以通过 LLM 调用来实现更复杂的审核)
        # TODO: 可以在这里添加更多的审核逻辑，比如：
        # - 检查选择的片段是否与目标内容/情感匹配
        # - 检查片段的视觉质量
        # - 检查叙事连贯性

        # 构建反馈
        if issues:
            feedback = "❌ 审核未通过，发现以下问题:\n"
            for i, issue in enumerate(issues, 1):
                feedback += f"\n问题 {i}:\n{issue}\n"

            suggestions.append("请根据上述问题调整你的 shot 选择")
            if "Duration mismatch" in str(issues) or "时长" in str(issues):
                suggestions.append("调整时间范围的起止点以匹配目标时长")
            if "OVERLAP" in str(issues) or "重叠" in str(issues):
                suggestions.append("选择一个不与已使用片段重叠的时间范围")

            feedback += "\n建议:\n" + "\n".join(f"- {s}" for s in suggestions)

            return {
                "approved": False,
                "feedback": feedback,
                "issues": issues,
                "suggestions": suggestions
            }
        else:
            return {
                "approved": True,
                "feedback": f"✅ 审核通过！Shot 选择符合要求。\n{finish_review}",
                "issues": [],
                "suggestions": []
            }

    def review_with_llm(self, shot_proposal: dict, context: dict, used_time_ranges: list = None) -> dict:
        """
        使用 LLM 进行更深入的内容审核。

        这个方法会调用 LLM 来评估选择的片段是否与目标内容和情感匹配。
        """
        # 首先进行基本审核
        basic_review = self.review(shot_proposal, context, used_time_ranges)

        if not basic_review["approved"]:
            return basic_review

        # 如果基本审核通过，进行 LLM 深度审核
        answer = shot_proposal.get("answer", "")
        target_content = context.get("content", "")
        target_emotion = context.get("emotion", "")

        review_prompt = f"""你是一个专业的视频剪辑审核员。请评估以下 shot 选择是否合适。

目标内容: {target_content}
目标情感: {target_emotion}
选择的片段: {answer}

请评估:
1. 这个时间范围的片段是否可能符合目标内容?
2. 这个片段是否可能传达目标情感?
3. 有什么潜在的问题或建议?

请用 JSON 格式回答:
{{"approved": true/false, "reason": "...", "suggestions": ["...", "..."]}}
"""

        try:
            messages = [
                {"role": "system", "content": "你是一个专业的视频剪辑审核员。"},
                {"role": "user", "content": review_prompt}
            ]

            # This is a text-only call, use the video version (no images needed)
            response = call_vllm_model_for_video(
                messages,
                endpoint=config.VLLM_AGENT_ENDPOINT,
                model_name=config.AGENT_MODEL,
                temperature=0.0,
                max_tokens=512,
                return_json=True,
            )

            if response and response.get("content"):
                llm_review = json.loads(response["content"])
                if not llm_review.get("approved", True):
                    basic_review["approved"] = False
                    basic_review["feedback"] += f"\n\nLLM 审核反馈:\n{llm_review.get('reason', '')}"
                    basic_review["suggestions"].extend(llm_review.get("suggestions", []))
        except Exception as e:
            print(f"LLM 审核出错: {e}")
            # LLM 审核失败时，仍然返回基本审核结果

        return basic_review
