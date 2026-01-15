#!/usr/bin/env python3
"""
Render video from shot_point.json

This script reads a shot_point.json file containing clip timestamps and
renders them into a single video file using ffmpeg.
"""

import argparse
import json
import os
import subprocess
import tempfile
from typing import List, Dict, Any


def hhmmss_to_seconds(time_str: str) -> float:
    """Convert HH:MM:SS.s or MM:SS.s to seconds."""
    parts = time_str.strip().split(':')
    if len(parts) == 3:
        h, m = int(parts[0]), int(parts[1])
        s = float(parts[2])
        return h * 3600 + m * 60 + s
    elif len(parts) == 2:
        m = int(parts[0])
        s = float(parts[1])
        return m * 60 + s
    else:
        return float(parts[0])


def get_video_framerate(video_path: str) -> float:
    """Get video framerate using ffprobe."""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            # Parse fraction like "24000/1001" or "24/1"
            fps_str = result.stdout.strip()
            if '/' in fps_str:
                num, denom = fps_str.split('/')
                return float(num) / float(denom)
            else:
                return float(fps_str)
    except Exception as e:
        print(f"Warning: Could not get video framerate: {e}")
    return 24.0  # Default fallback


def get_video_dimensions(video_path: str) -> tuple:
    """Get video width and height using ffprobe."""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'csv=p=0',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            width, height = result.stdout.strip().split(',')
            return (int(width), int(height))
    except Exception as e:
        print(f"Warning: Could not get video dimensions: {e}")
    return (1920, 1080)  # Default fallback


def parse_shot_scenes(shot_scenes_path: str, fps: float) -> List[float]:
    """
    Parse shot_scenes.txt and return list of scene cut timestamps in seconds.

    Args:
        shot_scenes_path: Path to shot_scenes.txt file
        fps: Video framerate

    Returns:
        List of timestamps (in seconds) where scene cuts occur
    """
    cut_points = []

    try:
        with open(shot_scenes_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) >= 2:
                    # Each line has start_frame and end_frame
                    # The end_frame of one scene is the cut point
                    end_frame = int(parts[1])
                    timestamp = end_frame / fps
                    cut_points.append(timestamp)
    except Exception as e:
        print(f"Warning: Could not parse shot_scenes.txt: {e}")
        return []

    # Sort and remove duplicates
    cut_points = sorted(set(cut_points))
    return cut_points


def adjust_clip_for_scene_cuts(
    start_sec: float,
    end_sec: float,
    cut_points: List[float],
    tolerance: float = 0.5
) -> tuple:
    """
    Adjust clip start time if there's a scene cut within the clip.

    If a scene cut point exists within the clip (excluding the exact start/end),
    snap the start time to the nearest cut point and maintain duration.

    Args:
        start_sec: Original start time in seconds
        end_sec: Original end time in seconds
        cut_points: List of scene cut timestamps
        tolerance: Minimum distance from start/end to consider a cut point (seconds)

    Returns:
        Tuple of (adjusted_start_sec, adjusted_end_sec)
    """
    if not cut_points:
        return start_sec, end_sec

    duration = end_sec - start_sec

    # Find cut points that are inside the clip (with tolerance)
    internal_cuts = [
        cp for cp in cut_points
        if start_sec + tolerance < cp < end_sec - tolerance
    ]

    if internal_cuts:
        # Snap to the first internal cut point
        new_start = internal_cuts[0]
        new_end = new_start + duration
        return new_start, new_end

    return start_sec, end_sec


def calculate_optimal_crop_center(
    protagonist_detection: Dict[str, Any],
    detection_width: int = None,
    detection_height: int = None,
    video_width: int = None,
    video_height: int = None
) -> tuple:
    """
    Calculate the optimal crop center based on protagonist bounding boxes.

    Args:
        protagonist_detection: Dictionary containing frame_detections with bounding_box info
        detection_width: Width of the detection image (if different from video)
        detection_height: Height of the detection image (if different from video)
        video_width: Original video width
        video_height: Original video height

    Returns:
        Tuple of (center_x, center_y) in original video coordinates,
        or None if no valid detections found
    """
    if not protagonist_detection or 'frame_detections' not in protagonist_detection:
        return None

    valid_boxes = []
    for frame_det in protagonist_detection['frame_detections']:
        if frame_det.get('protagonist_detected') and frame_det.get('bounding_box'):
            bbox = frame_det['bounding_box']
            # Calculate center of bounding box (in detection image coordinates)
            center_x = bbox['x'] + bbox['width'] / 2
            center_y = bbox['y'] + bbox['height'] / 2
            valid_boxes.append((center_x, center_y, bbox['width'], bbox['height']))

    if not valid_boxes:
        return None

    # Calculate weighted average center (larger boxes have more weight)
    total_weight = 0
    weighted_x = 0
    weighted_y = 0

    for center_x, center_y, width, height in valid_boxes:
        # Use box area as weight
        weight = width * height
        weighted_x += center_x * weight
        weighted_y += center_y * weight
        total_weight += weight

    if total_weight == 0:
        return None

    # Average center position (still in detection image coordinates)
    avg_x = weighted_x / total_weight
    avg_y = weighted_y / total_weight

    # Scale coordinates back to original video size if scaling info provided
    if detection_width and detection_height and video_width and video_height:
        scale_x = video_width / detection_width
        scale_y = video_height / detection_height
        avg_x = avg_x * scale_x
        avg_y = avg_y * scale_y

    return (avg_x, avg_y)


def extract_all_clips(
    shot_data: List[Dict[str, Any]],
    cut_points: List[float] = None,
    video_width: int = None,
    video_height: int = None,
    detection_short_side: int = None
) -> List[Dict[str, Any]]:
    """
    Extract all clips from shot_point.json in order.

    Args:
        shot_data: List of shot data from shot_point.json
        cut_points: Optional list of scene cut timestamps for adjustment
        video_width: Original video width (for scaling detection coordinates)
        video_height: Original video height (for scaling detection coordinates)
        detection_short_side: Size of short side used during detection (e.g., 360)

    Returns:
        A flat list of clips with start/end times in seconds.
    """
    all_clips = []

    # Calculate detection image dimensions if scaling info provided
    detection_width = None
    detection_height = None
    scale_x = 1.0
    scale_y = 1.0
    if video_width and video_height and detection_short_side:
        # Determine which is the short side
        if video_width < video_height:
            # Width is short side
            scale = detection_short_side / video_width
            detection_width = detection_short_side
            detection_height = int(video_height * scale)
        else:
            # Height is short side
            scale = detection_short_side / video_height
            detection_height = detection_short_side
            detection_width = int(video_width * scale)
        scale_x = video_width / detection_width
        scale_y = video_height / detection_height
        print(f"Detection image size: {detection_width}x{detection_height} (scaled from {video_width}x{video_height})")

    # Sort by section_idx, then shot_idx to ensure correct order
    sorted_data = sorted(shot_data, key=lambda x: (x.get('section_idx', 0), x.get('shot_idx', 0)))

    for shot in sorted_data:
        if shot.get('status') != 'success':
            continue

        section_idx = shot.get('section_idx', -1)
        shot_idx = shot.get('shot_idx', -1)

        # Calculate optimal crop center from protagonist detection
        crop_center = None
        if 'protagonist_detection' in shot:
            crop_center = calculate_optimal_crop_center(
                shot['protagonist_detection'],
                detection_width=detection_width,
                detection_height=detection_height,
                video_width=video_width,
                video_height=video_height
            )

        # Scale bounding boxes to video coordinates for visualization
        scaled_detections = None
        if 'protagonist_detection' in shot and shot['protagonist_detection'].get('frame_detections'):
            scaled_detections = []
            for frame_det in shot['protagonist_detection']['frame_detections']:
                if frame_det.get('protagonist_detected') and frame_det.get('bounding_box'):
                    bbox = frame_det['bounding_box']
                    scaled_bbox = {
                        'x': int(bbox['x'] * scale_x),
                        'y': int(bbox['y'] * scale_y),
                        'width': int(bbox['width'] * scale_x),
                        'height': int(bbox['height'] * scale_y)
                    }
                    scaled_detections.append({
                        'time_sec': frame_det['time_sec'],
                        'bounding_box': scaled_bbox
                    })

        for clip in shot.get('clips', []):
            start_sec = hhmmss_to_seconds(clip['start'])
            end_sec = hhmmss_to_seconds(clip['end'])

            # Adjust for scene cuts if cut_points provided
            original_start = start_sec
            original_end = end_sec
            if cut_points:
                start_sec, end_sec = adjust_clip_for_scene_cuts(start_sec, end_sec, cut_points)

            # Convert back to string format if adjusted
            def sec_to_hhmmss(sec: float) -> str:
                h = int(sec // 3600)
                m = int((sec % 3600) // 60)
                s = sec % 60
                return f"{h:02d}:{m:02d}:{s:06.3f}"

            all_clips.append({
                'section_idx': section_idx,
                'shot_idx': shot_idx,
                'start_sec': start_sec,
                'end_sec': end_sec,
                'duration': end_sec - start_sec,
                'start_str': sec_to_hhmmss(start_sec),
                'end_str': sec_to_hhmmss(end_sec),
                'original_start': original_start,
                'original_end': original_end,
                'adjusted': (start_sec != original_start or end_sec != original_end),
                'crop_center': crop_center,  # Add crop center information
                'scaled_detections': scaled_detections  # Add scaled detection info for visualization
            })

    return all_clips


def render_video_ffmpeg(
    video_path: str,
    clips: List[Dict[str, Any]],
    output_path: str,
    audio_path: str = None,
    audio_start_time: float = None,
    audio_duration: float = None,
    verbose: bool = False,
    show_labels: bool = True,
    label_position: str = "top-left",
    font_size: int = 32,
    font_color: str = "white",
    bg_color: str = "black@0.6",
    crop_ratio: str = None,
    original_audio_volume: float = 0.0,
    video_width: int = None,
    video_height: int = None,
    visualize_detections: bool = False
) -> bool:
    """
    Render video clips using ffmpeg concat demuxer.

    Args:
        video_path: Path to source video file
        clips: List of clip dictionaries with start_sec and end_sec
        output_path: Path for output video
        audio_path: Optional path to audio file to mix with video
        audio_start_time: Optional start time (in seconds) to crop audio from
        audio_duration: Optional duration (in seconds) to crop audio to
        verbose: Print ffmpeg output
        show_labels: Whether to overlay Section/Shot labels on video
        label_position: Position of labels ("top-left", "top-right", "bottom-left", "bottom-right")
        font_size: Font size for labels
        font_color: Font color for labels
        bg_color: Background color for label box (with opacity, e.g., "black@0.6")
        crop_ratio: Optional aspect ratio for center cropping (e.g., "9:16", "16:9", "1:1").
                   Keeps height unchanged and crops width to match the ratio.
                   If clips have crop_center info, uses dynamic crop centers instead of fixed center.
        original_audio_volume: Volume level for original video audio (0.0 or higher).
                              0.0 = muted (default), 1.0 = full volume, >1.0 = amplified.
                              Only applies when mixing with external audio.
        video_width: Video width in pixels (auto-detected if None)
        video_height: Video height in pixels (auto-detected if None)
        visualize_detections: Whether to draw bounding boxes and crop area on video

    Returns:
        True if successful, False otherwise
    """
    if not clips:
        print("Error: No clips to render")
        return False

    # Get video dimensions if not provided
    if video_width is None or video_height is None:
        video_width, video_height = get_video_dimensions(video_path)
        print(f"Detected video dimensions: {video_width}x{video_height}")
    else:
        print(f"Using provided video dimensions: {video_width}x{video_height}")

    # Determine label position coordinates
    position_map = {
        "top-left": ("10", "10"),
        "top-right": ("w-tw-10", "10"),
        "bottom-left": ("10", "h-th-10"),
        "bottom-right": ("w-tw-10", "h-th-10"),
    }
    label_x, label_y = position_map.get(label_position, ("10", "10"))

    # Parse crop ratio if provided
    crop_w_ratio = None
    crop_h_ratio = None
    if crop_ratio:
        try:
            parts = crop_ratio.split(':')
            if len(parts) == 2:
                crop_w_ratio, crop_h_ratio = int(parts[0]), int(parts[1])
                print(f"Crop ratio: {crop_ratio} (dynamic crop centers will be used if available)")
            else:
                print(f"Warning: Invalid crop ratio format '{crop_ratio}', expected format like '9:16'. Ignoring.")
        except ValueError:
            print(f"Warning: Could not parse crop ratio '{crop_ratio}'. Ignoring.")

    # Create temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        clip_files = []

        # Extract each clip to a temporary file
        print(f"Extracting {len(clips)} clips...")
        for i, clip in enumerate(clips):
            clip_file = os.path.join(temp_dir, f"clip_{i:04d}.mp4")
            clip_files.append(clip_file)

            start = clip['start_sec']
            duration = clip['duration']
            section_idx = clip.get('section_idx', 0)
            shot_idx = clip.get('shot_idx', 0)

            # Generate crop filter for this clip (if crop ratio is provided)
            crop_filter = None
            crop_x_px = None
            crop_width_px = None
            if crop_w_ratio and crop_h_ratio:
                # Calculate crop dimensions in pixels
                # Keep height unchanged (video_height), calculate width from ratio
                crop_width_px = int(video_height * crop_w_ratio / crop_h_ratio)
                crop_height_px = video_height

                # Check if this clip has a crop_center from protagonist detection
                crop_center = clip.get('crop_center')
                if crop_center:
                    # Use dynamic crop center based on protagonist position
                    center_x, center_y = crop_center
                    # Calculate crop x position: center the crop on protagonist
                    crop_x_px = int(center_x - crop_width_px / 2)
                    # Ensure crop stays within bounds: 0 <= x <= (video_width - crop_width)
                    crop_x_px = max(0, min(crop_x_px, video_width - crop_width_px))
                    crop_y_px = 0  # Keep y at 0 (top of frame)
                    crop_filter = f"crop={crop_width_px}:{crop_height_px}:{crop_x_px}:{crop_y_px}"
                    if verbose:
                        print(f"  Clip {i}: Using dynamic crop center at ({center_x:.1f}, {center_y:.1f}) -> crop_x={crop_x_px}")
                else:
                    # No crop center info, use default center crop
                    crop_x_px = int((video_width - crop_width_px) / 2)
                    crop_y_px = 0
                    crop_filter = f"crop={crop_width_px}:{crop_height_px}:{crop_x_px}:{crop_y_px}"

            # Build visualization filters if requested
            viz_filters = []
            if visualize_detections:
                # Draw crop area boundary (before cropping)
                if crop_x_px is not None and crop_width_px is not None:
                    # Draw crop area as a green rectangle
                    viz_filters.append(
                        f"drawbox=x={crop_x_px}:y=0:w={crop_width_px}:h={video_height}:color=green@0.3:t=fill"
                    )
                    # Draw crop area border
                    viz_filters.append(
                        f"drawbox=x={crop_x_px}:y=0:w={crop_width_px}:h={video_height}:color=green:t=4"
                    )

                # Draw crop center point
                crop_center = clip.get('crop_center')
                if crop_center:
                    center_x, center_y = crop_center
                    # Draw a crosshair at crop center
                    viz_filters.append(
                        f"drawbox=x={int(center_x-10)}:y={int(center_y)}:w=20:h=1:color=red:t=fill"
                    )
                    viz_filters.append(
                        f"drawbox=x={int(center_x)}:y={int(center_y-10)}:w=1:h=20:color=red:t=fill"
                    )

                # Draw bounding boxes for all detected frames in this clip
                scaled_detections = clip.get('scaled_detections', [])
                if scaled_detections:
                    # Draw all bounding boxes (we'll use average box for simplicity)
                    # Or draw the first/last detection
                    for det in scaled_detections[:3]:  # Limit to first 3 to avoid too many boxes
                        bbox = det['bounding_box']
                        viz_filters.append(
                            f"drawbox=x={bbox['x']}:y={bbox['y']}:w={bbox['width']}:h={bbox['height']}:color=yellow:t=3"
                        )

            # Build ffmpeg command
            if show_labels:
                # Create label text
                label_text = f"Section {section_idx + 1} | Shot {shot_idx + 1}"

                # Use drawtext filter to overlay label
                # Note: Need to escape special characters for ffmpeg filter
                # fontfile is required - try common system fonts
                drawtext_filter = (
                    f"drawtext=text='{label_text}':"
                    f"fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
                    f"fontsize={font_size}:"
                    f"fontcolor={font_color}:"
                    f"x={label_x}:y={label_y}:"
                    f"box=1:boxcolor={bg_color}:boxborderw=8"
                )

                # Combine filters: viz first, then crop (if enabled), then drawtext
                filter_chain = []
                if viz_filters:
                    filter_chain.extend(viz_filters)
                if crop_filter:
                    filter_chain.append(crop_filter)
                filter_chain.append(drawtext_filter)
                video_filter = ",".join(filter_chain)

                cmd = [
                    'ffmpeg',
                    '-y',  # Overwrite output
                    '-ss', str(start),
                    '-i', video_path,
                    '-t', str(duration),
                    '-vf', video_filter,
                    '-c:v', 'libx264',  # Re-encode for consistent format
                    '-c:a', 'aac',
                    '-preset', 'fast',
                    '-crf', '18',  # High quality
                    '-avoid_negative_ts', 'make_zero',
                    clip_file
                ]
            else:
                # No label overlay, but may have crop and/or viz
                filter_chain = []
                if viz_filters:
                    filter_chain.extend(viz_filters)
                if crop_filter:
                    filter_chain.append(crop_filter)

                if filter_chain:
                    video_filter = ",".join(filter_chain)
                    cmd = [
                        'ffmpeg',
                        '-y',  # Overwrite output
                        '-ss', str(start),
                        '-i', video_path,
                        '-t', str(duration),
                        '-vf', video_filter,
                        '-c:v', 'libx264',  # Re-encode for consistent format
                        '-c:a', 'aac',
                        '-preset', 'fast',
                        '-crf', '18',  # High quality
                        '-avoid_negative_ts', 'make_zero',
                        clip_file
                    ]
                else:
                    cmd = [
                        'ffmpeg',
                        '-y',  # Overwrite output
                        '-ss', str(start),
                        '-i', video_path,
                        '-t', str(duration),
                        '-c:v', 'libx264',  # Re-encode for consistent format
                        '-c:a', 'aac',
                        '-preset', 'fast',
                        '-crf', '18',  # High quality
                        '-avoid_negative_ts', 'make_zero',
                        clip_file
                    ]

            if verbose:
                label_info = f" [S{section_idx + 1}-Shot{shot_idx + 1}]" if show_labels else ""
                print(f"  [{i+1}/{len(clips)}] {clip['start_str']} - {clip['end_str']} ({duration:.2f}s){label_info}")

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE if not verbose else None,
                stderr=subprocess.PIPE if not verbose else None
            )

            if result.returncode != 0:
                print(f"Error extracting clip {i}: {clip}")
                if not verbose and result.stderr:
                    print(result.stderr.decode()[-500:])
                return False

        # Create concat list file
        concat_file = os.path.join(temp_dir, 'concat_list.txt')
        with open(concat_file, 'w') as f:
            for clip_file in clip_files:
                f.write(f"file '{clip_file}'\n")

        # Concatenate all clips
        print("Concatenating clips...")

        if audio_path and os.path.exists(audio_path):
            # First concatenate video clips
            temp_video = os.path.join(temp_dir, 'temp_video.mp4')
            cmd = [
                'ffmpeg',
                '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_file,
                '-c', 'copy',
                temp_video
            ]

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE if not verbose else None,
                stderr=subprocess.PIPE if not verbose else None
            )

            if result.returncode != 0:
                print("Error concatenating clips")
                if not verbose and result.stderr:
                    print(result.stderr.decode()[-500:])
                return False

            # Then mix with audio (with optional cropping)
            print(f"Mixing with audio: {audio_path}")

            # Check if we need to mix original audio with external audio
            if original_audio_volume > 0:
                print(f"Mixing original video audio at volume: {original_audio_volume:.2f}")

                if audio_start_time is not None and audio_duration is not None:
                    print(f"Audio crop: {audio_start_time:.2f}s - {audio_start_time + audio_duration:.2f}s (duration: {audio_duration:.2f}s)")
                    # Use filter_complex to mix original audio with external audio
                    filter_complex = f"[0:a]volume={original_audio_volume}[a0];[a0][1:a]amix=inputs=2:duration=shortest[aout]"
                    cmd = [
                        'ffmpeg',
                        '-y',
                        '-i', temp_video,
                        '-ss', str(audio_start_time),  # Audio start time
                        '-t', str(audio_duration),      # Audio duration
                        '-i', audio_path,
                        '-filter_complex', filter_complex,
                        '-map', '0:v:0',  # Video from first input
                        '-map', '[aout]',  # Mixed audio
                        '-c:v', 'copy',
                        '-c:a', 'aac',
                        '-shortest',  # End when shortest stream ends
                        output_path
                    ]
                else:
                    # No audio cropping, but mix audio
                    filter_complex = f"[0:a]volume={original_audio_volume}[a0];[a0][1:a]amix=inputs=2:duration=shortest[aout]"
                    cmd = [
                        'ffmpeg',
                        '-y',
                        '-i', temp_video,
                        '-i', audio_path,
                        '-filter_complex', filter_complex,
                        '-map', '0:v:0',  # Video from first input
                        '-map', '[aout]',  # Mixed audio
                        '-c:v', 'copy',
                        '-c:a', 'aac',
                        '-shortest',  # End when shortest stream ends
                        output_path
                    ]
            else:
                # Original behavior: replace video audio with external audio
                if audio_start_time is not None and audio_duration is not None:
                    print(f"Audio crop: {audio_start_time:.2f}s - {audio_start_time + audio_duration:.2f}s (duration: {audio_duration:.2f}s)")
                    cmd = [
                        'ffmpeg',
                        '-y',
                        '-i', temp_video,
                        '-ss', str(audio_start_time),  # Audio start time
                        '-t', str(audio_duration),      # Audio duration
                        '-i', audio_path,
                        '-c:v', 'copy',
                        '-c:a', 'aac',
                        '-map', '0:v:0',  # Video from first input
                        '-map', '1:a:0',  # Audio from second input
                        '-shortest',  # End when shortest stream ends
                        output_path
                    ]
                else:
                    # No audio cropping
                    cmd = [
                        'ffmpeg',
                        '-y',
                        '-i', temp_video,
                        '-i', audio_path,
                        '-c:v', 'copy',
                        '-c:a', 'aac',
                        '-map', '0:v:0',  # Video from first input
                        '-map', '1:a:0',  # Audio from second input
                        '-shortest',  # End when shortest stream ends
                        output_path
                    ]
        else:
            # Just concatenate without additional audio
            cmd = [
                'ffmpeg',
                '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_file,
                '-c', 'copy',
                output_path
            ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE if not verbose else None,
            stderr=subprocess.PIPE if not verbose else None
        )

        if result.returncode != 0:
            print("Error creating final video")
            if not verbose and result.stderr:
                print(result.stderr.decode()[-500:])
            return False

    return True


def print_clip_summary(clips: List[Dict[str, Any]]):
    """Print a summary of all clips to be rendered."""
    total_duration = sum(c['duration'] for c in clips)
    adjusted_count = sum(1 for c in clips if c.get('adjusted', False))
    crop_center_count = sum(1 for c in clips if c.get('crop_center') is not None)

    print("\n" + "=" * 60)
    print("Clip Summary")
    print("=" * 60)

    current_section = -1
    for clip in clips:
        if clip['section_idx'] != current_section:
            current_section = clip['section_idx']
            print(f"\n[Section {current_section}]")

        adjustment_marker = " [ADJUSTED]" if clip.get('adjusted', False) else ""
        crop_marker = ""
        if clip.get('crop_center'):
            cx, cy = clip['crop_center']
            crop_marker = f" [CROP_CENTER: ({cx:.1f}, {cy:.1f})]"
        print(f"  Shot {clip['shot_idx']}: {clip['start_str']} - {clip['end_str']} ({clip['duration']:.2f}s){adjustment_marker}{crop_marker}")

    print("\n" + "-" * 60)
    print(f"Total clips: {len(clips)}")
    print(f"Total duration: {total_duration:.2f}s ({total_duration/60:.2f} min)")
    if adjusted_count > 0:
        print(f"Adjusted clips (snapped to scene cuts): {adjusted_count}")
    if crop_center_count > 0:
        print(f"Clips with dynamic crop center: {crop_center_count}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Render video from shot_point.json')
    parser.add_argument(
        '--shot-json',
        type=str,
        required=True,
        help='Path to shot_point.json file'
    )
    parser.add_argument(
        '--shot-plan',
        type=str,
        default=None,
        help='Optional path to shot_plan.json file (for audio time range detection in short videos)'
    )
    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help='Path to source video file'
    )
    parser.add_argument(
        '--audio',
        type=str,
        default=None,
        help='Optional path to audio file to mix with video'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path for output video file'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show ffmpeg output'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only print clip summary without rendering'
    )
    parser.add_argument(
        '--no-labels',
        action='store_true',
        help='Disable Section/Shot label overlay on video'
    )
    parser.add_argument(
        '--label-position',
        type=str,
        default='top-left',
        choices=['top-left', 'top-right', 'bottom-left', 'bottom-right'],
        help='Position of Section/Shot labels (default: top-left)'
    )
    parser.add_argument(
        '--font-size',
        type=int,
        default=32,
        help='Font size for labels (default: 32)'
    )
    parser.add_argument(
        '--font-color',
        type=str,
        default='white',
        help='Font color for labels (default: white)'
    )
    parser.add_argument(
        '--bg-color',
        type=str,
        default='black@0.6',
        help='Background color for label box with opacity (default: black@0.6)'
    )
    parser.add_argument(
        '--shot-scenes',
        type=str,
        default=None,
        help='Optional path to shot_scenes.txt file for scene cut detection and adjustment'
    )
    parser.add_argument(
        '--crop-ratio',
        type=str,
        default=None,
        help='Optional aspect ratio for center cropping (e.g., "9:16", "16:9", "1:1"). Keeps height unchanged and crops width to match the ratio.'
    )
    parser.add_argument(
        '--detection-short-side',
        type=int,
        default=360,
        help='Short side size used during protagonist detection (default: 360). Used to scale detection coordinates to video coordinates.'
    )
    parser.add_argument(
        '--visualize-detections',
        action='store_true',
        help='Visualize protagonist detection results on video (draw bounding boxes, crop center, and crop area)'
    )
    parser.add_argument(
        '--original-audio-volume',
        type=float,
        default=0.0,
        help='Volume level for original video audio when mixing with external audio (0.0 or higher). 0.0 = muted (default), 1.0 = full volume, >1.0 = amplified (e.g., 2.0 = double volume).'
    )

    args = parser.parse_args()

    # Validate original audio volume
    if args.original_audio_volume < 0.0:
        print(f"Error: --original-audio-volume must be 0.0 or higher (got {args.original_audio_volume})")
        return 1

    # Check input files exist
    if not os.path.exists(args.shot_json):
        print(f"Error: Shot JSON file not found: {args.shot_json}")
        return 1

    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return 1

    # Load shot data
    print(f"Loading shot data from: {args.shot_json}")
    with open(args.shot_json, 'r', encoding='utf-8') as f:
        shot_data = json.load(f)

    # Parse scene cut points if provided
    cut_points = None
    if args.shot_scenes and os.path.exists(args.shot_scenes):
        print(f"Loading scene cut points from: {args.shot_scenes}")
        fps = 2
        print(f"Video framerate: {fps:.2f} fps")
        cut_points = parse_shot_scenes(args.shot_scenes, fps)
        print(f"Loaded {len(cut_points)} scene cut points")

    # Get video dimensions for coordinate scaling
    video_width, video_height = get_video_dimensions(args.video)
    print(f"Video dimensions: {video_width}x{video_height}")

    # Extract all clips (with coordinate scaling if detection_short_side provided)
    clips = extract_all_clips(
        shot_data,
        cut_points,
        video_width=video_width,
        video_height=video_height,
        detection_short_side=args.detection_short_side
    )

    if not clips:
        print("Error: No valid clips found in shot data")
        return 1

    # Detect audio time range from shot_plan if provided
    audio_start_time = None
    audio_duration = None

    if args.shot_plan and os.path.exists(args.shot_plan):
        print(f"Loading shot plan from: {args.shot_plan}")
        with open(args.shot_plan, 'r', encoding='utf-8') as f:
            shot_plan = json.load(f)

        # Extract time range from video_structure
        if 'video_structure' in shot_plan and len(shot_plan['video_structure']) > 0:
            # For short videos, typically there's one audio section
            section = shot_plan['video_structure'][0]
            start_time_str = section.get('start_time', '0')
            end_time_str = section.get('end_time', '0')

            audio_start_time = float(start_time_str)
            audio_end_time = float(end_time_str)
            audio_duration = audio_end_time - audio_start_time

            print(f"Detected audio time range from shot_plan:")
            print(f"  Start: {audio_start_time:.1f}s")
            print(f"  End: {audio_end_time:.1f}s")
            print(f"  Duration: {audio_duration:.1f}s")

    # Print summary
    print_clip_summary(clips)

    if args.dry_run:
        print("Dry run mode - skipping render")
        return 0

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Render video
    print(f"Rendering video to: {args.output}")
    if not args.no_labels:
        print(f"Labels enabled: position={args.label_position}, font_size={args.font_size}")
    if args.crop_ratio:
        print(f"Crop ratio specified: {args.crop_ratio}")
    if args.visualize_detections:
        print(f"Detection visualization enabled")
    if args.audio and args.original_audio_volume > 0:
        print(f"Original audio volume: {args.original_audio_volume:.2f}")
    success = render_video_ffmpeg(
        video_path=args.video,
        clips=clips,
        output_path=args.output,
        audio_path=args.audio,
        audio_start_time=audio_start_time,
        audio_duration=audio_duration,
        verbose=args.verbose,
        show_labels=not args.no_labels,
        label_position=args.label_position,
        font_size=args.font_size,
        font_color=args.font_color,
        bg_color=args.bg_color,
        crop_ratio=args.crop_ratio,
        original_audio_volume=args.original_audio_volume,
        video_width=video_width,
        video_height=video_height,
        visualize_detections=args.visualize_detections
    )

    if success:
        print(f"\nSuccess! Video saved to: {args.output}")
        return 0
    else:
        print("\nFailed to render video")
        return 1


if __name__ == '__main__':
    exit(main())
