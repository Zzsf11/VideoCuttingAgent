"""
Simple Video Captioning with Fixed Window

This script captions video frames using a fixed time window and VLLM model.
"""

import argparse
import base64
import copy
import functools
import json
import multiprocessing as mp
import os
import sys
from typing import Dict, List, Tuple

import requests
from tqdm import tqdm
from pprint import pprint
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from .. import config

# --------------------------------------------------------------------------- #
#                              Prompt templates                               #
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = "You are a professional video analyst specializing in scene segmentation and temporal video understanding."

CAPTION_PROMPT = """
**Role**: You are an expert video editor specializing in shot segmentation—grouping temporally adjacent frames into coherent visual shots.

**Input**:  
A sequence of video frames sampled uniformly every **0.5 seconds**, each annotated with its timestamp (e.g., `<0.5s>`, `<1.0s>`, `<1.5s>`, …).  
Each frame may include a brief visual caption or be raw (assumed to be analyzable for shot continuity).

**Task**:  
Aggregate consecutive frames into **continuous shots (segments)** based on visual continuity.  
Only start a *new* segment when there is an unambiguous **shot boundary**, such as:
- A hard **cut** (sudden visual discontinuity),
- A **camera angle or position change** (e.g., wide → close-up),
- A **scene change** (location, lighting, or subject shift),
- Or other clear editorial transition (e.g., fade, wipe—label accordingly).

**Critical Constraints** (non-negotiable):
1. ✅ **Minimum segment duration: 3.0 seconds** — shorter visual units must be merged into adjacent shots.
2. ✅ **Preserve shot integrity**: Do *not* split a single continuous shot—even if minor motion or lighting changes occur (e.g., panning, zooming, actor movement).
3. ✅ **Prefer extension over fragmentation**: When in doubt, extend the current segment rather than create a new one.

**Output Format (strict JSON schema)**:
```json
{
  "clip_overview": "A single-sentence high-level summary of the entire clip content and structure.",
  "segments": [
    {
      "id": 1,
      "start_time": 0.0,
      "end_time": 4.5,
      "visual_type": "Cut|Dissolve|Fade|Wipe|Motion (e.g., Pan/Zoom)|Static",
      "description": "Concise, objective description of visual content — include subjects, actions, setting, and camera behavior."
    }
  ]
}
```

**Note**:  
- Timestamps must be in **seconds**, with `start_time` inclusive and `end_time` exclusive (segments should be contiguous and non-overlapping).  
- `visual_type` should reflect *how* this shot begins relative to the prior one (e.g., `"Cut"` for abrupt change, `"Motion"` for continuous camera movement within one shot).  
- For the *first* segment, `visual_type` may be `"Static"` or `"Motion"` (no preceding transition).

"""




# --------------------------------------------------------------------------- #
#                           VLLM Model Call Function                          #
# --------------------------------------------------------------------------- #

def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def debug_print_messages(processed_messages: List[Dict], max_base64_len: int = 30):
    """
    Print message structure for debugging, truncating base64 content.
    
    Args:
        processed_messages: The messages to print
        max_base64_len: Max length of base64 string to show
    """
    import copy
    debug_msgs = copy.deepcopy(processed_messages)
    
    for msg in debug_msgs:
        if isinstance(msg.get("content"), list):
            for item in msg["content"]:
                if item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url.startswith("data:image"):
                        # Truncate base64 content
                        prefix = url[:50]  # Keep "data:image/jpeg;base64," prefix
                        item["image_url"]["url"] = f"{prefix}...[BASE64_TRUNCATED]"
    
    print("=" * 60)
    print("DEBUG: Message Structure")
    print("=" * 60)
    for i, msg in enumerate(debug_msgs):
        print(f"\n[Message {i}] role: {msg['role']}")
        if isinstance(msg.get("content"), list):
            for j, item in enumerate(msg["content"]):
                item_type = item.get("type", "unknown")
                if item_type == "text":
                    print(f"  [{j}] type: text, text: {item.get('text', '')!r}")
                elif item_type == "image_url":
                    print(f"  [{j}] type: image_url, url: {item.get('image_url', {}).get('url', '')}")
                else:
                    print(f"  [{j}] {item}")
        else:
            content = msg.get("content", "")
            if len(content) > 100:
                content = content[:100] + "..."
            print(f"  content: {content!r}")
    print("=" * 60)


def call_vllm_model(
    messages: List[Dict],
    model_name: str = None,
    endpoint: str = "http://localhost:8000/v1/chat/completions",
    return_json: bool = False,
    image_paths: List[str] = None,
    image_timestamps: List[float] = None,
    max_tokens: int = 4096,
    temperature: float = 0.1,
) -> Dict:
    """
    Call vllm local deployed model.
    
    Args:
        messages: List of message dicts
        model_name: Model name
        endpoint: API endpoint
        return_json: Request JSON output
        image_paths: List of image file paths
        image_timestamps: List of timestamps (in seconds) for each image
        max_tokens: Max tokens to generate
        temperature: Sampling temperature
    """
    processed_messages = []
    for msg in messages:
        has_images = msg["role"] == "user" and image_paths
        
        if has_images:
            content_array = []
            if msg["content"]:
                content_array.append({"type": "text", "text": msg["content"]})
            
            # Add images with interleaved timestamps
            for idx, img_path in enumerate(image_paths):
                try:
                    # Add timestamp text before each image (displayed with frame-level precision)
                    if image_timestamps and idx < len(image_timestamps):
                        timestamp_sec = image_timestamps[idx]
                        # print(f"Adding image with timestamp: {timestamp_sec:.3f} seconds")
                        # Use 3 decimal places to support frame-level precision
                        # e.g., fps=30 -> 0.033s/frame, fps=2 -> 0.5s/frame
                        content_array.append({
                            "type": "text",
                            "text": f"<{timestamp_sec:.3f} seconds>"
                        })
                    # Add image
                    img_base64 = encode_image_to_base64(img_path)
                    content_array.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                    })
                except Exception as e:
                    print(f"Warning: Failed to encode image {img_path}: {e}")
            
            processed_messages.append({"role": msg["role"], "content": content_array})
        else:
            processed_messages.append({"role": msg["role"], "content": msg["content"]})
    
    # Debug: print message structure (without full base64)
    if os.environ.get("DEBUG_VLLM", "0") == "1":
        debug_print_messages(processed_messages)
    
    payload = {
        "messages": processed_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    
    if model_name:
        payload["model"] = model_name
    if return_json:
        payload["response_format"] = {"type": "json_object"}
    
    try:
        response = requests.post(endpoint, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            return {"content": result["choices"][0]["message"]["content"]}
        return {"content": None}
    except Exception as e:
        print(f"Error calling vllm model: {e}")
        return {"content": None}


# --------------------------------------------------------------------------- #
#                            Visualization utils                              #
# --------------------------------------------------------------------------- #

def visualize_scene_boundaries(
    frame_folder: str,
    captions: Dict,
    output_path: str,
    width: int = 25,
    show_frame_num: bool = True,
    max_frames: int = None,
    source_fps: float = None,
    target_fps: float = None,
    max_seconds: float = None,
    start_seconds: float = None,
):
    """
    Visualize video frames with scene boundary markers.
    Following AutoShot visualization style.

    Args:
        frame_folder: Path to folder containing video frames
        captions: Caption results dict with scene_boundaries
        output_path: Path to save visualization image
        width: Number of frames per row
        show_frame_num: Whether to show frame numbers
        max_frames: Maximum frames to visualize (None = no limit)
        source_fps: FPS of the source frames in frame_folder (None = use config.VIDEO_FPS)
        target_fps: Target FPS for processing/visualization (None = use config.SHOT_DETECTION_FPS)
        max_seconds: Maximum video duration in seconds to visualize (None = no limit)
        start_seconds: Start time in seconds to visualize (None = 0)

    Returns:
        PIL Image object
    """
    if source_fps is None:
        source_fps = config.VIDEO_FPS
    if target_fps is None:
        target_fps = config.SHOT_DETECTION_FPS
    if start_seconds is None:
        start_seconds = 0.0

    # Collect all frame files
    all_frame_files = sorted(
        [f for f in os.listdir(frame_folder)
         if f.startswith("frame") and (f.endswith(".jpg") or f.endswith(".png"))],
        key=lambda x: float(x.split("_")[-1].rstrip(".jpg").rstrip(".png")),
    )

    if not all_frame_files:
        print("No frames found for visualization")
        return None

    # Calculate frame sampling interval based on source and target fps
    # e.g., source_fps=2, target_fps=1 -> sample every 2nd frame
    sample_interval = max(1, int(source_fps / target_fps))

    # Sample frames according to target_fps
    sampled_frame_files = all_frame_files[::sample_interval]

    # Filter frames by start_seconds and max_seconds (only visualize processed duration)
    if max_seconds is not None or start_seconds > 0:
        sampled_frame_files = [
            f for f in sampled_frame_files
            if start_seconds <= float(f.split("_")[-1].rstrip(".jpg").rstrip(".png")) / source_fps and
               (max_seconds is None or float(f.split("_")[-1].rstrip(".jpg").rstrip(".png")) / source_fps <= max_seconds)
        ]
    
    # Then apply max_frames limit if specified
    if max_frames is not None:
        frame_files = sampled_frame_files[:max_frames]
    else:
        frame_files = sampled_frame_files
    
    num_frames = len(frame_files)
    
    # Load frames into numpy array (like AutoShot)
    print(f"Loading {num_frames} frames for visualization (sampled from {len(all_frame_files)} total, interval={sample_interval})...")
    frames_list = []
    for frame_file in tqdm(frame_files, desc="Loading frames"):
        frame_path = os.path.join(frame_folder, frame_file)
        try:
            img = Image.open(frame_path).convert('RGB')
            # Resize to small size (48x27 like AutoShot)
            img = img.resize((48, 27), Image.Resampling.BILINEAR)
            frames_list.append(np.array(img))
        except Exception as e:
            print(f"Warning: Failed to load {frame_file}: {e}")
            frames_list.append(np.zeros((27, 48, 3), dtype=np.uint8))
    
    frames = np.array(frames_list)  # Shape: (N, 27, 48, 3)
    
    # Get frame timestamps (frame_index / source_fps = actual time in seconds)
    frame_indices = [float(f.split("_")[-1].rstrip(".jpg").rstrip(".png")) for f in frame_files]
    frame_timestamps = [idx / source_fps for idx in frame_indices]
    
    # Collect all scene boundaries from captions
    all_boundaries = []
    for clip_key, clip_data in captions.items():
        for boundary in clip_data.get("scene_boundaries", []):
            all_boundaries.append(boundary)
    all_boundaries.sort(key=lambda x: x.get("timestamp", 0))
    
    # Create binary prediction array (1 = boundary, 0 = not boundary)
    predictions = np.zeros(num_frames, dtype=np.float32)
    for boundary in all_boundaries:
        ts = boundary.get("timestamp", 0)
        # Find nearest frame
        nearest_idx = min(range(len(frame_timestamps)), 
                         key=lambda i: abs(frame_timestamps[i] - ts))
        if nearest_idx < num_frames:
            predictions[nearest_idx] = 1.0
    
    # Visualize using AutoShot style
    print(f"Generating visualization: {num_frames} frames, {len(all_boundaries)} boundaries")
    img = _visualize_autoshot_style(
        frames, 
        predictions=predictions,
        width=width,
        show_frame_num=show_frame_num,
        frame_timestamps=frame_timestamps,
    )
    
    # Save image
    img.save(output_path)
    print(f"Visualization saved to: {output_path}")
    
    return img


def _visualize_autoshot_style(
    frames: np.ndarray,
    predictions: np.ndarray,
    width: int = 25,
    show_frame_num: bool = True,
    frame_timestamps: List[float] = None,
):
    """
    Visualize frames using AutoShot style.
    
    Args:
        frames: numpy array of shape (N, H, W, C)
        predictions: binary predictions array (1 = boundary)
        width: number of frames per row
        show_frame_num: whether to show frame numbers
        frame_timestamps: optional timestamps for each frame
    
    Returns:
        PIL Image object
    """
    ih, iw, ic = frames.shape[1:]  # 27, 48, 3
    num_predictions = 1  # We only have one prediction channel
    
    # Pad frames to make divisible by width
    pad_with = width - len(frames) % width if len(frames) % width != 0 else 0
    frames = np.pad(frames, [(0, pad_with), (0, 1), (0, num_predictions), (0, 0)])
    predictions = np.pad(predictions, (0, pad_with))
    
    height = len(frames) // width
    
    # Reshape into grid
    img = frames.reshape([height, width, ih + 1, iw + num_predictions, ic])
    img_tmp = np.concatenate(np.split(
        np.concatenate(np.split(img, height), axis=2)[0], width
    ), axis=2)[0, :-1]
    
    img = Image.fromarray(img_tmp)
    draw = ImageDraw.Draw(img)
    
    # Load font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 8)
    except:
        font = ImageFont.load_default()
    
    # Draw frame numbers
    if show_frame_num:
        for h in range(height):
            for w in range(width):
                n = h * width + w
                if n >= len(frames) - pad_with:
                    break
                
                # Calculate background brightness for text color
                region = img_tmp[h * (ih + 1) + 2 : h * (ih + 1) + 10, 
                               w * (iw + 1) : w * (iw + 1) + 20, :]
                avg_c = region.mean() if region.size > 0 else 128
                text_color = (255, 255, 255) if avg_c < 128 else (0, 0, 0)
                
                # Show timestamp instead of frame index
                if frame_timestamps and n < len(frame_timestamps):
                    # Show precise timestamp (1 decimal place for readability in visualization)
                    label = f"{frame_timestamps[n]:.1f}s"
                else:
                    label = str(n)
                
                draw.text(
                    (w * (iw + num_predictions), h * (ih + 1) + 2),
                    label,
                    fill=text_color,
                    font=font
                )
    
    # Draw boundary predictions (red border around frames)
    # Scene boundaries are marked by drawing a red rectangle around the entire frame
    for i in range(len(predictions) - pad_with):
        x_pos = i % width
        y_pos = i // width

        pred_val = float(predictions[i])
        if pred_val > 0:
            # Draw red rectangle around the entire frame at boundary timestamp
            x_left = x_pos * (iw + num_predictions)
            x_right = x_left + iw - 1
            y_top = y_pos * (ih + 1)
            y_bottom = y_top + ih - 1

            # Draw red border (rectangle outline) around the frame
            draw.rectangle(
                [(x_left, y_top), (x_right, y_bottom)],
                outline=(255, 0, 0),
                width=2
            )
    
    return img


# --------------------------------------------------------------------------- #
#                               Helper utils                                  #
# --------------------------------------------------------------------------- #

def convert_seconds_to_hhmmss(seconds: float) -> str:
    h = int(seconds // 3600)
    seconds %= 3600
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02}"


def parse_srt_to_dict(srt_path: str) -> Dict[str, str]:
    """Parse .srt file to dict: '{startSec_endSec}': 'text'."""
    if not os.path.isfile(srt_path):
        return {}

    def ts_to_sec(ts: str) -> float:
        hh, mm, rest = ts.split(":")
        ss, ms = rest.split(",")
        return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0

    result = {}
    with open(srt_path, "r", encoding="utf-8") as fh:
        lines = [l.rstrip("\n") for l in fh]

    idx, n = 0, len(lines)
    while idx < n:
        if lines[idx].strip().isdigit():
            idx += 1
        if idx >= n:
            break
        if "-->" not in lines[idx]:
            idx += 1
            continue
        start_ts, end_ts = [t.strip() for t in lines[idx].split("-->")]
        start_sec, end_sec = int(ts_to_sec(start_ts)), int(ts_to_sec(end_ts))
        idx += 1

        subtitle_lines = []
        while idx < n and lines[idx].strip():
            subtitle_lines.append(lines[idx].strip())
            idx += 1
        subtitle = " ".join(subtitle_lines)
        key = f"{start_sec}_{end_sec}"
        result[key] = result.get(key, "") + " " + subtitle if key in result else subtitle
        idx += 1
    return result


def gather_clip_frames(
    frame_folder: str,
    window_secs: int,
    subtitle_file_path: str = None,
    max_minutes: float = None,
    source_fps: float = None,
    target_fps: float = None,
    start_time: float = None,
    end_time: float = None,
) -> List[Tuple[str, Dict]]:
    """
    Gather frames into fixed-length clips.

    Args:
        frame_folder: Path to folder containing video frames
        window_secs: Window length in seconds
        subtitle_file_path: Optional path to subtitle file
        max_minutes: Maximum video duration to process in minutes (default: config.VIDEO_MAX_MINUTES)
        source_fps: FPS of the source frames in frame_folder (default: config.VIDEO_FPS)
        target_fps: Target FPS for processing (default: config.SHOT_DETECTION_FPS)
        start_time: Start time in seconds (default: 0)
        end_time: End time in seconds (default: None, uses max_minutes or full video)

    Returns:
        List of (timestamp_key, {"files": [...], "timestamps": [...], "transcript": "..."})
    """
    # Use config value if not specified
    if max_minutes is None:
        max_minutes = config.VIDEO_MAX_MINUTES
    if source_fps is None:
        source_fps = config.VIDEO_FPS
    if target_fps is None:
        target_fps = config.SHOT_DETECTION_FPS
    if start_time is None:
        start_time = 0.0

    # Determine end time: prioritize end_time parameter, then max_minutes
    if end_time is None:
        max_seconds = max_minutes * 60
    else:
        max_seconds = end_time

    # Ensure start_time < end_time
    if start_time >= max_seconds:
        print(f"Error: start_time ({start_time}s) must be less than end_time ({max_seconds}s)")
        return []
    
    frame_files = sorted(
        [f for f in os.listdir(frame_folder) 
         if f.startswith("frame") and (f.endswith(".jpg") or f.endswith(".png"))],
        key=lambda x: float(x.split("_")[-1].rstrip(".jpg").rstrip(".png")),
    )
    if not frame_files:
        return []

    subtitle_map = parse_srt_to_dict(subtitle_file_path) if subtitle_file_path else {}

    # Calculate frame sampling interval based on source and target fps
    # e.g., source_fps=2, target_fps=1 -> sample every 2nd frame
    sample_interval = max(1, int(source_fps / target_fps))
    
    # Sample frames according to target_fps
    sampled_frame_files = frame_files[::sample_interval]
    print(f"Sampling frames: {len(frame_files)} total -> {len(sampled_frame_files)} sampled (interval={sample_interval}, source_fps={source_fps}, target_fps={target_fps})")

    # Map frame index to timestamps (seconds)
    # Frame filename format: frame_{frame_number}.jpg
    # Timestamp = frame_number / source_fps (source_fps is the fps of frames in the folder)
    frame_ts = [float(f.split("_")[-1].rstrip(".jpg").rstrip(".png")) / source_fps for f in sampled_frame_files]

    # Apply start_time and end_time filters
    original_min_ts = min(frame_ts) if frame_ts else 0
    original_max_ts = max(frame_ts) if frame_ts else 0
    frame_ts = [t for t in frame_ts if start_time <= t <= max_seconds]

    if not frame_ts:
        print(f"Warning: No frames within time range [{start_time}s, {max_seconds}s]")
        return []

    if original_min_ts < start_time or original_max_ts > max_seconds:
        actual_start = max(original_min_ts, start_time)
        actual_end = min(original_max_ts, max_seconds)
        print(f"Note: Processing video from {actual_start:.1f}s to {actual_end:.1f}s (filtered from [{original_min_ts:.1f}s, {original_max_ts:.1f}s])")

    # Rebuild mapping with filtered frames
    filtered_frame_files = [f for f in sampled_frame_files
                           if start_time <= float(f.split("_")[-1].rstrip(".jpg").rstrip(".png")) / source_fps <= max_seconds]
    frame_ts = [float(f.split("_")[-1].rstrip(".jpg").rstrip(".png")) / source_fps for f in filtered_frame_files]
    ts_to_file = dict(zip(frame_ts, filtered_frame_files))
    last_ts = int(max(frame_ts))

    result = []
    clip_start = int(start_time)  # Start from start_time instead of 0
    
    while clip_start <= last_ts:
        clip_end = min(clip_start + window_secs - 1, last_ts)

        # Collect frames and their timestamps in this window
        clip_files = []
        clip_timestamps = []
        for t in frame_ts:
            if clip_start <= t <= clip_end:
                clip_files.append(os.path.join(frame_folder, ts_to_file[t]))
                clip_timestamps.append(t)

        # Aggregate transcript
        transcript_parts = []
        for key, text in subtitle_map.items():
            s, e = map(int, key.split("_"))
            if s <= clip_end and e >= clip_start:
                transcript_parts.append(text)
        transcript = " ".join(transcript_parts).strip() or "No transcript."

        result.append((f"{clip_start}_{clip_end}", {
            "files": clip_files,
            "timestamps": clip_timestamps,
            "transcript": transcript
        }))
        clip_start += window_secs

    return result


# --------------------------------------------------------------------------- #
#                        Caption single clip                                  #
# --------------------------------------------------------------------------- #

def _caption_clip(task: Tuple[str, Dict], ckpt_folder: str) -> Tuple[str, dict]:
    """Caption one clip. Returns (timestamp_key, parsed_json)."""
    timestamp, info = task
    files = info["files"]
    timestamps = info.get("timestamps", [])  # These are absolute timestamps
    transcript = info["transcript"]

    start_sec = float(timestamp.split("_")[0])
    end_sec = float(timestamp.split("_")[1])
    clip_duration = end_sec - start_sec
    clip_start_time = convert_seconds_to_hhmmss(start_sec)
    clip_end_time = convert_seconds_to_hhmmss(end_sec)
    
    # Convert absolute timestamps to relative timestamps (starting from 0)
    relative_timestamps = [t - start_sec for t in timestamps]

    # Check cache
    cache_path = os.path.join(ckpt_folder, f"{timestamp}.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cached = json.load(f)
            cached.setdefault("clip_start_time", clip_start_time)
            cached.setdefault("clip_end_time", clip_end_time)
            return timestamp, cached

    # Build prompt with clip duration info
    prompt = CAPTION_PROMPT.replace("TRANSCRIPT_PLACEHOLDER", transcript)
    prompt = prompt.replace("CLIP_DURATION_PLACEHOLDER", str(int(clip_duration)))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    tries = 3
    while tries:
        tries -= 1
        resp = call_vllm_model(
            messages,
            endpoint=config.VLLM_ENDPOINT,
            model_name=config.VIDEO_ANALYSIS_MODEL,
            return_json=True,
            image_paths=files,
            image_timestamps=relative_timestamps,  # Pass relative timestamps (starting from 0)
            max_tokens=config.VIDEO_ANALYSIS_MODEL_MAX_TOKEN,
        )["content"]
        
        if resp is None:
            continue
        try:
            parsed = json.loads(resp)
            
            # Validate required fields (new format uses clip_overview and segments)
            if "clip_overview" not in parsed and "clip_description" not in parsed:
                print(f"Warning: Missing 'clip_overview' in response for {timestamp}. Retrying...")
                continue
            
            # Add metadata
            parsed["clip_start_time"] = clip_start_time
            parsed["clip_end_time"] = clip_end_time
            parsed["start_sec"] = start_sec
            parsed["end_sec"] = end_sec
            
            # Convert segments to scene_boundaries format for visualization compatibility
            # New format: segments with start_time/end_time
            # Old format: scene_boundaries with timestamp
            if "segments" in parsed:
                scene_boundaries = []
                for seg in parsed["segments"]:
                    # Create boundary at the start of each segment (except the first one)
                    if seg.get("id", 1) > 1:  # Skip first segment, it starts at 0
                        boundary = {
                            "timestamp": start_sec + seg.get("start_time", 0),  # Convert to absolute time
                            "type": seg.get("visual_type", "Cut"),
                            "description": seg.get("description", "")
                        }
                        scene_boundaries.append(boundary)
                parsed["scene_boundaries"] = scene_boundaries
            elif "scene_boundaries" not in parsed:
                parsed["scene_boundaries"] = []
            else:
                # Old format: convert relative timestamps to absolute
                for boundary in parsed["scene_boundaries"]:
                    if "timestamp" in boundary:
                        relative_ts = boundary["timestamp"]
                        boundary["timestamp"] = start_sec + relative_ts
            
            with open(cache_path, "w") as f:
                json.dump(parsed, f, indent=2)
            return timestamp, parsed
        except json.JSONDecodeError as e:
            print(f"Warning: JSON decode error for {timestamp}: {e}")
            continue
    
    return timestamp, {}


# --------------------------------------------------------------------------- #
#                     Main processing function                                #
# --------------------------------------------------------------------------- #

def process_video(
    frame_folder: str,
    output_folder: str,
    window_secs: int = None,
    subtitle_file: str = None,
    num_workers: int = 8,
    visualize: bool = True,
    max_minutes: float = None,
    source_fps: float = None,
    target_fps: float = None,
    start_time: float = None,
    end_time: float = None,
):
    """
    Process video with fixed window captioning.

    Args:
        frame_folder: Path to folder containing video frames
        output_folder: Path to save caption outputs
        window_secs: Fixed window length in seconds (default: config.CLIP_SECS)
        subtitle_file: Optional path to subtitle file (.srt)
        num_workers: Number of parallel workers
        visualize: Whether to generate visualization
        max_minutes: Maximum video duration to process in minutes (default: config.VIDEO_MAX_MINUTES)
        source_fps: FPS of the source frames in frame_folder (default: config.VIDEO_FPS)
        target_fps: Target FPS for processing (default: config.SHOT_DETECTION_FPS)
        start_time: Start time in seconds (default: 0)
        end_time: End time in seconds (default: None, uses max_minutes)
    """
    # Use config values if not specified
    if window_secs is None:
        window_secs = config.CLIP_SECS
    if max_minutes is None:
        max_minutes = config.VIDEO_MAX_MINUTES
    if source_fps is None:
        source_fps = config.VIDEO_FPS
    if target_fps is None:
        target_fps = config.SHOT_DETECTION_FPS
    
    ckpt_folder = os.path.join(output_folder, "ckpt")
    os.makedirs(ckpt_folder, exist_ok=True)

    print(f"Using fixed window captioning with {window_secs}s window")
    if start_time is not None or end_time is not None:
        st = start_time if start_time is not None else 0.0
        et = end_time if end_time is not None else max_minutes * 60
        print(f"Processing time range: {st:.1f}s to {et:.1f}s")
    else:
        print(f"Max video duration: {max_minutes} minutes ({max_minutes * 60:.0f} seconds)")
    print(f"Source FPS: {source_fps}, Target FPS: {target_fps}")
    clips = gather_clip_frames(
        frame_folder,
        window_secs,
        subtitle_file,
        max_minutes=max_minutes,
        source_fps=source_fps,
        target_fps=target_fps,
        start_time=start_time,
        end_time=end_time
    )
    print(f"Total clips: {len(clips)}")

    caption_fn = functools.partial(_caption_clip, ckpt_folder=ckpt_folder)

    # Parallel captioning
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(caption_fn, clips),
            total=len(clips),
            desc="Captioning"
        ))

    # Sort and save results
    results = sorted(results, key=lambda x: float(x[0].split("_")[0]))
    
    # Collect all scene boundaries and captions
    captions = {}
    all_boundaries = []
    
    for ts, parsed in results:
        if parsed:
            captions[ts] = {
                "clip_overview": parsed.get("clip_overview", parsed.get("clip_description", "")),
                "segments": parsed.get("segments", []),
                "scene_boundaries": parsed.get("scene_boundaries", []),
                "clip_start_time": parsed.get("clip_start_time", ""),
                "clip_end_time": parsed.get("clip_end_time", ""),
                "start_sec": parsed.get("start_sec", 0),
                "end_sec": parsed.get("end_sec", 0),
            }
            # Collect boundaries for global view
            for boundary in parsed.get("scene_boundaries", []):
                all_boundaries.append(boundary)

    # Sort all boundaries by timestamp
    all_boundaries.sort(key=lambda x: x.get("timestamp", 0))

    # Save outputs
    output_path = os.path.join(output_folder, "captions.json")
    with open(output_path, "w") as f:
        json.dump(captions, f, indent=2, ensure_ascii=False)
    
    boundaries_path = os.path.join(output_folder, "scene_boundaries.json")
    with open(boundaries_path, "w") as f:
        json.dump({
            "total_boundaries": len(all_boundaries),
            "boundaries": all_boundaries
        }, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(captions)} captions to {output_path}")
    print(f"Saved {len(all_boundaries)} scene boundaries to {boundaries_path}")
    
    # Generate shot_scenes.txt format (start_frame end_frame per line)
    # Collect all segments with their absolute time ranges
    all_segments = []
    for ts, clip_data in captions.items():
        clip_start_sec = clip_data.get("start_sec", 0)
        
        # If segments are available, use them
        if clip_data.get("segments"):
            for seg in clip_data.get("segments", []):
                seg_start_sec = clip_start_sec + seg.get("start_time", 0)
                seg_end_sec = clip_start_sec + seg.get("end_time", 0)
                all_segments.append({
                    "start_sec": seg_start_sec,
                    "end_sec": seg_end_sec,
                    "description": seg.get("description", ""),
                    "visual_type": seg.get("visual_type", "")
                })
        # Fallback: if no segments but scene_boundaries exist, reconstruct segments
        elif clip_data.get("scene_boundaries"):
            boundaries = sorted(clip_data.get("scene_boundaries", []), key=lambda x: x.get("timestamp", 0))
            # Add start and end of clip as implicit boundaries
            clip_end_sec = clip_data.get("end_sec", clip_start_sec + window_secs)
            
            current_start = clip_start_sec
            for b in boundaries:
                b_time = b.get("timestamp", 0)
                if b_time > current_start:
                    all_segments.append({
                        "start_sec": current_start,
                        "end_sec": b_time,
                        "description": b.get("description", ""),
                        "visual_type": b.get("type", "")
                    })
                    current_start = b_time
            
            # Add final segment
            if current_start < clip_end_sec:
                all_segments.append({
                    "start_sec": current_start,
                    "end_sec": clip_end_sec,
                    "description": "End of clip segment",
                    "visual_type": "End"
                })

    # Sort by start time and convert to frame indices
    all_segments.sort(key=lambda x: x["start_sec"])
    
    # Merge overlapping or adjacent segments from different clips
    # Logic: 
    # 1. Sort segments by start time
    # 2. Iterate through segments
    # 3. If current segment starts before or exactly when previous ends (with small tolerance), merge them?
    #    Actually, for shot detection, we usually want distinct shots.
    #    However, since we process in windows, we might have split a shot across windows.
    #    If the last shot of window N and first shot of window N+1 are actually the same shot,
    #    we should merge them. But how do we know?
    #    The prompt asks to "Aggregate consecutive frames into continuous shots".
    #    If the model is consistent, it should detect a cut at the boundary if there is one.
    #    If there is no cut at the window boundary, it implies the shot continues.
    
    merged_segments = []
    if all_segments:
        current_seg = all_segments[0].copy()
        
        for next_seg in all_segments[1:]:
            # Check for continuity
            # If next segment starts roughly where current ends (within small epsilon)
            # AND there is no explicit "Cut" or visual transition marked at the start of next_seg
            # (Note: our current data structure might not fully capture "start type" for the first segment of a clip perfectly
            # unless we look at visual_type)
            
            time_gap = next_seg["start_sec"] - current_seg["end_sec"]
            
            # If segments are adjacent (gap is small, e.g. < 0.1s)
            if abs(time_gap) < 0.1:
                # Heuristic: If the next segment starts with "Motion" or "Static" (continuation) 
                # rather than "Cut", we might merge.
                # But simpler logic for now: just merge if they touch, to avoid artificial cuts at window boundaries?
                # NO, that would merge everything.
                # We should only merge if the boundary is NOT a real shot change.
                # But we don't have a "is_shot_change" flag easily available across windows unless we infer it.
                
                # Let's look at the visual_type of the next segment.
                # If it says "Cut", "Dissolve", "Wipe", it's a new shot.
                # If it says "Motion", "Static", or is empty, it might be a continuation.
                
                visual_type = next_seg.get("visual_type", "").lower()
                is_transition = any(t in visual_type for t in ["cut", "dissolve", "fade", "wipe"])
                
                if not is_transition:
                    # Merge
                    current_seg["end_sec"] = next_seg["end_sec"]
                    # Update description? Maybe append or keep longest? Keep first for now.
                else:
                    # It's a new shot, push current and start new
                    merged_segments.append(current_seg)
                    current_seg = next_seg.copy()
            else:
                # Gap exists or overlap is weird, treat as separate
                merged_segments.append(current_seg)
                current_seg = next_seg.copy()
        
        merged_segments.append(current_seg)
    
    # Convert seconds to frame indices (using source_fps)
    shot_scenes_path = os.path.join(output_folder, "shot_scenes.txt")
    with open(shot_scenes_path, "w") as f:
        for seg in merged_segments:
            start_frame = int(seg["start_sec"] * source_fps)
            end_frame = int(seg["end_sec"] * source_fps)
            # Ensure start < end
            if end_frame > start_frame:
                f.write(f"{start_frame} {end_frame}\n")
    
    print(f"Saved {len(merged_segments)} shots to {shot_scenes_path}")
    
    # Generate visualization
    if visualize:
        # Determine actual time range for visualization
        if end_time is not None:
            vis_max_seconds = end_time
        else:
            vis_max_seconds = max_minutes * 60

        if start_time is not None:
            vis_start = start_time
        else:
            vis_start = 0.0

        print(f"\nGenerating visualization (time range: {vis_start:.1f}s - {vis_max_seconds:.1f}s)...")
        vis_path = os.path.join(output_folder, "scene_boundaries_vis.png")
        visualize_scene_boundaries(
            frame_folder=frame_folder,
            captions=captions,
            output_path=vis_path,
            width=25,
            show_frame_num=True,
            source_fps=source_fps,
            target_fps=target_fps,
            max_seconds=vis_max_seconds,
            start_seconds=vis_start,
        )
    
    return captions
def main():
    parser = argparse.ArgumentParser(description="Fixed window video captioning with VLLM")
    parser.add_argument("--frame_folder", type=str, required=True,
                        help="Path to folder containing video frames")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Path to save caption outputs")
    parser.add_argument("--window_secs", type=int, default=None,
                        help=f"Window length in seconds (default: {config.CLIP_SECS})")
    parser.add_argument("--subtitle_file", type=str, default=None,
                        help="Optional path to subtitle file (.srt)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    parser.add_argument("--max_minutes", type=float, default=None,
                        help=f"Maximum video duration to process in minutes (default: {config.VIDEO_MAX_MINUTES})")
    parser.add_argument("--source_fps", type=float, default=None,
                        help=f"FPS of source frames in frame_folder (default: {config.VIDEO_FPS})")
    parser.add_argument("--target_fps", type=float, default=None,
                        help=f"Target FPS for processing/visualization (default: {config.SHOT_DETECTION_FPS})")
    parser.add_argument("--start_time", type=float, default=None,
                        help="Start time in seconds (default: 0)")
    parser.add_argument("--end_time", type=float, default=None,
                        help="End time in seconds (default: None, uses max_minutes)")
    parser.add_argument("--visualize", action="store_true", default=True,
                        help="Generate visualization (default: True)")
    parser.add_argument("--no-visualize", action="store_false", dest="visualize",
                        help="Disable visualization")

    args = parser.parse_args()
    
    process_video(
        frame_folder=args.frame_folder,
        output_folder=args.output_folder,
        window_secs=args.window_secs,
        subtitle_file=args.subtitle_file,
        num_workers=args.num_workers,
        visualize=args.visualize,
        max_minutes=args.max_minutes,
        source_fps=args.source_fps,
        target_fps=args.target_fps,
        start_time=args.start_time,
        end_time=args.end_time,
    )


if __name__ == "__main__":
    main()
