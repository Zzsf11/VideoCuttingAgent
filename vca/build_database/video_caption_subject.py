import copy
import functools
import json
import multiprocessing as mp
import os
from typing import Dict, List, Tuple
import base64
import requests

from tqdm import tqdm

# Use relative import since this file is inside vca package
from .. import config

# --------------------------------------------------------------------------- #
#                              Prompt templates                               #
# --------------------------------------------------------------------------- #

messages = [
    {
        "role": "system",
        "content": ""
    },
    {
        "role": "user",
        "content": "",
    },
]


# CAPTION_PROMPT = """There are consecutive frames from a video. Please understand the video clip with the given transcript then output JSON in the template below.

# Transcript of current clip:
# TRANSCRIPT_PLACEHOLDER

# Output template:
# {
#   "subject_registry": {
#     <subject_i>: {
#       "name": <fill with short identity if name is unknown>,
#       "appearance": <list of appearance descriptions>,
#       "identity": <list of identity descriptions>,
#       "first_seen": <timestamp>
#     },
#     ...
#   "clip_description": <smooth and detailed natural narration of the video clip>,
#   "shot type": <shot type>,
#   "emotion": <emotion>
# }
# """

CAPTION_PROMPT = """There are consecutive frames from a video. Please understand the video clip with the given transcript then output JSON in the template below.

Transcript of current clip:
TRANSCRIPT_PLACEHOLDER

Output template:
{
  "clip_description": <smooth and detailed natural narration of the video clip>,
  "shot_type": <shot type>,
  "emotion": <emotion>
}
"""


# MERGE_PROMPT = """You are given several partial `new_subject_registry` JSON objects extracted from different clips of the *same* video. They may contain duplicated subjects with slightly different IDs or descriptions.

# Task:
# 1. Merge these partial registries into one coherent `subject_registry`.
# 2. Preserve all unique subjects.
# 3. If two subjects obviously refer to the same person, merge them
#    (keep earliest `first_seen` time and union all fields).

# Input (list of JSON objects):
# REGISTRIES_PLACEHOLDER

# Return *only* the merged `subject_registry` JSON object.
# """

MERGE_PROMPT = """You are given several partial `new_subject_registry` JSON objects extracted from different clips of the *same* video. They may contain duplicated subjects with slightly different IDs or descriptions.

Task:
1. Merge these partial registries into one coherent `subject_registry`.
2. Preserve all unique subjects.
3. If two subjects obviously refer to the same person, merge them
   (keep earliest `first_seen` time and union all fields).

Input 
"""

    # Prepare merge prompt
# SCENE_MERGE_PROMPT = """You are given several caption JSONs from consecutive sub-clips of the SAME scene. 
# Please merge them into one coherent scene description.

# Sub-clip captions:
# SUBCLIPS_PLACEHOLDER

# Output a JSON in this format:
# {
#   "clip_description": <merged smooth narration of the entire scene>,
#   "shot_type": <most representative shot type>,
#   "emotion": <dominant emotion across the scene>,
#   "subject_registry": <merged subject registry>
# }
# """
SCENE_MERGE_PROMPT = """You are given several caption JSONs from consecutive sub-clips of the SAME scene. 
Please merge them into one coherent scene description.

Sub-clip captions:
SUBCLIPS_PLACEHOLDER

Output a JSON in this format:
{
  "clip_description": <merged smooth narration of the entire scene>,
  "shot_type": <most representative shot type>,
  "emotion": <dominant emotion across the scene>,
}
"""

SYSTEM_PROMPT = "You are a helpful assistant."

# --------------------------------------------------------------------------- #
#                           VLLM Model Call Function                          #
# --------------------------------------------------------------------------- #
def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_vllm_model(
    messages: List[Dict],
    model_name: str = None,
    endpoint: str = "http://localhost:8000/v1/chat/completions",
    return_json: bool = False,
    image_paths: List[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
) -> Dict:
    """
    Call vllm local deployed model.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        model_name: Model name (optional, will use default from vllm server)
        endpoint: vllm server endpoint URL
        return_json: Whether to request JSON output
        image_paths: List of image file paths to include
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    
    Returns:
        Dict with 'content' key containing the model response
    """
    # Prepare messages with images if provided
    processed_messages = []
    for msg in messages:
        # Check if this message needs multimodal format (has images)
        has_images = msg["role"] == "user" and image_paths
        
        if has_images:
            # Use array format for multimodal content
            content_array = []
            
            # Add text content
            if msg["content"]:
                content_array.append({
                    "type": "text",
                    "text": msg["content"]
                })
            
            # Add images
            for img_path in image_paths:
                try:
                    img_base64 = encode_image_to_base64(img_path)
                    content_array.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}"
                        }
                    })
                except Exception as e:
                    print(f"Warning: Failed to encode image {img_path}: {e}")
            
            processed_messages.append({
                "role": msg["role"],
                "content": content_array
            })
        else:
            # Use simple string format for text-only messages
            processed_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
    
    # Prepare request payload
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
        
        # Extract content from response
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"]
            return {"content": content}
        else:
            return {"content": None}
    except requests.exceptions.HTTPError as e:
        print(f"Error calling vllm model: {e}")
        print(f"Response text: {response.text}")
        print(f"Payload preview (first 500 chars): {str(payload)[:500]}")
        return {"content": None}
    except Exception as e:
        print(f"Error calling vllm model: {e}")
        return {"content": None}

# --------------------------------------------------------------------------- #
#                               Helper utils                                  #
# --------------------------------------------------------------------------- #
def convert_seconds_to_hhmmss(seconds: float) -> str:
    h = int(seconds // 3600)
    seconds %= 3600
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02}"


def gather_frames_from_time_ranges(
    frame_folder: str, time_ranges: List[Tuple[int, int, str]]
) -> Dict[str, Dict]:
    """Return a dict keyed by 't1_t2' -> {files, transcript}."""
    frame_files = sorted(
        [f for f in os.listdir(frame_folder) if f.endswith(".jpg")],
        key=lambda x: float(x.split("_")[-1].rstrip(".jpg")),
    )
    result = {}
    for t1, t2, text in time_ranges:
        files = frame_files[t1 : t2 + 1]
        result[f"{t1}_{t2}"] = {
            "files": [os.path.join(frame_folder, f) for f in files],
            "transcript": text or "No transcript.",
        }
    return result

def parse_shot_scenes(shot_scenes_path: str) -> List[Tuple[int, int]]:
    """
    Parse shot_scenes.txt file to get list of (start_frame, end_frame) tuples.
    Frame numbers are based on SHOT_DETECTION_FPS.
    """
    scenes = []
    if not os.path.isfile(shot_scenes_path):
        return scenes
    
    with open(shot_scenes_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                start_frame = int(parts[0])
                end_frame = int(parts[1])
                scenes.append((start_frame, end_frame))
    return scenes


def gather_clip_frames_from_scenes(
    video_frame_folder: str,
    shot_scenes_path: str,
    clip_secs: int,
    subtitle_file_path: str = None
) -> List[Tuple[str, Dict]]:
    """
    Gather frames based on shot scenes. If a scene is longer than clip_secs,
    split it into smaller clips.
    
    Returns:
        List of (timestamp_key, {"files": [...], "transcript": "...", "scene_id": int, "is_sub_clip": bool})
    """
    # Get all frame files
    frame_files = sorted(
        [
            f for f in os.listdir(video_frame_folder)
            if f.startswith("frame") and (f.endswith(".jpg") or f.endswith(".png"))
        ],
        key=lambda x: float(x.split("_")[-1].rstrip(".jpg").rstrip(".png")),
    )
    if not frame_files:
        return []

    # Optional subtitle information
    subtitle_map = (
        parse_srt_to_dict(subtitle_file_path) if subtitle_file_path else {}
    )

    # Map frame numbers (from filename) to file paths
    # Frame filename format: frame_{frame_number}.jpg
    frame_num_to_file = {}
    for f in frame_files:
        frame_num = int(f.split("_")[-1].rstrip(".jpg").rstrip(".png"))
        frame_num_to_file[frame_num] = os.path.join(video_frame_folder, f)

    # Parse shot scenes
    scenes = parse_shot_scenes(shot_scenes_path)
    if not scenes:
        print("Warning: No scenes found in shot_scenes.txt, falling back to time-based clips")
        return gather_clip_frames(video_frame_folder, clip_secs, subtitle_file_path)

    result = []

    for scene_id, (start_frame, end_frame) in enumerate(scenes):
        # Convert frame numbers to seconds
        # shot detection frames are based on SHOT_DETECTION_FPS
        scene_start_sec = start_frame / config.SHOT_DETECTION_FPS
        scene_end_sec = end_frame / config.SHOT_DETECTION_FPS
        scene_duration = scene_end_sec - scene_start_sec

        # Determine if we need to split this scene
        if scene_duration <= clip_secs:
            # Process entire scene as one clip
            clip_files = [
                frame_num_to_file[fn]
                for fn in range(start_frame, end_frame + 1)
                if fn in frame_num_to_file
            ]

            # Aggregate transcript
            transcript_parts: List[str] = []
            for key, text in subtitle_map.items():
                s, e = map(int, key.split("_"))
                if s <= scene_end_sec and e >= scene_start_sec:
                    transcript_parts.append(text)
            transcript = " ".join(transcript_parts).strip() or "No transcript."

            result.append((
                f"{int(scene_start_sec)}_{int(scene_end_sec)}",
                {
                    "files": clip_files,
                    "transcript": transcript,
                    "scene_id": scene_id,
                    "is_sub_clip": False
                }
            ))
        else:
            # Split scene into sub-clips of clip_secs length
            sub_clips = []
            clip_start_sec = scene_start_sec
            sub_clip_idx = 0
            
            while clip_start_sec < scene_end_sec:
                clip_end_sec = min(clip_start_sec + clip_secs, scene_end_sec)
                
                # Convert time back to frame numbers for extraction
                clip_start_frame = int(clip_start_sec * config.SHOT_DETECTION_FPS)
                clip_end_frame = int(clip_end_sec * config.SHOT_DETECTION_FPS)
                
                clip_files = [
                    frame_num_to_file[fn]
                    for fn in range(clip_start_frame, clip_end_frame + 1)
                    if fn in frame_num_to_file
                ]

                # Aggregate transcript for this sub-clip
                transcript_parts: List[str] = []
                for key, text in subtitle_map.items():
                    s, e = map(int, key.split("_"))
                    if s <= clip_end_sec and e >= clip_start_sec:
                        transcript_parts.append(text)
                transcript = " ".join(transcript_parts).strip() or "No transcript."

                sub_clips.append((
                    f"{int(clip_start_sec)}_{int(clip_end_sec)}_scene{scene_id}_sub{sub_clip_idx}",
                    {
                        "files": clip_files,
                        "transcript": transcript,
                        "scene_id": scene_id,
                        "is_sub_clip": True,
                        "sub_clip_idx": sub_clip_idx
                    }
                ))
                
                clip_start_sec = clip_end_sec
                sub_clip_idx += 1
            
            result.extend(sub_clips)

    return result


def gather_clip_frames(
    video_frame_folder, clip_secs: int, subtitle_file_path: str = None
) -> Dict[str, Dict]:
    # Fix possible typo in the earlier list-comprehension and gather frames again
    frame_files = sorted(
        [
            f for f in os.listdir(video_frame_folder)
            if f.startswith("frame") and (f.endswith(".jpg") or f.endswith(".png"))
        ],
        key=lambda x: float(x.split("_")[-1].rstrip(".jpg").rstrip(".png")),
    )
    if not frame_files:
        return {}

    # Optional subtitle information
    subtitle_map = (
        parse_srt_to_dict(subtitle_file_path) if subtitle_file_path else {}
    )

    # Map timestamps → file names for quick lookup
    frame_ts = [float(f.split("_")[-1].rstrip(".jpg").rstrip(".png")) / config.VIDEO_FPS for f in frame_files]
    ts_to_file = dict(zip(frame_ts, frame_files))
    last_ts = int(max(frame_ts))

    result = []

    # Iterate over fixed-length clips
    clip_start = 0
    while clip_start <= last_ts:
        clip_end = min(clip_start + clip_secs - 1, last_ts)

        # Collect frames that fall inside the current clip
        clip_files = [
            os.path.join(video_frame_folder, ts_to_file[t])
            for t in frame_ts
            if clip_start <= t <= clip_end
        ]

        # Aggregate transcript text overlapping the clip interval
        transcript_parts: List[str] = []
        for key, text in subtitle_map.items():
            s, e = map(int, key.split("_"))
            if s <= clip_end and e >= clip_start:  # overlap check
                transcript_parts.append(text)
        transcript = " ".join(transcript_parts).strip() or "No transcript."

        result.append((
                f"{clip_start}_{clip_end}", 
                {"files": clip_files, "transcript": transcript}
        ))

        clip_start += clip_secs
    return result


# --------------------------------------------------------------------------- #
#                   Subtitle (.srt) parsing helper function                    #
# --------------------------------------------------------------------------- #
def _timestamp_to_seconds(ts: str) -> float:
    """Convert 'HH:MM:SS,mmm' to seconds (float)."""
    hh, mm, rest = ts.split(":")
    ss, ms = rest.split(",")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0


def parse_srt_to_dict(srt_path: str) -> Dict[str, str]:
    """
    Parse an .srt file and return a mapping
    '{startSec_endSec}': 'subtitle text'.
    """
    if not os.path.isfile(srt_path):
        return {}

    result: Dict[str, str] = {}
    with open(srt_path, "r", encoding="utf-8") as fh:
        lines = [l.rstrip("\n") for l in fh]

    idx = 0
    n = len(lines)
    while idx < n:
        # Skip sequential index if present
        if lines[idx].strip().isdigit():
            idx += 1
        if idx >= n:
            break

        # Time-range line
        if "-->" not in lines[idx]:
            idx += 1
            continue
        start_ts, end_ts = [t.strip() for t in lines[idx].split("-->")]
        start_sec = int(_timestamp_to_seconds(start_ts))
        end_sec = int(_timestamp_to_seconds(end_ts))
        idx += 1

        # Collect subtitle text (may span multiple lines)
        subtitle_lines: List[str] = []
        while idx < n and lines[idx].strip():
            subtitle_lines.append(lines[idx].strip())
            idx += 1
        subtitle = " ".join(subtitle_lines)
        key = f"{start_sec}_{end_sec}"
        if key in result:  # append if duplicate key
            result[key] += " " + subtitle
        else:
            result[key] = subtitle
        # Skip blank line separating entries
        idx += 1
    return result


# --------------------------------------------------------------------------- #
#                   Merge sub-clip captions for a scene                       #
# --------------------------------------------------------------------------- #
def merge_scene_captions(sub_clip_captions: List[dict]) -> dict:
    """
    Merge multiple sub-clip captions from the same scene into one coherent caption.
    
    Args:
        sub_clip_captions: List of caption dicts from sub-clips of the same scene
        
    Returns:
        Merged caption dict with combined description and merged subject_registry
    """
    if not sub_clip_captions:
        return {}
    
    if len(sub_clip_captions) == 1:
        return sub_clip_captions[0]
    

    
    send_messages = copy.deepcopy(messages)
    send_messages[0]["content"] = SYSTEM_PROMPT
    send_messages[1]["content"] = SCENE_MERGE_PROMPT.replace(
        "SUBCLIPS_PLACEHOLDER", json.dumps(sub_clip_captions, indent=2)
    )
    
    tries = 3
    while tries:
        tries -= 1
        resp = call_vllm_model(
            send_messages,
            endpoint=config.VLLM_ENDPOINT,
            model_name=config.VIDEO_ANALYSIS_MODEL,
            return_json=True,
            max_tokens=config.VIDEO_ANALYSIS_MODEL_MAX_TOKEN,
        )["content"]
        if resp is None:
            continue
        try:
            merged = json.loads(resp)
            # Preserve time information from first and last clip
            merged["clip_start_time"] = sub_clip_captions[0].get("clip_start_time", "")
            merged["clip_end_time"] = sub_clip_captions[-1].get("clip_end_time", "")
            return merged
        except json.JSONDecodeError:
            continue
    
    # Fallback: simple concatenation
    print("Warning: Scene caption merge failed, using simple concatenation")
    descriptions = [c.get("clip_description", "") for c in sub_clip_captions]
    # registries = [c.get("subject_registry", {}) for c in sub_clip_captions]
    
    return {
        "clip_description": " ".join(descriptions),
        "shot_type": sub_clip_captions[0].get("shot_type", ""),
        "emotion": sub_clip_captions[0].get("emotion", ""),
        # "subject_registry": merge_subject_registries(registries),
        "clip_start_time": sub_clip_captions[0].get("clip_start_time", ""),
        "clip_end_time": sub_clip_captions[-1].get("clip_end_time", ""),
    }


# --------------------------------------------------------------------------- #
#                        LLM wrappers (single clip)                           #
# --------------------------------------------------------------------------- #
def _caption_clip(task: Tuple[str, Dict], caption_ckpt_folder) -> Tuple[str, dict]:
    """LLM call for one clip. Returns (timestamp_key, parsed_json)."""
    timestamp, info = task
    files, transcript = info["files"], info["transcript"]

    clip_start_time = convert_seconds_to_hhmmss(float(timestamp.split("_")[0]))
    clip_end_time = convert_seconds_to_hhmmss(float(timestamp.split("_")[1]))

    send_messages = copy.deepcopy(messages)
    send_messages[0]["content"] = SYSTEM_PROMPT
    send_messages[1]["content"] = CAPTION_PROMPT.replace(
        "TRANSCRIPT_PLACEHOLDER", transcript)

    if os.path.exists(os.path.join(caption_ckpt_folder, f"{timestamp}.json")):
        # If the caption already exists, skip processing
        with open(os.path.join(caption_ckpt_folder, f"{timestamp}.json"), "r") as f:
            return timestamp, json.load(f)

    tries = 3
    while tries:
        tries -= 1
        resp = call_vllm_model(
            send_messages,
            endpoint=config.VLLM_ENDPOINT,
            model_name=config.VIDEO_ANALYSIS_MODEL,
            return_json=True,
            image_paths=files,
            max_tokens=config.VIDEO_ANALYSIS_MODEL_MAX_TOKEN,
        )["content"]
        if resp is None:
            continue
        try:
            assert isinstance(resp, str), f"Response must be a JSON string instead of {type(resp)}:{resp}."
            parsed = json.loads(resp)
            
            # Validate required fields
            if "clip_description" not in parsed:
                print(f"Warning: Missing 'clip_description' in response for {timestamp}. Retrying...")
                continue
            elif "shot_type" not in parsed:
                print(f"Warning: Missing 'shot_type' in response for {timestamp}. Retrying...")
                continue
            elif "emotion" not in parsed:
                print(f"Warning: Missing 'emotion' in response for {timestamp}. Retrying...")
                continue
            # if "subject_registry" not in parsed:
            #     print(f"Warning: Missing 'subject_registry' in response for {timestamp}. Setting to empty dict.")
            #     parsed["subject_registry"] = {}
            
            
            # Add transcript to description
            parsed["clip_description"] += f"\n\nTranscript during this video clip: {transcript}."
            parsed["clip_start_time"] = clip_start_time
            parsed["clip_end_time"] = clip_end_time
            
            resp = json.dumps(parsed)
            with open(os.path.join(caption_ckpt_folder, f"{timestamp}.json"), "w") as f:
                f.write(resp)
            return timestamp, parsed
        except json.JSONDecodeError:
            continue
    return timestamp, {}  # give up


# --------------------------------------------------------------------------- #
#                  LLM wrapper – merge subject registries                     #
# --------------------------------------------------------------------------- #
def merge_subject_registries(registries: List[dict], batch_size: int = 10) -> dict:
    """
    Merge all `new_subject_registry` dicts using batch processing.
    If there are too many registries, merge them in batches recursively.
    """
    if not registries:
        return {}
    
    # Filter out empty registries
    registries = [r for r in registries if r]
    
    if not registries:
        return {}
    
    # If only one registry, return it directly
    if len(registries) == 1:
        return registries[0]
    
    # If there are too many registries, merge in batches first
    if len(registries) > batch_size:
        print(f"Merging {len(registries)} registries in batches of {batch_size}...")
        merged_batches = []
        for i in range(0, len(registries), batch_size):
            batch = registries[i:i + batch_size]
            merged_batch = merge_subject_registries(batch, batch_size)
            if merged_batch:
                merged_batches.append(merged_batch)
        # Recursively merge the batch results
        return merge_subject_registries(merged_batches, batch_size)
    
    # Merge current batch with LLM
    send_messages = copy.deepcopy(messages)
    send_messages[0]["content"] = SYSTEM_PROMPT
    send_messages[1]["content"] = MERGE_PROMPT.replace(
        "REGISTRIES_PLACEHOLDER", json.dumps(registries)
    )

    tries = 3
    while tries:
        tries -= 1
        resp = call_vllm_model(
            send_messages,
            endpoint=config.VLLM_ENDPOINT,
            model_name=config.VIDEO_ANALYSIS_MODEL,
            return_json=True,
            max_tokens=config.VIDEO_ANALYSIS_MODEL_MAX_TOKEN,
        )["content"]
        if resp is None:
            continue
        try:
            return json.loads(resp)
        except json.JSONDecodeError:
            continue
    
    # Fallback: if LLM merge fails, do simple dict merge
    print("Warning: LLM merge failed, falling back to simple merge")
    merged = {}
    for registry in registries:
        merged.update(registry)
    return merged


# --------------------------------------------------------------------------- #
#                     Process one video (parallel caption)                    #
# --------------------------------------------------------------------------- #
def process_video(
    frame_folder: str,
    output_caption_folder: str,
    subtitle_file_path: str = None,
    shot_scenes_path: str = None,
):
    """
    Process video and generate captions.
    
    Args:
        frame_folder: Path to folder containing video frames
        output_caption_folder: Path to save caption outputs
        subtitle_file_path: Optional path to subtitle file (.srt)
        shot_scenes_path: Optional path to shot_scenes.txt file for scene-based processing
    """
    caption_ckpt_folder = os.path.join(output_caption_folder, "ckpt")
    os.makedirs(caption_ckpt_folder, exist_ok=True)

    # Use scene-based processing if shot_scenes_path is provided
    if shot_scenes_path and os.path.isfile(shot_scenes_path):
        print(f"Using scene-based processing with {shot_scenes_path}")
        clips = gather_clip_frames_from_scenes(
            frame_folder, shot_scenes_path, config.CLIP_SECS, subtitle_file_path
        )
    else:
        print("Using time-based clip processing")
        clips = gather_clip_frames(frame_folder, config.CLIP_SECS, subtitle_file_path)

    caption_clip = functools.partial(
        _caption_clip,
        caption_ckpt_folder=caption_ckpt_folder,
    )
    # ---------------- Parallel captioning --------------- #
    with mp.Pool(8) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(caption_clip, clips),
                total=len(clips),
                desc=f"Captioning {frame_folder}",
            )
        )

    # ---------------- Process results and merge scene sub-clips ---------------- #
    partial_registries = []
    frame_captions = {}
    
    # Sort results by timestamp
    def get_sort_key(x):
        timestamp = x[0]
        # Extract numeric part before any _scene suffix
        parts = timestamp.split("_scene")
        if len(parts) > 1:
            # Has scene suffix, use start time
            return float(parts[0].split("_")[0])
        else:
            # Regular timestamp
            return float(timestamp.split("_")[0])
    
    results = sorted(results, key=get_sort_key)
    
    # Group sub-clips by scene_id if using scene-based processing
    if shot_scenes_path and os.path.isfile(shot_scenes_path):
        scene_groups = {}
        for ts, parsed in results:
            if not parsed:
                continue
            
            # Check if this is a sub-clip by looking at the timestamp format
            if "_scene" in ts and "_sub" in ts:
                # Extract scene_id from timestamp (format: {start}_{end}_scene{id}_sub{idx})
                scene_part = ts.split("_scene")[1]
                scene_id = int(scene_part.split("_sub")[0])
                
                if scene_id not in scene_groups:
                    scene_groups[scene_id] = []
                scene_groups[scene_id].append((ts, parsed))
            else:
                # Regular clip (entire scene fits in one clip)
                frame_captions[ts] = {
                    "caption": parsed["clip_description"],
                    "shot_type": parsed.get("shot_type", ""),
                    "emotion": parsed.get("emotion", ""),
                }
                # partial_registries.append(parsed["subject_registry"])
        
        # Merge sub-clips for each scene
        for scene_id in sorted(scene_groups.keys()):
            # sub_clips = sorted(scene_groups[scene_id], key=lambda x: x[0])
            sub_clips = scene_groups[scene_id]
            sub_clip_captions = [parsed for _, parsed in sub_clips]
            
            print(f"Merging {len(sub_clip_captions)} sub-clips for scene {scene_id}")
            merged_caption = merge_scene_captions(sub_clip_captions)
            
            # Create a timestamp key for the merged scene
            # Use the time range from first sub-clip start to last sub-clip end
            first_ts = sub_clips[0][0].split("_scene")[0]  # e.g., "0_30"
            last_ts = sub_clips[-1][0].split("_scene")[0]  # e.g., "30_60"
            start_time = first_ts.split("_")[0]
            end_time = last_ts.split("_")[1]
            merged_ts = f"{start_time}_{end_time}"
            
            frame_captions[merged_ts] = {
                "caption": merged_caption.get("clip_description", ""),
                "shot_type": merged_caption.get("shot_type", ""),
                "emotion": merged_caption.get("emotion", ""),
            }
            partial_registries.append(merged_caption.get("subject_registry", {}))
    else:
        # Time-based processing (original behavior)
        for ts, parsed in results:
            if parsed:
                frame_captions[ts] = {
                    "caption": parsed["clip_description"],
                    "shot_type": parsed.get("shot_type", ""),
                    "emotion": parsed.get("emotion", ""),
                }
                partial_registries.append(parsed["subject_registry"])

    # ---------------- Merge subject registries ---------- #
    merged_registry = merge_subject_registries(partial_registries)
    frame_captions["subject_registry"] = merged_registry

    with open(
        os.path.join(output_caption_folder, "captions.json"), "w"
    ) as f:
        json.dump(frame_captions, f, indent=4)


def process_video_lite(
    output_caption_folder: str,
    subtitle_file_path: str,
):
    """
    Process video in LITE_MODE using SRT subtitles.
    """
    captions = parse_srt_to_dict(subtitle_file_path)
    frame_captions = {}
    for key, text in captions.items():
        frame_captions[key] = {
            "caption": f"\n\nTranscript during this video clip: {text}.",
        }
    frame_captions["subject_registry"] = {}
    with open(
        os.path.join(output_caption_folder, "captions.json"), "w"
    ) as f:
        json.dump(frame_captions, f, indent=4)

# --------------------------------------------------------------------------- #
#                                    main                                     #
# --------------------------------------------------------------------------- #
def main():
    frame_folder = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Database/VLOG_Lisbon_20min/frames"
    output_caption_folder = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Database/test"
    subtitle_file_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Database/subtitles.srt"
    shot_scenes_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Database/VLOG_Lisbon_20min/frames/shot_scenes.txt"
    process_video(
        frame_folder,
        output_caption_folder,
        subtitle_file_path=subtitle_file_path,
        shot_scenes_path=shot_scenes_path,
    )

if __name__ == "__main__":
    main()