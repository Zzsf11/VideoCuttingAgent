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


CAPTION_PROMPT = """
[Role]
You are an experienced film editor who excels at translating visuals into narrative prose.

[Task]
Study the given frames together with the transcript and produce a JSON caption that captures both what is seen and how it is filmed.

[Input]
Transcript of current clip:
TRANSCRIPT_PLACEHOLDER

[Output]
{
  "clip_description": <4-6 sentences describing key visuals, staging, and overall mood>,
  "shot_type": <dominant shot type>,
  "emotion": <predominant emotional tone>
}

[Guidelines]
- Highlight the main characters, their actions, setting, and any notable props or visual motifs.
- Describe cinematography choices such as shot composition, camera movement, lighting, and color style.
- Explain how these visual elements support the emotional tone; note body language, pacing, or ambient sound cues when relevant.
- Use vivid yet concise language; avoid speculation beyond what the visuals and transcript imply.
- Do not mention that you were given frames or a prompt; write as an objective narrator.
"""


SCENE_MERGE_PROMPT = """
[Role]
You are a senior movie analyst who specializes in summarizing cinematic sequences.

[Task]
You will receive caption JSONs from consecutive sub-clips of the same scene. Merge them into one coherent description that reflects both the narrative flow and the filmmaking techniques.

[Input]
Sub-clip captions:
SUBCLIPS_PLACEHOLDER

[Output]
{
  "clip_description": <merged narration of the full scene>,
  "shot_type": <most representative shot type>,
  "emotion": <dominant emotion across the scene>,
}

[Guidelines]
- Preserve the chronological order and clearly communicate how the action evolves across the sub-clips.
- Emphasize key visual beats: character actions, staging, setting changes, and notable props or motifs.
- Discuss filmmaking craft—shot composition, camera motion, lighting, color, and editing rhythm—when they impact the viewer's perception.
- Convey the emotional progression or tension shifts across the scene, citing gestures, dialogue tone, or environmental cues when relevant.
- Keep the narration smooth and cinematic, avoiding redundant details from individual sub-clips.
"""

WHOLE_VIDEO_MERGE_PROMPT = \
"""
[Role]
You are a top-tier movie script analysis expert.

[Task]
You will receive a list of caption JSONs from consecutive sub-clips of the video.  Please merge them into one coherent scene description.

[Input]
List of sub-clip captions:
SUBCLIPS_PLACEHOLDER

[Output]
{
  "clip_description": <merged smooth narration of the entire video>,
  "emotion": <dominant emotion across the video>,
}

[Guidelines]
- The merged description should be a smooth, detailed narration of the entire video.
- The description should be coherent and logically connected.
- Keep the narration concise: aim for 4-6 sentences and stay under roughly 500 English words.
- Focus on the most important people, actions, emotions, and transitions; avoid repeating minor details.
- Ignore any non-diegetic or meta information such as advertisements, opening logos, production credits, or distributor notes—describe only what appears within the narrative footage.
"""

SYSTEM_PROMPT = "You are a helpful assistant."

MIN_SCENE_DURATION_SECONDS_FOR_SUMMARY = 3

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


def parse_hhmmss_to_seconds(time_str: str) -> int:
    if not time_str:
        return 0
    parts = time_str.split(":")
    if len(parts) != 3:
        return 0
    try:
        h, m, s = map(int, parts)
        return h * 3600 + m * 60 + s
    except ValueError:
        return 0


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
        Merged caption dict with combined description
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
    
    return {
        "clip_description": " ".join(descriptions),
        "shot_type": sub_clip_captions[0].get("shot_type", ""),
        "emotion": sub_clip_captions[0].get("emotion", ""),
        "clip_start_time": sub_clip_captions[0].get("clip_start_time", ""),
        "clip_end_time": sub_clip_captions[-1].get("clip_end_time", ""),
    }


def merge_whole_video_captions(all_clip_captions: List[dict]) -> dict:
    """
    Merge all clip captions from the entire video into one overall summary.
    
    Args:
        all_clip_captions: List of caption dicts from all clips in the video
        
    Returns:
        Merged caption dict with overall video summary
    """
    if not all_clip_captions:
        return {}
    
    if len(all_clip_captions) == 1:
        return all_clip_captions[0]
    
    print(f"Merging {len(all_clip_captions)} clips into whole video summary...")
    
    send_messages = copy.deepcopy(messages)
    send_messages[0]["content"] = SYSTEM_PROMPT
    send_messages[1]["content"] = WHOLE_VIDEO_MERGE_PROMPT.replace(
        "SUBCLIPS_PLACEHOLDER", json.dumps(all_clip_captions, indent=2)
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
            merged["clip_start_time"] = all_clip_captions[0].get("clip_start_time", "")
            merged["clip_end_time"] = all_clip_captions[-1].get("clip_end_time", "")
            return merged
        except json.JSONDecodeError:
            continue
    
    # Fallback: simple concatenation
    print("Warning: Whole video caption merge failed, using simple concatenation")
    descriptions = [c.get("clip_description", "") for c in all_clip_captions]
    
    return {
        "clip_description": " ".join(descriptions),
        "shot_type": all_clip_captions[0].get("shot_type", ""),
        "emotion": all_clip_captions[0].get("emotion", ""),
        "clip_start_time": all_clip_captions[0].get("clip_start_time", ""),
        "clip_end_time": all_clip_captions[-1].get("clip_end_time", ""),
    }


def merge_captions_map_reduce(all_clip_captions: List[dict], batch_size=None) -> dict:
    if not all_clip_captions:
        return {}

    if batch_size is None:
        batch_size = getattr(config, "WHOLE_VIDEO_SUMMARY_BATCH_SIZE", 0)

    if not batch_size or batch_size <= 0:
        batch_size = len(all_clip_captions)

    current_level = all_clip_captions

    while len(current_level) > 1:
        next_level = []
        for idx in range(0, len(current_level), batch_size):
            batch = current_level[idx : idx + batch_size]
            if len(batch) == 1:
                next_level.append(batch[0])
                continue

            merged = merge_whole_video_captions(batch)

            if not merged:
                descriptions = [c.get("clip_description", "") for c in batch]
                merged = {
                    "clip_description": " ".join(descriptions),
                    "shot_type": batch[0].get("shot_type", ""),
                    "emotion": batch[0].get("emotion", ""),
                }

            merged.setdefault("clip_start_time", batch[0].get("clip_start_time", ""))
            merged.setdefault("clip_end_time", batch[-1].get("clip_end_time", ""))

            next_level.append(merged)

        current_level = next_level

    return current_level[0]


# --------------------------------------------------------------------------- #
#                        LLM wrappers (single clip)                           #
# --------------------------------------------------------------------------- #
def _caption_clip(task: Tuple[str, Dict], caption_ckpt_folder) -> Tuple[str, dict]:
    """LLM call for one clip. Returns (timestamp_key, parsed_json)."""
    timestamp, info = task
    files, transcript = info["files"], info["transcript"]

    # Extract time information from timestamp
    # Handle both regular timestamps (e.g., "0_30") and scene-based timestamps (e.g., "0_30_scene0_sub0")
    timestamp_parts = timestamp.split("_scene")[0] if "_scene" in timestamp else timestamp
    clip_start_time = convert_seconds_to_hhmmss(float(timestamp_parts.split("_")[0]))
    clip_end_time = convert_seconds_to_hhmmss(float(timestamp_parts.split("_")[1]))

    send_messages = copy.deepcopy(messages)
    send_messages[0]["content"] = SYSTEM_PROMPT
    send_messages[1]["content"] = CAPTION_PROMPT.replace(
        "TRANSCRIPT_PLACEHOLDER", transcript)

    if os.path.exists(os.path.join(caption_ckpt_folder, f"{timestamp}.json")):
        # If the caption already exists, skip processing
        with open(os.path.join(caption_ckpt_folder, f"{timestamp}.json"), "r") as f:
            cached_data = json.load(f)
            # Ensure time fields are present (for backward compatibility with old checkpoints)
            if "clip_start_time" not in cached_data:
                cached_data["clip_start_time"] = clip_start_time
            if "clip_end_time" not in cached_data:
                cached_data["clip_end_time"] = clip_end_time
            return timestamp, cached_data

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
                    "clip_start_time": parsed.get("clip_start_time", ""),
                    "clip_end_time": parsed.get("clip_end_time", ""),
                }
        
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
                "clip_start_time": merged_caption.get("clip_start_time", ""),
                "clip_end_time": merged_caption.get("clip_end_time", ""),
            }
    else:
        # Time-based processing (original behavior)
        for ts, parsed in results:
            if parsed:
                frame_captions[ts] = {
                    "caption": parsed["clip_description"],
                    "shot_type": parsed.get("shot_type", ""),
                    "emotion": parsed.get("emotion", ""),
                    "clip_start_time": parsed.get("clip_start_time", ""),
                    "clip_end_time": parsed.get("clip_end_time", ""),
                }

    with open(
        os.path.join(output_caption_folder, "captions.json"), "w"
    ) as f:
        json.dump(frame_captions, f, indent=4)
    
    # ---------------- Merge all clips to generate whole video summary ---------------- #
    print("\nGenerating whole video summary...")
    
    # Prepare all clip captions for merging (sorted by time)
    def _clip_duration_seconds(ts: str, caption_data: Dict[str, str]) -> float:
        start_time_str = caption_data.get("clip_start_time", "")
        end_time_str = caption_data.get("clip_end_time", "")
        if start_time_str and end_time_str:
            start_sec = parse_hhmmss_to_seconds(start_time_str)
            end_sec = parse_hhmmss_to_seconds(end_time_str)
            if end_sec >= start_sec:
                return float(end_sec - start_sec)

        base_ts = ts.split("_scene")[0]
        parts = base_ts.split("_")
        if len(parts) >= 2:
            try:
                start_sec = float(parts[0])
                end_sec = float(parts[1])
                return max(0.0, end_sec - start_sec)
            except ValueError:
                return 0.0
        return 0.0

    sorted_clips = sorted(frame_captions.items(), key=lambda x: float(x[0].split("_")[0]))

    summary_clips = []
    skipped_for_summary = 0
    for ts, caption_data in sorted_clips:
        duration_seconds = _clip_duration_seconds(ts, caption_data)
        if duration_seconds >= MIN_SCENE_DURATION_SECONDS_FOR_SUMMARY:
            summary_clips.append((ts, caption_data))
        else:
            skipped_for_summary += 1

    if skipped_for_summary:
        print(
            f"Skipping {skipped_for_summary} clip(s) shorter than "
            f"{MIN_SCENE_DURATION_SECONDS_FOR_SUMMARY} seconds for video summary"
        )

    if not summary_clips:
        summary_clips = sorted_clips

    all_clip_data = []
    for ts, caption_data in summary_clips:
        all_clip_data.append({
            "clip_description": caption_data.get("caption", ""),
            "shot_type": caption_data.get("shot_type", ""),
            "emotion": caption_data.get("emotion", ""),
            "clip_start_time": caption_data.get("clip_start_time", ""),
            "clip_end_time": caption_data.get("clip_end_time", ""),
        })
    
    # Generate whole video summary
    video_summary = merge_captions_map_reduce(
        all_clip_data,
        batch_size=getattr(config, "WHOLE_VIDEO_SUMMARY_BATCH_SIZE", 0),
    )
    
    # Save video summary
    with open(
        os.path.join(output_caption_folder, "video_summary.json"), "w"
    ) as f:
        json.dump(video_summary, f, indent=4)
    
    print(f"Video summary saved to {os.path.join(output_caption_folder, 'video_summary.json')}")


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