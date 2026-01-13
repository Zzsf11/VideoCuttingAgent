import os
import cv2
import json
import base64
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from openai import OpenAI
import re
import requests
from typing import List, Dict
from .. import config
import functools
from typing import Tuple
import copy
import glob

# Import scene merge and analysis functions
from .scene_merge import OptimizedSceneSegmenter, load_shots, save_scenes
from .scene_analysis_video import SceneVideoAnalyzer

# 定义核心 Prompt (保持之前设计的结构化指令)
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

SYSTEM_PROMPT = "You are a helpful assistant."

SHOT_PROMPT = """
[Role]
You are an expert film data archivist. Your task is to analyze the provided video frames of a SINGLE SHOT and populate a structured database.
You must be OBJECTIVE and PRECISE. Do not hallucinate narrative context not visible in the frame.

[Input]
Transcript of current clip(The begingning name is speacker name):
TRANSCRIPT_PLACEHOLDER

[Output Schema]
You must return a single valid JSON object strictly following this structure:

{
  "spatio_temporal": {
    "location_type": "Select one: [Interior, Exterior, Hybrid, Abstract/Space]",
    "environment_tags": ["List", "of", "3-5", "static", "elements", "defining", "the", "place", "e.g., 'Brick Wall', 'Forest', 'Office Desk'"],
    "time_state": "Select one: [Day, Night, Dawn/Dusk, Unclear]",
    "lighting_mood": "Select one: [Daylight, Night, Sunset, Neon, Low-key, High-key, Artificial]",
    "color_palette": "Dominant color vibe (e.g., 'Warm Orange', 'Cold Blue')"
  },
  "entities": {
    "character_count": "Integer (or 'Crowd' if > 10)",
    "active_characters": [
      {
        "visual_id": "Short descriptor (e.g., 'Man_A', 'Woman_in_Red')",
        "appearance": "Key visual traits (e.g., 'Black tuxedo, short hair', 'Dirty ragged clothes')",
        "facial_expression": "Current emotion (e.g., 'Angry', 'Terrified', 'Neutral')"
      }
    ],
    "key_props": ["List of objects that are being used or are visually dominant"]
  },
  "action_atoms": {
    "primary_action": "The main verb occurring in the shot (e.g., 'Running', 'Slapping', 'Driving')",
    "interaction_type": "Select one: [Solo, Person-to-Person, Person-to-Object, None]",
    "event_summary": "Detailed description of the event occurring in the shot. Especially note the characters' actions and interactions."
  },
  "cinematography": {
    "shot_scale": "Select one: [Extreme Close-up, Close-up, Medium Shot, Full Shot, Wide Shot, Extreme Wide Shot]",
    "camera_movement": "Select one: [Static, Pan, Tilt, Zoom-in, Zoom-out, Tracking/Dolly, Hand-held Shake]",
    "composition_note": "Brief note on framing (e.g., 'Over-the-shoulder', 'Symmetrical', 'Low Angle Power Shot')",
    "angle": "Select one: [Eye-level, Low Angle, High Angle, Dutch Angle]"
  },
  "narrative_analysis": {
    "narrative_function": "Select best fit: [Establishment (Setting the scene), Progression (Advancing action), Reaction (Emotional response), Insert (Focus on detail)]",
    "shot_purpose": "One sentence analysis of WHY this shot exists (e.g., 'To show the protagonist's hesitation before entering the room.')",
    "mood": "Emotional tone adjectives"
  }
}

[Guidelines]
1. **Clustering Cues**: The 'environment_tags' and 'lighting_color' are crucial for algorithmically grouping shots into scenes. Be consistent.
2. **Entities**: Using the character name from transcript. If you don't know a name, use a visual descriptor (e.g., "Man in Black").
3. **Format**: Return ONLY the JSON object. No markdown blocks.
"""

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

def gather_clip_frames_from_long_shots(
    video_frame_folder: str,
    long_shot_boundaries_path: str,  # 原 shot_scenes_path
    clip_secs: int,
    subtitle_file_path: str = None
) -> List[Tuple[str, Dict]]:
    """
    Gather frames based on physical shot boundaries (long_shots). 
    No merging is performed to ensure independence for metadata extraction.
    If a long_shot is longer than clip_secs, it is split into sub-clips.
    
    Returns:
        List of (timestamp_key, {
            "files": [...], 
            "transcript": "...", 
            "long_shot_id": int, 
            "is_sub_clip": bool
        })
    """
    # 1. Get all frame files and sort them
    frame_files = sorted(
        [
            f for f in os.listdir(video_frame_folder)
            if f.startswith("frame") and (f.endswith(".jpg") or f.endswith(".png"))
        ],
        key=lambda x: float(x.split("_")[-1].rstrip(".jpg").rstrip(".png")),
    )
    if not frame_files:
        return []

    # 2. Optional subtitle information
    subtitle_map = (
        parse_srt_to_dict(subtitle_file_path) if subtitle_file_path else {}
    )

    # 3. Map frame numbers to file paths
    # Frame filename format: frame_{frame_number}.jpg
    frame_num_to_file = {}
    for f in frame_files:
        frame_num = int(f.split("_")[-1].rstrip(".jpg").rstrip(".png"))
        frame_num_to_file[frame_num] = os.path.join(video_frame_folder, f)

    # 4. Parse long shots (raw shot boundaries)
    # 假设 parse_shot_scenes 函数依然可用，或者你已将其改名为 parse_long_shots
    # 这里为了代码兼容性，我们假设它返回的是 [(start_frame, end_frame), ...]
    long_shots = parse_shot_scenes(long_shot_boundaries_path)

    if not long_shots:
        print("Warning: No shot boundaries found, falling back to time-based clips")
        return gather_clip_frames(video_frame_folder, clip_secs, subtitle_file_path)

    # ========================================================
    # [REMOVED] Merging logic has been removed as requested.
    # We now process the raw long_shots directly.
    # ========================================================
    
    result = []

    for shot_id, (start_frame, end_frame) in enumerate(long_shots):
        # Convert frame numbers to seconds
        shot_start_sec = start_frame / config.SHOT_DETECTION_FPS
        shot_end_sec = end_frame / config.SHOT_DETECTION_FPS
        shot_duration = shot_end_sec - shot_start_sec

        # Determine if we need to split this long_shot (due to VLM context limit)
        if shot_duration <= clip_secs:
            # Case A: Process the entire long_shot as one clip
            clip_files = [
                frame_num_to_file[fn]
                for fn in range(start_frame, end_frame)
                if fn in frame_num_to_file
            ]

            # Aggregate transcript
            transcript_parts: List[str] = []
            for key, text in subtitle_map.items():
                s, e = map(int, key.split("_"))
                # Check overlap with the shot duration
                if s <= shot_end_sec and e >= shot_start_sec:
                    transcript_parts.append(text)
            transcript = " ".join(transcript_parts).strip() or "No transcript."

            result.append((
                f"{int(shot_start_sec)}_{int(shot_end_sec)}",
                {
                    "files": clip_files,
                    "transcript": transcript,
                    "long_shot_id": shot_id,  # Renamed from scene_id
                    "is_sub_clip": False,
                    "frame_range": (start_frame, end_frame-1)
                }
            ))
        else:
            # Case B: Split overly long shot into sub-clips
            sub_clips = []
            clip_start_sec = shot_start_sec
            sub_clip_idx = 0
            
            while clip_start_sec < shot_end_sec:
                clip_end_sec = min(clip_start_sec + clip_secs, shot_end_sec)
                
                # Convert time back to frame numbers for extraction
                clip_start_frame = int(clip_start_sec * config.SHOT_DETECTION_FPS)
                clip_end_frame = int(clip_end_sec * config.SHOT_DETECTION_FPS)
                
                clip_files = [
                    frame_num_to_file[fn]
                    for fn in range(clip_start_frame, clip_end_frame)
                    if fn in frame_num_to_file
                ]

                # Aggregate transcript for this sub-clip
                transcript_parts: List[str] = []
                for key, text in subtitle_map.items():
                    s, e = map(int, key.split("_"))
                    if s <= clip_end_sec and e >= clip_start_sec:
                        transcript_parts.append(text)
                transcript = " ".join(transcript_parts).strip() or "No transcript."

                # Unique key: time + shot_id + sub_idx
                key_str = f"{int(clip_start_sec)}_{int(clip_end_sec)}_shot{shot_id}_sub{sub_clip_idx}"
                
                sub_clips.append((
                    key_str,
                    {
                        "files": clip_files,
                        "transcript": transcript,
                        "long_shot_id": shot_id,    # Renamed from scene_id
                        "is_sub_clip": True,
                        "sub_clip_idx": sub_clip_idx,
                        "frame_range": (clip_start_frame, clip_end_frame-1)
                    }
                ))
                
                clip_start_sec = clip_end_sec
                sub_clip_idx += 1
            
            result.extend(sub_clips)

    return result

def parse_json_safely(text):
    """鲁棒的 JSON 解析"""
    text = text.strip()
    # 去除 Markdown 代码块标记
    if text.startswith("```"):
        text = re.sub(r"^```json\s*|^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 尝试提取第一个 { ... }
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
        return None
    
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
    
def _caption_clip(task: Tuple[str, Dict], caption_ckpt_folder) -> Tuple[str, dict]:
    """LLM call for one clip. Returns (timestamp_key, parsed_json)."""
    timestamp, info = task
    files, transcript, frame_range = info["files"], info["transcript"], info["frame_range"]

    # Extract time information from timestamp
    # Handle both regular timestamps (e.g., "0_30") and scene-based timestamps (e.g., "0_30_scene0_sub0")
    timestamp_parts = timestamp.split("_scene")[0] if "_scene" in timestamp else timestamp
    clip_start_time = convert_seconds_to_hhmmss(float(timestamp_parts.split("_")[0]))
    clip_end_time = convert_seconds_to_hhmmss(float(timestamp_parts.split("_")[1]))

    send_messages = copy.deepcopy(messages)
    send_messages[0]["content"] = SYSTEM_PROMPT
    send_messages[1]["content"] = SHOT_PROMPT.replace(
        "TRANSCRIPT_PLACEHOLDER", transcript)
    
    resp = call_vllm_model(
        send_messages,
        endpoint=config.VLLM_ENDPOINT,
        model_name=config.VIDEO_ANALYSIS_MODEL,
        return_json=True,
        image_paths=files,
        max_tokens=config.VIDEO_ANALYSIS_MODEL_MAX_TOKEN,
    )["content"]
    json_data = parse_json_safely(resp)

    save_path = os.path.join(caption_ckpt_folder, f"{timestamp}.json")
    if json_data:
        # Add duration and frame_range only if json_data is valid
        json_data["duration"] = {
            "clip_start_time": clip_start_time,
            "clip_end_time": clip_end_time
        }
        json_data["frame_range"] = frame_range

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        return None # Success
    else:
        # 解析失败，保存原始内容以便 Debug
        with open(save_path.replace('.json', '.txt'), 'w', encoding='utf-8') as f:
            f.write(resp if resp else "No response from model")
        return f"JSON Parse Error in {timestamp_parts}.json"

def process_video(
    frame_folder: str,
    output_caption_folder: str,
    subtitle_file_path: str = None,
    long_shots_path: str = None,
    video_type: str = "film",
):
    """
    Process video and generate captions.

    Args:
        frame_folder: Path to folder containing video frames
        output_caption_folder: Path to save caption outputs
        subtitle_file_path: Optional path to subtitle file (.srt)
        long_shots_path: Optional path to shot_scenes.txt file for scene-based processing
        video_type: Type of video ("film" or "vlog"). For vlog, subtitles will not be auto-searched.
    """
    caption_ckpt_folder = os.path.join(output_caption_folder, "ckpt")
    os.makedirs(caption_ckpt_folder, exist_ok=True)

    # 1. 读取 shot_scenes.txt，获取视频片段列表
    clips = gather_clip_frames_from_long_shots(
            frame_folder, long_shots_path, config.CLIP_SECS, subtitle_file_path
        )
    # 3. 构造 vLLM 请求消息
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

    # # ============ 单线程串行版本 ============ 
    # results = []
    # for i, clip in enumerate(tqdm(clips, total=len(clips), desc=f"Captioning {frame_folder}")):
    #     try:
    #         # 串行处理每个clip
    #         result = caption_clip(clip)
    #         results.append(result)
            
    #         # 添加调试信息
    #         print(f"已处理 clip {i+1}/{len(clips)}: {clip}")
    #         if result:  # 根据你的实际情况调整
    #             print(f"  结果: {result[:50]}...")  # 只显示前50个字符避免过长
    #     except Exception as e:
    #         print(f"处理 clip {i+1} 时出错: {e}")
    #         # 可以选择追加空结果或重新抛出异常
    #         results.append(None)  # 或者根据需求处理

    # # 如果你需要保持结果的顺序（原imap_unordered是无序的）
    # # 但既然改为串行了，结果就是按顺序的

    # ============ Step 2: Scene Merge ============
    print("\n" + "="*50)
    print("Step 2: Merging shots into scenes...")
    print("="*50)

    scenes_dir = os.path.join(output_caption_folder, "scenes")
    scenes_output = os.path.join(scenes_dir, "scene_0.json")

    if not os.path.exists(scenes_output):
        # Load shots from ckpt folder
        shots = load_shots(caption_ckpt_folder)
        print(f"Loaded {len(shots)} shots from {caption_ckpt_folder}")

        if shots:
            # Initialize segmenter
            segmenter = OptimizedSceneSegmenter()

            # Merge shots into scenes
            merged_scenes = segmenter.segment(
                shots,
                threshold=getattr(config, 'SCENE_SIMILARITY_THRESHOLD', 0.5),
                max_scene_duration_secs=getattr(config, 'MAX_SCENE_DURATION_SECS', 300)
            )

            print(f"Merged {len(shots)} shots into {len(merged_scenes)} scenes")

            # Save scenes
            save_scenes(merged_scenes, scenes_dir)
            print(f"Scenes saved to {scenes_dir}")
        else:
            print("Warning: No shots found to merge")
    else:
        print(f"Scenes already exist at {scenes_dir}, skipping merge")

    # ============ Step 3: Scene Video Analysis ============
    print("\n" + "="*50)
    print("Step 3: Analyzing scenes with video understanding...")
    print("="*50)

    scene_summaries_dir = os.path.join(output_caption_folder, "scene_summaries_video")
    first_summary = os.path.join(scene_summaries_dir, "scene_0.json")

    if os.path.exists(scenes_dir) and not os.path.exists(first_summary):
        # Get parent directory to find frames and subtitle
        # frame_folder structure: .../Video/{video_id}/frames
        # We need to go up from output_caption_folder to find frames
        video_base_dir = os.path.dirname(output_caption_folder)

        # For vlog, don't search for subtitles; for film, check for subtitle files
        subtitle_to_use = None
        if video_type != "vlog":
            # Check for subtitle files with character names first
            subtitle_candidates = [
                os.path.join(video_base_dir, "subtitles_with_characters.srt"),
                os.path.join(video_base_dir, "subtitles.srt")
            ]
            for candidate in subtitle_candidates:
                if os.path.exists(candidate):
                    subtitle_to_use = candidate
                    break

        # Initialize analyzer
        analyzer = SceneVideoAnalyzer(
            frames_dir=frame_folder,
            subtitle_file=subtitle_to_use
        )

        # Create output directory
        os.makedirs(scene_summaries_dir, exist_ok=True)

        # Get all scene files
        scene_files = sorted(glob.glob(os.path.join(scenes_dir, "scene_*.json")))

        print(f"Processing {len(scene_files)} scenes...")

        # Process each scene
        success_count = 0
        for scene_file in tqdm(scene_files, desc="Analyzing scenes"):
            scene_name = os.path.basename(scene_file)
            output_file = os.path.join(scene_summaries_dir, scene_name)

            result = analyzer.process_file(scene_file, output_file)
            if result == "Success":
                success_count += 1

        print(f"Scene analysis completed: {success_count}/{len(scene_files)} scenes processed")
        print(f"Scene summaries saved to {scene_summaries_dir}")
    elif os.path.exists(first_summary):
        print(f"Scene summaries already exist at {scene_summaries_dir}, skipping analysis")
    else:
        print(f"Warning: Scenes directory not found at {scenes_dir}, skipping scene analysis")

    print("\n" + "="*50)
    print("Video processing complete!")
    print("="*50)