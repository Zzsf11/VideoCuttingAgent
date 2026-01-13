import os
import json
import copy
import re
import numpy as np
from pprint import pprint
from typing import Annotated as A
from vca.build_database.video_caption import (
    convert_seconds_to_hhmmss,
    SYSTEM_PROMPT,
    messages as caption_messages,
    parse_srt_to_dict,
)
from vca import config
from vca.func_call_shema import as_json_schema
from vca.func_call_shema import doc as D
from vca.vllm_calling import call_vllm_model, get_vllm_embeddings
from vca.Reviewer import ReviewerAgent




'''
Logic:
    1. Abstract of editted video by video summery & audio summery & user instruction, give theme of video and narrative logic and the rough of each structure
    2. For each section:

        Time duration (music structure part): (对应的音乐部分)
        Section type: 具体描述这个部分应该表现的画面


        a. Given the music partial analysis and the theme of video and the sturcture, generate a detailed proposal for this section
        b. Call [Fetch video clip] to retrival the top K candidate clips by the similarity structure proposal and caption embedding
        c. Call [Trim the clip] to 
TOOLS:
    1. [Fetch video clip] 根据指定的线索去检索相关的片段，返回标注信息
    2. [Render the edited video] 渲染整个视频成为video


    # 3. [Generate structure proposal] 使用audio和video的整体概述结合user的instruction生成一个大概的叙事逻辑和故事剪辑结构，并给出每个部分的时间长度

    # [Trim the clip] Trim the clip by give time duration and text instruction and clip frame
    # [Match the audio] Using some music detectors and video clip detectors to align the timeline for harmany 



'''
class StopException(Exception):
    """
    Stop Execution by raising this exception (Signal that the task is Finished).
    """


def finish(
    answer: A[str, D("Output the final shot time range. Must be exactly ONE continuous clip.")],
    video_path: A[str, D("Path to the source video file")] = "",
    output_path: A[str, D("Path to save the edited video")] = "",
    target_length_sec: A[float, D("Expected total length in seconds")] = 0.0,
    section_idx: A[int, D("Current section index. Auto-injected.")] = -1,
    shot_idx: A[int, D("Current shot index. Auto-injected.")] = -1,
    protagonist_frame_data: A[list, D("Frame-by-frame protagonist detection data. Auto-injected.")] = None
) -> str:
    """
    Call this function to finalize the shot selection and save the result.
    NOTE: You MUST call review_finish first to validate the shot before calling this function.

    IMPORTANT: Only accepts ONE continuous time range.
    Example: [shot: 00:10:00 to 00:10:07.3] for a 7.3s target duration.

    Returns:
        str: Success message with saved result, or error message if parsing fails.
    """
    def hhmmss_to_seconds(time_str: str) -> float:
        """Convert HH:MM:SS, HH:MM:SS.s, HH:MM:SS:FF, or MM:SS to seconds."""
        parts = time_str.strip().split(':')
        if len(parts) == 4:
            # HH:MM:SS:FF format (with frame number)
            h, m, s, f = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            # Assume VIDEO_FPS from config, default to 24 if not available
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
    shot_pattern = re.compile(r'\[?shot[\s_]*\d*:\s*([0-9:.]+)\s+to\s+([0-9:.]+)\]?', re.IGNORECASE)
    matches = shot_pattern.findall(answer)

    if not matches:
        return "Error: Could not parse shot time ranges. Please provide format: [shot: HH:MM:SS to HH:MM:SS]"

    # Support multiple shots for stitching (with constraints)
    max_shots_allowed = getattr(config, 'MAX_SHOTS_PER_CLIP', 3)
    if len(matches) > max_shots_allowed:
        return f"Error: Too many shots detected ({len(matches)}). Maximum allowed: {max_shots_allowed}"

    # Parse all time ranges and validate
    clips = []
    total_duration = 0
    for i, (start_time, end_time) in enumerate(matches):
        try:
            start_sec = hhmmss_to_seconds(start_time)
            end_sec = hhmmss_to_seconds(end_time)
            duration = end_sec - start_sec

            if duration <= 0:
                return f"Error: Shot {i+1} has invalid duration (start: {start_time}, end: {end_time})"

            clips.append({
                'start_sec': start_sec,
                'end_sec': end_sec,
                'duration': duration,
                'start_time': start_time,
                'end_time': end_time
            })
            total_duration += duration
        except Exception as e:
            return f"Error parsing shot {i+1} time range: {str(e)}"

    # Validate time continuity and gaps for multi-shot stitching
    if len(clips) > 1:
        max_gap = getattr(config, 'MAX_STITCH_GAP_SEC', 2.0)
        for i in range(len(clips) - 1):
            gap = clips[i+1]['start_sec'] - clips[i]['end_sec']
            if gap < 0:
                return f"Error: Overlapping shots detected. Shot {i+1} ends at {clips[i]['end_time']}, but shot {i+2} starts at {clips[i+1]['start_time']}"
            if gap > max_gap:
                return f"Error: Time gap ({gap:.2f}s) between shot {i+1} and {i+2} exceeds maximum ({max_gap}s). Shots must maintain visual continuity."

    # Use total duration for validation
    duration = total_duration

    # For result building, we'll use the first and last timestamps
    start_sec = clips[0]['start_sec']
    end_sec = clips[-1]['end_sec']
    start_time = clips[0]['start_time']
    end_time = clips[-1]['end_time']

    # Auto-trim if slightly over target (within 1 second)
    duration_diff = duration - target_length_sec
    if 0 < duration_diff <= 1.0:
        # Only auto-trim the last clip's end time
        clips[-1]['end_sec'] = clips[-1]['end_sec'] - duration_diff
        clips[-1]['duration'] = clips[-1]['end_sec'] - clips[-1]['start_sec']
        clips[-1]['end_time'] = seconds_to_hhmmss(clips[-1]['end_sec'])

        # Recalculate total duration
        duration = sum(c['duration'] for c in clips)

        end_sec = clips[-1]['end_sec']
        end_time = clips[-1]['end_time']
        print(f"Auto-trimmed by {duration_diff:.2f}s. New end: {end_time}")

    # Build result data with all clips
    result_clips = []
    for i, clip in enumerate(clips):
        result_clips.append({
            "shot": i + 1,
            "start": seconds_to_hhmmss(clip['start_sec']),
            "end": seconds_to_hhmmss(clip['end_sec']),
            "duration": round(clip['duration'], 2)
        })

    result_data = {
        "status": "success",
        "section_idx": section_idx,
        "shot_idx": shot_idx,
        "total_duration": round(duration, 2),
        "target_duration": target_length_sec,
        "num_clips": len(clips),
        "is_stitched": len(clips) > 1,
        "clips": result_clips
    }

    # Add protagonist frame detection data if available
    if protagonist_frame_data:
        result_data["protagonist_detection"] = {
            "method": "vlm",
            "total_frames_analyzed": len(protagonist_frame_data),
            "frames_with_protagonist": sum(1 for f in protagonist_frame_data if f.get("protagonist_detected", False)),
            "protagonist_ratio": round(
                sum(1 for f in protagonist_frame_data if f.get("protagonist_detected", False)) / len(protagonist_frame_data),
                3
            ) if protagonist_frame_data else 0.0,
            "frame_detections": protagonist_frame_data
        }

    # Save result to output_path if provided
    if output_path:
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                try:
                    all_results = json.load(f)
                except json.JSONDecodeError:
                    all_results = []
        else:
            all_results = []

        all_results.append(result_data)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"Result saved to {output_path}")

    success_msg = f"Successfully created edited video: {seconds_to_hhmmss(start_sec)} to {seconds_to_hhmmss(end_sec)} ({duration:.2f}s)"
    print(success_msg)

    return json.dumps(result_data, ensure_ascii=False)


def get_related_shot(
        related_scenes: A[list, D("List of scene indices to search. Optional - you can specify nearby scenes within allowed range.")] = None,
        scene_folder_path: A[str, D("Path to the folder containing scene JSON files. Auto-injected.")] = None,
        recommended_scenes: A[list, D("Recommended scene indices from shot_plan. Auto-injected.")] = None
) -> str:
    """
    Retrieves shot information from specified scenes.

    You can optionally specify which scenes to search by passing a 'related_scenes' list.
    However, you can only explore scenes within ±SCENE_EXPLORATION_RANGE of the recommended scenes.

    Example:
    - If recommended scenes are [8, 12] and SCENE_EXPLORATION_RANGE=3
    - You can search scenes 5-11 (around 8) and 9-15 (around 12)
    - Searching scene 50 would be REJECTED

    If you don't specify scenes, the system will use the recommended scenes automatically.

    Returns:
        str: A formatted string containing the shot information from the requested scenes.
        IMPORTANT: Select segments within shot boundaries to avoid visual discontinuities.

    Notes:
        - If you can't find suitable shots in recommended scenes, try nearby scenes
        - Going too far from recommended scenes may result in mismatched content
    """

    # Validate scene range if agent requested specific scenes
    if related_scenes and recommended_scenes:
        from vca import config
        allowed_range = getattr(config, 'SCENE_EXPLORATION_RANGE', 3)

        # Get total scene count by checking available scene files
        max_scene_idx = 0
        if scene_folder_path and os.path.isdir(scene_folder_path):
            import glob
            scene_files = glob.glob(os.path.join(scene_folder_path, "scene_*.json"))
            if scene_files:
                # Extract scene numbers from filenames
                scene_numbers = []
                for f in scene_files:
                    basename = os.path.basename(f)  # e.g., "scene_42.json"
                    try:
                        num = int(basename.replace("scene_", "").replace(".json", ""))
                        scene_numbers.append(num)
                    except ValueError:
                        continue
                max_scene_idx = max(scene_numbers) if scene_numbers else 0

        # Build allowed scene set with boundary constraints
        allowed_scenes = set()
        for rec_scene in recommended_scenes:
            for offset in range(-allowed_range, allowed_range + 1):
                scene_idx = rec_scene + offset
                # Ensure scene index is within valid range [0, max_scene_idx]
                if scene_idx >= 0 and (max_scene_idx == 0 or scene_idx <= max_scene_idx):
                    allowed_scenes.add(scene_idx)

        # Check if all requested scenes are within allowed range
        invalid_scenes = [s for s in related_scenes if s not in allowed_scenes]
        if invalid_scenes:
            return (
                f"❌ Error: Cannot search scenes {invalid_scenes} - they are outside the allowed range.\n"
                f"Recommended scenes: {recommended_scenes}\n"
                f"Allowed exploration range: ±{allowed_range} scenes\n"
                f"Valid scenes you can search: {sorted(list(allowed_scenes))}\n"
                f"Please select scenes within the allowed range or omit the 'related_scenes' parameter to use defaults."
            )

        print(f"📍 Agent exploring nearby scenes: {related_scenes} (recommended: {recommended_scenes})")
    elif not related_scenes and recommended_scenes:
        # Use recommended scenes if agent didn't specify
        related_scenes = recommended_scenes
        print(f"📍 Using recommended scenes: {related_scenes}")

    if not related_scenes:
        return "Error: No scenes specified and no recommended scenes available."

    all_shots_info = []

    for scene_idx in related_scenes:
        scene_file = os.path.join(scene_folder_path, f"scene_{scene_idx}.json")
        if os.path.exists(scene_file):
            try:
                with open(scene_file, 'r', encoding='utf-8') as f:
                    scene_data = json.load(f)

                scene_time_range = scene_data.get('time_range', {})
                scene_start = scene_time_range.get('start_seconds', '00:00:00')
                scene_end = scene_time_range.get('end_seconds', '00:00:00')

                all_shots_info.append(f"\n=== Scene {scene_idx} ({scene_start} - {scene_end}) ===")

                shots_data = scene_data.get('shots_data', [])
                for shot in shots_data:
                    duration = shot.get('duration', {})
                    start_time = duration.get('clip_start_time', '')
                    end_time = duration.get('clip_end_time', '')

                    action = shot.get('action_atoms', {})
                    event_summary = action.get('event_summary', '')

                    narrative = shot.get('narrative_analysis', {})
                    shot_purpose = narrative.get('shot_purpose', '')
                    mood = narrative.get('mood', '')

                    shot_info = f"[{start_time} - {end_time}] {event_summary}"
                    if mood:
                        shot_info += f" (Mood: {mood})"

                    all_shots_info.append(shot_info)

            except Exception as e:
                print(f"Warning: Failed to load scene {scene_idx}: {e}")
                all_shots_info.append(f"Scene {scene_idx}: Failed to load - {e}")
        else:
            all_shots_info.append(f"Scene {scene_idx}: File not found")

    result = "\n".join(all_shots_info)
    return f"Here are the available shots from related scenes {related_scenes}:\n{result}"


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
            # Assume VIDEO_FPS from config, default to 24 if not available
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


def trim_shot(
    time_range: A[str, D("The time range to analyze ('HH:MM:SS to HH:MM:SS'). This tool will analyze the ENTIRE range and provide scene breakdowns within it.")],
    frame_path: A[str, D("The path to the video frames file.")] = "",
    transcript_path: A[str, D("Optional path to an .srt transcript file; subtitles in this range will be injected into the prompt.")] = "",
    original_shot_boundaries: A[list, D("List of original shot boundaries from source material. Auto-injected.")] = None,
) -> str:
    """
    Analyze a video clip time range and return detailed scene information and usability assessment.
        
    Returns:
        A JSON string with structure:
        {
            "analyzed_range": "HH:MM:SS to HH:MM:SS",  # The full range you requested, must longer that 3.0s
            "total_duration_sec": float,                # Total duration
            "usability_assessment": "...",              # Overall evaluation
            "recommended_usage": "...",                 # How to use this clip
            "internal_scenes": [...]                    # Scene breakdowns (for reference)
        }
        
        The "internal_scenes" are fine-grained descriptions to help you understand what's  happening INSIDE the analyzed range.
        Use them to decide whether to use the full range, a subset, or refine with another call.

    
    Args:
        time_range: String in format 'HH:MM:SS to HH:MM:SS' - the range to analyze
        frame_path: Path to the video frames directory
    """
    def hhmmss_to_seconds(time_str: str) -> float:
        """Convert HH:MM:SS, HH:MM:SS.s, HH:MM:SS:FF, MM:SS, or MM:SS.s to seconds (supports decimal seconds and frame numbers)."""
        parts = time_str.strip().split(':')
        if len(parts) == 4:
            # HH:MM:SS:FF format (with frame number)
            h, m, s, f = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            # Assume VIDEO_FPS from config, default to 24 if not available
            fps = getattr(config, 'VIDEO_FPS', 24) or 24
            return h * 3600 + m * 60 + s + (f / fps)
        elif len(parts) == 3:
            h, m = int(parts[0]), int(parts[1])
            s = float(parts[2])  # Support decimal seconds like "35.5"
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m = int(parts[0])
            s = float(parts[1])  # Support decimal seconds like "35.5"
            return m * 60 + s
        else:
            return float(parts[0])

    def _extract_subtitles_in_range(srt_path: str, start_s: float, end_s: float, max_chars: int = 1500) -> str:
        """Reuse video_caption.parse_srt_to_dict() and only do range filtering + formatting here."""
        if not srt_path:
            return ""
        if not os.path.exists(srt_path):
            return ""
        try:
            subtitle_map = parse_srt_to_dict(srt_path)
            if not subtitle_map:
                return ""

            # parse_srt_to_dict() truncates timestamps to int seconds for keys.
            # Align the filtering to the same granularity to avoid boundary misses.
            import math
            start_i = int(start_s)
            end_i = int(math.ceil(end_s))
            if end_i <= start_i:
                end_i = start_i + 1


            picked = []  # list[tuple[int, str]]
            for key, text in subtitle_map.items():
                try:
                    s_sec, e_sec = map(int, key.split("_"))
                except Exception as e:
                    continue

                # Overlap check in integer-second domain (half-open interval)
                if s_sec >= end_i or e_sec <= start_i:
                    continue
                t = re.sub(r"\s+", " ", (text or "")).strip()
                if t:
                    picked.append((s_sec, t))

            if not picked:
                # Show a few sample keys from subtitle_map for debugging
                sample_keys = list(subtitle_map.keys())[:5]
                return ""

            picked.sort(key=lambda x: x[0])
            joined = " ".join(t for _, t in picked)
            joined = re.sub(r"\s+", " ", joined).strip()
            if len(joined) > max_chars:
                joined = joined[:max_chars].rsplit(' ', 1)[0] + "…"
            return joined
        except Exception as e:
            import traceback
            traceback.print_exc()
            return ""

    
    # Parse the time range string: 'HH:MM:SS to HH:MM:SS' or 'HH:MM:SS.s to HH:MM:SS.s'
    match = re.search(r'([\d:.]+)\s+to\s+([\d:.]+)', time_range, re.IGNORECASE)
    
    if not match:
        return f"Error: Could not parse time range '{time_range}'."
    
    start_time_str = match.group(1)
    end_time_str = match.group(2)
    
    # Convert to seconds
    start_sec = hhmmss_to_seconds(start_time_str)
    end_sec = hhmmss_to_seconds(end_time_str)
    
    # Convert seconds to HH:MM:SS format for display
    clip_start_time = convert_seconds_to_hhmmss(start_sec)
    clip_end_time = convert_seconds_to_hhmmss(end_sec)

    subtitles_context = _extract_subtitles_in_range(transcript_path, start_sec, end_sec)
    
    # Prepare messages for VIDEO_ANALYSIS_MODEL
    # Use the same prompt template as video_caption.py
    send_messages = copy.deepcopy(caption_messages)
    send_messages[0]["content"] = SYSTEM_PROMPT

    DENSE_CAPTION_PROMPT = """
    [Role]
    You are an expert Cinematographer and Video Editor specializing in shot boundary detection and emotional analysis.

    [Task]
    Identify KEY CUT POINTS where significant visual or narrative changes occur, and provide quality/emotion analysis for each segment.
    
    [What Constitutes a Key Cut Point]
    1. **Hard Cut**: Camera angle, framing, or location changes completely (different shot)
    2. **Scene Transition**: Change in time, place, or context
    3. **Significant Action Shift**: Major plot beat or dramatic action change (NOT minor movements)
    4. **Emotional Pivot**: Clear shift in mood or tone of the scene

    [What is NOT a Cut Point]
    - Minor head turns, gestures, or expressions within the same shot
    - Slight camera movements (pan, tilt) in continuous shots
    - Background changes that don't affect the main subject

    [Output Format]
    Return a JSON object:
    {
    "total_analyzed_duration": <float>,
    "segments": [
        {
        "timestamp": "<start_HH:MM:SS> to <end_HH:MM:SS>",
        "cut_type": "<hard_cut | scene_transition | action_shift | emotional_pivot>",
        "content_description": "<Factual description: Subject, Action, Camera angle (close-up/medium/wide/etc.), Environment>",
        "visual_quality": {
            "score": <1-5>,
            "notes": "<e.g., 'Sharp focus, stable shot' | 'Motion blur present' | 'Low lighting' | 'Excellent composition'>"
        },
        "emotion": {
            "mood": "<e.g., tense, melancholic, hopeful, aggressive, calm, mysterious>",
            "intensity": "<low | medium | high>",
            "narrative_function": "<e.g., 'builds suspense', 'reveals character emotion', 'establishes setting'>"
        },
        "character_presence": {
            "main_character_visible": <true | false>,
            "character_view": "<e.g., 'close-up', 'medium shot', 'long shot', 'not visible'>"
        },
        "editor_recommendation": "<e.g., 'Ideal for action sequence', 'Good emotional beat', 'Use as reaction shot', 'Transition material'>"
        }
    ]
    }

    [Quality Score Guide]
    - 5: Excellent - Sharp, well-lit, stable, professional composition
    - 4: Good - Minor imperfections but highly usable
    - 3: Acceptable - Noticeable issues but still usable
    - 2: Poor - Significant quality issues (blur, noise, bad framing)
    - 1: Unusable - Major technical problems

    [Guidelines]
    - **CRITICAL**: Each segment MUST have a meaningful duration (≥1.0s). For example, "00:00:00 to 00:00:03" is valid, but "00:00:00 to 00:00:00" is INVALID.
    - **Timestamps are RELATIVE to the clip start**: The first frame of the provided video clip is 00:00:00, and you must mark segments relative to this start time.
    - Prioritize SIGNIFICANT cuts only; avoid over-segmentation
    - Be precise with timestamps - mark the exact moment where the cut occurs
    - Segments should COLLECTIVELY cover the ENTIRE duration of the provided video clip
    - Emotion analysis should reflect what's visually conveyed, not assumed
    - Output ONLY valid JSON, no additional text

    [Example]
    For a 6-second clip showing: (1) close-up of person A for 2s, (2) cut to person B for 3s, (3) wide shot for 1s:
    ```json
    {
      "total_analyzed_duration": 6.0,
      "segments": [
        {"timestamp": "00:00:00 to 00:00:02", ...},
        {"timestamp": "00:00:02 to 00:00:05", ...},
        {"timestamp": "00:00:05 to 00:00:06", ...}
      ]
    }
    ```
    """

    if subtitles_context:
        DENSE_CAPTION_PROMPT += f"\n\n[Subtitles in this range]\n{subtitles_context}\n"
    send_messages[1]["content"] = DENSE_CAPTION_PROMPT

    # DEBUG: Print info about what we're sending to VLM
    print(f"[DEBUG trim_shot] Calling VLM with:")
    print(f"  - video_path: {frame_path}")
    print(f"  - video_start_time: {start_sec:.2f}s")
    print(f"  - video_end_time: {end_sec:.2f}s")
    print(f"  - expected_duration: {end_sec - start_sec:.2f}s")
    print(f"  - video_fps: {config.VIDEO_FPS}")
    print(f"  - use_local_clipping: True")

    # Call VIDEO_ANALYSIS_MODEL with the clip frames
    tries = 3
    while tries > 0:
        tries -= 1
        resp = call_vllm_model(
            send_messages,
            endpoint=config.VLLM_ENDPOINT,
            model_name=config.VIDEO_ANALYSIS_MODEL,
            return_json=False,
            video_path=frame_path,
            video_fps=config.VIDEO_FPS,  # Critical for temporal grounding - tells model the FPS used for frame extraction
            do_sample_frames=False,  # Don't re-sample - we already have pre-extracted frames
            max_tokens=config.VIDEO_ANALYSIS_MODEL_MAX_TOKEN,
            use_local_clipping=True, # True - clip the video from ori video 
            video_start_time=start_sec,
            video_end_time=end_sec,
        )
        
        if resp is None or resp.get("content") is None:
            if tries == 0:
                return f"Error: Failed to generate caption for time range {time_range}."
            continue
        
        try:
            content = resp["content"].strip()
            
            # Debug: print the raw content to help diagnose issues
            if not content:
                print(f"Warning: Empty content from model for time range {time_range}")
                if tries == 0:
                    return f"Error: Empty response from model for time range {time_range}."
                continue
            
            # Try to extract JSON from markdown code blocks if present
            # Pattern: ```json ... ``` or ``` ... ```
            json_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
            if json_block_match:
                content = json_block_match.group(1).strip()
            
            # Try to parse JSON
            parsed = json.loads(content)

            # DEBUG: Print first segment to diagnose timestamp issues
            if isinstance(parsed, dict) and "segments" in parsed and len(parsed["segments"]) > 0:
                first_seg = parsed["segments"][0]
                print(f"[DEBUG trim_shot] VLM returned first segment timestamp: {first_seg.get('timestamp', 'N/A')}")
                print(f"[DEBUG trim_shot] Request range: {start_sec:.2f}s to {end_sec:.2f}s (duration: {end_sec - start_sec:.2f}s)")

            # Handle "segments" format (from DENSE_CAPTION_PROMPT)
            if isinstance(parsed, dict) and "segments" in parsed:
                result = {
                    "analyzed_range": f"{clip_start_time} to {clip_end_time}",
                    "total_duration_sec": end_sec - start_sec,
                    "usability_assessment": "See segment details with quality scores and emotions.",
                    "internal_scenes": []
                }

                # Optional: Get detailed protagonist detection data using VLM
                frame_protagonist_data = []
                if getattr(config, 'ENABLE_TRIM_SHOT_CHARACTER_ANALYSIS', False):
                    print(f"[trim_shot] Calling VLM for detailed character analysis...")
                    try:
                        # Create ReviewerAgent instance to access protagonist detection
                        # Pass None for frame_folder_path to force video decoding
                        # (trim_shot's frame_path is actually video_path, not a frames directory)
                        from vca.Reviewer import ReviewerAgent
                        reviewer = ReviewerAgent(
                            frame_folder_path=None,  # Force video decoding, not using pre-extracted frames
                            video_path=frame_path     # frame_path is actually video_path in trim_shot context
                        )

                        frame_protagonist_data = reviewer.get_protagonist_frame_data(
                            video_path=frame_path,
                            time_range=time_range,
                            main_character_name=getattr(config, 'MAIN_CHARACTER_NAME', 'the main character'),
                            sample_fps=getattr(config, 'TRIM_SHOT_CHARACTER_SAMPLE_FPS', 1.0),
                            min_box_size=getattr(config, 'VLM_MIN_BOX_SIZE', 50),
                        )
                        print(f"[trim_shot] Got {len(frame_protagonist_data)} frame detections")
                    except Exception as e:
                        print(f"[trim_shot] Warning: VLM character analysis failed: {e}")
                        import traceback
                        traceback.print_exc()

                for seg in parsed["segments"]:
                    # Build comprehensive description from new format
                    desc_parts = []

                    # Cut type
                    if seg.get("cut_type"):
                        desc_parts.append(f"[{seg['cut_type'].upper()}]")

                    # Content description
                    if seg.get("content_description"):
                        desc_parts.append(seg["content_description"])

                    # Visual quality info
                    visual_quality = seg.get("visual_quality", {})
                    quality_score = visual_quality.get("score", "N/A")
                    quality_notes = visual_quality.get("notes", "")

                    # Emotion info
                    emotion = seg.get("emotion", {})
                    mood = emotion.get("mood", "")
                    intensity = emotion.get("intensity", "")
                    narrative_func = emotion.get("narrative_function", "")

                    # Editor recommendation
                    editor_rec = seg.get("editor_recommendation", "")

                    # Get base character_presence from VLM scene analysis
                    character_presence = seg.get("character_presence", {})

                    scene = {
                        "scene_time": seg.get("timestamp", ""),
                        "description": " ".join(desc_parts),
                        "cut_type": seg.get("cut_type", ""),
                        "visual_quality": {
                            "score": quality_score,
                            "notes": quality_notes
                        },
                        "emotion": {
                            "mood": mood,
                            "intensity": intensity,
                            "narrative_function": narrative_func
                        },
                        "character_presence": character_presence,
                        "editor_recommendation": editor_rec,
                        "duration_sec": 0
                    }

                    # Calculate absolute timestamps and duration
                    seg_start_sec = None
                    seg_end_sec = None
                    if "timestamp" in seg:
                        range_match = re.search(r'([0-9:.]+)\s+to\s+([0-9:.]+)', seg["timestamp"], re.IGNORECASE)
                        if range_match:
                            try:
                                # Timestamps from model are relative to the clip start (00:00:00)
                                # We need to convert them to absolute timestamps
                                s_rel = hhmmss_to_seconds(range_match.group(1))
                                e_rel = hhmmss_to_seconds(range_match.group(2))

                                s_abs = start_sec + s_rel
                                e_abs = start_sec + e_rel

                                seg_start_sec = s_abs
                                seg_end_sec = e_abs

                                scene["scene_time"] = f"{convert_seconds_to_hhmmss(s_abs)} to {convert_seconds_to_hhmmss(e_abs)}"
                                scene["duration_sec"] = round(e_abs - s_abs, 2)
                            except ValueError:
                                pass

                    # Enhance character_presence with VLM frame-level detection data
                    if frame_protagonist_data and seg_start_sec is not None and seg_end_sec is not None:
                        # Find frames that fall within this segment's time range
                        segment_frames = [
                            f for f in frame_protagonist_data
                            if seg_start_sec <= f["time_sec"] < seg_end_sec
                        ]

                        if segment_frames:
                            # Calculate protagonist presence ratio for this segment
                            protagonist_frames = [f for f in segment_frames if f["protagonist_detected"]]
                            protagonist_ratio = len(protagonist_frames) / len(segment_frames) if segment_frames else 0.0

                            # Calculate average bounding box size
                            box_sizes = []
                            for f in protagonist_frames:
                                bbox = f.get("bounding_box")
                                if bbox:
                                    box_size = min(bbox.get("width", 0), bbox.get("height", 0))
                                    if box_size > 0:
                                        box_sizes.append(box_size)
                            avg_box_size = sum(box_sizes) / len(box_sizes) if box_sizes else 0

                            # Enhance character_presence with VLM detection data
                            character_presence["vlm_protagonist_detected"] = protagonist_ratio > 0.5
                            character_presence["vlm_protagonist_ratio"] = round(protagonist_ratio, 2)
                            character_presence["vlm_avg_box_size"] = round(avg_box_size, 1) if avg_box_size > 0 else None
                            character_presence["vlm_frame_count"] = len(segment_frames)
                            character_presence["vlm_detection_summary"] = (
                                f"{len(protagonist_frames)}/{len(segment_frames)} frames with protagonist "
                                f"(ratio: {protagonist_ratio:.1%})"
                            )

                            # Update the scene's character_presence
                            scene["character_presence"] = character_presence

                            # If protagonist ratio meets threshold, enhance editor recommendation
                            min_protagonist_ratio = getattr(config, 'MIN_PROTAGONIST_RATIO', 0.7)
                            if protagonist_ratio >= min_protagonist_ratio:
                                current_rec = scene.get("editor_recommendation", "")
                                # Add strong recommendation prefix if ratio is high
                                if protagonist_ratio >= 0.9:
                                    scene["editor_recommendation"] = f"⭐ HIGHLY RECOMMENDED: Protagonist clearly visible ({protagonist_ratio:.0%} presence). {current_rec}"
                                else:
                                    scene["editor_recommendation"] = f"✅ RECOMMENDED: Good protagonist presence ({protagonist_ratio:.0%}). {current_rec}"
                            else:
                                # Add warning if protagonist ratio is low
                                current_rec = scene.get("editor_recommendation", "")
                                scene["editor_recommendation"] = f"⚠️ Low protagonist presence ({protagonist_ratio:.0%}). {current_rec}"

                    result["internal_scenes"].append(scene)


                # Validate that internal_scenes cover the requested time range
                total_requested_duration = end_sec - start_sec
                covered_duration = sum(scene.get("duration_sec", 0) for scene in result["internal_scenes"])
                coverage_ratio = covered_duration / total_requested_duration if total_requested_duration > 0 else 0

                min_coverage_ratio = 0.5  # Require at least 50% coverage
                if coverage_ratio < min_coverage_ratio:
                    print(f"⚠️ trim_shot output validation failed:")
                    print(f"   Requested: {total_requested_duration:.2f}s ({clip_start_time} to {clip_end_time})")
                    print(f"   Covered: {covered_duration:.2f}s (ratio: {coverage_ratio:.1%})")
                    print(f"   Scenes returned: {len(result['internal_scenes'])}")

                    # Print scene details for debugging
                    for i, scene in enumerate(result["internal_scenes"]):
                        scene_time = scene.get("scene_time", "unknown")
                        scene_dur = scene.get("duration_sec", 0)
                        print(f"   Scene {i+1}: {scene_time} ({scene_dur:.2f}s)")

                    if tries > 0:
                        print(f"   Retrying... ({tries} attempts remaining)")
                        continue
                    else:
                        print(f"   ⚠️ Max retries reached. Returning partial result.")
                        # Add warning to result
                        result["usability_assessment"] = (
                            f"⚠️ WARNING: Model only provided {coverage_ratio:.0%} coverage of requested range. "
                            f"Scenes may be incomplete or improperly segmented. Consider calling trim_shot with a different time range."
                        )

                return json.dumps(result, indent=4, ensure_ascii=False)
            
        except json.JSONDecodeError as e:
            print(f"JSON decode error for time range {time_range}: {e}")
            print(f"Raw content (first 500 chars): {resp['content'][:500]}")
            if tries == 0:
                return f"Error: Failed to parse model response for time range {time_range}. Content: {resp['content'][:200]}"
            continue
        except Exception as e:
            print(f"Unexpected error processing response for time range {time_range}: {e}")
            if tries == 0:
                return f"Error: Unexpected error processing response: {str(e)}"
            continue
    
    return f"Error: Failed to generate caption for time range {time_range} after multiple attempts."


class EditorCoreAgent:
    def __init__(self, video_caption_path, video_scene_path, audio_caption_path, output_path, max_iterations, video_path=None, frame_folder_path=None, transcript_path: str = None):
        self.tools = [get_related_shot, trim_shot, review_clip, finish]
        self.name_to_function_map = {tool.__name__: tool for tool in self.tools}
        self.function_schemas = [
            {"function": as_json_schema(func), "type": "function"}
            for func in self.name_to_function_map.values()
        ]
        self.video_scene_path = video_scene_path
        self.audio_db = json.load(open(audio_caption_path, 'r', encoding='utf-8'))
        self.max_iterations = max_iterations
        self.frame_folder_path = frame_folder_path
        self.video_path = video_path
        self.transcript_path = transcript_path
        self.output_path = output_path
        self.current_target_length = None  # Will be set during run()
        self.messages = self._construct_messages()
        # Track used time ranges to avoid duplicate clip selection
        self.used_time_ranges = []  # List of (start_sec, end_sec) tuples
        self.current_section_idx = None
        self.current_shot_idx = None
        self.current_related_scenes = []  # Will be set during run() for each shot
        self.attempted_time_ranges = set()  # Track attempted trim_shot time ranges to avoid duplicate calls
        self.duplicate_call_count = 0  # Count consecutive duplicate calls
        self.max_duplicate_calls = 3  # Max duplicates before restart

        # Initialize ReviewerAgent for finish validation
        self.reviewer = ReviewerAgent(
            frame_folder_path=frame_folder_path,
            video_path=video_path
        )
        # Current shot context for reviewer
        self.current_shot_context = {}

    def _construct_messages(self):
        messages = [
            {
                "role": "system",
                "content": f"""You are an expert video editor specializing in creating emotionally engaging highlight reels synchronized with music.

Your workflow follows the THINK → ACT → OBSERVE loop:
• THINK: Analyze what you've learned so far and plan your next action
• ACT: Call ONE tool to gather more information or finalize your selection
• OBSERVE: Carefully review the tool's output before proceeding

Key principles:
1. Focus on finding shots with the MAIN CHARACTER in iconic, emotionally powerful moments
2. Prioritize visual continuity and smooth transitions between shots
3. Be flexible - if perfect matches don't exist, find the best available alternative
4. Never call the same tool with identical parameters twice
5. When uncertain, trust your judgment and proceed confidently with the best option available"""
            },
            {
                "role": "user",
                "content": \
        """
========================================
MISSION: Select the Best Video Clip
========================================

[Your Goal]
Find ONE continuous video clip (or multiple nearby shots that can be stitched together) that:
✓ Features the MAIN CHARACTER in a compelling, iconic moment
✓ Matches the target duration: VIDEO_LENGTH_PLACEHOLDER seconds
✓ Aligns with the target emotion: CURRENT_VIDEO_EMOTION_PLACEHOLDER
✓ Fits the narrative content: CURRENT_VIDEO_CONTENT_PLACEHOLDER
✓ Synchronizes well with the music: BACKGROUND_MUSIC_PLACEHOLDER

[Available Tools & Usage]

1. **get_related_shot** - Your starting point
   Purpose: Search the video database for scenes matching your requirements
   When to use: At the beginning to find candidate scenes
   What it returns: A list of shots from specified scenes with descriptions
   Parameters:
      - related_scenes (optional): List of scene indices to search
        * If not specified, uses recommended scenes: RECOMMENDED_SCENES_PLACEHOLDER
        * You CAN explore nearby scenes within ±{getattr(config, 'SCENE_EXPLORATION_RANGE', 3)} range
        * Example: If recommended is [8], you can search [5, 6, 7, 8, 9, 10, 11]
        * Searching too far (e.g., scene 50) will be REJECTED
   Pro tip: Start with recommended scenes, expand to nearby scenes if needed

2. **trim_shot** - Your analysis tool
   Purpose: Get detailed frame-by-frame analysis of a specific time range
   When to use: When you've identified a promising time range and need details
   What it returns: Scene breakdowns with:
      - Content description for each internal scene
      - Visual quality scores (1-5, aim for ≥4)
      - Protagonist presence ratio (aim for ≥{config.MIN_PROTAGONIST_RATIO * 100:.0f}%)
      - Emotion/mood analysis
      - Editor recommendations
   Pro tip: Call this on slightly longer ranges (e.g., target + 2-3 seconds) to see context

3. **review_clip** - Your validation checkpoint
   Purpose: Check if your selected time range is valid
   When to use: ALWAYS call this right before finish()
   What it checks:
      - No overlap with previously used footage
      - Protagonist appears in enough frames
   Pro tip: This is mandatory - never skip it!

4. **finish** - Your final submission
   Purpose: Submit your final shot selection
   When to use: After review_clip confirms your selection is valid
   Format: [shot: HH:MM:SS to HH:MM:SS]
   Note: You can submit ONE continuous clip OR multiple short clips that form a coherent sequence

[Recommended Workflow]

Step 1: EXPLORE
→ Call get_related_shot to see available footage
→ Read the shot descriptions and identify 2-3 promising candidates

Step 2: ANALYZE
→ Call trim_shot on your best candidate (use target duration + 2s as range)
→ Review the "internal_scenes" carefully:
   * Check protagonist_ratio for each scene
   * Check visual_quality scores
   * Check emotion/mood alignment
   * Read editor_recommendation notes

Step 3: REFINE (if needed)
→ If the range isn't perfect, call trim_shot again with adjusted boundaries
→ Look for adjacent scenes that could extend a good short clip
→ Consider stitching 2-3 nearby shots if they maintain visual continuity

Step 4: VALIDATE
→ Call review_clip with your selected time range
→ If it fails, adjust and try again
→ If it passes, proceed immediately to finish

Step 5: SUBMIT
→ Call finish with your final selection in format: [shot: HH:MM:SS to HH:MM:SS]

[Critical Selection Criteria]

🎯 PRIORITY 1: Main Character Presence
- The protagonist must be CLEARLY VISIBLE and the FOCAL POINT of the shot
- Aim for protagonist_ratio ≥ {config.MIN_PROTAGONIST_RATIO * 100:.0f}% (can go as low as 40% if emotion is very strong)
- Prefer close-ups (CU), medium close-ups (MCU), or medium shots (MS)
- AVOID: Wide shots where the character is a tiny distant figure
- AVOID: Shots of minor characters, extras, or crowd scenes without the protagonist

🎬 PRIORITY 2: Visual Quality & Emotion
- Visual quality score should be ≥ 4 (accept 3 if emotion is perfect)
- Emotion/mood must align with target: CURRENT_VIDEO_EMOTION_PLACEHOLDER
- Look for iconic, memorable moments - dramatic expressions, powerful actions

🧩 PRIORITY 3: Continuity (when stitching multiple shots)
- If combining multiple shots, they MUST maintain visual continuity
- Check that character position/action flows naturally between shots
- Time gaps between shots should be < 2 seconds
- All shots should be from the same scene or closely related scenes

[Flexibility Guidelines - READ CAREFULLY]

❌ OLD RESTRICTIVE APPROACH: "I can't find a perfect 7.3s shot, so I'm stuck"

✅ NEW FLEXIBLE APPROACH: "I found a great 5s shot - let me extend it" OR "I'll use the best available option"

**When you can't find a perfect match:**

Option A: Extend a great shorter clip
- Found an amazing 4s shot? → Extend it to target duration by including adjacent frames
- Example: If 00:10:00-00:10:04 is perfect, try 00:10:00-00:10:07 to hit 7s target

Option B: Stitch nearby shots
- Found 2-3 short shots in the same scene? → Combine them
- Example: [00:10:00-00:10:03] + [00:10:03-00:10:06] = one 6s clip
- REQUIREMENT: Must maintain visual continuity (same location, character action flows naturally)

Option C: Accept close-enough duration
- Target is 7s but best clip is 5s? → That's acceptable if quality/emotion are strong
- Minimum acceptable: {getattr(config, 'MIN_ACCEPTABLE_SHOT_DURATION', 2.0)}s
- Can be ±{getattr(config, 'ALLOW_DURATION_TOLERANCE', 1.0)}s off target

Option D: Prioritize emotion over exact content
- Can't find the exact action described in CURRENT_VIDEO_CONTENT_PLACEHOLDER?
- → Find a shot with the SAME EMOTION and PROTAGONIST that fits the music
- Example: If script says "Batman punches enemy" but you only find "Batman kicks enemy" - that's fine!
- The music emotion matters more than literal content match

Option E: Use music as your guide
- When content description is too specific to find:
- → Let the music guide you: BACKGROUND_MUSIC_PLACEHOLDER
- → Find shots that match the music's energy, rhythm, and mood
- → Ensure protagonist is present and the shot feels "right" for that moment

**Your decision hierarchy:**
1st: Main character present + strong emotion match
2nd: Visual quality + good duration match
3rd: Perfect content match (this is LEAST important)

[Common Mistakes to Avoid]

❌ Calling trim_shot with the exact same time range twice
❌ Selecting shots without the main character
❌ Being too rigid about exact content matching
❌ Forgetting to call review_clip before finish
❌ Giving up because "perfect match not found" (there's always a best option!)
❌ Selecting long/wide shots where protagonist is too small
❌ Stitching shots with visual discontinuity (different locations, jarring cuts)

[Output Format]

When you call finish, use this exact format:
[shot: HH:MM:SS to HH:MM:SS]

Examples:
- Single continuous clip: [shot: 00:13:28 to 00:13:35.5]
- Stitched clips: [shot: 00:13:28 to 00:13:31] (pause) [shot: 00:13:34 to 00:13:35.5]

========================================
Ready? Start with get_related_shot!
========================================
        """
            },


        ]

        return messages

    # ------------------------------------------------------------------ #
    # Helper methods
    # ------------------------------------------------------------------ #
    def _append_tool_msg(self, tool_call_id, name, content, msgs):
        msgs.append(
            {
                "tool_call_id": tool_call_id,
                "role": "tool",
                "name": name,
                "content": content,
            }
        )

    def _exec_tool(self, tool_call, msgs):
        name = tool_call["function"]["name"]
        if name not in self.name_to_function_map:
            self._append_tool_msg(tool_call["id"], name, f"Invalid function name: {name!r}", msgs)
            return False

        # Parse arguments
        try:
            args = json.loads(tool_call["function"]["arguments"])
        except json.JSONDecodeError as exc:
            raise StopException(f"Error decoding arguments: {exc!s}")

        # Inject system-provided parameters
        if "topk" in args:
            if config.OVERWRITE_CLIP_SEARCH_TOPK > 0:
                args["topk"] = config.OVERWRITE_CLIP_SEARCH_TOPK

        # For get_related_shot, inject scene_folder_path and recommended_scenes
        if name == "get_related_shot":
            agent_requested_scenes = args.get("related_scenes", [])
            if agent_requested_scenes and isinstance(agent_requested_scenes, list):
                # Agent explicitly requested specific scenes - will be validated in function
                print(f"📍 Agent requested scenes: {agent_requested_scenes}")
            else:
                # No agent request, will use default recommended related scenes in function
                print(f"📍 No specific scenes requested, will use recommended: {self.current_related_scenes}")

            # Inject both scene_folder_path and recommended_scenes for validation
            args["scene_folder_path"] = self.video_scene_path
            args["recommended_scenes"] = self.current_related_scenes

        # For trim_shot, inject frame_folder parameter and check for duplicate calls
        if name == "trim_shot":
            if self.frame_folder_path:
                args["frame_path"] = self.video_path
                if self.transcript_path:
                    args["transcript_path"] = self.transcript_path
            else:
                self._append_tool_msg(
                    tool_call["id"],
                    name,
                    "Error: frame_folder_path not configured in agent.",
                    msgs
                )
                return False

            # Check for duplicate time range calls to prevent infinite loops
            time_range = args.get("time_range", "")
            # Normalize the time range for comparison (remove extra spaces)
            normalized_range = " ".join(time_range.split())

            if normalized_range in self.attempted_time_ranges:
                self.duplicate_call_count += 1
                print(f"⚠️ Duplicate call detected ({self.duplicate_call_count}/{self.max_duplicate_calls}): {normalized_range}")

                if self.duplicate_call_count >= self.max_duplicate_calls:
                    print(f"🔄 Max duplicate calls reached. Restarting conversation for this shot...")
                    return "RESTART"  # Signal to restart the conversation

                # Return a helpful message instead of calling the tool again
                self._append_tool_msg(
                    tool_call["id"],
                    name,
                    f"Warning: You have already analyzed '{time_range}'. "
                    f"Duplicate call {self.duplicate_call_count}/{self.max_duplicate_calls}. "
                    f"Call 'finish' NOW with your best selection, or conversation will restart.",
                    msgs
                )
                return False

            # Reset duplicate counter on new time range
            self.duplicate_call_count = 0
            # Record this time range as attempted
            self.attempted_time_ranges.add(normalized_range)
        
        # For review_clip, inject used_time_ranges
        if name == "review_clip":
            args["used_time_ranges"] = self.used_time_ranges
            print(f"📍 Checking overlap against {len(self.used_time_ranges)} used clips")

        # For finish, first call ReviewerAgent to validate
        if name == "finish":
            args["video_path"] = self.video_path or ""
            args["output_path"] = self.output_path or ""
            args["target_length_sec"] = self.current_target_length or 0.0
            args["section_idx"] = self.current_section_idx if self.current_section_idx is not None else -1
            args["shot_idx"] = self.current_shot_idx if self.current_shot_idx is not None else -1
            # Note: protagonist_frame_data will be set after face quality check

            # Call ReviewerAgent to validate before executing finish
            shot_proposal = {
                "answer": args.get("answer", ""),
                "target_length_sec": self.current_target_length or 0.0
            }

            # Face quality check (optional, controlled by config.ENABLE_FACE_QUALITY_CHECK)
            if config.ENABLE_FACE_QUALITY_CHECK:
                time_match = re.search(r'\[?shot[\s_]*\d*:\s*([0-9:.]+)\s+to\s+([0-9:.]+)\]?', shot_proposal["answer"], re.IGNORECASE)
                if time_match:
                    time_range = f"{time_match.group(1)} to {time_match.group(2)}"

                    # Choose face quality check method based on config
                    face_check_method = getattr(config, 'FACE_QUALITY_CHECK_METHOD', 'traditional')

                    if face_check_method == 'vlm':
                        # Use VLM method for protagonist focus analysis
                        print(f"🎬 Using VLM face quality check method...")
                        face_check = self.reviewer.check_face_quality_vlm(
                            video_path=args.get("video_path", ""),
                            time_range=time_range,
                            main_character_name=getattr(config, 'MAIN_CHARACTER_NAME', 'the main character'),
                            min_protagonist_ratio=getattr(config, 'MIN_PROTAGONIST_RATIO', 0.7),
                            sample_fps=getattr(config, 'VLM_FACE_CHECK_SAMPLE_FPS', 1.0),
                            min_box_size=getattr(config, 'VLM_MIN_BOX_SIZE', 50),
                        )
                    else:
                        # Use traditional face_recognition method
                        print(f"👤 Using traditional face quality check method...")
                        face_check = self.reviewer.check_face_quality(
                            video_path=args.get("video_path", ""),
                            time_range=time_range,
                            max_break_ratio=config.FACE_QUALITY_MAX_BREAK_RATIO,
                            min_face_size=config.FACE_QUALITY_MIN_FACE_SIZE,
                            sample_fps=getattr(config, 'FACE_QUALITY_SAMPLE_FPS', 2.0),
                        )

                    # Store for debugging/trace
                    self.current_shot_context["face_quality"] = face_check
                    self.current_shot_context["face_quality_method"] = face_check_method

                    if "❌" in face_check or "FAILED" in face_check:
                        self._append_tool_msg(
                            tool_call["id"],
                            name,
                            f"Review Failed - Face quality check ({face_check_method}) did not pass:\n{face_check}",
                            msgs
                        )
                        return False

                    print(f"✅ Face quality check ({face_check_method}) passed")

                    # If VLM check passed, get detailed frame-by-frame detection data
                    if face_check_method == 'vlm':
                        print(f"📊 Collecting detailed protagonist detection data for frames...")
                        protagonist_frame_data = self.reviewer.get_protagonist_frame_data(
                            video_path=args.get("video_path", ""),
                            time_range=time_range,
                            main_character_name=getattr(config, 'MAIN_CHARACTER_NAME', 'the main character'),
                            sample_fps=getattr(config, 'VLM_FACE_CHECK_SAMPLE_FPS', 1.0),
                            min_box_size=getattr(config, 'VLM_MIN_BOX_SIZE', 50),
                        )
                        self.current_shot_context["protagonist_frame_data"] = protagonist_frame_data
                        print(f"✅ Collected {len(protagonist_frame_data)} frame detections")
                        # NOW set the protagonist_frame_data in args after it's been collected
                        args["protagonist_frame_data"] = protagonist_frame_data
            else:
                print(f"⚠️  Face quality check is disabled (config.ENABLE_FACE_QUALITY_CHECK=False)")

            # Set protagonist_frame_data if not already set (for cases where face quality check is disabled)
            # Also overwrite if the value is None but we have data in context
            if "protagonist_frame_data" not in args or args.get("protagonist_frame_data") is None:
                context_data = self.current_shot_context.get("protagonist_frame_data", None)
                if context_data is not None:
                    args["protagonist_frame_data"] = context_data
                    print(f"✅ Set protagonist_frame_data from context: {len(context_data)} detections")
                else:
                    args["protagonist_frame_data"] = None

            review_result = self.reviewer.review(
                shot_proposal=shot_proposal,
                context=self.current_shot_context,
                used_time_ranges=self.used_time_ranges
            )

            if not review_result["approved"]:
                # Review failed, return feedback to agent for iteration
                print(f"🔍 Reviewer check failed:")
                print(review_result["feedback"])
                self._append_tool_msg(
                    tool_call["id"],
                    name,
                    f"Review Failed - Please adjust your selection:\n{review_result['feedback']}",
                    msgs
                )
                return False  # Continue iteration

        # Call the tool
        try:
            print(f"Calling function `{name}` with args: {args}")
            result = self.name_to_function_map[name](**args)
            print("Result: ", result)
            self._append_tool_msg(tool_call["id"], name, result, msgs)

            # Check if finish was successful
            if name == "finish":
                # Parse result as JSON and check status
                try:
                    result_data = json.loads(result)
                    if result_data.get("status") == "success":
                        # Record used time ranges to prevent duplicate selection
                        clips = result_data.get("clips", [])
                        for clip in clips:
                            start_str = clip.get("start", "")
                            end_str = clip.get("end", "")
                            if start_str and end_str:
                                # Convert to seconds for comparison
                                def hhmmss_to_sec(t):
                                    parts = t.strip().split(':')
                                    if len(parts) == 3:
                                        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                                    elif len(parts) == 2:
                                        return int(parts[0]) * 60 + float(parts[1])
                                    return float(parts[0])
                                start_sec = hhmmss_to_sec(start_str)
                                end_sec = hhmmss_to_sec(end_str)
                                self.used_time_ranges.append((start_sec, end_sec))
                                print(f"📍 Recorded used time range: {start_str} to {end_str}")
                        print(f"Section completed successfully: {result}")
                        return True  # Signal to break the current section loop
                except json.JSONDecodeError:
                    # If not JSON, check for success message in string
                    if "Successfully validated shot selection" in result:
                        print(f"Section completed successfully: {result}")
                        return True
            
            return False
        except StopException as exc:  # graceful stop
            print(f"Finish task with message: '{exc!s}'")
            raise

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self, instruction, shot_plan_path: str) -> list[dict]:
        """
        Run the ReAct-style loop with OpenAI Function Calling.

        Args:
            instruction: The user instruction for video editing.
            shot_plan_path: Path to a pre-generated shot_plan.json file.
        """

        # Load shot plan from file
        print(f"Loading shot plan from: {shot_plan_path}")
        with open(shot_plan_path, 'r', encoding='utf-8') as f:
            structure_proposal = json.load(f)

        # Check if output file already exists, add random suffix if so
        if self.output_path and os.path.exists(self.output_path):
            import random
            import string
            base, ext = os.path.splitext(self.output_path)
            random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
            self.output_path = f"{base}_{random_suffix}{ext}"
            print(f"⚠️ Output file already exists. Using new path: {self.output_path}")

        # Store original output path and create section-specific paths
        original_output_path = self.output_path
        print("structure_proposal: ", structure_proposal)
        for sec_idx, sec_cur in enumerate(structure_proposal['video_structure']):
            print(f"\n{'='*60}")
            print(f"Processing Section {sec_idx + 1}/{len(structure_proposal['video_structure'])}")
            print(f"{'='*60}\n")
            # Set current section for reporting
            self.current_section_idx = sec_idx
            
            # Load shot_plan from sec_cur (loaded from file)
            shot_plan = sec_cur.get('shot_plan')
            if not shot_plan:
                print(f"Error: No shot_plan found for section {sec_idx}")
                continue
            print("Using shot plan from file")
            for idx, shot in enumerate(shot_plan['shots']):
                max_shot_restarts = 3  # Max restarts per shot
                for restart_attempt in range(max_shot_restarts):
                    if restart_attempt > 0:
                        print(f"\n🔄 Restart attempt {restart_attempt + 1}/{max_shot_restarts} for Shot {idx + 1}")

                    msgs = copy.deepcopy(self.messages)
                    print(f"\n{'='*60}")
                    print(f"Processing Shot {idx + 1}/{len(shot_plan['shots'])}")
                    print(f"{'='*60}\n")
                    # Use the same output file for all shots (results will be appended)
                    print("shot plan: ", shot)
                    self.output_path = original_output_path
                    print(f"Output path: {self.output_path}")
                    # Set current shot for reporting
                    self.current_shot_idx = idx
                    # Reset attempted time ranges and duplicate counter for each new shot/restart
                    self.attempted_time_ranges = set()
                    self.duplicate_call_count = 0
                    msgs[-1]["content"] = msgs[-1]["content"].replace("VIDEO_LENGTH_PLACEHOLDER", str(shot['time_duration']))
                    # msgs[-1]["content"] = msgs[-1]["content"].replace("WHOLE_VIDEO_CONTENT_PLACEHOLDER", overall_theme).replace("WHOLE_VIDEO_NARRATIVE_LOGIC_PLACEHOLDER", narrative_logic)
                    msgs[-1]["content"] = msgs[-1]["content"].replace("CURRENT_VIDEO_CONTENT_PLACEHOLDER", shot['content']).replace("CURRENT_VIDEO_EMOTION_PLACEHOLDER", shot['emotion'])
                    # Get corresponding audio section's detailed analysis
                    audio_section = self.audio_db['sections'][sec_idx]
                    detailed_analysis = audio_section.get('detailed_analysis', {})

                    # Build audio section info string
                    audio_info_parts = []

                    # Add section name and description if available
                    if 'name' in audio_section:
                        audio_info_parts.append(f"Section: {audio_section['name']}")
                    if 'description' in audio_section:
                        audio_info_parts.append(f"Description: {audio_section['description']}")

                    # Add summary if available
                    if isinstance(detailed_analysis, dict) and 'summary' in detailed_analysis:
                        audio_info_parts.append(f"Summary: {detailed_analysis['summary']}")

                    # Add the specific shot/section caption
                    if isinstance(detailed_analysis, dict) and 'sections' in detailed_analysis:
                        sections_list = detailed_analysis['sections']
                        if isinstance(sections_list, list) and idx < len(sections_list):
                            audio_info_parts.append(f"Shot caption: {sections_list[idx]}")
                        elif isinstance(sections_list, dict) and str(idx) in sections_list:
                            audio_info_parts.append(f"Shot caption: {sections_list[str(idx)]}")

                    audio_section_info = "\n".join(audio_info_parts) if audio_info_parts else "No audio information available"
                    msgs[-1]["content"] = msgs[-1]["content"].replace("BACKGROUND_MUSIC_PLACEHOLDER", audio_section_info)
                    self.current_target_length = shot['time_duration']
                    # Set current shot context for ReviewerAgent
                    self.current_shot_context = {
                        "content": shot.get('content', ''),
                        "emotion": shot.get('emotion', ''),
                        "section_idx": sec_idx,
                        "shot_idx": idx,
                        "time_duration": shot.get('time_duration', 0)
                    }
                    related_scene_value = shot.get('related_scene', [])
                    # Add recommended scenes to prompt
                    recommended_scenes_str = str(related_scene_value) if related_scene_value else "None specified"
                    msgs[-1]["content"] = msgs[-1]["content"].replace("RECOMMENDED_SCENES_PLACEHOLDER", recommended_scenes_str)
                    # Ensure related_scenes is always a list
                    if isinstance(related_scene_value, int):
                        self.current_related_scenes = [related_scene_value]
                    elif isinstance(related_scene_value, list):
                        self.current_related_scenes = related_scene_value
                    else:
                        self.current_related_scenes = []

                    should_restart = False  # Track if restart is needed
                    section_completed = False  # Track if shot completed successfully

                    for i in range(self.max_iterations):
                        if i == self.max_iterations - 1:
                            msgs.append(
                                {
                                    "role": "user",
                                    "content": "Please call the `finish` function to finish the task.",
                                }
                            )

                        # Retry loop for both model call and tool execution
                        max_model_retries = 2
                        max_tool_retries = 2
                        tool_execution_success = False

                        for tool_retry in range(max_tool_retries):
                            msgs_snapshot = copy.deepcopy(msgs)

                            # Call model with retry mechanism
                            response = None
                            context_length_error = False
                            for model_retry in range(max_model_retries):
                                try:
                                    response = call_vllm_model(
                                        msgs,
                                        endpoint=config.VLLM_AGENT_ENDPOINT,
                                        model_name=config.AGENT_MODEL,
                                        temperature=0.0,
                                        max_tokens=config.AGENT_MODEL_MAX_TOKEN,
                                        tools=self.function_schemas,
                                        tool_choice="auto",
                                        return_json=False,
                                    )
                                    if response is not None:
                                        break
                                    else:
                                        print(f"⚠️  Model returned None, retrying ({model_retry + 1}/{max_model_retries})...")
                                except Exception as e:
                                    error_msg = str(e).lower()
                                    # Check for context length exceeded error
                                    if "context length" in error_msg or "too large" in error_msg or "max_tokens" in error_msg:
                                        print(f"🔄 Context length exceeded: {e}")
                                        context_length_error = True
                                        break
                                    print(f"⚠️  Model call failed: {e}, retrying ({model_retry + 1}/{max_model_retries})...")
                                    if model_retry == max_model_retries - 1:
                                        raise

                            # If context length error, trigger restart
                            if context_length_error:
                                print(f"🔄 Triggering restart due to context overflow...")
                                should_restart = True
                                break

                            if response is None:
                                print(f"❌ Model call failed after {max_model_retries} retries.")
                                msgs[:] = msgs_snapshot
                                break

                            response.setdefault("role", "assistant")
                            msgs.append(response)
                            print("#### Iteration: ", i, f"(Tool retry: {tool_retry + 1}/{max_tool_retries})" if tool_retry > 0 else "")
                            print(response)

                            # Execute any requested tool calls
                            tool_execution_failed = False

                            try:
                                tool_calls = response.get("tool_calls", [])
                                if tool_calls is None:
                                    tool_calls = []

                                if not tool_calls:
                                    content = response.get("content", "")
                                    # Check if model returned a final shot answer (not just mentioned it in reasoning)
                                    # Must match pattern like "[shot: 00:10:00 to 00:10:07]" at the END of content
                                    # or be a short response (< 500 chars) with the pattern
                                    final_shot_pattern = re.search(r'\[shot:\s*[\d:.]+\s+to\s+[\d:.]+\s*\]', content, re.IGNORECASE)
                                    is_short_response = len(content) < 500
                                    is_final_answer = final_shot_pattern and (is_short_response or content.strip().endswith(']'))

                                    if is_final_answer:
                                        print("✅ Model returned final answer. Task completed.")
                                        section_completed = True
                                        tool_execution_success = True
                                        break
                                    elif not tool_calls:
                                        # Model didn't call any tool and didn't provide final answer
                                        # This is likely a malformed response - prompt it to use tools
                                        print("⚠️ Model did not call any tool. Adding prompt to use tools...")
                                        msgs.append({
                                            "role": "user",
                                            "content": "You must call a tool function (get_related_shot, trim_shot, or finish). Do not output your reasoning as text - use the tool_calls format."
                                        })

                                for tool_call in tool_calls:
                                    is_finished = self._exec_tool(tool_call, msgs)
                                    if is_finished == "RESTART":
                                        should_restart = True
                                        break
                                    if is_finished:
                                        section_completed = True
                                        break

                                if should_restart:
                                    print("🔄 Restarting conversation for current shot...")
                                    break

                                tool_execution_success = True

                            except StopException:
                                return msgs
                            except Exception as e:
                                print(f"❌ Error executing tool calls: {e}")
                                import traceback
                                traceback.print_exc()
                                tool_execution_failed = True

                            # If tool execution succeeded or we're on the last retry, exit retry loop
                            if tool_execution_success or tool_retry == max_tool_retries - 1:
                                break

                            # Tool execution failed, rollback and retry
                            if tool_execution_failed:
                                print(f"🔄 Rolling back messages and retrying...")
                                msgs[:] = msgs_snapshot
                                continue

                        # Check if restart was triggered
                        if should_restart:
                            break  # Break out of iteration loop to restart

                        # If section is completed successfully, move to next shot
                        if section_completed:
                            print(f"Shot {idx + 1} completed. Moving to next shot...")
                            break

                    # End of iteration loop - check if we should restart or continue
                    if section_completed:
                        break  # Break out of restart loop, move to next shot
                    if not should_restart:
                        print(f"⚠️ Max iterations reached for Shot {idx + 1}. Moving to next shot.")
                        break  # Break out of restart loop

                # End of restart loop

            # End of shot loop
            print(f"\nSection {sec_idx + 1} completed. All shots processed.")

        return msgs


def main():
    # Batman Begins + Way Down We Go (short video test)
    # video_caption_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Video/Batman_Begins/captions/captions.json"
    # video_scene_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Video/Batman_Begins/captions/scene_summaries"
    # audio_caption_path = "//public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Audio/Way_Down_We_Go/captions_max4s/captions_short.json"
    # frame_folder_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Video/Batman_Begins/frames"
    # video_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/Dataset/Video/Movie/Batman_Begins.mp4"
    # transcript_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Video/Batman_Begins/subtitles_with_characters.srt"
    # shot_plan_output = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/output_batman_waywe_go_short/shot_plan_gemini.json"
    # output_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/output_batman_waywe_go_short/shot_point_gemini.json"
    # Instruction = """A visceral montage emphasizing the sheer physicality and iconography of the Dark Knight, synchronizing the song's heavy, stomping bass with the brutal precision of Batman's combat and the mechanical roar of the Tumbler as he dominates the Gotham night."""

    # The Dart Knight Rises + Way Down We Go (short video test)
    video_caption_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Video/The_Dark_Knight/captions/captions.json"
    video_scene_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Video/The_Dark_Knight/captions/scene_summaries_video"
    audio_caption_path = "//public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Audio/Way_Down_We_Go/captions/captions_short.json"
    frame_folder_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Video/The_Dark_Knight/frames"
    video_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/Dataset/Video/Movie/The_Dark_Knight.mkv"
    transcript_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Video/The_Dark_Knight/subtitles_with_characters.srt"
    shot_plan_output = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Output/The_Dark_Knight_Way_Down_We_Go/object/shot_plan_A_mesmerizing_showcase_of_the_Jokers_chaotic_prese_ad387f66.json"
    output_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/output_dark_knight_way_down_we_go_short/shot_point_gemini_joker.json"
    Instruction = "A mesmerizing showcase of the Joker's chaotic presence, capturing his unsettling mannerisms—the constant licking of scarred lips, the unpredictable head tilts, the disturbingly calm smile amidst carnage. Sync his erratic movements and explosive laughter to the song's building tension, highlighting close-ups of his smeared makeup, the purple coat swirling as he orchestrates destruction, and those haunting green eyes that embody pure anarchy." \


    agent = EditorCoreAgent(
        video_caption_path,
        video_scene_path,
        audio_caption_path,
        output_path,
        max_iterations=30,
        video_path=video_path,
        frame_folder_path=frame_folder_path,
        transcript_path=transcript_path
    )
    messages = agent.run(Instruction, shot_plan_path=shot_plan_output)

    # if messages:
    #     for m in messages:
    #         import pdb; pdb.set_trace()
    #         if m.get('role') == 'assistant':
    #             print(m)


if __name__ == "__main__":
    main()

