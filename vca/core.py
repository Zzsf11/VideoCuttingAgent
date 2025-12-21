import os
import json
import copy
import re
import numpy as np
from pprint import pprint
from typing import Annotated as A
from vca.build_database.video_caption_ori import (
    convert_seconds_to_hhmmss, 
    CAPTION_PROMPT,
    SYSTEM_PROMPT,
    messages as caption_messages
)
from vca.build_database.build_vectorDB import init_single_video_db
from nano_vectordb import NanoVectorDB
from vca import config
from vca.func_call_shema import as_json_schema
from vca.func_call_shema import doc as D
from vca.vllm_calling import call_vllm_model, get_vllm_embeddings




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

TOPK = 16

class StopException(Exception):
    """
    Stop Execution by raising this exception (Signal that the task is Finished).
    """


def finish(
    answer: A[str, D("Render the edited video to the user.")],
    video_path: A[str, D("Path to the source video file")] = "",
    output_path: A[str, D("Path to save the edited video")] = "",
    target_length_sec: A[float, D("Expected total length in seconds")] = 0.0
) -> str:
    """
    Call this function after generating the detailed proposal for the video editing.
    Validates the total duration and renders the final video if valid.
    
    Returns:
        str: Success message if video is rendered, or error message if duration doesn't match.
    """
    def hhmmss_to_seconds(time_str: str) -> float:
        """Convert HH:MM:SS or MM:SS to seconds."""
        parts = time_str.strip().split(':')
        if len(parts) == 3:
            h, m, s = [int(x) for x in parts]
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m, s = [int(x) for x in parts]
            return m * 60 + s
        else:
            return float(parts[0])
    
    # Parse the answer to extract shot time ranges
    # Expected format: "shot 1: 00:10:00 to 00:10:05, shot 2: 00:29:30 to 00:29:35, ..."
    shot_pattern = re.compile(r'shot[\s_]*\d+:\s*\[?([0-9:]+)\s+to\s+([0-9:]+)\]?', re.IGNORECASE)
    matches = shot_pattern.findall(answer)
    
    if not matches:
        return "Error: Could not parse shot time ranges from the answer. Please provide time ranges in the format: {shot 1: HH:MM:SS to HH:MM:SS, shot 2: HH:MM:SS to HH:MM:SS, ...}"
    
    # Calculate total duration and collect clips
    total_duration = 0.0
    clips = []
    
    for i, (start_time, end_time) in enumerate(matches, 1):
        try:
            start_sec = hhmmss_to_seconds(start_time)
            end_sec = hhmmss_to_seconds(end_time)
            duration = end_sec - start_sec
            
            if duration <= 0:
                return f"Error: Shot {i} has invalid duration (start: {start_time}, end: {end_time}). End time must be greater than start time."
            
            total_duration += duration
            clips.append({
                'start': start_time,
                'end': end_time,
                'start_sec': start_sec,
                'end_sec': end_sec,
                'duration': duration
            })
        except Exception as e:
            return f"Error parsing shot {i} time range ({start_time} to {end_time}): {str(e)}"
    
    # Prepare duration summaries for feedback
    duration_lines = [
        f"shot {idx}: {clip['start']} to {clip['end']} ({clip['duration']:.2f}s)"
        for idx, clip in enumerate(clips, 1)
    ]
    short_clips = [
        (idx, clip)
        for idx, clip in enumerate(clips, 1)
        if clip['duration'] < 3
    ]

    # Check if total duration matches target length (allow 0.5 second tolerance)
    if abs(total_duration - target_length_sec) > 0.5:
        short_info = (
            "none"
            if not short_clips
            else ", ".join(
                [
                    f"shot {idx} ({clip['duration']:.2f}s)"
                    for idx, clip in short_clips
                ]
            )
        )
        shot_details = "\n".join(duration_lines)
        return (
            f"Error: Total duration ({total_duration:.2f} seconds) does not match target length "
            f"({target_length_sec:.2f} seconds). Please adjust your shot selections. Current shots: {len(clips)}, "
            f"Total gap: {abs(total_duration - target_length_sec):.2f} seconds.\n"
            f"Current shot durations:\n{shot_details}\n"
            f"Shots shorter than 3 seconds: {short_info}."
        )
    
    # If validation passes, cut and concatenate video clips
    if not video_path or not os.path.exists(video_path):
        return f"Error: Source video file not found: {video_path}"
    
    try:
        import subprocess
        import tempfile
        
        # Create temporary directory for clip files
        temp_dir = tempfile.mkdtemp()
        clip_files = []
        concat_list_path = os.path.join(temp_dir, 'concat_list.txt')
        
        # Extract each clip using ffmpeg
        for i, clip in enumerate(clips):
            clip_output = os.path.join(temp_dir, f'clip_{i:03d}.mp4')
            
            # Use ffmpeg to extract clip with precise timing
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output files
                '-ss', str(clip['start_sec']),  # Start time
                '-i', video_path,  # Input file
                '-t', str(clip['duration']),  # Duration
                '-c:v', 'libx264',  # Video codec
                '-c:a', 'aac',  # Audio codec
                '-strict', 'experimental',
                clip_output
            ]
            
            print(f"Extracting clip {i+1}/{len(clips)}: {clip['start']} to {clip['end']}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                return f"Error extracting clip {i+1}: {result.stderr}"
            
            clip_files.append(clip_output)
        
        # Create concat list file for ffmpeg
        with open(concat_list_path, 'w') as f:
            for clip_file in clip_files:
                f.write(f"file '{clip_file}'\n")
        
        # Concatenate all clips
        print(f"Concatenating {len(clip_files)} clips into final video...")
        concat_cmd = [
            'ffmpeg',
            '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_list_path,
            '-c', 'copy',
            output_path
        ]
        
        result = subprocess.run(concat_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return f"Error concatenating clips: {result.stderr}"
        
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir)
        
        success_msg = f"Successfully created edited video at {output_path}. Total duration: {total_duration:.2f} seconds, {len(clips)} clips."
        if short_clips:
            short_detail = ", ".join(
                [
                    f"shot {idx} ({clip['duration']:.2f}s, {clip['start']} to {clip['end']})"
                    for idx, clip in short_clips
                ]
            )
            success_msg += f" Warning: shots shorter than 3 seconds detected: {short_detail}."
        print(success_msg)
        
        return success_msg
    
    except Exception as e:
        return f"Error during video processing: {str(e)}"


def get_related_shot(
        database: A[NanoVectorDB, D("The database object that supports querying with embeddings.")],
        event_description: A[str, D("A textual description of the event to search for.")],
        top_k: A[int, D("The maximum number of top results to retrieve. Just use the default value.")] = 16,
        related_scenes: A[list, D("List of scene indices to restrict the search scope. If empty, search all.")] = None,
        scene_folder_path: A[str, D("Path to the folder containing scene JSON files.")] = None
) -> tuple:
    """
    Searches for events in a video clip database based on a given event description and retrieves the top-k most relevant video clip captions.

    Returns:
        str: A formatted string containing the concatenated captions of the searched video clip scripts.

    Notes:
        - This function utilizes the vLLM Embedding Service to generate embeddings for the input text.
        - Use default values for `top_k` to limit the number of results returned.
        - When related_scenes is provided, the search is restricted to clips within those scenes' time ranges.
    """
    # Build allowed time ranges from related_scenes
    allowed_time_ranges = []
    if related_scenes and scene_folder_path:
        for scene_idx in related_scenes:
            scene_file = os.path.join(scene_folder_path, f"scene_{scene_idx}.json")
            if os.path.exists(scene_file):
                try:
                    with open(scene_file, 'r', encoding='utf-8') as f:
                        scene_data = json.load(f)
                    time_range = scene_data.get('time_range', {})
                    # Parse time strings like "00:01:17" to seconds
                    start_str = time_range.get('start_seconds', '00:00:00')
                    end_str = time_range.get('end_seconds', '00:00:00')

                    def hhmmss_to_seconds(time_str):
                        parts = time_str.strip().split(':')
                        if len(parts) == 3:
                            h, m, s = [int(x) for x in parts]
                            return h * 3600 + m * 60 + s
                        elif len(parts) == 2:
                            m, s = [int(x) for x in parts]
                            return m * 60 + s
                        else:
                            return float(parts[0])

                    start_sec = hhmmss_to_seconds(start_str)
                    end_sec = hhmmss_to_seconds(end_str)
                    allowed_time_ranges.append((start_sec, end_sec))
                except Exception as e:
                    print(f"Warning: Failed to load scene {scene_idx}: {e}")

    # 获取对应片段的数据
    embedding_data = get_vllm_embeddings(
        input_text=event_description,
        endpoint=config.VLLM_EMBEDDING_ENDPOINT
    )
    # Extract the embedding vector from the response
    embedding = np.array(embedding_data[0]['embedding'])

    # Query more results if we need to filter, to ensure we get enough after filtering
    query_top_k = top_k * 3 if allowed_time_ranges else top_k
    results = database.query(embedding, top_k=query_top_k)

    # Filter results based on allowed time ranges
    if allowed_time_ranges:
        def is_in_allowed_range(clip_start, clip_end):
            for range_start, range_end in allowed_time_ranges:
                # Check if clip overlaps with the allowed range
                if clip_start < range_end and clip_end > range_start:
                    return True
            return False

        filtered_results = []
        for data in results:
            clip_start = data.get('time_start_secs', 0)
            clip_end = data.get('time_end_secs', clip_start)
            if is_in_allowed_range(clip_start, clip_end):
                filtered_results.append(data)
                if len(filtered_results) >= top_k:
                    break
        results = filtered_results

    captions = [
        (data['time_start_secs'], data.get('caption', data.get('event_summary', '')))
        for data in results
    ]
    captions = sorted(captions, key=lambda x: x[0])
    captions = "\n".join([cap[1] for cap in captions])

    if related_scenes and allowed_time_ranges:
        return f"Here is the searched video clip scripts (restricted to scenes {related_scenes}):\n\n" + captions
    return f"Here is the searched video clip scripts:\n\n" + captions

def trim_shot(
    time_range: A[str, D("The time range to analyze (e.g., '00:13:28 to 00:13:40'). This tool will analyze the ENTIRE range and provide scene breakdowns within it.")],
    frame_path: A[str, D("The path to the video frames file.")] = "",
) -> str:
    """
    Analyze a video clip time range and return detailed scene information.
    
    IMPORTANT: This tool provides ANALYSIS of the given time range, not a list of ready-to-use shots.
    
    Returns:
        A JSON string with structure:
        {
            "analyzed_range": "HH:MM:SS to HH:MM:SS",  # The full range you requested
            "total_duration_sec": float,                # Total duration
            "usability_assessment": "...",              # Overall evaluation
            "recommended_usage": "...",                 # How to use this clip
            "internal_scenes": [...]                    # Scene breakdowns (for reference)
        }
        
        The "internal_scenes" are fine-grained descriptions to help you understand what's 
        happening INSIDE the analyzed range. They are NOT meant to be used as separate shots.
        Use them to decide whether to use the full range, a subset, or refine with another call.
    
    Args:
        time_range: String in format 'HH:MM:SS to HH:MM:SS' - the range to analyze
        frame_path: Path to the video frames directory
    """
    def hhmmss_to_seconds(time_str: str) -> float:
        """Convert HH:MM:SS or MM:SS to seconds."""
        parts = time_str.strip().split(':')
        if len(parts) == 3:
            h, m, s = [int(x) for x in parts]
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m, s = [int(x) for x in parts]
            return m * 60 + s
        else:
            return float(parts[0])

    
    # Parse the time range string: 'HH:MM:SS to HH:MM:SS'
    match = re.search(r'([\d:]+)\s+to\s+([\d:]+)', time_range, re.IGNORECASE)
    
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
    
    # Prepare messages for VIDEO_ANALYSIS_MODEL
    # Use the same prompt template as video_caption.py
    send_messages = copy.deepcopy(caption_messages)
    send_messages[0]["content"] = SYSTEM_PROMPT

    DENSE_CAPTION_PROMPT = """
[Role]
You are an expert Video Logger and Editor. 
Your specialty is "Granular Video Segmentation": identifying precise boundaries where visual shots change or where the narrative action shifts significantly.

[Task]
Analyze the provided frames to segment the video into distinct, coherent clips.
1. **Identify Boundaries:** Look for visual cuts (camera changes) or distinct shifts in subject action (plot beats).
2. **Describe:** For each segment, write a factual, dense visual caption.

[Segmentation Rules]
- **Visual Cut:** Create a new segment when the camera angle, framing, or location changes completely.
- **Action Shift:** Within a continuous shot, if the subject completes one action and starts a distinctly different one (e.g., "stops running" -> "starts drinking water"), create a split.
- **Avoid Micro-splitting:** Do not split for minor gestures (e.g., blinking, turning head slightly) unless it changes the meaning.

[Output]
Return a JSON object with a list of segments:
{
  "total_analyzed_duration": <float>,
  "segments": [
    {
      "timestamp": "<start_HH:MM:SS> to <end_HH:MM:SS>",
      "visual_details": "<Detailed description: clearly state the Subject, their specific Movements, the Camera angle, and the Environment. No flowery language.>",
      "editor_notes": "<Brief note on usability: e.g., 'Good stabilizer shot', 'Contains motion blur', 'Good for reaction cut'.>"
    },
    ...
  ]
}

[Guidelines]
- **Accuracy is Paramount:** If you are unsure if a cut happened, prioritize the continuity of the action.
- **Strict JSON:** Do not include any text before or after the JSON block.
"""
    send_messages[1]["content"] = DENSE_CAPTION_PROMPT
    
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
            
            # Handle "segments" format (from DENSE_CAPTION_PROMPT)
            if isinstance(parsed, dict) and "segments" in parsed:
                result = {
                    "analyzed_range": f"{clip_start_time} to {clip_end_time}",
                    "total_duration_sec": end_sec - start_sec,
                    "usability_assessment": "See segment details.",
                    "recommended_usage": "Select continuous segments based on action shifts.",
                    "internal_scenes": []
                }
                
                for seg in parsed["segments"]:
                    # Construct description
                    desc_parts = []
                    if seg.get("segment_type"):
                        desc_parts.append(f"[{seg['segment_type']}]")
                    if seg.get("primary_action"):
                        desc_parts.append(seg["primary_action"])
                    if seg.get("visual_details"):
                        desc_parts.append(seg["visual_details"])
                    if seg.get("editor_notes"):
                        desc_parts.append(f"(Note: {seg['editor_notes']})")
                        
                    scene = {
                        "scene_time": seg.get("timestamp", ""),
                        "description": " ".join(desc_parts),
                        "duration_sec": 0
                    }
                    
                    # Calculate absolute timestamps and duration
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
                                
                                scene["scene_time"] = f"{convert_seconds_to_hhmmss(s_abs)} to {convert_seconds_to_hhmmss(e_abs)}"
                                scene["duration_sec"] = round(e_abs - s_abs, 2)
                            except ValueError:
                                pass
                    
                    result["internal_scenes"].append(scene)
                
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


class DVDCoreAgent:
    def __init__(self, video_db_path, video_caption_path, video_scene_path, audio_caption_path, output_path, max_iterations, video_path=None, frame_folder_path=None):
        self.tools = [get_related_shot, trim_shot, finish]
        self.name_to_function_map = {tool.__name__: tool for tool in self.tools}
        self.function_schemas = [
            {"function": as_json_schema(func), "type": "function"}
            for func in self.name_to_function_map.values()
        ]
        self.video_db = init_single_video_db(video_caption_path, video_db_path, config.AOAI_EMBEDDING_LARGE_DIM)
        self.video_scene_path = video_scene_path
        self.audio_db = json.load(open(audio_caption_path, 'r', encoding='utf-8'))
        self.max_iterations = max_iterations
        self.frame_folder_path = frame_folder_path
        self.video_path = video_path
        self.output_path = output_path
        self.current_target_length = None  # Will be set during run()
        self.messages = self._construct_messages()
        # Note: no trim overlap or redundancy restrictions; tracking removed per user request
        self.current_section_idx = None
        self.current_shot_idx = None
        self.current_related_scenes = []  # Will be set during run() for each shot

    def _construct_messages(self):
        messages = [
            {
                "role": "system",
                "content": """You are a helpful assistant who edits videos by sequentially invoking tools. Follow the THINK → ACT → OBSERVE loop:
  • THOUGHT: Reason step-by-step about which function to call next.
  • ACTION: Call exactly one tool that help you get more information about the video editing.
  • OBSERVATION: Summarize the tool's output.
You MUST plan extensively before each tool call, and reflect extensively on the outcomes of the previous tool calls.
Only pass arguments that come verbatim from the user or from earlier tool outputs—never invent them. Continue the loop until the user's query is fully resolved, then end your turn with the final answer. If you are uncertain about code structure or video content, use the available tools to inspect rather than guessing. Plan carefully before each call and reflect on every result. Do not rely solely on blind tool calls, as this degrades reasoning quality. Timestamps may be formatted as 'HH:MM:SS' or 'MM:SS'."""
            },
            {
                "role": "user",
                "content": \
"""
[Role]
You are a senior video editor who plans narrative-driven highlight reels.

[Task]
Inspect the script and storyline, using the given tools to select aligned clips in given video material. Your goal is to create a coherent, emotionally engaging edited video that matches the provided creative brief. And if you find it impossible to select and get some desired the shot in the current video, clearly state the reason and give the suggestion for revise the shot plan.

[Tools]
• `get_related_shot`: retrieve candidate video clips from the database for contextual exploration.
• `trim_shot`: **IMPORTANT**: This tool analyzes a time range and returns scene descriptions to help you understand what's happening INSIDE the analyzed clip.
• `finish`: present the final timestamped editing plan once all required clips are selected and refined.

[Workflow]
1. Review the global brief and initial observations about the video.
2. Use `get_related_shot` to surface promising segments aligned with the target theme, narrative logic, and emotion, filtering out clips whose narrative jumps conflict with the current storyline.
3. For each promising segment, call `trim_shot` with a time range (e.g., "00:13:28 to 00:13:40") to get detailed scene breakdown to understand the content.
4. Based on trim_shot's output, decide your shot selection:
   - If the full analyzed range works well → use it as one shot
   - If only part of it fits → select a continuous subset (e.g., first 3 scenes combined)
   - If you need more precision → call trim_shot again with a narrower range
6. Repeat 2–5 until the desired runtime and storytelling flow are covered.
7. Conclude with `finish`, summarizing the final ordered shot list with exact timestamps.

[Input Brief]
- Target edited video length: VIDEO_LENGTH_PLACEHOLDER seconds.
- Target edited video content: CURRENT_VIDEO_CONTENT_PLACEHOLDER
- Target edited video emotion: CURRENT_VIDEO_EMOTION_PLACEHOLDER
- Background music: BACKGROUND_MUSIC_PLACEHOLDER

[Output]
Provide a detailed timestamped plan in the format:
{shot 1: start_time to end_time, shot 2: start_time to end_time, ...}

[Guidelines]
- Think aloud about why each tool call is necessary before executing it, and reflect on the observations afterwards.
- Ground every argument in user-provided data or tool outputs; never fabricate timestamps or descriptions.
- **Understanding trim_shot output**:
  * The "analyzed_range" and "total_duration_sec" describe the ENTIRE clip you requested
  * The "internal_scenes" are fine-grained breakdowns for REFERENCE - they help you understand what's inside, but are NOT meant to be used as separate shots
  * Always read "recommended_usage" for guidance on how to select usable portions
  * To create a shot, use a TIME RANGE that makes sense (≥3s), not individual scene times
- **Shot selection strategy**:
  * After reviewing trim_shot output, decide on a continuous time range for your shot
  * Combine multiple internal_scenes if needed to reach ≥3s duration
  * Example: If scenes 1-4 (spanning 00:13:28 to 00:13:35) form a coherent beat, your shot is "00:13:28 to 00:13:35", NOT four separate 1-2s shots
- Continuity check: discard retrieved clips that introduce abrupt jumps or contradict the evolving narrative before moving forward.
- Maintain consistent formatting for timestamps (HH:MM:SS or MM:SS) and keep the final plan aligned with the requested length and narrative arc.
- Ensure every selected shot lasts at least 3 seconds; if trim_shot shows good content but it's fragmented, select a continuous range that combines multiple scenes.
- When a clear story beat is not found, expand the search window progressively (e.g., ±10–20s, then ±30–45s) and attempt multiple expansions before moving on.
- If you find a related scene but some shots/beats are missing, perform a neighborhood search centered on that segment by expanding on both sides (e.g., ±10–20s, then ±30s) to capture the surrounding context.
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
        if "database" in args:
            args["database"] = self.video_db

        if "topk" in args:
            if config.OVERWRITE_CLIP_SEARCH_TOPK > 0:
                args["topk"] = config.OVERWRITE_CLIP_SEARCH_TOPK

        # For get_related_shot, inject database, related_scenes and scene_folder_path
        if name == "get_related_shot":
            args["database"] = self.video_db
            args["related_scenes"] = self.current_related_scenes
            args["scene_folder_path"] = self.video_scene_path

        # For trim_shot, inject frame_folder parameter
        if name == "trim_shot":
            if self.frame_folder_path:
                args["frame_path"] = self.video_path
            else:
                self._append_tool_msg(
                    tool_call["id"], 
                    name, 
                    "Error: frame_folder_path not configured in agent.", 
                    msgs
                )
                return False
            # Note: Overlap restrictions removed per user request
            # Redundant trim restrictions removed per user request
        
        # For finish, inject video_path, output_path, and target_length_sec
        if name == "finish":
            args["video_path"] = self.video_path or ""
            args["output_path"] = self.output_path or ""
            args["target_length_sec"] = self.current_target_length or 0.0

        # Call the tool
        try:
            print(f"Calling function `{name}` with args: {args}")
            result = self.name_to_function_map[name](**args)
            print("Result: ", result)
            self._append_tool_msg(tool_call["id"], name, result, msgs)

            # Check if finish was successful
            if name == "finish" and result.startswith("Successfully created edited video"):
                print(f"Section completed successfully: {result}")
                return True  # Signal to break the current section loop
            
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
                msgs = copy.deepcopy(self.messages)
                print(f"\n{'='*60}")
                print(f"Processing Shot {idx + 1}/{len(shot_plan['shots'])}")
                print(f"{'='*60}\n")
                # Set shot-specific output path: <base>_section_<sec_idx>_shot_<idx>.mp4
                print("shot plan: ", shot)
                base_path, ext = os.path.splitext(original_output_path)
                self.output_path = f"{base_path}_section_{sec_idx}_shot_{idx}{ext}"
                print(f"Shot output path: {self.output_path}")
                # Set current shot for reporting
                self.current_shot_idx = idx
                # No per-shot trim history tracking
                msgs[-1]["content"] = msgs[-1]["content"].replace("VIDEO_LENGTH_PLACEHOLDER", str(shot['time_duration']))
                # msgs[-1]["content"] = msgs[-1]["content"].replace("WHOLE_VIDEO_CONTENT_PLACEHOLDER", overall_theme).replace("WHOLE_VIDEO_NARRATIVE_LOGIC_PLACEHOLDER", narrative_logic)
                msgs[-1]["content"] = msgs[-1]["content"].replace("CURRENT_VIDEO_CONTENT_PLACEHOLDER", shot['content']).replace("CURRENT_VIDEO_EMOTION_PLACEHOLDER", shot['emotion'])
                # Get corresponding audio section's detailed analysis
                audio_section_info = str('summary: ' + self.audio_db['sections'][sec_idx].get('detailed_analysis', '')['summary']) + "\n" + 'section_caption: ' + str(self.audio_db['sections'][sec_idx].get('detailed_analysis', '')['sections'][idx])
                msgs[-1]["content"] = msgs[-1]["content"].replace("BACKGROUND_MUSIC_PLACEHOLDER", audio_section_info)
                self.current_target_length = shot['time_duration']
                self.current_related_scenes = shot.get('related_scene', [])

                for i in range(self.max_iterations):
                    # msgs[-1]["content"] = msgs[-1]["content"].replace("VIDEO_THEME_PLACEHOLDER", overall_theme).replace("VIDEO_NARRATIVE_LOGIC_PLACEHOLDER", narrative_logic)
                    if i == self.max_iterations - 1:
                        msgs.append(
                            {
                                "role": "user",
                                "content": "Please call the `finish` function to finish the task.",
                            }
                        )

                    # Retry loop for both model call and tool execution
                    # If tool execution fails, we rollback and retry the entire model call
                    max_model_retries = 2  # Retry model call if it returns None
                    max_tool_retries = 2   # Retry entire iteration if tool execution fails
                    tool_execution_success = False
                    
                    for tool_retry in range(max_tool_retries):
                        # Save snapshot of msgs before making any changes
                        msgs_snapshot = copy.deepcopy(msgs)
                        
                        # Call model with retry mechanism
                        response = None
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
                                    break  # Success, exit model retry loop
                                else:
                                    print(f"⚠️  Model returned None, retrying model call ({model_retry + 1}/{max_model_retries})...")
                            except Exception as e:
                                print(f"⚠️  Model call failed with error: {e}, retrying ({model_retry + 1}/{max_model_retries})...")
                                if model_retry == max_model_retries - 1:
                                    raise
                        
                        # If all model retries failed, skip this iteration entirely
                        if response is None:
                            print(f"❌ Model call failed after {max_model_retries} retries. Skipping iteration {i}.")
                            # Restore original msgs and remove finish prompt if added
                            msgs[:] = msgs_snapshot
                            if i == self.max_iterations - 1 and msgs and msgs[-1].get("content") == "Please call the `finish` function to finish the task.":
                                msgs.pop()
                            break  # Exit tool retry loop
                        
                        # Add response to msgs
                        response.setdefault("role", "assistant")
                        msgs.append(response)
                        print("#### Iteration: ", i, f"(Tool retry: {tool_retry + 1}/{max_tool_retries})" if tool_retry > 0 else "")
                        print(response)
                        
                        # Execute any requested tool calls
                        section_completed = False
                        tool_execution_failed = False
                        
                        try:
                            tool_calls = response.get("tool_calls", [])
                            if tool_calls is None:
                                print("⚠️  Warning: tool_calls is None, treating as empty list")
                                tool_calls = []
                            
                            for tool_call in tool_calls:
                                is_finished = self._exec_tool(tool_call, msgs)
                                if is_finished:
                                    section_completed = True
                                    break
                            
                            # If we reach here, tool execution was successful
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
                            if tool_execution_failed and tool_retry == max_tool_retries - 1:
                                print(f"❌ Tool execution failed after {max_tool_retries} retries. Moving to next iteration.")
                            break
                        
                        # Tool execution failed, rollback and retry
                        if tool_execution_failed:
                            print(f"🔄 Rolling back messages and retrying model call ({tool_retry + 1}/{max_tool_retries})...")
                            msgs[:] = msgs_snapshot  # Rollback to snapshot
                            continue  # Retry the model call
                    
                    # If model call failed completely, skip to next iteration
                    if response is None:
                        continue
                    
                    # If section is completed successfully, move to next section
                    if section_completed:
                        print(f"Section {sec_idx + 1} completed. Moving to next section...")
                        break
        
                # Restore original output path
                print(f"\nAll sections processed. Section videos saved with '_section_N' suffix.")

        return msgs


def single_run_wrapper(info) -> dict:
    qid, video_db_path, video_caption_path, question = info
    agent = DVDCoreAgent(video_db_path, video_caption_path, question)
    msgs = agent.run()
    return {qid: msgs}


def main():
    video_caption_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Database/Batman_Begins_2005_1080p_BluRay_x264_YIFY/captions/captions.json"
    video_scene_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Database/Batman_Begins_2005_1080p_BluRay_x264_YIFY/captions/scene_summaries_video"
    audio_caption_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Database/CallofSilence/captions/captions.json"
    video_db_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Database/Batman_Begins_2005_1080p_BluRay_x264_YIFY/vdb.json"
    frame_folder_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Database/Batman_Begins_2005_1080p_BluRay_x264_YIFY/frames"
    video_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/Dataset/Video/Batman.Begins.2005.1080p.BluRay.x264.YIFY.mp4"
    Audio_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/Dataset/Audio/Call_of_Slience/CallofSilence.mp3"

    shot_plan_output = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/output_batman_waywe_go/shot_plan.json"

    output_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/output_batman_waywe_go"

    Instruction = """Give me a video that show the growth of batman from a young boy to a mature man."""

    agent = DVDCoreAgent(
        video_db_path,
        video_caption_path,
        video_scene_path,
        audio_caption_path,
        output_path,
        max_iterations=100,
        video_path=video_path,
        frame_folder_path=frame_folder_path
    )
    messages = agent.run(Instruction, shot_plan_path=shot_plan_output)

    # if messages:
    #     for m in messages:
    #         import pdb; pdb.set_trace()
    #         if m.get('role') == 'assistant':
    #             print(m)


if __name__ == "__main__":
    main()

