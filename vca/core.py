import os
import json
import copy
import re
import numpy as np
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

GENERATE_STRUCTURE_PROPOSAL_PROMPT = """
You are a professional video editor. Your task is to reorganize and assemble a large collection of video material into a short video. The editing should:
1. Align the rhythm and emotional progression of the video with the development of the accompanying music;
2. Seamlessly integrate the main theme and subject matter of the video materials;
3. Satisfy the user's specific editing instructions.

Requirements:
- Ensure that all selected video content is consistent with the main theme and narrative style.
- The final video should display clear emotional and rhythmic variation as well as story progression and visual engagement.
- Avoid any irrelevant or off-theme content, monotonous emotion or pacing, or lack of narrative/story.

Your goal is to design a detailed and coherent organizational plan for the short video, specifying the emotional arc and content of each segment.

1. A summary of the clippings's content: VIDEO_SUMMARY_PLACEHOLDER
2. A summary of the audio's content: AUDIO_SUMMARY_PLACEHOLDER
3. The structure of the audio: AUDIO_STRUCTURE_PLACEHOLDER
4. The user's editing instruction: INSTRUCTION_PLACEHOLDER
Based on these inputs, please analyze and provide the following information about the target video in a structured format:
{
    "overall_theme": "The overall theme of the video",
    "narrative_logic": "The narrative logic of the video",
    "video_structure": [
        {   
            "content": "The detailed description of the content in this segment",
            "start_time": "The start time of the segment",
            "end_time": "The end time of the segment",
            "audio_section": "The section in given audio",
            "emotion": "The emotion of the segment",
        }
        ...
    ]
"""


GENERATE_SHOT_PLAN_PROMPT = """
[Role]
You are a senior video editor and story architect who translates music structure into cinematic story beats aligned with the provided storyline.

[Task]
Map each analyzed music segment to exactly one film shot so the resulting sequence reads as a coherent mini-arc that locks tightly to rhythm, phrasing, and emotional flow.

[Inputs]
- Whole-video summary for theme continuity: VIDEO_SUMMARY_PLACEHOLDER
- Detailed per-segment music analysis: AUDIO_SUMMARY_PLACEHOLDER
- Current video section brief: VIDEO_SECTION_INFO_PLACEHOLDER

[Workflow]
1. Internalize the three inputs so the section plan aligns with global narrative and musical intent.
2. For every music part (in order), design one shot that preserves continuity of subject, geography, and emotional trajectory.
3. Define the shot's visual content, story beat, and narrative function (setup/development/payoff/button/bridge) while ensuring the duration (>= 3.0s) tracks the music window.
4. Specify the dominant emotion, key visuals, shot type (e.g., CU/MCU/WS + camera move), and explicit music sync/transitional intent so the editor can implement without guessing.
5. Validate the full list for uninterrupted mini-arc flow and fidelity to the soundtrack's major cues (downbeats, motif changes, swells, drops).

[Guidelines]
- Enforce strict one-to-one mapping between music segments and shots; no merges or splits.
- Keep durations realistic relative to each music segment (small deviations only for continuity or rhythm).
- Reject shots that would break continuity or introduce subject/action jumps.
- Prioritize concrete visual details (blocking, composition, motion, lighting) over vague adjectives.
- Strictly adhere to the given storyline; do NOT design shots for content outside the provided narrative. Follow the storyline's plot development precisely when designing each shot.


[Output]
Return STRICT JSON ONLY with this schema (no extra text):
{
    "shots": [
        {
            "id": <int, same as in AUDIO_SUMMARY_PLACEHOLDER>,
            "time_duration": <float, duration in seconds>,
            "content": "<detailed description of on-screen action>",
            "story_beat": "<precise beat for this moment>",
            "narrative_function": "setup|development|payoff|button|bridge",
            "emotion": "<dominant emotion>",
            "visuals": "<key visual elements / composition / movement>",
            "shot_type": "<e.g., CU/MCU/WS + tracking/static/handheld>"
        },
        ...
    ]
}
"""

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


def get_video_clip_frame(
        database: A[NanoVectorDB, D("The database object that supports querying with embeddings.")],
        event_description: A[str, D("A textual description of the event to search for.")],
        top_k: A[int, D("The maximum number of top results to retrieve. Just use the default value.")] = 16
) -> tuple:
    """
    Searches for events in a video clip database based on a given event description and retrieves the top-k most relevant video clip captions.

    Returns:
        str: A formatted string containing the concatenated captions of the searched video clip scripts.

    Notes:
        - This function utilizes the vLLM Embedding Service to generate embeddings for the input text.
        - Use default values for `top_k` to limit the number of results returned.
    """
    # 获取对应片段的数据
    embedding_data = get_vllm_embeddings(
        input_text=event_description,
        endpoint=config.VLLM_EMBEDDING_ENDPOINT
    )
    # Extract the embedding vector from the response
    embedding = np.array(embedding_data[0]['embedding'])
    
    results = database.query(embedding, top_k=top_k)
    captions = [
    (data['time_start_secs'], data['caption'])
    for i, data in enumerate(results)
    ]
    captions = sorted(captions, key=lambda x: x[0])
    captions = "\n".join([cap[1] for cap in captions])
    return f"Here is the searched video clip scripts:\n\n" + captions

def trim_video_clip(
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


def generate_structure_proposal(video_summary_path, audio_db, user_instruction):
    """
    Generate a structure proposal for the video editing.
    
    Args:
        video_summary_path: Path to video summary JSON file
        audio_caption_path: Path to audio caption JSON file
        user_instruction: User's editing instruction
        
    Returns:
        str: Structure proposal response from LLM
    """
    # Read video summary
    with open(video_summary_path, 'r', encoding='utf-8') as f:
        video_summary_data = json.load(f)
    video_summary = video_summary_data.get('clip_description', '')
    
    
    # Extract overall analysis summary
    audio_summary = audio_db.get('overall_analysis', {}).get('summary', '')
    
    # Extract sections without detailed_analysis
    sections = audio_db.get('sections', [])
    filtered_sections = []
    for section in sections:
        # Create a copy of section without detailed_analysis
        section_copy = {
            'name': section.get('name', ''),
            'description': section.get('description', ''),
            'Start_Time': section.get('Start_Time', ''),
            'End_Time': section.get('End_Time', '')
        }
        filtered_sections.append(section_copy)
    
    # Convert audio structure to string format
    audio_structure = json.dumps(filtered_sections, indent=2, ensure_ascii=False)
    
    # Construct the prompt using the template
    prompt = GENERATE_STRUCTURE_PROPOSAL_PROMPT
    prompt = prompt.replace("VIDEO_SUMMARY_PLACEHOLDER", video_summary)
    prompt = prompt.replace("AUDIO_SUMMARY_PLACEHOLDER", audio_summary)
    prompt = prompt.replace("AUDIO_STRUCTURE_PLACEHOLDER", audio_structure)
    prompt = prompt.replace("INSTRUCTION_PLACEHOLDER", user_instruction)
    
    # Prepare messages for the LLM

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    # Call the LLM model
    response = call_vllm_model(
        messages,
        endpoint=config.VLLM_AGENT_ENDPOINT,
        model_name=config.AGENT_MODEL,
        temperature=0.0,
        max_tokens=config.AGENT_MODEL_MAX_TOKEN,
    )
    
    if response is None:
        return None
    
    # Extract content from response
    content = response.get('content', '')
    
    return content 

def generate_structure_proposal_all_caption(video_caption_path, audio_db, user_instruction):
    """
    Generate a structure proposal for the video editing.
    
    Args:
        video_summary_path: Path to video summary JSON file
        audio_caption_path: Path to audio caption JSON file
        user_instruction: User's editing instruction
        
    Returns:
        str: Structure proposal response from LLM
    """
    # Read video summary
    with open(video_caption_path, 'r', encoding='utf-8') as f:
        video_caption_data = json.load(f)
    video_caption = video_caption_data.get('clip_description', '')
    
    
    # Extract overall analysis summary
    audio_summary = audio_db.get('overall_analysis', {}).get('summary', '')
    
    # Extract sections without detailed_analysis
    sections = audio_db.get('sections', [])
    filtered_sections = []
    for section in sections:
        # Create a copy of section without detailed_analysis
        section_copy = {
            'name': section.get('name', ''),
            'description': section.get('description', ''),
            'Start_Time': section.get('Start_Time', ''),
            'End_Time': section.get('End_Time', '')
        }
        filtered_sections.append(section_copy)
    
    # Convert audio structure to string format
    audio_structure = json.dumps(filtered_sections, indent=2, ensure_ascii=False)
    
    # Construct the prompt using the template
    prompt = GENERATE_STRUCTURE_PROPOSAL_PROMPT
    prompt = prompt.replace("VIDEO_SUMMARY_PLACEHOLDER", video_caption)
    prompt = prompt.replace("AUDIO_SUMMARY_PLACEHOLDER", audio_summary)
    prompt = prompt.replace("AUDIO_STRUCTURE_PLACEHOLDER", audio_structure)
    prompt = prompt.replace("INSTRUCTION_PLACEHOLDER", user_instruction)
    
    # Prepare messages for the LLM

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    # Call the LLM model
    response = call_vllm_model(
        messages,
        endpoint=config.VLLM_AGENT_ENDPOINT,
        model_name=config.AGENT_MODEL,
        temperature=0.0,
        max_tokens=config.AGENT_MODEL_MAX_TOKEN,
    )
    
    if response is None:
        return None
    
    # Extract content from response
    content = response.get('content', '')
    
    return content 

        
def generate_shot_plan(video_summary_path: str, music_detailed_structure, video_section_proposal) -> str | None:
    """
    Generate a one-to-one shot mapping for each music part using GENERATE_MUSIC_SHOT_MAPPING_PROMPT.

    - video_summary_path: path to video_summary.json (provides VIDEO_CONTEXT)
    - music_detailed_structure: list/dict or JSON string describing music parts (from captions.json detailed sections)
    - user_instruction: editing brief
    """
    # Read video summary context
    try:
        with open(video_summary_path, 'r', encoding='utf-8') as f:
            video_summary_data = json.load(f)
        video_context = video_summary_data.get('clip_description', '')
    except Exception:
        video_context = ''

    # Normalize music structure to JSON string
    if isinstance(music_detailed_structure, (dict, list)):
        music_json = json.dumps(music_detailed_structure, ensure_ascii=False, indent=2)
    else:
        music_json = str(music_detailed_structure or '')

    prompt = GENERATE_SHOT_PLAN_PROMPT
    prompt = prompt.replace("VIDEO_SUMMARY_PLACEHOLDER", video_context)
    prompt = prompt.replace("AUDIO_SUMMARY_PLACEHOLDER", music_json)
    prompt = prompt.replace("VIDEO_SECTION_INFO_PLACEHOLDER", str(video_section_proposal))

    messages = [
        {"role": "user", "content": prompt}
    ]

    response = call_vllm_model(
        messages,
        endpoint=config.VLLM_AGENT_ENDPOINT,
        model_name=config.AGENT_MODEL,
        temperature=0.0,
        max_tokens=getattr(config, 'AGENT_MODEL_MAX_TOKEN', 2048),
    )
    if response is None:
        return None
    return response.get('content', '')


def parse_structure_proposal_output(output: str):
    """
    解析generate_structure_proposal输出的结果。

    Args:
        output (str): generate_structure_proposal返回的字符串，通常为模型生成的结构提案。

    Returns:
        dict or list: 尝试将输出解析为结构化数据（dict或list），解析失败则返回原始字符串。
    """
    import json
    import re

    # 尝试直接json解析
    try:
        result = json.loads(output)
        return result
    except Exception:
        pass

    # 尝试从代码块中提取json字符串
    json_block_re = re.compile(r"```(?:json)?\n(.*?)```", re.DOTALL | re.IGNORECASE)
    match = json_block_re.search(output)
    if match:
        json_str = match.group(1)
        try:
            result = json.loads(json_str)
            return result
        except Exception:
            pass

    # 尝试定位到貌似json的部分开始(比如以{或[开头)
    json_start = min(
        [i for i in (output.find("{"), output.find("[")) if i != -1] or [None]
    )
    if json_start is not None:
        json_candidate = output[json_start:]
        try:
            result = json.loads(json_candidate)
            return result
        except Exception:
            pass

    # 再尝试提取所有大括号内容
    brackets = re.findall(r'({.*})', output, re.DOTALL)
    for b in brackets:
        try:
            result = json.loads(b)
            return result
        except Exception:
            continue

    # 最后返回原始字符串结果
    return output


def parse_shot_plan_output(output: str) -> dict | None:
    if not output:
        return None
    text = output.strip()
    # Strip ```json ... ``` fences if present (keep parser simple but robust to common wrapping)
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        return None

class DVDCoreAgent:
    def __init__(self, video_db_path, video_caption_path, video_summary_path, audio_caption_path, output_path, max_iterations, video_path=None, frame_folder_path=None):
        self.tools = [get_video_clip_frame, trim_video_clip, finish]
        self.name_to_function_map = {tool.__name__: tool for tool in self.tools}
        self.function_schemas = [
            {"function": as_json_schema(func), "type": "function"}
            for func in self.name_to_function_map.values()
        ]
        self.video_db = init_single_video_db(video_caption_path, video_db_path, config.AOAI_EMBEDDING_LARGE_DIM)
        self.video_summary_path = video_summary_path
        self.audio_db = json.load(open(audio_caption_path, 'r', encoding='utf-8'))
        self.max_iterations = max_iterations
        self.frame_folder_path = frame_folder_path
        self.video_path = video_path
        self.output_path = output_path
        self.current_target_length = None  # Will be set during run()
        self.messages = self._construct_messages()
        # Note: no trim overlap or redundancy restrictions; tracking removed per user request

        # Reporting/logging state
        self._agent_log = []
        self.current_section_idx = None
        self.current_shot_idx = None
        try:
            out_dir = os.path.dirname(self.output_path) if self.output_path else os.getcwd()
            self.report_root = os.path.join(out_dir or ".", "reports")
            os.makedirs(self.report_root, exist_ok=True)
        except Exception:
            self.report_root = os.path.join(os.getcwd(), "reports")
            try:
                os.makedirs(self.report_root, exist_ok=True)
            except Exception:
                pass

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
• `get_video_clip_frame`: retrieve candidate video clips from the database for contextual exploration.
• `trim_video_clip`: **IMPORTANT**: This tool analyzes a time range and returns scene descriptions to help you understand what's happening INSIDE the analyzed clip.
• `finish`: present the final timestamped editing plan once all required clips are selected and refined.

[Workflow]
1. Review the global brief and initial observations about the video.
2. Use `get_video_clip_frame` to surface promising segments aligned with the target theme, narrative logic, and emotion, filtering out clips whose narrative jumps conflict with the current storyline.
3. For each promising segment, call `trim_video_clip` with a time range (e.g., "00:13:28 to 00:13:40") to get detailed scene breakdown to understand the content.
4. Based on trim_video_clip's output, decide your shot selection:
   - If the full analyzed range works well → use it as one shot
   - If only part of it fits → select a continuous subset (e.g., first 3 scenes combined)
   - If you need more precision → call trim_video_clip again with a narrower range
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
- **Understanding trim_video_clip output**:
  * The "analyzed_range" and "total_duration_sec" describe the ENTIRE clip you requested
  * The "internal_scenes" are fine-grained breakdowns for REFERENCE - they help you understand what's inside, but are NOT meant to be used as separate shots
  * Always read "recommended_usage" for guidance on how to select usable portions
  * To create a shot, use a TIME RANGE that makes sense (≥3s), not individual scene times
- **Shot selection strategy**:
  * After reviewing trim_video_clip output, decide on a continuous time range for your shot
  * Combine multiple internal_scenes if needed to reach ≥3s duration
  * Example: If scenes 1-4 (spanning 00:13:28 to 00:13:35) form a coherent beat, your shot is "00:13:28 to 00:13:35", NOT four separate 1-2s shots
- Continuity check: discard retrieved clips that introduce abrupt jumps or contradict the evolving narrative before moving forward.
- Maintain consistent formatting for timestamps (HH:MM:SS or MM:SS) and keep the final plan aligned with the requested length and narrative arc.
- Ensure every selected shot lasts at least 3 seconds; if trim_video_clip shows good content but it's fragmented, select a continuous range that combines multiple scenes.
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

    def _append_agent_log(self, entry: dict) -> None:
        try:
            # shallow copy with small truncation for large fields
            def _truncate(v):
                try:
                    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
                except Exception:
                    s = str(v)
                if len(s) > 4000:
                    return s[:4000] + "\n...[truncated]"
                return s
            safe_entry = {}
            for k, v in entry.items():
                if k in ("args", "result", "content", "data"):
                    safe_entry[k] = _truncate(v)
                else:
                    safe_entry[k] = v
            self._agent_log.append(safe_entry)
        except Exception:
            pass

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
        
        # For trim_video_clip, inject frame_folder parameter
        if name == "trim_video_clip":
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
            # Prepare a JSON-serializable view of args for logging (strip DB object)
            try:
                args_for_log = {k: ("[database]" if k == "database" else v) for k, v in args.items()}
            except Exception:
                args_for_log = {}
            print(f"Calling function `{name}` with args: {args}")
            result = self.name_to_function_map[name](**args)
            print("Result: ", result)
            self._append_tool_msg(tool_call["id"], name, result, msgs)
            # Log tool call to agent timeline
            try:
                self._append_agent_log({
                    "type": "tool_call",
                    "section": self.current_section_idx,
                    "shot": self.current_shot_idx,
                    "tool": name,
                    "args": args_for_log,
                    "result": result,
                })
            except Exception:
                pass
            # Persist per-shot artifacts for visualization
            try:
                if name == "get_video_clip_frame":
                    shot_dir = self._ensure_shot_report_dir()
                    self._write_text_file(os.path.join(shot_dir, "retrieval.txt"), str(result))
                elif name == "trim_video_clip":
                    shot_dir = self._ensure_shot_report_dir()
                    self._write_text_file(os.path.join(shot_dir, "segments.json"), str(result))
                    self._render_shot_report(str(result))
                # Update timelines incrementally after every tool call
                self._render_agent_timeline()
                self._render_shot_timeline(self.current_section_idx, self.current_shot_idx)
            except Exception:
                pass
            # Record successful trim ranges
            if name == "trim_video_clip":
                try:
                    requested_range = args.get("time_range", "")
                    match = re.search(r"([\d:]+)\s+to\s+([\d:]+)", requested_range, re.IGNORECASE)
                    if match:
                        def _to_sec(t: str) -> float:
                            parts = t.strip().split(':')
                            if len(parts) == 3:
                                h, m, s = [int(x) for x in parts]
                                return h * 3600 + m * 60 + s
                            if len(parts) == 2:
                                m, s = [int(x) for x in parts]
                                return m * 60 + s
                            return float(parts[0])
                        req_start = _to_sec(match.group(1))
                        req_end = _to_sec(match.group(2))
                except Exception:
                    pass
            
            # Check if finish was successful
            if name == "finish" and result.startswith("Successfully created edited video"):
                print(f"Section completed successfully: {result}")
                return True  # Signal to break the current section loop
            
            return False
        except StopException as exc:  # graceful stop
            print(f"Finish task with message: '{exc!s}'")
            raise

    # ------------------------------- Reporting utils ------------------------------- #
    def _time_str_to_sec(self, t: str) -> float:
        parts = t.strip().split(':')
        if len(parts) == 3:
            try:
                h, m, s = [int(x) for x in parts]
                return h * 3600 + m * 60 + s
            except Exception:
                try:
                    h, m, s = [float(x) for x in parts]
                    return h * 3600 + m * 60 + s
                except Exception:
                    return 0.0
        if len(parts) == 2:
            try:
                m, s = [int(x) for x in parts]
                return m * 60 + s
            except Exception:
                try:
                    m, s = [float(x) for x in parts]
                    return m * 60 + s
                except Exception:
                    return 0.0
        try:
            return float(parts[0])
        except Exception:
            return 0.0

    def _ensure_shot_report_dir(self) -> str:
        sec = self.current_section_idx if self.current_section_idx is not None else 0
        shot = self.current_shot_idx if self.current_shot_idx is not None else 0
        shot_dir = os.path.join(self.report_root, f"section_{sec}_shot_{shot}")
        try:
            os.makedirs(shot_dir, exist_ok=True)
        except Exception:
            pass
        return shot_dir

    def _copy_frame_for_time(self, seconds: float, dst_dir: str, basename: str) -> str | None:
        if not self.frame_folder_path:
            return None
        fps = getattr(config, "SHOT_DETECTION_FPS", None) or getattr(config, "VIDEO_FPS", 0) or 0
        if fps <= 0:
            return None
        frame_index = int(seconds * fps)
        # Try png then jpg
        fname = f"frame_{frame_index:06d}.png"
        src_png = os.path.join(self.frame_folder_path, fname)
        src_jpg = os.path.join(self.frame_folder_path, f"frame_{frame_index:06d}.jpg")
        src_path = src_png if os.path.exists(src_png) else (src_jpg if os.path.exists(src_jpg) else None)
        if not src_path:
            return None
        import shutil
        dst_path = os.path.join(dst_dir, f"{basename}.png")
        try:
            shutil.copyfile(src_path, dst_path)
            return dst_path
        except Exception:
            return None

    def _write_text_file(self, path: str, content: str) -> None:
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception:
            pass

    def _render_shot_report(self, segments_json: str) -> None:
        """Render an HTML report for current shot using trim_video_clip segments JSON."""
        try:
            segments = json.loads(segments_json)
            if not isinstance(segments, list):
                segments = [segments]
        except Exception:
            # Not JSON; nothing to render
            return
        sec = self.current_section_idx if self.current_section_idx is not None else 0
        shot = self.current_shot_idx if self.current_shot_idx is not None else 0
        shot_dir = self._ensure_shot_report_dir()
        # Copy a representative frame per segment and build HTML blocks
        html_blocks = []
        for i, seg in enumerate(segments):
            tr = str(seg.get("time_range", "")).strip()
            start_s = end_s = None
            if tr:
                m = re.search(r"([0-9:.]+)\s+to\s+([0-9:.]+)", tr, re.IGNORECASE)
                if m:
                    start_s = self._time_str_to_sec(m.group(1))
                    end_s = self._time_str_to_sec(m.group(2))
            rep_time = None
            if start_s is not None and end_s is not None and end_s > start_s:
                rep_time = (start_s + end_s) / 2.0
            elif start_s is not None:
                rep_time = start_s
            img_path = None
            if rep_time is not None:
                copied = self._copy_frame_for_time(rep_time, shot_dir, f"segment_{i:02d}")
                if copied:
                    img_path = os.path.relpath(copied, self.report_root)
            # Escape text fields for HTML safety
            def esc(x: str) -> str:
                try:
                    return (x or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                except Exception:
                    return str(x)
            story_beat = esc(seg.get("story_beat", ""))
            shot_type = esc(seg.get("shot_type", ""))
            emotion = esc(seg.get("emotion", ""))
            clip_desc = esc(seg.get("clip_description", ""))
            music_align = esc(seg.get("music_alignment", ""))
            pacing = esc(seg.get("pacing_suggestion", ""))
            edit_point = esc(seg.get("edit_point_suggestion", ""))
            continuity = esc(seg.get("continuity_notes", ""))
            risk = esc(seg.get("risk_notes", ""))
            time_label = esc(tr)
            img_html = f"<img src='{img_path}' alt='segment frame' style='width:100%;max-width:560px;border-radius:6px;border:1px solid #333'/>" if img_path else "<div style='width:100%;max-width:560px;height:315px;background:#111;border:1px solid #333;border-radius:6px;display:flex;align-items:center;justify-content:center;color:#777'>No frame</div>"
            block = f"""
            <div style='display:flex;gap:16px;align-items:flex-start;margin:18px 0;padding-bottom:18px;border-bottom:1px solid #2a2a2a'>
              <div style='flex:0 0 auto'>{img_html}</div>
              <div style='flex:1 1 auto'>
                <div style='font-weight:700;color:#ddd;margin-bottom:6px'>时间段: {time_label}</div>
                <div style='color:#eee;margin:6px 0'><b>故事节拍</b>: {story_beat}</div>
                <div style='color:#bbb;margin:6px 0'><b>镜头/情绪</b>: {shot_type} · {emotion}</div>
                <details style='margin-top:8px'>
                  <summary style='cursor:pointer;color:#9ad'>更多细节</summary>
                  <div style='color:#ccc;white-space:pre-wrap;margin-top:8px'>
                    <div><b>画面描述</b>: {clip_desc}</div>
                    <div style='margin-top:6px'><b>音乐对齐</b>: {music_align}</div>
                    <div style='margin-top:6px'><b>节奏建议</b>: {pacing}</div>
                    <div style='margin-top:6px'><b>剪辑点建议</b>: {edit_point}</div>
                    <div style='margin-top:6px'><b>连贯性</b>: {continuity}</div>
                    <div style='margin-top:6px'><b>风险提示</b>: {risk}</div>
                  </div>
                </details>
              </div>
            </div>
            """
            html_blocks.append(block)
        # Wrap into a full HTML page for this shot
        title = f"Section {sec} · Shot {shot}"
        page_html = f"""
        <!doctype html>
        <html lang='zh-CN'>
        <head>
          <meta charset='utf-8'>
          <meta name='viewport' content='width=device-width, initial-scale=1'>
          <title>{title}</title>
          <style>
            body {{ background:#0b0b0b; color:#eaeaea; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', 'Liberation Sans', sans-serif; margin:0; padding:24px; }}
            a {{ color:#8ab4f8; }}
            .container {{ max-width: 1080px; margin: 0 auto; }}
            .header {{ margin-bottom: 18px; }}
            .meta {{ color:#aaa; }}
          </style>
        </head>
        <body>
          <div class='container'>
            <div class='header'>
              <h2 style='margin:0 0 6px 0'>{title}</h2>
              <div class='meta'>可视化：分段画面 + 文字说明</div>
              <div style='margin-top:10px'>
                <a href='../index.html'>返回索引</a> · <a href='trajectory.html'>查看此镜头轨迹</a>
              </div>
            </div>
            {''.join(html_blocks)}
          </div>
        </body>
        </html>
        """
        # Write shot page
        shot_page_path = os.path.join(self._ensure_shot_report_dir(), "report.html")
        try:
            with open(shot_page_path, 'w', encoding='utf-8') as f:
                f.write(page_html)
        except Exception:
            pass
        # Track for index creation
        if not hasattr(self, "_report_index_entries"):
            self._report_index_entries = []
        rel_path = os.path.relpath(shot_page_path, self.report_root)
        self._report_index_entries.append({
            "title": title,
            "href": rel_path
        })

    def _escape_html(self, x: str) -> str:
        try:
            return (x or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        except Exception:
            return str(x)

    def _render_agent_timeline(self) -> None:
        """Render an HTML page visualizing the agent's execution (messages, tools, results)."""
        log_json_path = os.path.join(self.report_root, "agent_log.json")
        try:
            with open(log_json_path, 'w', encoding='utf-8') as f:
                json.dump(self._agent_log, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        blocks = []
        for idx, entry in enumerate(self._agent_log):
            etype = entry.get("type", "")
            section = entry.get("section", "-")
            shot = entry.get("shot", "-")
            title = f"[{idx:03d}] {etype} · section {section} · shot {shot}"
            content = self._escape_html(entry.get("content", ""))
            result = self._escape_html(entry.get("result", ""))
            tool = self._escape_html(entry.get("tool", ""))
            args = self._escape_html(entry.get("args", ""))
            data = self._escape_html(entry.get("data", ""))
            detail_rows = []
            if tool:
                detail_rows.append(f"<div><b>tool</b>: {tool}</div>")
            if args:
                detail_rows.append(f"<div><b>args</b>: <pre style='white-space:pre-wrap'>{args}</pre></div>")
            if content:
                detail_rows.append(f"<div><b>assistant</b>: <pre style='white-space:pre-wrap'>{content}</pre></div>")
            if data:
                detail_rows.append(f"<div><b>data</b>: <pre style='white-space:pre-wrap'>{data}</pre></div>")
            if result:
                detail_rows.append(f"<div><b>result</b>: <pre style='white-space:pre-wrap'>{result}</pre></div>")
            detail_html = "".join(detail_rows) or "<div style='color:#777'>No details</div>"
            blocks.append(f"""
            <details style='margin:10px 0; padding:12px; border:1px solid #2a2a2a; border-radius:8px; background:#101010'>
              <summary style='cursor:pointer; color:#cfd; font-weight:600'>{title}</summary>
              <div style='margin-top:10px; color:#ddd'>{detail_html}</div>
            </details>
            """)
        html = f"""
        <!doctype html>
        <html lang='zh-CN'>
        <head>
          <meta charset='utf-8'>
          <meta name='viewport' content='width=device-width, initial-scale=1'>
          <title>Agent 执行轨迹</title>
          <style>
            body {{ background:#0b0b0b; color:#eaeaea; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', 'Liberation Sans', sans-serif; margin:0; padding:24px; }}
            a {{ color:#8ab4f8; }}
            .container {{ max-width: 1080px; margin: 0 auto; }}
          </style>
        </head>
        <body>
          <div class='container'>
            <h2 style='margin:0 0 10px 0'>Agent 执行轨迹</h2>
            <div style='color:#aaa;margin-bottom:12px'>包括 assistant 消息、tool 调用与结果、结构/镜头计划等</div>
            {''.join(blocks)}
          </div>
        </body>
        </html>
        """
        try:
            with open(os.path.join(self.report_root, "agent_timeline.html"), 'w', encoding='utf-8') as f:
                f.write(html)
        except Exception:
            pass

    def _write_report_index(self) -> None:
        entries = getattr(self, "_report_index_entries", []) or []
        items_html = []
        # Insert agent timeline link if exists
        try:
            timeline_rel = "agent_timeline.html"
            if os.path.exists(os.path.join(self.report_root, timeline_rel)):
                items_html.append(f"<li style='margin:8px 0'><a href='{timeline_rel}'>Agent 执行轨迹</a></li>")
        except Exception:
            pass
        for e in entries:
            t = e.get("title", "")
            h = e.get("href", "")
            items_html.append(f"<li style='margin:8px 0'><a href='{h}'>{t}</a></li>")
        index_html = f"""
        <!doctype html>
        <html lang='zh-CN'>
        <head>
          <meta charset='utf-8'>
          <meta name='viewport' content='width=device-width, initial-scale=1'>
          <title>可视化报告索引</title>
          <style>
            body {{ background:#0b0b0b; color:#eaeaea; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', 'Liberation Sans', sans-serif; margin:0; padding:24px; }}
            a {{ color:#8ab4f8; }}
            .container {{ max-width: 960px; margin: 0 auto; }}
          </style>
        </head>
        <body>
          <div class='container'>
            <h2 style='margin:0 0 12px 0'>可视化报告索引</h2>
            <ul style='list-style:disc;padding-left:20px'>
              {''.join(items_html)}
            </ul>
          </div>
        </body>
        </html>
        """
        try:
            with open(os.path.join(self.report_root, "index.html"), 'w', encoding='utf-8') as f:
                f.write(index_html)
        except Exception:
            pass

    def _render_shot_timeline(self, section: int | None, shot: int | None) -> None:
        try:
            sec = section if section is not None else 0
            sh = shot if shot is not None else 0
            shot_dir = self._ensure_shot_report_dir()
            blocks = []
            for idx, entry in enumerate(self._agent_log):
                if entry.get("section") != section or entry.get("shot") != shot:
                    continue
                etype = entry.get("type", "")
                title = f"[{idx:03d}] {etype}"
                content = self._escape_html(entry.get("content", ""))
                result = self._escape_html(entry.get("result", ""))
                tool = self._escape_html(entry.get("tool", ""))
                args = self._escape_html(entry.get("args", ""))
                data = self._escape_html(entry.get("data", ""))
                detail_rows = []
                if tool:
                    detail_rows.append(f"<div><b>tool</b>: {tool}</div>")
                if args:
                    detail_rows.append(f"<div><b>args</b>: <pre style='white-space:pre-wrap'>{args}</pre></div>")
                if content:
                    detail_rows.append(f"<div><b>assistant</b>: <pre style='white-space:pre-wrap'>{content}</pre></div>")
                if data:
                    detail_rows.append(f"<div><b>data</b>: <pre style='white-space:pre-wrap'>{data}</pre></div>")
                if result:
                    detail_rows.append(f"<div><b>result</b>: <pre style='white-space:pre-wrap'>{result}</pre></div>")
                detail_html = "".join(detail_rows) or "<div style='color:#777'>No details</div>"
                blocks.append(f"""
                <details style='margin:10px 0; padding:12px; border:1px solid #2a2a2a; border-radius:8px; background:#101010'>
                  <summary style='cursor:pointer; color:#cfd; font-weight:600'>{title}</summary>
                  <div style='margin-top:10px; color:#ddd'>{detail_html}</div>
                </details>
                """)
            html = f"""
            <!doctype html>
            <html lang='zh-CN'>
            <head>
              <meta charset='utf-8'>
              <meta name='viewport' content='width=device-width, initial-scale=1'>
              <title>Section {sec} · Shot {sh} · 轨迹</title>
              <style>
                body {{ background:#0b0b0b; color:#eaeaea; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', 'Liberation Sans', sans-serif; margin:0; padding:24px; }}
                a {{ color:#8ab4f8; }}
                .container {{ max-width: 1080px; margin: 0 auto; }}
              </style>
            </head>
            <body>
              <div class='container'>
                <h2 style='margin:0 0 10px 0'>Section {sec} · Shot {sh} · 轨迹</h2>
                <div style='color:#aaa;margin-bottom:12px'><a href='../index.html'>返回索引</a> · <a href='report.html'>返回镜头报告</a></div>
                {''.join(blocks)}
              </div>
            </body>
            </html>
            """
            with open(os.path.join(shot_dir, "trajectory.html"), 'w', encoding='utf-8') as f:
                f.write(html)
        except Exception:
            pass

    def _ensure_section_report_dir(self, sec_idx: int) -> str:
        d = os.path.join(self.report_root, f"section_{sec_idx}")
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            pass
        return d

    def _render_structure_proposal_report(self, structure_proposal: dict) -> None:
        if not isinstance(structure_proposal, dict):
            return
        overall_theme = self._escape_html(structure_proposal.get('overall_theme', ''))
        narrative_logic = self._escape_html(structure_proposal.get('narrative_logic', ''))
        video_structure = structure_proposal.get('video_structure', []) or []
        cards = []
        for i, seg in enumerate(video_structure):
            content = self._escape_html(str(seg.get('content', '')))
            st = self._escape_html(str(seg.get('start_time', '')))
            et = self._escape_html(str(seg.get('end_time', '')))
            audio = self._escape_html(str(seg.get('audio_section', '')))
            emo = self._escape_html(str(seg.get('emotion', '')))
            cards.append(f"""
            <div style='border:1px solid #2a2a2a;border-radius:10px;padding:12px;margin:12px 0;background:#101010'>
              <div style='color:#9ad;margin-bottom:6px'>Section {i}</div>
              <div style='color:#eee;white-space:pre-wrap'><b>内容</b>: {content}</div>
              <div style='color:#bbb;margin-top:6px'>
                <b>时间</b>: {st} → {et} &nbsp;&nbsp; <b>音乐段落</b>: {audio} &nbsp;&nbsp; <b>情绪</b>: {emo}
              </div>
            </div>
            """)
        html = f"""
        <!doctype html>
        <html lang='zh-CN'>
        <head>
          <meta charset='utf-8'>
          <meta name='viewport' content='width=device-width, initial-scale=1'>
          <title>Structure Proposal</title>
          <style>
            body {{ background:#0b0b0b; color:#eaeaea; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', 'Liberation Sans', sans-serif; margin:0; padding:24px; }}
            a {{ color:#8ab4f8; }}
            .container {{ max-width: 1080px; margin: 0 auto; }}
            .header {{ margin-bottom: 12px; }}
            .meta {{ color:#aaa; }}
          </style>
        </head>
        <body>
          <div class='container'>
            <div class='header'>
              <h2 style='margin:0 0 6px 0'>结构提案（Proposal）</h2>
              <div class='meta'><a href='index.html'>返回索引</a></div>
            </div>
            <div style='margin:8px 0;color:#ddd'><b>主题</b>: {overall_theme}</div>
            <div style='margin:8px 0;color:#ddd'><b>叙事逻辑</b>: {narrative_logic}</div>
            {''.join(cards)}
          </div>
        </body>
        </html>
        """
        try:
            out_path = os.path.join(self.report_root, "proposal.html")
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(html)
            # add to index
            if not hasattr(self, "_report_index_entries"):
                self._report_index_entries = []
            rel = os.path.relpath(out_path, self.report_root)
            self._report_index_entries.append({"title": "结构提案（Proposal）", "href": rel})
        except Exception:
            pass

    def _render_shot_plan_report(self, sec_idx: int, shot_plan: dict) -> None:
        if not isinstance(shot_plan, dict):
            return
        shots = shot_plan.get('shots', []) or []
        rows = []
        for s in shots:
            sid = self._escape_html(str(s.get('id', '')))
            dur = self._escape_html(str(s.get('time_duration', '')))
            content = self._escape_html(str(s.get('content', '')))
            beat = self._escape_html(str(s.get('story_beat', '')))
            func = self._escape_html(str(s.get('narrative_function', '')))
            emo = self._escape_html(str(s.get('emotion', '')))
            visuals = self._escape_html(str(s.get('visuals', '')))
            stype = self._escape_html(str(s.get('shot_type', '')))
            rows.append(f"""
              <tr>
                <td style='padding:8px 10px;border-bottom:1px solid #232323;color:#ddd'>{sid}</td>
                <td style='padding:8px 10px;border-bottom:1px solid #232323;color:#ddd'>{dur}</td>
                <td style='padding:8px 10px;border-bottom:1px solid #232323;color:#ddd'>{stype}</td>
                <td style='padding:8px 10px;border-bottom:1px solid #232323;color:#ddd'>{emo}</td>
                <td style='padding:8px 10px;border-bottom:1px solid #232323;color:#ddd'>{func}</td>
              </tr>
              <tr>
                <td colspan='5' style='padding:0 10px 14px 10px;border-bottom:1px solid #232323'>
                  <div style='color:#9ad'><b>内容</b></div>
                  <div style='color:#eaeaea;white-space:pre-wrap'>{content}</div>
                  <div style='color:#9ad;margin-top:6px'><b>故事节拍</b></div>
                  <div style='color:#eaeaea;white-space:pre-wrap'>{beat}</div>
                  <div style='color:#9ad;margin-top:6px'><b>视觉要点</b></div>
                  <div style='color:#eaeaea;white-space:pre-wrap'>{visuals}</div>
                </td>
            </tr>
            """)
        table = f"""
          <table style='width:100%;border-collapse:collapse;margin-top:10px'>
            <thead>
              <tr>
                <th style='text-align:left;padding:8px 10px;border-bottom:2px solid #2f2f2f;color:#aaa'>ID</th>
                <th style='text-align:left;padding:8px 10px;border-bottom:2px solid #2f2f2f;color:#aaa'>时长(s)</th>
                <th style='text-align:left;padding:8px 10px;border-bottom:2px solid #2f2f2f;color:#aaa'>镜头</th>
                <th style='text-align:left;padding:8px 10px;border-bottom:2px solid #2f2f2f;color:#aaa'>情绪</th>
                <th style='text-align:left;padding:8px 10px;border-bottom:2px solid #2f2f2f;color:#aaa'>叙事功能</th>
              </tr>
            </thead>
            <tbody>
              {''.join(rows)}
            </tbody>
          </table>
        """
        html = f"""
        <!doctype html>
        <html lang='zh-CN'>
        <head>
          <meta charset='utf-8'>
          <meta name='viewport' content='width=device-width, initial-scale=1'>
          <title>Section {sec_idx} Shot Plan</title>
          <style>
            body {{ background:#0b0b0b; color:#eaeaea; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', 'Liberation Sans', sans-serif; margin:0; padding:24px; }}
            a {{ color:#8ab4f8; }}
            .container {{ max-width: 1080px; margin: 0 auto; }}
            .header {{ margin-bottom: 12px; }}
          </style>
        </head>
        <body>
          <div class='container'>
            <div class='header'>
              <h2 style='margin:0 0 6px 0'>Section {sec_idx} · Shot Plan</h2>
              <div><a href='../index.html'>返回索引</a></div>
            </div>
            {table}
          </div>
        </body>
        </html>
        """
        try:
            sec_dir = self._ensure_section_report_dir(sec_idx)
            out_path = os.path.join(sec_dir, "shot_plan.html")
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(html)
            if not hasattr(self, "_report_index_entries"):
                self._report_index_entries = []
            rel = os.path.relpath(out_path, self.report_root)
            self._report_index_entries.append({"title": f"Section {sec_idx} · Shot Plan", "href": rel})
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self, instruction) -> list[dict]:
        """
        Run the ReAct-style loop with OpenAI Function Calling.
        """

        structure_proposal = generate_structure_proposal(self.video_summary_path, self.audio_db, instruction) # 第一次生产结构proposal，按照音乐的整体段落
        structure_proposal = parse_structure_proposal_output(structure_proposal)
        overall_theme = structure_proposal['overall_theme']
        narrative_logic = structure_proposal['narrative_logic']
        # Log + visualize structure proposal
        self._append_agent_log({
            "type": "structure_proposal",
            "section": None,
            "shot": None,
            "data": structure_proposal,
        })
        try:
            self._render_structure_proposal_report(structure_proposal)
            self._render_agent_timeline()
        except Exception:
            pass

        # TODO: 对每个section再做一次generate_structure_proposal，按照section的详细内容
        
        # Store original output path and create section-specific paths
        original_output_path = self.output_path
        print("structure_proposal: ", structure_proposal)
        for sec_idx, sec_cur in enumerate(structure_proposal['video_structure']):
            print(f"\n{'='*60}")
            print(f"Processing Section {sec_idx + 1}/{len(structure_proposal['video_structure'])}")
            print(f"{'='*60}\n")
            # Set current section for reporting
            self.current_section_idx = sec_idx
            
            # 计算视频区段长度（秒）
            start_time = sec_cur.get('start_time', '00:00')
            end_time = sec_cur.get('end_time', '00:00')
            def time_str_to_sec(t):
                parts = t.split(':')
                if len(parts) == 3:
                    h, m, s = [int(x) for x in parts]
                    return h * 3600 + m * 60 + s
                elif len(parts) == 2:
                    m, s = [int(x) for x in parts]
                    return m * 60 + s
                else:
                    return 0
            length_sec = abs(time_str_to_sec(end_time) - time_str_to_sec(start_time))
            
            # Set current target length for finish function validation
            
            # Set section-specific output path
            # e.g., output.mp4 -> output_section_0.mp4, output_section_1.mp4, etc.
            base_path, ext = os.path.splitext(original_output_path)
            self.output_path = f"{base_path}_section_{sec_idx}{ext}"
            print(f"Section {sec_idx + 1} output path: {self.output_path}")
            print(f"Target duration: {length_sec} seconds")
            print(f"Content: {sec_cur.get('content', 'N/A')}")
            print(f"Emotion: {sec_cur.get('emotion', 'N/A')}\n")
            
            

            shot_plan = generate_shot_plan(self.video_summary_path, self.audio_db['sections'][sec_idx].get('detailed_analysis', '')['sections'], sec_cur)
            shot_plan = parse_shot_plan_output(shot_plan)
            self._append_agent_log({
                "type": "shot_plan",
                "section": sec_idx,
                "shot": None,
                "data": shot_plan,
            })
            try:
                self._render_shot_plan_report(sec_idx, shot_plan)
                self._render_agent_timeline()
            except Exception:
                pass
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
                        # Log assistant message and intended tool calls
                        try:
                            self._append_agent_log({
                                "type": "assistant_message",
                                "section": sec_idx,
                                "shot": idx,
                                "content": response.get("content", ""),
                                "data": {"tool_calls": response.get("tool_calls", [])}
                            })
                            # Update timelines on every assistant turn
                            self._render_agent_timeline()
                            self._render_shot_timeline(self.current_section_idx, self.current_shot_idx)
                        except Exception:
                            pass
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
                    
        # Render agent timeline and write report index page linking to all reports
        try:
            self._render_agent_timeline()
        except Exception:
            pass
        try:
            self._write_report_index()
        except Exception:
            pass
        return msgs


# class SupervisorAgent(DVDCoreAgent):
#     def __init__(self, video_db_path, video_caption_path, video_summary_path, audio_caption_path, output_path, max_iterations, video_path=None, frame_folder_path=None):
#         super().__init__(video_db_path, video_caption_path, video_summary_path, audio_caption_path, output_path, max_iterations, video_path, frame_folder_path)
#         self.messages = self._construct_messages()

#     def _construct_messages(self):
#         messages = [
#             {
#                 "role": "system",


def single_run_wrapper(info) -> dict:
    qid, video_db_path, video_caption_path, question = info
    agent = DVDCoreAgent(video_db_path, video_caption_path, question)
    msgs = agent.run()
    return {qid: msgs}


def main():
    video_caption_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Database/Batman_Begins_2005_1080p_BluRay_x264_YIFY/captions/captions.json"
    video_summary_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Database/Batman_Begins_2005_1080p_BluRay_x264_YIFY/captions/video_summary.json"
    audio_caption_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Database/CallofSilence/captions/captions.json"
    video_db_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Database/Batman_Begins_2005_1080p_BluRay_x264_YIFY/vdb.json"
    frame_folder_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Database/Batman_Begins_2005_1080p_BluRay_x264_YIFY/frames"
    video_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/Dataset/Video/Batman.Begins.2005.1080p.BluRay.x264.YIFY.mp4"
    Audio_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/Dataset/Audio/Call_of_Slience/CallofSilence.mp3"

    output_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/output.mp4"

    Instruction = """Give me a video that show the growth of batman from a young boy to a mature man."""
    
    # trim_video_clip(time_range="00:13:57 to 00:14:16", frame_path=video_path)
    # generate_structure_proposal_all_caption(video_summary_path, audio_caption_path, Instruction)
    
    agent = DVDCoreAgent(
        video_db_path, 
        video_caption_path, 
        video_summary_path,
        audio_caption_path, 
        output_path,
        max_iterations=100,
        video_path=video_path,
        frame_folder_path=frame_folder_path
    )
    # print("Generating structure proposal...")
    # structure_proposal = generate_structure_proposal(video_summary_path, audio_caption_path, Instruction)
    # structure_proposal = parse_structure_proposal_output(structure_proposal)
    # print("Structure proposal: ", structure_proposal)
    messages = agent.run(Instruction)

    # if messages:
    #     for m in messages:
    #         import pdb; pdb.set_trace()
    #         if m.get('role') == 'assistant':
    #             print(m)


if __name__ == "__main__":
    main()

