import os
import json
import copy
import re
import argparse
from pathlib import Path
import numpy as np
from typing import Annotated as A, Any
from vca.build_database.video_caption import (
    convert_seconds_to_hhmmss, 
    SYSTEM_PROMPT,
    messages as caption_messages
)
from vca import config
from vca.func_call_shema import as_json_schema
from vca.func_call_shema import doc as D
from vca.vllm_calling import call_vllm_model

from pprint import pprint



GENERATE_STRUCTURE_PROPOSAL_PROMPT = """
VIDEO_SUMMARY_PLACEHOLDER

You are a professional tiktok editor specializing in creating short video.

**YOUR PRIMARY GOAL:**
Select 8-15 scenes that create the BEST MATCH between:
1. User's creative vision (instruction)
2. Music's energy and rhythm
3. Visual excitement and iconic moments

**USER'S CREATIVE VISION:**: INSTRUCTION_PLACEHOLDER


**SELECTION STRATEGY:**

**Step 1: Understand the Instruction's Core Elements**
Before selecting ANY scene, identify what the instruction emphasizes:
- **Visual Style**: What kind of visuals? (e.g., "visceral", "elegant", "chaotic", "intimate")
- **Key Elements**: What specific elements are mentioned? (e.g., "combat", "Tumbler", "relationship", "cityscape")
- **Energy Level**: What's the overall intensity? (e.g., "explosive action" vs "quiet reflection")
- **Emotional Tone**: What feeling should dominate? (e.g., "powerful", "melancholic", "triumphant")

**Step 2: Match Scenes to Instruction + Music**
For each scene you consider, ask:
1. **Does this scene's VISUAL STYLE match what the instruction describes?**
   - Example: If instruction says "visceral combat", does this scene show intense physical action?
   - Example: If instruction says "emotional intimacy", does this scene show close character moments?

2. **Does this scene contain ELEMENTS explicitly mentioned in the instruction?**
   - Example: If instruction mentions "Tumbler/Batmobile", prioritize vehicle scenes
   - Example: If instruction mentions "relationship", prioritize character interaction scenes

3. **Does this scene's ENERGY LEVEL match instruction + music?**
   - High-energy music + "combat" instruction → Dynamic action scenes with movement
   - Low-energy music + "reflection" instruction → Quiet character moments
   - Build-up music + "tension" instruction → Escalating threat or preparation scenes

4. **Does this scene feature the MAIN CHARACTER in a way that fits the instruction?**
   - If instruction emphasizes "physicality" → Character must be actively moving/fighting
   - If instruction emphasizes "iconography" → Character must be visually striking/memorable
   - If instruction emphasizes "emotion" → Character's expression must be prominent

**Step 3: Prioritize Based on Alignment Score**
Rate each scene's alignment with instruction:
- ⭐⭐⭐ **PERFECT MATCH**: Scene embodies multiple core elements from instruction
  - Example: "visceral combat" instruction → Batman fighting multiple enemies in brutal hand-to-hand combat
- ⭐⭐ **GOOD MATCH**: Scene contains 1-2 core elements from instruction
  - Example: "visceral combat" instruction → Batman standing ready for battle (static but iconic)
- ⭐ **WEAK MATCH**: Scene has main character but doesn't match instruction's style/energy
  - Example: "visceral combat" instruction → Bruce Wayne sitting quietly (wrong energy level)

**Choose scenes with ⭐⭐⭐ or ⭐⭐ alignment. Avoid ⭐ scenes.**

**Scene Selection Guidelines:**
1. **Visual Variety**: Mix different shot types (action, close-ups, wide shots) while maintaining instruction alignment
2. **Main Character Focus**: Protagonist should be the PRIMARY visual subject in most scenes
3. **Distribution**: Select scenes from different parts of the video for variety
4. **Total Count**: Pick 8-15 scenes total
5. **Available Scenes**: TOTAL_SCENE_COUNT_PLACEHOLDER scenes total

**No Hallucination**: Only use scenes explicitly described in the input.

**INPUT DATA:**
- Audio Summary: AUDIO_SUMMARY_PLACEHOLDER
- Audio Description: AUDIO_STRUCTURE_PLACEHOLDER

**OUTPUT (JSON):**
{
    "overall_theme": "Describe how your selected scenes match the instruction's vision",
    "narrative_logic": "Explain how scenes will sync with music progression",
    "emotion": "Overall emotional tone that aligns with instruction",
    "related_scenes": [8-15 scene indices with BEST instruction+music alignment]
}

"""



GENERATE_SHOT_PLAN_PROMPT = """
RELATED_VIDEO_PLACEHOLDER

[Role]
You are a professional music video editor creating a shot-by-shot plan that achieves the BEST MATCH between user's creative vision and music rhythm.

[YOUR PRIMARY GOAL]
For EACH music segment, select the shot that creates the STRONGEST ALIGNMENT with:
1. User's creative vision (instruction below)
2. This specific music segment's energy/emotion
3. Visual impact and character presence

[USER'S CREATIVE VISION]
INSTRUCTION_PLACEHOLDER


[Task]
Map each music segment to ONE shot by finding the BEST MATCH for that specific moment.

[Inputs]
- Music segments with detailed analysis: AUDIO_SUMMARY_PLACEHOLDER
- Creative direction from user: See USER'S CREATIVE VISION above
- Visual guidance: VIDEO_SECTION_INFO_PLACEHOLDER
- Available scenes: Provided above

[How to Select the Right Shot - Step by Step]

**For EACH music segment:**

**STEP 1: Understand This Music Segment**
Read the music segment's description carefully:
- What's the energy level? (explosive, building, calm, intense)
- What's the emotional tone? (triumphant, melancholic, aggressive, hopeful)
- What's the rhythm/pacing? (fast cuts, smooth flow, climactic peak)

**STEP 2: Extract Instruction's Key Requirements**
Re-read the instruction and identify:
- **Core Visual Style**: What kind of visuals dominate? (e.g., "visceral", "elegant", "chaotic")
- **Key Elements**: What's explicitly mentioned? (e.g., "combat", "Tumbler", "cityscape", "relationships")
- **Overall Energy**: What's the baseline intensity? (high-energy action vs contemplative emotion)
- **Character Focus**: How should the character be shown? (in action, in emotion, iconographic)

**STEP 3: Find the Best Shot Match**
For each available scene, calculate its MATCH SCORE:

**Match Score = Visual Style Match + Element Match + Energy Match + Character Match**

1. **Visual Style Match** (0-3 points)
   - Does this shot's visual style match what the instruction describes?
   - Example: Instruction says "visceral combat" → Shot shows brutal hand-to-hand fighting = 3 points
   - Example: Instruction says "visceral combat" → Shot shows character sitting quietly = 0 points

2. **Element Match** (0-3 points)
   - Does this shot contain elements explicitly mentioned in the instruction?
   - Example: Instruction mentions "Tumbler" → Shot shows Batmobile chase = 3 points
   - Example: Instruction mentions "Tumbler" → Shot has no vehicles = 0 points

3. **Energy Match** (0-3 points)
   - Does this shot's energy level match both instruction AND current music segment?
   - Example: Instruction = "intense", Music = "explosive climax", Shot = dramatic fight = 3 points
   - Example: Instruction = "intense", Music = "explosive climax", Shot = quiet dialogue = 0 points

4. **Character Match** (0-3 points)
   - Does the main character appear in a way that fits the instruction?
   - Example: Instruction emphasizes "physicality" → Character actively fighting = 3 points
   - Example: Instruction emphasizes "physicality" → Character lying still = 0 points

**Target Score: Aim for 9-12 points (perfect match)**
**Acceptable: 6-8 points (good match)**
**Avoid: 0-5 points (poor match)**

**STEP 4: Consider Music Sync**
- Fast-paced music → Choose shots with dynamic movement or quick visual changes
- Slow build-up → Choose shots with tension, preparation, or escalating action
- Emotional peak → Choose shots with character close-ups showing intense expression
- Bass drop/climax → Choose shots with explosive action or dramatic reveals

**STEP 5: Maintain Visual Flow**
- Ensure smooth transitions between consecutive shots
- Vary shot types (wide, medium, close-up) for visual interest
- Use different scenes across the video to avoid repetition
- Keep main character as primary visual subject in most shots

[Constraints]
- Every shot MUST use content from the provided related_scenes
- Duration must EXACTLY match the music segment
- One shot per music segment - no combining or splitting
- Describe only what's actually visible in the selected scene
- Distribute scenes evenly - avoid using the same scene repeatedly

[Output Format]
Return STRICT JSON ONLY:
{
    "shots": [
        {
            "id": <int, matching music segment id>,
            "time_duration": <float, EXACT duration from music segment>,
            "content": "<Detailed visual description of what's happening in this shot>",
            "story_beat": "<The narrative/emotional moment this shot represents>",
            "emotion": "<Energy level and mood that matches music + instruction>",
            "visuals": "<Camera angle, movement, lighting, composition details>",
            "related_scene": <int, the scene index being used>
        },
        ...
    ]
}

[Quality Checklist Before Submitting]
For each shot, verify:
✅ Does this shot's visual style match the instruction's core aesthetic?
✅ Does this shot contain elements mentioned in the instruction (if applicable)?
✅ Does this shot's energy match both the instruction AND this music segment?
✅ Is the main character present and shown in a way that fits the instruction?
✅ Does the shot duration exactly match the music segment?

**No Hallucination**: Only use scenes and content explicitly described in the input.

"""

def parse_structure_proposal_output(output: str):
    """
    解析generate_structure_proposal输出的结果。

    期望格式:
    {
        "overall_theme": "Visual style and mood",
        "narrative_logic": "How visual energy matches music progression",
        "emotion": "Visual mood/energy level",
        "related_scenes": [list of scene indices]
    }

    Args:
        output (str): generate_structure_proposal返回的字符串，通常为模型生成的结构提案。

    Returns:
        dict: 解析后的结构化数据，包含必需字段。解析失败则返回None。
    """
    import json
    import re

    def validate_structure(data):
        """验证解析结果是否符合预期格式"""
        if not isinstance(data, dict):
            return False

        # 检查必需字段
        required_fields = ['overall_theme', 'narrative_logic', 'emotion', 'related_scenes']
        for field in required_fields:
            if field not in data:
                print(f"Warning: Missing required field '{field}'")
                return False

        # 验证 related_scenes 是列表
        if not isinstance(data['related_scenes'], list):
            print(f"Warning: 'related_scenes' must be a list, got {type(data['related_scenes'])}")
            return False

        # 验证场景索引都是整数
        for idx, scene_id in enumerate(data['related_scenes']):
            if not isinstance(scene_id, int):
                print(f"Warning: Scene index at position {idx} is not an integer: {scene_id}")
                return False

        return True

    # 尝试直接json解析
    try:
        result = json.loads(output)
        if validate_structure(result):
            return result
        print("Direct JSON parse succeeded but validation failed")
    except Exception as e:
        print(f"Direct JSON parse failed: {e}")

    # 尝试从代码块中提取json字符串
    json_block_re = re.compile(r"```(?:json)?\n(.*?)```", re.DOTALL | re.IGNORECASE)
    match = json_block_re.search(output)
    if match:
        json_str = match.group(1)
        try:
            result = json.loads(json_str)
            if validate_structure(result):
                return result
            print("Code block JSON parse succeeded but validation failed")
        except Exception as e:
            print(f"Code block JSON parse failed: {e}")

    # 尝试定位到貌似json的部分开始(比如以{或[开头)
    json_start = min(
        [i for i in (output.find("{"), output.find("[")) if i != -1] or [None]
    )
    if json_start is not None:
        json_candidate = output[json_start:]
        try:
            result = json.loads(json_candidate)
            if validate_structure(result):
                return result
            print("JSON candidate parse succeeded but validation failed")
        except Exception as e:
            print(f"JSON candidate parse failed: {e}")

    # 再尝试提取所有大括号内容
    brackets = re.findall(r'({.*})', output, re.DOTALL)
    for b in brackets:
        try:
            result = json.loads(b)
            if validate_structure(result):
                return result
        except Exception:
            continue

    # 所有解析尝试都失败
    print("All parsing attempts failed. Raw output:")
    print(output[:500])  # Print first 500 chars for debugging
    return None

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

def load_scene_summaries(scene_folder_path: str) -> tuple[str, int]:
    """
    Load all scene_caption.scene_summary from the scene_summaries_video folder
    and concatenate them into a complete video material description.

    Args:
        scene_folder_path: Path to the scene summaries folder

    Returns:
        tuple: (Concatenated scene summaries, Number of loaded scenes)
    """
    scene_summaries = []

    # Get all scene_*.json files
    scene_files = []
    for filename in os.listdir(scene_folder_path):
        if filename.startswith('scene_') and filename.endswith('.json'):
            scene_files.append(filename)

    # Sort by scene number
    def get_scene_number(filename):
        # Extract number X from scene_X.json
        try:
            return int(filename.replace('scene_', '').replace('.json', ''))
        except ValueError:
            return float('inf')

    scene_files.sort(key=get_scene_number)

    # Read each scene file and extract scene_summary
    for filename in scene_files:
        filepath = os.path.join(scene_folder_path, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                scene_data = json.load(f)

            # Extract scene_caption.scene_summary
            video_analysis = scene_data.get('video_analysis', {})
            scene_caption = video_analysis.get('scene_caption', {})

            # Check is_usable field, only load usable scenes
            scene_classification = scene_caption.get('scene_classification', {})
            is_usable = scene_classification.get('is_usable', True)  # Default to True for backward compatibility
            if not is_usable:
                print(f"Skipping {filename}: scene is not usable (reason: {scene_classification.get('unusable_reason', 'unknown')})")
                continue

            # Check importance_score field, skip scenes with score < 3
            importance_score = scene_classification.get('importance_score', 5)  # Default to 5 for backward compatibility
            if importance_score < 3:
                print(f"Skipping {filename}: importance_score ({importance_score}) is below threshold (3)")
                continue

            scene_summary = scene_caption.get('scene_summary', {})

            if scene_summary:
                # Get basic scene info
                scene_id = scene_data.get('scene_id', 'Unknown')
                time_range = scene_data.get('time_range', {})
                start_time = time_range.get('start_seconds', 'N/A')
                end_time = time_range.get('end_seconds', 'N/A')

                # Build scene summary text
                narrative = scene_summary.get('narrative', '')
                key_event = scene_summary.get('key_event', '')
                location = scene_summary.get('location', '')
                time_state = scene_summary.get('time', '')

                summary_text = f"[Scene {scene_id}] ({start_time} - {end_time})\n"
                summary_text += f"Location: {location}, Time: {time_state}\n"
                summary_text += f"Key Event: {key_event}\n"
                summary_text += f"Narrative: {narrative}\n"

                scene_summaries.append(summary_text)

        except Exception as e:
            print(f"Warning: Failed to read {filename}: {e}")
            continue

    # Concatenate all scene summaries
    full_summary = "\n".join(scene_summaries)
    print(f"Loaded {len(scene_summaries)} scene summaries from {scene_folder_path}")

    return full_summary, len(scene_summaries)


class StopException(Exception):
    """
    Stop Execution by raising this exception (Signal that the task is Finished).
    """

def finish(
    answer: A[str, D("Render the edited video to the user.")],
    output_path: A[str, D("Path to save the shot plan")] = "",
) -> str:
    """
    Call this function after generating the executable shot plan.
    Args:
        answer (str): The generated shot plan in string format:
                {
                    "time_duration": <float, duration in seconds from draft shot plan>,
                    "content": "<detailed description of on-screen action and visual staging>",
                    "story_beat": "<specific narrative moment>",
                    "emotion": "<primary emotional tone>",
                    "visuals": "<key visual elements, composition, and camera movement>",
                }
        output_path (str, optional): The file path to save the shot plan. Defaults to .

    Returns:
        str: Success message if video is rendered, or error message if duration doesn't match.
    """
    if output_path:
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write("=== FINAL SHOT PLAN ===\n")
            f.write(answer + "\n")
            f.write("\n")
        return f"Final shot plan saved to {output_path}."
    else:
        return "Error: output_path not provided for saving the shot plan."

def inspect_clip_details(
    time_range: A[str, D("The time range to analyze (e.g., '00:13:28 to 00:13:40'). This tool will analyze the ENTIRE range and provide scene breakdowns within it.")],
    frame_path: A[str, D("The path to the video frames file.")] = "",
) -> str:
    """
    Analyze a video clip time range and return detailed caption.
    
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
You are an expert Video Logger and Editor. Identify meaningful visual beats without fragmenting the footage into micro-moments.

[Task]
Analyze the provided frames and segment the video into coherent clips that an editor can use directly.
1. Identify Boundaries: Only split when there is an actual camera cut or a clear narrative beat change.
2. Describe: For each segment, write a factual, dense visual caption.

[Segmentation Rules]x
- Minimum Duration: Aim for segments that are at least ~3 seconds long. Only create a shorter segment when there is a hard cut or a dramatic visual change that cannot be merged.
- Merge Micro-Actions: If the camera and setting stay consistent, cover the entire action in a single segment even if the subject performs multiple small movements.
- Target Density: Produce roughly 4-8 segments for every 60 seconds of footage. Prioritize editorial usefulness over sheer granularity.

[Output]
Return a JSON object with a list of segments:
{
    "total_analyzed_duration": <float>,
    "segments": [
        {
            "id": "<int, sequential segment number>",
            "timestamp": "<HH:MM:SS to HH:MM:SS>",
            "visual_details": "<Detailed description: clearly state the Subject, their specific Movements, the Camera angle, and the Environment. No flowery language.>",
            "editor_notes": "<Brief note on usability: e.g., 'Good stabilizer shot', 'Contains motion blur', 'Good for reaction cut'.>"
        },
        ...
    ]
}

[Guidelines]
- **Accuracy is Paramount:** If you are unsure if a cut happened, keep the action within the existing segment.
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

def generate_structure_proposal(
    video_scene_path: A[str, D("Path to scene_summaries_video folder containing scene JSON files.")],
    audio_caption_path: A[str, D("Path to captions.json describing the audio segments.")],
    user_instruction: A[str, D("Editing brief provided by the user.")],
) -> str | None:
    """
    Generate a structure proposal for the video editing based on scene summaries.

    Args:
        video_scene_path: Path to scene_summaries_video folder containing scene JSON files
        audio_caption_path: Path to audio caption JSON file
        user_instruction: User's editing instruction

    Returns:
        str: Structure proposal response from LLM
    """
    # Load and concatenate all scene summaries
    video_summary, scene_count = load_scene_summaries(video_scene_path)
    max_scene_index = scene_count - 1 if scene_count > 0 else 0

    if isinstance(audio_caption_path, str):
        with open(audio_caption_path, 'r', encoding='utf-8') as f:
            audio_caption_data = json.load(f)
    else:
        audio_caption_data = audio_caption_path

    # Extract overall analysis summary
    audio_summary = audio_caption_data.get('overall_analysis', {}).get('summary', '')

    # Extract sections without detailed_analysis
    sections = audio_caption_data.get('sections', [])
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
    prompt = prompt.replace("TOTAL_SCENE_COUNT_PLACEHOLDER", str(scene_count))
    prompt = prompt.replace("MAX_SCENE_INDEX_PLACEHOLDER", str(max_scene_index))
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


def check_scene_distribution(
    structure_proposal: dict,
    total_scene_count: int,
    concentration_threshold: float = 0.7
) -> tuple[bool, str]:
    """
    Simplified check - only verify basic structure validity for the new flat format.
    No strict scene distribution or overlap requirements for visual-focused music videos.

    Expected format:
    {
        "overall_theme": "...",
        "narrative_logic": "...",
        "emotion": "...",
        "related_scenes": [list of scene indices]
    }

    Args:
        structure_proposal: Parsed script structure (flat format)
        total_scene_count: Total number of scenes
        concentration_threshold: (unused but kept for compatibility)

    Returns:
        tuple: (Whether the check passed, Feedback message)
    """
    # Check if structure_proposal is valid
    if not structure_proposal or not isinstance(structure_proposal, dict):
        return False, "Invalid structure proposal format."

    # Get related_scenes from the flat structure
    related_scenes = structure_proposal.get('related_scenes', [])

    if not related_scenes:
        return False, "No related_scenes found in proposal."

    # Check scene count (should be 8-15 as per prompt)
    if len(related_scenes) < 8:
        return False, f"Too few scenes selected: {len(related_scenes)}. Need at least 8 scenes."

    if len(related_scenes) > 15:
        print(f"Warning: More than 15 scenes selected ({len(related_scenes)}). This might be too many.")

    # Check that scenes are valid indices
    for scene_id in related_scenes:
        if not isinstance(scene_id, int):
            return False, f"Invalid scene index (not an integer): {scene_id}"

        if scene_id < 0:
            return False, f"Invalid scene index (negative): {scene_id}"

        if scene_id >= total_scene_count:
            return False, f"Scene index {scene_id} exceeds total scene count ({total_scene_count})"

    # All good - this is a visually-focused music video, no strict rules needed
    print(f"[Scene Check] Proposal contains {len(related_scenes)} scenes selected based on visual appeal.")
    print(f"[Scene Check] Selected scenes: {related_scenes}")
    return True, f"Scene selection looks good - {len(related_scenes)} visually appealing scenes selected."


def generate_structure_proposal_with_retry(
    video_scene_path: str,
    audio_caption_path: str,
    user_instruction: str,
    max_retries: int = 2  # Reduced from 5 - less strict checking needed
) -> str | None:
    """
    Structure generation function with basic validation.
    Only retries if parsing fails or basic structure is invalid.
    """
    # Get total scene count from loaded scenes (excluding unusable and low importance scenes)
    _, scene_count = load_scene_summaries(video_scene_path)

    # First generation
    content = generate_structure_proposal(video_scene_path, audio_caption_path, user_instruction)
    if content is None:
        return None

    # Try to parse and validate basic structure
    for retry in range(max_retries):
        try:
            parsed = parse_structure_proposal_output(content)
            if parsed is None:
                print(f"[Retry {retry+1}/{max_retries}] Parsing failed, regenerating...")
                content = generate_structure_proposal(video_scene_path, audio_caption_path, user_instruction)
                continue

            # Basic validation only
            passed, feedback = check_scene_distribution(parsed, scene_count)

            if passed:
                print(f"[Validation] Passed - proposal ready for shot planning")
                return content
            else:
                print(f"[Validation] Failed (Retry {retry+1}/{max_retries})")
                print(f"Feedback: {feedback}")

                if retry < max_retries - 1:
                    # Simple retry without complex feedback
                    content = generate_structure_proposal(video_scene_path, audio_caption_path, user_instruction)
                else:
                    print("[Warning] Maximum retries reached, using current result")
                    return content

        except Exception as e:
            print(f"[Retry {retry+1}/{max_retries}] Processing error: {e}")
            if retry < max_retries - 1:
                content = generate_structure_proposal(video_scene_path, audio_caption_path, user_instruction)
            else:
                return content

    return content


def generate_shot_plan(
    music_detailed_structure: A[list | dict | str, D("Detailed per-segment music analysis for current section.")],
    video_section_proposal: A[dict, D("Section brief extracted from structure proposal.")],
    scene_folder_path: A[str | None, D("Path to scene summaries folder.")] = None,
    user_instruction: A[str, D("User's editing instruction.")] = "",
) -> str | None:
    """
    Generate a one-to-one shot mapping for each music part using GENERATE_MUSIC_SHOT_MAPPING_PROMPT.

    - music_detailed_structure: list/dict or JSON string describing music parts (from captions.json detailed sections)
    - video_section_proposal: dict describing the current video section (from structure proposal)
    - scene_folder_path: path to folder containing scene_*.json files
    """

    # Normalize music structure to JSON string
    if isinstance(music_detailed_structure, (dict, list)):
        music_json = json.dumps(music_detailed_structure, ensure_ascii=False, indent=2)
    else:
        music_json = str(music_detailed_structure or '')

    prompt = GENERATE_SHOT_PLAN_PROMPT
    prompt = prompt.replace("AUDIO_SUMMARY_PLACEHOLDER", music_json)
    prompt = prompt.replace("VIDEO_SECTION_INFO_PLACEHOLDER", str(video_section_proposal))
    prompt = prompt.replace("INSTRUCTION_PLACEHOLDER", user_instruction)

    # Load related scenes based on related_scenes indices from video_section_proposal
    related_video_context = ""
    related_scenes = video_section_proposal.get("related_scenes", []) if isinstance(video_section_proposal, dict) else []
    if related_scenes and scene_folder_path:
        scene_descriptions = []
        for scene_idx in related_scenes:
            scene_file = os.path.join(scene_folder_path, f"scene_{scene_idx}.json")
            if os.path.exists(scene_file):
                try:
                    with open(scene_file, 'r', encoding='utf-8') as f:
                        scene_data = json.load(f)
                    video_analysis = scene_data.get('video_analysis', {})
                    scene_caption = video_analysis.get('scene_caption', {})
                    scene_summary = scene_caption.get('scene_summary', '')
                    if scene_summary:
                        scene_descriptions.append(f"Scene {scene_idx}: {scene_summary}")
                except Exception:
                    pass
        related_video_context = "\n".join(scene_descriptions)

    prompt = prompt.replace("RELATED_VIDEO_PLACEHOLDER", related_video_context)


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


class Screenwriter:
    def __init__(self, video_scene_path, audio_caption_path, output_path, max_iterations, video_path=None, frame_folder_path=None):
        self.tools = [
            inspect_clip_details,
            finish,
        ]
        self.name_to_function_map = {tool.__name__: tool for tool in self.tools}
        self.function_schemas = [
            {"function": as_json_schema(func), "type": "function"}
            for func in self.name_to_function_map.values()
        ]
        self.video_scene_path = video_scene_path
        self.audio_caption_path = audio_caption_path
        self.audio_db = json.load(open(audio_caption_path, 'r', encoding='utf-8'))
        self.max_iterations = max_iterations
        self.frame_folder_path = frame_folder_path
        self.video_path = video_path
        self.output_path = output_path
        self.current_target_length = None  # Will be set during run()
        self.messages = self._construct_messages()
        # Note: no trim overlap or redundancy restrictions; tracking removed per user request

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

    def _construct_messages(self):
        messages = [
            {
            "role": "system",
            "content": """You are a professional video editing assistant that edits videos by sequentially invoking tools. Follow the THINK → ACT → OBSERVE loop:
      • THOUGHT: Reason step-by-step about which function to call next and why.
      • ACTION: Call exactly one tool that helps you gather more information about the video editing task.
      • OBSERVATION: Summarize and analyze the tool's output.
    You MUST plan extensively before each tool call and reflect thoroughly on the outcomes of previous tool calls.
    Only pass arguments that come directly from the user or from earlier tool outputs—never fabricate them. Continue the loop until the user's request is fully satisfied, then conclude with the final answer. If you are uncertain about the video content or structure, use the available tools to inspect rather than guessing. Plan carefully before each call and reflect on every result. Do not rely solely on blind tool calls, as this degrades reasoning quality. Timestamps may be formatted as 'HH:MM:SS' or 'MM:SS'."""
            },
            {
            "role": "user",
            "content": \
"""
[Role]
You are an expert Video Conform Editor. Your goal is to convert a "Draft Shot Plan" into a strict, executable "Final Shot Plan" using ONLY available video assets.

[Task]
Validate and finalize the draft plan against the video database. Real footage is the only truth. If a drafted scene is unavailable, replace it with the most relevant existing clip or discard it. NEVER invent filenames or timestamps.

[Tools]
• `inspect_clip_details`: Verify and enrich clip details. Returns detailed captions.
• `finish`: Submit the finalized plan.

[Workflow]
1. Analyze: Break down the `Draft shot plan` into visual beats.
2. Search & Match (Iterative):
    For each shot:
    a. Query: Create a focused visual query based on the draft beat.
    c. Verify:
        - Strong Match: Use `inspect_clip_details` if needed, then lock `clip_id` and `time_range`.
        - Partial Match: Prioritize real footage. Revise the shot description to match the actual clip.
        - No Match: Broaden query. If still failing, drop or redesign the beat based on available footage.
3. Finalize: Ensure logical flow and rhythm alignment.
4. Output: Call `finish`.

[Input Brief]
- Draft shot plan: SHOT_PLAN_PLACEHOLDER
- Background music: BACKGROUND_MUSIC_PLACEHOLDER

[Constraints]
- The "Final Shot Plan" must be grounded in actual footage details (use `inspect_clip_details` when uncertain).
- Adhere to the draft's narrative and emotional arc, but strictly ground every detail in actual footage.
- Develop the narrative by expanding on emotions and plot points. Do NOT output a single isolated shot; instead, construct a coherent sequence of multiple shots that align with the music and tell a story.
"""
            },
        ]

        return messages


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

        # For inspect_clip_details, inject frame_folder parameter
        if name == "inspect_clip_details":
            if self.frame_folder_path:
                # args["frame_path"] = self.frame_folder_path
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
            args["output_path"] = self.output_path or ""
        # Call the tool
        try:
            print(f"Calling function `{name}` with args: {args}")
            result = self.name_to_function_map[name](**args)
            print("Result: ", result)
            self._append_tool_msg(tool_call["id"], name, result, msgs)
            # Record successful trim ranges
            if name == "inspect_clip_details":
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
            if name == "finish" and result.startswith("Final shot plan saved to"):
                print(f"Shot completed successfully: {result}")
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

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self, instruction) -> list[dict]:
        """
        Run the ReAct-style loop with OpenAI Function Calling.
        """
        structure_proposal = generate_structure_proposal_with_retry(self.video_scene_path, self.audio_caption_path, instruction)
        structure_proposal = parse_structure_proposal_output(structure_proposal)
        pprint(structure_proposal, width=150)

        # Check if this is a short video (single audio section)
        audio_sections = self.audio_db.get('sections', [])

        print("\n" + "="*60)
        print("DETECTED SHORT VIDEO MODE")
        print("="*60)
        print(f"Single audio section found: {audio_sections[0].get('name', 'Unknown')}")
        print(f"Time range: {audio_sections[0].get('Start_Time', '0')} - {audio_sections[0].get('End_Time', '0')}")
        print("Skipping section structure generation, directly creating shot plan...")
        print("="*60 + "\n")

        # For short videos, directly create shot plan without section subdivision
        audio_section = audio_sections[0]

        # Calculate duration
        def time_str_to_sec(t):
            if isinstance(t, (int, float)):
                return float(t)
            parts = str(t).split(':')
            if len(parts) == 3:
                h, m, s = [float(x) for x in parts]
                return h * 3600 + m * 60 + s
            elif len(parts) == 2:
                m, s = [float(x) for x in parts]
                return m * 60 + s
            else:
                try:
                    return float(parts[0])
                except ValueError:
                    return 0

        start_time = time_str_to_sec(audio_section.get('Start_Time', 0))
        end_time = time_str_to_sec(audio_section.get('End_Time', 0))
        duration = end_time - start_time



        print(f"Target duration: {duration:.1f} seconds")

        # Generate shot plan directly
        shot_plan = generate_shot_plan(
            audio_section.get('detailed_analysis', {}).get('sections', []),
            structure_proposal,
            self.video_scene_path,
            instruction
        )
        shot_plan = parse_shot_plan_output(shot_plan)
        pprint(shot_plan, width=150)

        # Create output data with single section
        import datetime
        output_data = {
            "instruction": instruction,  # Add explicit instruction field
            "metadata": {
                "created_at": datetime.datetime.now().isoformat(),
                "video_path": self.video_path if hasattr(self, 'video_path') else None,
                "audio_path": self.audio_caption_path,
                "video_scene_path": self.video_scene_path,
            },
            "overall_theme": f"Short video for {audio_section.get('name', 'audio section')}",
            "narrative_logic": instruction,
            "video_structure": [{
                **structure_proposal,
                "start_time": start_time,
                "end_time": end_time,
                "shot_plan": shot_plan
            }]
        }

        print("\nShort video shot plan generated successfully!")


        # Shot processed
        print(f"\nShot processed.")

        # Save complete output as valid JSON file
        if self.output_path:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"\nComplete shot plan saved to {self.output_path}")

        return output_data



def main():
    def _norm_name(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

    def _resolve_video_assets(
        video_path: str,
        video_scene_path: str | None,
        frame_folder_path: str | None,
        video_caption_path: str | None,
    ) -> tuple[str, str | None, str | None]:
        """Resolve video-related database paths from a raw video path.

        Returns:
            (resolved_video_scene_path, resolved_frame_folder_path, resolved_video_caption_path)
        """
        if video_scene_path and frame_folder_path and video_caption_path:
            return video_scene_path, frame_folder_path, video_caption_path

        if not video_path:
            raise ValueError("--video-path is required")

        repo_root = Path(__file__).resolve().parents[1]
        video_db_root = repo_root / "video_database" / "Video"
        if not video_db_root.exists():
            raise FileNotFoundError(
                f"Cannot find video database root at: {video_db_root}. "
                "Run from the repo workspace or pass --video-scene-path manually."
            )

        stem = Path(video_path).stem
        target_norm = _norm_name(stem)

        # Find best matching folder under video_database/Video
        match_dir: Path | None = None
        if (video_db_root / stem).is_dir():
            match_dir = video_db_root / stem
        else:
            for child in video_db_root.iterdir():
                if not child.is_dir():
                    continue
                if _norm_name(child.name) == target_norm:
                    match_dir = child
                    break

        if match_dir is None:
            raise FileNotFoundError(
                f"Cannot resolve video database folder for video '{stem}'. "
                f"Expected something like: {video_db_root}/<VIDEO_NAME>/ . "
                "Please pass --video-scene-path (and optionally --frame-folder-path)."
            )

        # Resolve paths with simple, existing conventions
        captions_dir = match_dir / "captions"
        candidate_scene_dirs = [
            captions_dir / "scene_summaries_video",
            captions_dir / "scene_summaries",
        ]
        resolved_scene_dir: Path | None = None
        for cand in candidate_scene_dirs:
            if cand.is_dir():
                resolved_scene_dir = cand
                break

        if video_scene_path is None:
            if resolved_scene_dir is None:
                raise FileNotFoundError(
                    f"Cannot find scene summaries folder under: {captions_dir}. "
                    "Tried: scene_summaries_video/, scene_summaries/. "
                    "Please pass --video-scene-path manually."
                )
            video_scene_path = str(resolved_scene_dir)

        if frame_folder_path is None:
            frames_dir = match_dir / "frames"
            frame_folder_path = str(frames_dir) if frames_dir.is_dir() else None

        if video_caption_path is None:
            captions_json = captions_dir / "captions.json"
            video_caption_path = str(captions_json) if captions_json.is_file() else None

        return video_scene_path, frame_folder_path, video_caption_path

    parser = argparse.ArgumentParser(
        description="Generate a short-video shot plan from video scene summaries and short audio captions."
    )
    parser.add_argument(
        "--video-scene-path",
        default=None,
        help="Path to scene summaries folder containing scene_*.json files. If omitted, inferred from --video-path.",
    )
    parser.add_argument(
        "--audio-caption-path",
        default="/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Audio/Way_Down_We_Go/captions_max4s/captions_short.json",
        help="Path to captions_short.json describing the audio segments.",
    )
    parser.add_argument(
        "--frame-folder-path",
        default=None,
        help="Path to extracted frames folder. If omitted, inferred from --video-path when possible.",
    )
    parser.add_argument(
        "--video-path",
        required=True,
        help="Path to the source video file (required).",
    )
    parser.add_argument(
        "--output-path",
        default="/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/output_dark_knight_way_down_we_go_short/shot_plan_gemini_joker.json",
        help="Output path to save the generated shot plan JSON.",
    )
    parser.add_argument(
        "--instruction",
        default="A montage that captures the crazy and chaotic of Joker's character.",
        help="User instruction / creative brief.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Maximum iterations for the agent loop (kept for compatibility).",
    )
    args = parser.parse_args()

    resolved_video_scene_path, resolved_frame_folder_path, _resolved_video_caption_path = _resolve_video_assets(
        video_path=args.video_path,
        video_scene_path=args.video_scene_path,
        frame_folder_path=args.frame_folder_path,
        video_caption_path=None,
    )

    agent = Screenwriter(
        video_scene_path=resolved_video_scene_path,
        audio_caption_path=args.audio_caption_path,
        output_path=args.output_path,
        max_iterations=args.max_iterations,
        video_path=args.video_path,
        frame_folder_path=resolved_frame_folder_path,
    )
    # print("Generating structure proposal...")
    # structure_proposal = generate_structure_proposal(video_summary_path, audio_caption_path, Instruction)
    # structure_proposal = parse_structure_proposal_output(structure_proposal)
    # print("Structure proposal: ", structure_proposal)
    messages = agent.run(args.instruction)

    # if messages:
    #     for m in messages:
    #         import pdb; pdb.set_trace()
    #         if m.get('role') == 'assistant':
    #             print(m)


if __name__ == "__main__":
    main()

