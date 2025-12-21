import os
import json
import copy
import re
import numpy as np
from typing import Annotated as A, Any
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
from vca.core import (
    parse_structure_proposal_output,
    parse_shot_plan_output
)
from pprint import pprint



GENERATE_STRUCTURE_PROPOSAL_PROMPT = """
VIDEO_SUMMARY_PLACEHOLDER

You are a professional screenwriter. Design a minutes-long video script structure based on the provided footage and background music.

**PRIORITY ORDER:**
1. **User Instruction** — All story decisions must serve this goal.
2. **Video Content** — Only use characters/events explicitly present.
3. **Audio Arc** — Determines pacing and intensity.

**SCENE RULES:**

1. **Full Coverage**: Use scenes from Scene 0 to MAX_SCENE_INDEX_PLACEHOLDER (total: TOTAL_SCENE_COUNT_PLACEHOLDER). Must cover early, middle, and late scenes.

2. **Representative Selection**: Pick 3-5 representative scenes per section (not exhaustive lists).
   - Prioritize: action, conflict, emotional climaxes, visually striking moments
   - Avoid: mundane activities, slow transitions, minor backstory without visual impact

3. **NO SCENE OVERLAP BETWEEN SECTIONS** (CRITICAL):
   - Each scene index can ONLY appear in ONE section's related_scenes.
   - If Section A uses Scene 15, Section B and all other sections CANNOT use Scene 15.
   - Plan the scene allocation across all sections BEFORE writing the output to ensure no duplicates.
   - This rule has NO exceptions.

4. **Short-Form Video Pacing** (CRITICAL):
   - This is a minutes-long short video, NOT a full movie edit. Every second counts.
   - **Intro must hook immediately**: Start with impactful, core-narrative scenes. NO slow childhood flashbacks, quiet dialogue-only moments, or peripheral backstory unless directly tied to the climax.
   - **Cut non-essential setup**: If a scene only provides context but lacks visual/emotional punch, SKIP IT.
   - **Prioritize payoff over buildup**: In short videos, audiences expect quick engagement. Favor scenes with clear conflict, stakes, or visual spectacle over gradual world-building.

5. **Scene Distribution by Audio Phase** (do NOT cluster in early scenes):
   - Intro/Verse → Early scenes (setup)
   - Build-up → Middle scenes (development)
   - Chorus/Climax → Later scenes (conflict, payoff)
   - Outro → Final scenes (closure)

6. **No Hallucination**: Only use content explicitly described in input materials.

**INPUT DATA:**
- Audio Summary: AUDIO_SUMMARY_PLACEHOLDER
- Audio Structure: AUDIO_STRUCTURE_PLACEHOLDER
- User Instruction: INSTRUCTION_PLACEHOLDER

**OUTPUT (JSON):**
{
    "overall_theme": "Theme reflecting the User Instruction",
    "narrative_logic": "How audio arc maps to beginning/middle/end of scene range",
    "video_structure": [
        {
            "content": "Concrete scene description: actions, characters, visuals. Detailed and specific.",
            "audio_section": "e.g., 'Chorus 1'",
            "emotion": "Dominant tone",
            "start_time": "seconds",
            "end_time": "seconds",
            "related_scenes": [5 representative scene indices]
        }
    ]
}

"""



GENERATE_SHOT_PLAN_PROMPT = """
RELATED_VIDEO_PLACEHOLDER

[Role]
You are a professional screenwriter tasked with creating a detailed shot plan that synchronizes the narrative storyline with the background music. This shot plan will serve as a part of full script for video editing.

[Task]
Map each analyzed music segment to a corresponding shot based EXCLUSIVELY on the provided scene information and proposal content. The resulting sequence must form a coherent narrative arc that aligns with the music's rhythm while using ONLY existing visual content from the scenes.

[Inputs]
- Detailed analysis of each music segment: AUDIO_SUMMARY_PLACEHOLDER
- Current video section narrative (from proposal): VIDEO_SECTION_INFO_PLACEHOLDER
- Related video scenes: Provided above with their visual descriptions

[Strict Content Constraints]
**CRITICAL: You must ONLY use content that explicitly appears in the provided inputs. DO NOT fabricate, imagine, or invent any visual elements, actions, characters, objects, or narrative events that are not directly described in the scene information or proposal.**

- Every visual element mentioned in "content" and "visuals" fields MUST come from the related_scenes descriptions
- Every narrative beat in "story_beat" MUST align with the proposal's section narrative
- If a music segment requires content not available in the scenes, describe it using the closest matching available scene content
- Never add characters, locations, objects, or events not present in the source materials

[Workflow]
1. Carefully read and understand ALL provided scene descriptions and the proposal narrative.
2. For each music segment (in chronological order), select the most appropriate scene content that matches both the narrative requirement and music mood.
3. Design each shot using ONLY the visual elements explicitly described in the selected scene - do not embellish or add imaginary details.
4. Ensure the duration matches the music segment's length (minimum 3.0s).
5. Verify that every shot description can be directly traced back to the provided scene information.

[Guidelines]
- Maintain strict one-to-one correspondence between music segments and shots; no combining or subdividing.
- Keep shot durations realistic and proportional to each music segment.
- **Every shot must reference a specific related_scene index, and the content must accurately reflect that scene's description.**
- If the proposal mentions a narrative element, use only scenes that actually contain visual content supporting that element.
- Do not create transitions, actions, or visual details beyond what the scene descriptions explicitly provide.

[Output]
Return STRICT JSON ONLY with this schema:
{
    "shots": [
        {
            "id": <int, matching the segment id in music segment>,
            "time_duration": <float, duration in seconds>,
            "content": "<detailed description of on-screen action and visual staging>",
            "story_beat": "<specific narrative moment>",
            "emotion": "<primary emotional tone>",
            "visuals": "<key visual elements, composition, and camera movement>",
            "related_scene": "<int, one the most relevant scene index from related_scenes>"
        },
        ...
    ]
}
"""

def load_scene_summaries(scene_folder_path: str) -> str:
    """
    Load all scene_caption.scene_summary from the scene_summaries_video folder
    and concatenate them into a complete video material description.

    Args:
        scene_folder_path: Path to the scene summaries folder

    Returns:
        str: Concatenated scene summaries
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

    return full_summary


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

def search_video_library(
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
    # Get the embedding data for the clip
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

[Segmentation Rules]
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
    video_summary = load_scene_summaries(video_scene_path)

    # Count total number of scenes for coverage guidance
    scene_count = 0
    for filename in os.listdir(video_scene_path):
        if filename.startswith('scene_') and filename.endswith('.json'):
            scene_count += 1
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
    Check whether the scene distribution in the generated script structure is reasonable.

    Args:
        structure_proposal: Parsed script structure
        total_scene_count: Total number of scenes
        concentration_threshold: Concentration threshold, exceeding this ratio is considered too concentrated

    Returns:
        tuple: (Whether the check passed, Feedback message)
    """
    video_structure = structure_proposal.get('video_structure', [])
    if not video_structure:
        return False, "No video_structure found in proposal."

    # Check for scene overlap between sections
    section_scenes = []  # List of (section_index, set of scenes)
    overlapping_scenes = {}  # scene_index -> list of section indices that use it

    for sec_idx, section in enumerate(video_structure):
        related_scenes = section.get('related_scenes', [])
        scene_set = set(related_scenes)
        section_scenes.append((sec_idx, scene_set))

        # Track which sections use each scene
        for scene in related_scenes:
            if scene not in overlapping_scenes:
                overlapping_scenes[scene] = []
            overlapping_scenes[scene].append(sec_idx)

    # Find scenes that appear in multiple sections
    duplicated_scenes = {
        scene: sections
        for scene, sections in overlapping_scenes.items()
        if len(sections) > 1
    }

    # Collect all scene indices
    all_scenes = []
    for section in video_structure:
        related_scenes = section.get('related_scenes', [])
        all_scenes.extend(related_scenes)

    if not all_scenes:
        return False, "No related_scenes found in any section."

    # Calculate scene distribution
    unique_scenes = set(all_scenes)
    max_scene_idx = max(unique_scenes) if unique_scenes else 0
    min_scene_idx = min(unique_scenes) if unique_scenes else 0

    # Check 1: Whether scenes cover a sufficient range
    coverage_range = max_scene_idx - min_scene_idx
    expected_range = total_scene_count * 0.5  # Expect to cover at least 50% of the scene range

    # Check 2: Whether scenes are too concentrated in the first half
    mid_point = total_scene_count // 2
    scenes_in_first_half = sum(1 for s in unique_scenes if s < mid_point)
    scenes_in_second_half = sum(1 for s in unique_scenes if s >= mid_point)

    # Check 3: Whether scenes are concentrated within a range of 30 scenes
    # Use sliding window check
    window_size = min(30, total_scene_count // 2)
    max_concentration = 0
    concentrated_range = (0, 0)

    for start in range(0, total_scene_count - window_size + 1):
        end = start + window_size
        scenes_in_window = sum(1 for s in unique_scenes if start <= s < end)
        concentration = scenes_in_window / len(unique_scenes) if unique_scenes else 0
        if concentration > max_concentration:
            max_concentration = concentration
            concentrated_range = (start, end)

    # Generate feedback
    issues = []

    # Check 0: Scene overlap between sections (HIGHEST PRIORITY)
    if duplicated_scenes:
        overlap_details = []
        for scene, sections in sorted(duplicated_scenes.items()):
            section_names = [f"Section {s+1}" for s in sections]
            overlap_details.append(f"Scene {scene} appears in {', '.join(section_names)}")
        issues.append(f"CRITICAL - Scene overlap detected! Each scene can only appear in ONE section. Conflicts:\n    " + "\n    ".join(overlap_details))

    if coverage_range < expected_range:
        issues.append(f"Insufficient scene coverage: Currently covering Scene {min_scene_idx} to Scene {max_scene_idx} (range={coverage_range}), but there are {total_scene_count} scenes in total, expecting to cover at least {int(expected_range)} scene range.")

    if scenes_in_second_half == 0:
        issues.append(f"Second half scenes completely unused! Scenes {mid_point} to {total_scene_count-1} were not selected, please ensure coverage of key plot points in the middle and later portions.")
    elif scenes_in_first_half > scenes_in_second_half * 2:
        issues.append(f"Unbalanced scene distribution: First half ({scenes_in_first_half} scenes) far exceeds second half ({scenes_in_second_half} scenes), please increase usage of later scenes.")

    if max_concentration > concentration_threshold:
        issues.append(f"Scenes too concentrated: {int(max_concentration*100)}% of scenes are concentrated in Scene {concentrated_range[0]} to Scene {concentrated_range[1]} range, please diversify scene selection.")

    if issues:
        feedback = "Scene distribution check failed with the following issues:\n" + "\n".join(f"- {issue}" for issue in issues)
        return False, feedback

    return True, "Scene distribution check passed"


def generate_structure_proposal_with_retry(
    video_scene_path: str,
    audio_caption_path: str,
    user_instruction: str,
    max_retries: int = 5
) -> str | None:
    """
    Structure generation function with scene distribution check and retry mechanism.

    If the generated structure has unreasonable scene distribution, it will provide feedback and regenerate.
    """
    # Get total scene count
    scene_count = 0
    for filename in os.listdir(video_scene_path):
        if filename.startswith('scene_') and filename.endswith('.json'):
            scene_count += 1

    # First generation
    content = generate_structure_proposal(video_scene_path, audio_caption_path, user_instruction)
    if content is None:
        return None

    # Try to parse and check
    for retry in range(max_retries):
        try:
            parsed = parse_structure_proposal_output(content)
            if parsed is None:
                print(f"[Retry {retry+1}/{max_retries}] Parsing failed, regenerating...")
                content = generate_structure_proposal(video_scene_path, audio_caption_path, user_instruction)
                continue

            passed, feedback = check_scene_distribution(parsed, scene_count)

            if passed:
                print(f"[Scene Distribution Check] Passed")
                return content
            else:
                print(f"[Scene Distribution Check] Failed (Retry {retry+1}/{max_retries})")
                print(f"Feedback: {feedback}")

                if retry < max_retries - 1:
                    # Construct regeneration request with feedback
                    content = _regenerate_with_feedback(
                        video_scene_path, audio_caption_path, user_instruction,
                        content, feedback, scene_count
                    )
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


def _regenerate_with_feedback(
    video_scene_path: str,
    audio_caption_path: str,
    user_instruction: str,
    previous_output: str,
    feedback: str,
    scene_count: int
) -> str | None:
    """
    Regenerate structure proposal based on feedback.
    """
    video_summary = load_scene_summaries(video_scene_path)
    max_scene_index = scene_count - 1 if scene_count > 0 else 0

    if isinstance(audio_caption_path, str):
        with open(audio_caption_path, 'r', encoding='utf-8') as f:
            audio_caption_data = json.load(f)
    else:
        audio_caption_data = audio_caption_path

    audio_summary = audio_caption_data.get('overall_analysis', {}).get('summary', '')
    sections = audio_caption_data.get('sections', [])
    filtered_sections = []
    for section in sections:
        section_copy = {
            'name': section.get('name', ''),
            'description': section.get('description', ''),
            'Start_Time': section.get('Start_Time', ''),
            'End_Time': section.get('End_Time', '')
        }
        filtered_sections.append(section_copy)
    audio_structure = json.dumps(filtered_sections, indent=2, ensure_ascii=False)

    # 构造带反馈的 prompt
    prompt = GENERATE_STRUCTURE_PROPOSAL_PROMPT
    prompt = prompt.replace("TOTAL_SCENE_COUNT_PLACEHOLDER", str(scene_count))
    prompt = prompt.replace("MAX_SCENE_INDEX_PLACEHOLDER", str(max_scene_index))
    prompt = prompt.replace("VIDEO_SUMMARY_PLACEHOLDER", video_summary)
    prompt = prompt.replace("AUDIO_SUMMARY_PLACEHOLDER", audio_summary)
    prompt = prompt.replace("AUDIO_STRUCTURE_PLACEHOLDER", audio_structure)
    prompt = prompt.replace("INSTRUCTION_PLACEHOLDER", user_instruction)

    retry_prompt = f"""
{prompt}

---

**IMPORTANT: Your previous attempt was rejected due to scene distribution issues.**

**Feedback on your previous attempt:**
{feedback}

**You MUST fix these issues in your new response:**
1. **NO SCENE OVERLAP**: Each scene index can ONLY appear in ONE section's related_scenes. If Scene X is used in Section A, it CANNOT be used in any other section. This is the HIGHEST priority rule.
2. Ensure scenes are distributed across the ENTIRE range (Scene 0 to Scene {max_scene_index})
3. Include scenes from the LATER portions of the footage (Scene {scene_count//2} onwards)
4. Do NOT cluster most scenes in one small range
5. Each audio section should draw from its corresponding part of the timeline

**Before generating output, mentally verify:**
- List all scenes you plan to use for each section
- Check that NO scene appears in more than one section
- If you find any overlap, reassign scenes to different sections

Generate a corrected JSON response now:
"""

    messages = [{"role": "user", "content": retry_prompt}]

    response = call_vllm_model(
        messages,
        endpoint=config.VLLM_AGENT_ENDPOINT,
        model_name=config.AGENT_MODEL,
        temperature=0.3,  # Slightly increase temperature to get different results
        max_tokens=config.AGENT_MODEL_MAX_TOKEN,
    )

    if response is None:
        return None

    return response.get('content', '')


def generate_shot_plan(
    music_detailed_structure: A[list | dict | str, D("Detailed per-segment music analysis for current section.")],
    video_section_proposal: A[dict, D("Section brief extracted from structure proposal.")],
    scene_folder_path: A[str | None, D("Path to scene summaries folder.")] = None,
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
    def __init__(self, video_db_path, video_caption_path, video_scene_path, audio_caption_path, output_path, max_iterations, video_path=None, frame_folder_path=None):
        self.tools = [
            search_video_library,
            inspect_clip_details,
            finish,
        ]
        self.name_to_function_map = {tool.__name__: tool for tool in self.tools}
        self.function_schemas = [
            {"function": as_json_schema(func), "type": "function"}
            for func in self.name_to_function_map.values()
        ]
        self.video_db = init_single_video_db(video_caption_path, video_db_path, config.AOAI_EMBEDDING_LARGE_DIM)
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
• `search_video_library`: Semantic search for clips. Returns `clip_id`, `time_range`, and `visual_summary`.
• `inspect_clip_details`: Verify and enrich clip details. Returns detailed captions.
• `finish`: Submit the finalized plan.

[Workflow]
1. Analyze: Break down the `Draft shot plan` into visual beats.
2. Search & Match (Iterative):
    For each shot:
    a. Query: Create a focused visual query based on the draft beat.
    b. Search: Call `search_video_library`.
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
- The "Final Shot Plan" MUST include valid `clip_id` and `exact_timestamps` for every shot.
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

        # Inject system-provided parameters
        if "database" in args or name == "search_video_library":
            args["database"] = self.video_db
        
        if "topk" in args or "top_k" in args:
            key = "top_k" if "top_k" in args else "topk"
            if config.OVERWRITE_CLIP_SEARCH_TOPK > 0:
                args[key] = config.OVERWRITE_CLIP_SEARCH_TOPK
        
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

        # Use version with scene distribution check, will automatically retry if distribution is unreasonable
        structure_proposal = generate_structure_proposal_with_retry(self.video_scene_path, self.audio_caption_path, instruction)
        structure_proposal = parse_structure_proposal_output(structure_proposal)
        overall_theme = structure_proposal['overall_theme']
        narrative_logic = structure_proposal['narrative_logic']

        # TODO: Run generate_structure_proposal again for each section based on detailed section content

        # Initialize output data structure for valid JSON format
        # Each section in video_structure will contain its corresponding shot_plan
        output_data = {
            "overall_theme": structure_proposal.get('overall_theme', ''),
            "narrative_logic": structure_proposal.get('narrative_logic', ''),
            "video_structure": []
        }

        pprint(structure_proposal, width=150)

        for sec_idx, sec_cur in enumerate(structure_proposal['video_structure']):
            print(f"\n{'='*60}")
            print(f"Processing Section {sec_idx + 1}/{len(structure_proposal['video_structure'])}")
            print(f"{'='*60}\n")
            
            # Calculate video section length (in seconds)
            start_time = sec_cur.get('start_time', 0)
            end_time = sec_cur.get('end_time', 0)
            def time_str_to_sec(t):
                # Handle if already a number (float or int)
                if isinstance(t, (int, float)):
                    return float(t)
                # Handle string format (HH:MM:SS or MM:SS)
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
            length_sec = abs(time_str_to_sec(end_time) - time_str_to_sec(start_time))
            
            # Set current target length for finish function validation
            
            print(f"Section {sec_idx + 1} output path: {self.output_path}")
            print(f"Target duration: {length_sec} seconds")
            print(f"Content: {sec_cur.get('content', 'N/A')}")
            print(f"Emotion: {sec_cur.get('emotion', 'N/A')}\n")
            
            

            shot_plan = generate_shot_plan(
                self.audio_db['sections'][sec_idx].get('detailed_analysis', '')['sections'],
                sec_cur,
                self.video_scene_path,
            )
            shot_plan = parse_shot_plan_output(shot_plan)
            pprint(shot_plan, width=150)

            # Combine section info with its shot_plan
            section_with_shots = {
                **sec_cur,  # Include all original section fields
                "shot_plan": shot_plan
            }
            output_data["video_structure"].append(section_with_shots)
            # for idx, shot in enumerate(shot_plan['shots']):
            #     msgs = copy.deepcopy(self.messages)
            #     print(f"\n{'='*60}")
            #     print(f"Processing Shot {idx + 1}/{len(shot_plan['shots'])}")
            #     print(f"{'='*60}\n")
            #     print(f"Shot output path: {self.output_path}")
            #     msgs[-1]["content"] = msgs[-1]["content"].replace("SHOT_PLAN_PLACEHOLDER", json.dumps(shot, indent=2, ensure_ascii=False))
            #     audio_section_info = str({k: v for k, v in (json.loads(self.audio_db['sections'][sec_idx].get('detailed_analysis', '')['sections'][idx]) if isinstance(self.audio_db['sections'][sec_idx].get('detailed_analysis', '')['sections'][idx], str) else self.audio_db['sections'][sec_idx].get('detailed_analysis', '')['sections'][idx]).items()})
            #     msgs[-1]["content"] = msgs[-1]["content"].replace("BACKGROUND_MUSIC_PLACEHOLDER", audio_section_info)
            #     self.current_target_length = shot['time_duration']

            #     for i in range(self.max_iterations):
            #         # msgs[-1]["content"] = msgs[-1]["content"].replace("VIDEO_THEME_PLACEHOLDER", overall_theme).replace("VIDEO_NARRATIVE_LOGIC_PLACEHOLDER", narrative_logic)
            #         if i == self.max_iterations - 1:
            #             msgs.append(
            #                 {
            #                     "role": "user",
            #                     "content": "Please call the `finish` function to finish the task.",
            #                 }
            #             )

            #         # Retry loop for both model call and tool execution
            #         # If tool execution fails, we rollback and retry the entire model call
            #         max_model_retries = 2  # Retry model call if it returns None
            #         max_tool_retries = 2   # Retry entire iteration if tool execution fails
            #         tool_execution_success = False
                    
            #         for tool_retry in range(max_tool_retries):
            #             # Save snapshot of msgs before making any changes
            #             msgs_snapshot = copy.deepcopy(msgs)
                        
            #             # Call model with retry mechanism
            #             response = None
            #             for model_retry in range(max_model_retries):
            #                 try:
            #                     response = call_vllm_model(
            #                         msgs,
            #                         endpoint=config.VLLM_AGENT_ENDPOINT,
            #                         model_name=config.AGENT_MODEL,
            #                         temperature=0.0,
            #                         max_tokens=config.AGENT_MODEL_MAX_TOKEN,
            #                         tools=self.function_schemas,
            #                         tool_choice="auto",
            #                         return_json=False,
            #                     )
            #                     if response is not None:
            #                         break  # Success, exit model retry loop
            #                     else:
            #                         print(f"⚠️  Model returned None, retrying model call ({model_retry + 1}/{max_model_retries})...")
            #                 except Exception as e:
            #                     print(f"⚠️  Model call failed with error: {e}, retrying ({model_retry + 1}/{max_model_retries})...")
            #                     if model_retry == max_model_retries - 1:
            #                         raise
                        
            #             # If all model retries failed, skip this iteration entirely
            #             if response is None:
            #                 print(f"❌ Model call failed after {max_model_retries} retries. Skipping iteration {i}.")
            #                 # Restore original msgs and remove finish prompt if added
            #                 msgs[:] = msgs_snapshot
            #                 if i == self.max_iterations - 1 and msgs and msgs[-1].get("content") == "Please call the `finish` function to finish the task.":
            #                     msgs.pop()
            #                 break  # Exit tool retry loop
                        
            #             # Add response to msgs
            #             response.setdefault("role", "assistant")
            #             # Clean up extra newlines in content to prevent accumulation
            #             if response.get("content"):
            #                 response["content"] = response["content"].rstrip('\n') + '\n'
            #             msgs.append(response)
            #             print("#### Iteration: ", i, f"(Tool retry: {tool_retry + 1}/{max_tool_retries})" if tool_retry > 0 else "")
            #             pprint(response, width=150)
                        
            #             # Execute any requested tool calls
            #             section_completed = False
            #             tool_execution_failed = False
                        
            #             try:
            #                 tool_calls = response.get("tool_calls", [])
            #                 if tool_calls is None:
            #                     print("⚠️  Warning: tool_calls is None, treating as empty list")
            #                     tool_calls = []
                            
            #                 for tool_call in tool_calls:
            #                     is_finished = self._exec_tool(tool_call, msgs)
            #                     if is_finished: 
            #                         section_completed = True
            #                         break
                            
            #                 # If we reach here, tool execution was successful
            #                 tool_execution_success = True
                            
            #             except StopException:
            #                 return msgs
            #             except Exception as e:
            #                 print(f"❌ Error executing tool calls: {e}")
            #                 import traceback
            #                 traceback.print_exc()
            #                 tool_execution_failed = True
                        
            #             # If tool execution succeeded or we're on the last retry, exit retry loop
            #             if tool_execution_success or tool_retry == max_tool_retries - 1:
            #                 if tool_execution_failed and tool_retry == max_tool_retries - 1:
            #                     print(f"❌ Tool execution failed after {max_tool_retries} retries. Moving to next iteration.")
            #                 break
                        
            #             # Tool execution failed, rollback and retry
            #             if tool_execution_failed:
            #                 print(f"🔄 Rolling back messages and retrying model call ({tool_retry + 1}/{max_tool_retries})...")
            #                 msgs[:] = msgs_snapshot  # Rollback to snapshot
            #                 continue  # Retry the model call
                    
            #         # If model call failed completely, skip to next iteration
            #         if response is None:
            #             continue
                    
            #         # If section is completed successfully, move to next section
            #         if section_completed:
            #             print(f"Shot {idx + 1} completed.")
            #             break

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
    video_caption_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Database/Batman_Begins_2005_1080p_BluRay_x264_YIFY/captions/captions.json"
    video_scene_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Database/Batman_Begins_2005_1080p_BluRay_x264_YIFY/captions/scene_summaries_video"
    # audio_caption_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Database/Luv(sic)Pt2/captions/captions.json"
    audio_caption_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Database/Way_Down_We_Go-Kaleo#1NrOG/captions/captions.json"
    video_db_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Database/Batman_Begins_2005_1080p_BluRay_x264_YIFY/vdb.json"
    frame_folder_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Database/Batman_Begins_2005_1080p_BluRay_x264_YIFY/frames"
    video_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/Dataset/Video/Batman_Begins_2005_1080p_BluRay_x264_YIFY.mp4"
    Audio_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/Dataset/Audio/Mianhuicai/Mianhuicai.mp3"

    output_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/output_batman_waywe_go/shot_plan.json"

    # Instruction = """Give me a video that show the beauty of Lisbon city, including its famous landmarks, vibrant culture, and scenic views. The video should be engaging and visually appealing, capturing the essence of Lisbon through dynamic shots and smooth transitions. Please ensure the video aligns well with the rhythm and mood of the background music provided."""
    Instruction = """This script effectively portrays the growth of Batman through trauma, training, and moral choices, while implicitly addressing Bruce Wayne’s dual identity. With clearer contrasts between Bruce’s personal life and his role as Batman, the narrative would more directly emphasize the struggle of living a double life."""
    # video_db = init_single_video_db(video_caption_path, video_db_path, config.AOAI_EMBEDDING_LARGE_DIM)
    # a = get_video_clip_frame(video_db, "A woman is sitting in a car with the man and express do.", 16)
    # Test inspect_clip_details function
    # result = inspect_clip_details("00:10:00 to 00:10:29", frame_folder_path)
    # print(result)
    # a = generate_structure_proposal(video_scene_path, audio_caption_path, Instruction)

    agent = Screenwriter(
        video_db_path,
        video_caption_path,
        video_scene_path,  # Use scene summary folder path
        audio_caption_path,
        output_path,
        max_iterations=20,
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

