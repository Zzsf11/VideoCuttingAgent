import os
import json
import copy
import re
import numpy as np
from typing import Annotated as A, Any
from vca.build_database.video_caption import (
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
You are a professional screenwriter. Your task is to write a script based on the provided video and background music materials. The script should:
1. Align the rhythm and emotional progression of the video with the development of the accompanying music;
2. Seamlessly integrate the main theme and subject matter of the video materials;
3. Satisfy the user's specific editing instructions.

Requirements:
- Ensure that all selected video content is consistent with the main theme and narrative style.
- The final video should display clear emotional and rhythmic variation as well as story progression and visual engagement.
- Avoid any irrelevant or off-theme content, monotonous emotion or pacing, or lack of narrative/story.
- The narrative structure coud be non-linear, but must maintain logical coherence and no repetitive content.

Your goal is to design a detailed and coherent organizational plan for the short video, specifying the emotional arc and content of each segment.

1. A summary of the video material: VIDEO_SUMMARY_PLACEHOLDER
2. A summary of the background music: AUDIO_SUMMARY_PLACEHOLDER
3. The structure of the background music: AUDIO_STRUCTURE_PLACEHOLDER
4. The user's instruction: INSTRUCTION_PLACEHOLDER
Based on these inputs, please analyze and provide the following information about the target video in a structured format:
{
    "overall_theme": "The overall theme of the video",
    "narrative_logic": "The narrative logic of the video",
    "video_structure": [
        {   
            "content": "The detailed description of the content in this segment",
            "key_words": "The key words can be used to search relevant video materials",
            "audio_section": "The section in given audio",
            "emotion": "The emotion of the segment",
            "start_time": "The start time of the segment",
            "end_time": "The end time of the segment",
        }
        ...
    ]
}
"""


GENERATE_SHOT_PLAN_PROMPT = """
[Role]
You are a professional screenwriter tasked with creating a detailed shot plan that synchronizes the narrative storyline with the background music. This shot plan will serve as a part of full script for video editing.

[Task]
Map each analyzed music segment to a corresponding shot, ensuring the resulting sequence forms a coherent narrative arc that aligns seamlessly with the music's rhythm, phrasing, and emotional progression.

[Inputs]
- Detailed analysis of each music segment: AUDIO_SUMMARY_PLACEHOLDER
- Current video section narrative: VIDEO_SECTION_INFO_PLACEHOLDER
- Related video material context: RELATED_VIDEO_PLACEHOLDER

[Workflow]
1. Thoroughly understand all three inputs to ensure the section aligns with both the global narrative and musical intent.
2. For each music segment (in chronological order), design a corresponding shot that maintains continuity of subject, spatial coherence, and emotional trajectory.
3. Specify the shot's visual content, narrative beat, and function (setup/development/payoff/button/bridge) while ensuring the duration (minimum 3.0s) matches the music segment's length.
4. Clearly define the dominant emotion, key visual elements, and explicit music synchronization points so the editor can execute with precision.
5. Review the complete shot sequence to ensure uninterrupted narrative flow and fidelity to the soundtrack's critical moments (downbeats, motif transitions, crescendos, breaks).

[Guidelines]
- Maintain strict one-to-one correspondence between music segments and shots; no combining or subdividing.
- Keep shot durations realistic and proportional to each music segment (minor adjustments permitted only for narrative continuity or rhythmic impact).
- Strictly follow the provided section narrative; design shots exclusively for the given narrative content. Ensure each shot directly advances the story as outlined.
- The related video material context comes from keyword-based retrieval; some captions may not actually fit this section's storyline. Treat them purely as loose reference and let the section narrative remain your primary guide.
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
        },
        ...
    ]
}
"""


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
    video_summary_path: A[str, D("Path to video_summary.json for clip overview.")],
    audio_caption_path: A[str, D("Path to captions.json describing the audio segments.")],
    user_instruction: A[str, D("Editing brief provided by the user.")],
) -> str | None:
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


def generate_shot_plan(
    music_detailed_structure: A[list | dict | str, D("Detailed per-segment music analysis for current section.")],
    video_section_proposal: A[dict, D("Section brief extracted from structure proposal.")],
    video_db: A[NanoVectorDB | None, D("Vector database used to retrieve related video clips.")] = None,
) -> str | None:
    """
    Generate a one-to-one shot mapping for each music part using GENERATE_MUSIC_SHOT_MAPPING_PROMPT.

    - music_detailed_structure: list/dict or JSON string describing music parts (from captions.json detailed sections)
    - video_section_proposal: dict describing the current video section (from structure proposal)
    """

    # Normalize music structure to JSON string
    if isinstance(music_detailed_structure, (dict, list)):
        music_json = json.dumps(music_detailed_structure, ensure_ascii=False, indent=2)
    else:
        music_json = str(music_detailed_structure or '')

    prompt = GENERATE_SHOT_PLAN_PROMPT
    prompt = prompt.replace("AUDIO_SUMMARY_PLACEHOLDER", music_json)
    prompt = prompt.replace("VIDEO_SECTION_INFO_PLACEHOLDER", str(video_section_proposal))

    # Retrieve related clips based on provided keywords so the model grounds its plan.
    related_video_context = ""
    keywords = video_section_proposal.get("key_words") if isinstance(video_section_proposal, dict) else None
    if keywords and video_db is not None:
        query = keywords if isinstance(keywords, str) else ", ".join(map(str, keywords))
        try:
            related_video_context = search_video_library(video_db, query)
        except Exception as exc:
            related_video_context = f"Error retrieving related clips: {exc}"

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
    def __init__(self, video_db_path, video_caption_path, video_summary_path, audio_caption_path, output_path, max_iterations, video_path=None, frame_folder_path=None):
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
        self.video_summary_path = video_summary_path
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

        structure_proposal = generate_structure_proposal(self.video_summary_path, self.audio_caption_path, instruction)  # 第一次生产结构proposal，按照音乐的整体段落
        structure_proposal = parse_structure_proposal_output(structure_proposal)
        overall_theme = structure_proposal['overall_theme']
        narrative_logic = structure_proposal['narrative_logic']

        # TODO: 对每个section再做一次generate_structure_proposal，按照section的详细内容
        
        # Clear output file
        if self.output_path and os.path.exists(self.output_path):
            os.remove(self.output_path)

        pprint(structure_proposal, width=150)
        # Write structure_proposal to output file
        if self.output_path:
            with open(self.output_path, 'w', encoding='utf-8') as f:
                f.write("=== STRUCTURE PROPOSAL ===\n")
                f.write(json.dumps(structure_proposal, indent=2, ensure_ascii=False))
                f.write("\n\n")
        for sec_idx, sec_cur in enumerate(structure_proposal['video_structure']):
            print(f"\n{'='*60}")
            print(f"Processing Section {sec_idx + 1}/{len(structure_proposal['video_structure'])}")
            print(f"{'='*60}\n")
            
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
            
            print(f"Section {sec_idx + 1} output path: {self.output_path}")
            print(f"Target duration: {length_sec} seconds")
            print(f"Content: {sec_cur.get('content', 'N/A')}")
            print(f"Emotion: {sec_cur.get('emotion', 'N/A')}\n")
            
            

            shot_plan = generate_shot_plan(
                self.audio_db['sections'][sec_idx].get('detailed_analysis', '')['sections'],
                sec_cur,
                self.video_db,
            )
            shot_plan = parse_shot_plan_output(shot_plan)
            pprint(shot_plan, width=150)
            if self.output_path:
                with open(self.output_path, 'a', encoding='utf-8') as f:
                    f.write("=== Draft_shot_plan ===\n")
                    f.write(json.dumps(shot_plan, indent=2, ensure_ascii=False))
                    f.write("\n\n")
            for idx, shot in enumerate(shot_plan['shots']):
                msgs = copy.deepcopy(self.messages)
                print(f"\n{'='*60}")
                print(f"Processing Shot {idx + 1}/{len(shot_plan['shots'])}")
                print(f"{'='*60}\n")
                print(f"Shot output path: {self.output_path}")
                msgs[-1]["content"] = msgs[-1]["content"].replace("SHOT_PLAN_PLACEHOLDER", json.dumps(shot, indent=2, ensure_ascii=False))
                audio_section_info = str({k: v for k, v in (json.loads(self.audio_db['sections'][sec_idx].get('detailed_analysis', '')['sections'][idx]) if isinstance(self.audio_db['sections'][sec_idx].get('detailed_analysis', '')['sections'][idx], str) else self.audio_db['sections'][sec_idx].get('detailed_analysis', '')['sections'][idx]).items()})
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
                        print("#### Iteration: ", i, f"(Tool retry: {tool_retry + 1}/{max_tool_retries})" if tool_retry > 0 else "")
                        pprint(response, width=150)
                        
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
                        print(f"Shot {idx + 1} completed.")
                        break
        
                # Shot processed
                print(f"\nShot processed. Result appended to {self.output_path}.")
                    
        return msgs



def main():
    video_caption_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Database/Batman_Begins_2005_1080p_BluRay_x264_YIFY/captions/captions.json"
    video_summary_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Database/Batman_Begins_2005_1080p_BluRay_x264_YIFY/captions/video_summary.json"
    audio_caption_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Database/CallofSilence/captions/captions.json"
    video_db_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Database/Batman_Begins_2005_1080p_BluRay_x264_YIFY/vdb.json"
    frame_folder_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Database/Batman_Begins_2005_1080p_BluRay_x264_YIFY/frames"
    video_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/Dataset/Video/Batman.Begins.2005.1080p.BluRay.x264.YIFY.mp4"
    Audio_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/Dataset/Audio/Call_of_Slience/CallofSilence.mp3"

    output_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/shot_plan.json"

    Instruction = """Give me a video that show the growth of batman from a young boy to a mature man."""
    
    # video_db = init_single_video_db(video_caption_path, video_db_path, config.AOAI_EMBEDDING_LARGE_DIM)
    # a = get_video_clip_frame(video_db, "A woman is sitting in a car with the man and express do.", 16)
    # Test inspect_clip_details function
    # result = inspect_clip_details("00:10:00 to 00:10:29", frame_folder_path)
    # print(result)
    # a = generate_structure_proposal(video_summary_path, audio_caption_path, Instruction)
    
    agent = Screenwriter(
        video_db_path, 
        video_caption_path, 
        video_summary_path,
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

