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
from vca.vllm_calling import call_vllm_model
from vca.core import (
    get_video_clip_frame,
    finish,
    parse_structure_proposal_output,
    parse_shot_plan_output
)

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
}
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


class StopException(Exception):
    """
    Stop Execution by raising this exception (Signal that the task is Finished).
    """

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
    video_summary_path: A[str, D("Path to video_summary.json for scene context.")],
    music_detailed_structure: A[list | dict | str, D("Detailed per-segment music analysis for current section.")],
    video_section_proposal: A[dict, D("Section brief extracted from structure proposal.")],
    retrieved_context: A[str, D("Retrieved video clips context from get_video_clip_frame tool.")] = "",
) -> str | None:
    """
    Generate a one-to-one shot mapping for each music part using GENERATE_MUSIC_SHOT_MAPPING_PROMPT.

    - video_summary_path: path to video_summary.json (provides VIDEO_CONTEXT)
    - music_detailed_structure: list/dict or JSON string describing music parts (from captions.json detailed sections)
    - user_instruction: editing brief
    - retrieved_context: context retrieved from video database
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
    
    if retrieved_context:
        prompt += f"\n\n[Retrieved Video Context]\n{retrieved_context}\n\nPlease use the above retrieved video context to select the best shots."

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
            generate_structure_proposal,
            generate_shot_plan,
            get_video_clip_frame,
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

        # Logging state only (visualization removed)
        self._agent_log = []
        self.current_section_idx = None
        self.current_shot_idx = None

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
            def _truncate(value):
                try:
                    serialized = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                except Exception:
                    serialized = str(value)
                return serialized if len(serialized) <= 4000 else serialized[:4000] + "\n...[truncated]"

            safe_entry = {}
            for key, value in entry.items():
                if key in {"args", "result", "content", "data"}:
                    safe_entry[key] = _truncate(value)
                else:
                    safe_entry[key] = value
            self._agent_log.append(safe_entry)
        except Exception:
            pass

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
    You are a senior screenwriter who analyzes video material, audio material, and user instructions to provide an executable plan for video editing.

    [Task]
    Inspect the audio and video materials along with the user's instruction. Use available tools to help refine the rough proposal and shot plan to ensure alignment with both video and audio materials, making the shot plan executable.

    [Tools]
    • `generate_structure_proposal`: Generate a high-level structure proposal for the entire video based on video summary, audio structure, and user instruction.
    • `generate_shot_plan`: Generate a rough shot plan for each music segment using the audio structure and brief description of the current video section.
    • `get_video_clip_frame`: Retrieve candidate video clips from the database for contextual exploration.
    • `finish`: Present the final shot list with timestamps.

    [Workflow]
    1. Quickly draft a high-level proposal (as a rough guide only) by calling `generate_structure_proposal`.
    2. For each audio section, call `get_video_clip_frame` to retrieve concrete candidate clips and timestamps; build a concise retrieved_context from the best matches.
    3. Call `generate_shot_plan` with the section brief and retrieved_context to produce an executable, grounded shot list for that section.
    4. Validate each shot against retrieved_context and music timing; if a shot lacks evidence or timing drifts, revise using more targeted retrieval.
    5. Conclude with `finish`, summarizing the final ordered shot list with exact timestamps.

    [Input Brief]
    - Target edited video length: VIDEO_LENGTH_PLACEHOLDER seconds.
    - Target edited video content: CURRENT_VIDEO_CONTENT_PLACEHOLDER
    - Target edited video emotion: CURRENT_VIDEO_EMOTION_PLACEHOLDER
    - Background music: BACKGROUND_MUSIC_PLACEHOLDER

    [Output]
    Provide a detailed, executable proposal.

    [Guidelines]
    - Treat the proposal as advisory; when it conflicts with retrieved footage or music timing, prioritize retrieved evidence and audio segmentation.
    - Think aloud about why each tool call is necessary before executing it, and reflect on the observations afterwards.
    - The output content must tightly align with both the video material and audio material. Never fabricate timestamps or descriptions.
    - The results from `generate_shot_plan` and `generate_structure_proposal` may not be perfect initially and require refinement through tool-assisted verification.
    - Ensure all timestamps and content references are grounded in actual material retrieved from the tools.
    - Every shot must cite at least one retrieved time window (e.g., “HH:MM:SS to HH:MM:SS”) either inside the content or as a clearly marked time_range; otherwise revise before proceeding.
    - Anchor shot durations to the corresponding audio section window (allowing small tolerance, e.g., ±0.5s) to maintain sync.
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
        if "database" in args:
            args["database"] = self.video_db
        
        if "topk" in args or "top_k" in args:
            key = "top_k" if "top_k" in args else "topk"
            if config.OVERWRITE_CLIP_SEARCH_TOPK > 0:
                args[key] = config.OVERWRITE_CLIP_SEARCH_TOPK
        
        # For trim_video_clip, inject frame_folder parameter
        if name == "trim_video_clip":
            if self.frame_folder_path:
                args["frame_path"] = self.frame_folder_path
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

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self, instruction) -> list[dict]:
        """
        Run the ReAct-style loop with OpenAI Function Calling.
        """

        # structure_proposal = generate_structure_proposal(self.video_summary_path, self.audio_caption_path, instruction)  # 第一次生产结构proposal，按照音乐的整体段落
        # structure_proposal = parse_structure_proposal_output(structure_proposal)
        # overall_theme = structure_proposal['overall_theme']
        # narrative_logic = structure_proposal['narrative_logic']
        # # Log + visualize structure proposal
        # self._append_agent_log({
        #     "type": "structure_proposal",
        #     "section": None,
        #     "shot": None,
        #     "data": structure_proposal,
        # })

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
            for idx, shot in enumerate(shot_plan['shots']):
                msgs = copy.deepcopy(self.messages)
                print(f"\n{'='*60}")
                print(f"Processing Shot {idx + 1}/{len(shot_plan['shots'])}")
                print(f"{'='*60}\n")
                # Set shot-specific output path: <base>_section_<sec_idx>_shot_<idx>.mp4
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
                    
        return msgs



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
    
    # video_db = init_single_video_db(video_caption_path, video_db_path, config.AOAI_EMBEDDING_LARGE_DIM)
    # a = get_video_clip_frame(video_db, "A woman is sitting in a car with the man and express do.", 16)
    # Test trim_video_clip function
    # result = trim_video_clip("00:10:00 to 00:10:29", frame_folder_path)
    # print(result)
    # a = generate_structure_proposal(video_summary_path, audio_caption_path, Instruction)
    
    agent = Screenwriter(
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

