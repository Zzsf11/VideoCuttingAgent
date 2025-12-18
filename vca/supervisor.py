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


class StopException(Exception):
    """
    Stop Execution by raising this exception (Signal that the task is Finished).
    """

def Review_Proposal(proposal, ):
    """
    Review Execution by raising this exception (Signal that the task is Finished).
    """


def Review_shot_plan(shot_plan, ):
    """
    Review Execution by raising this exception (Signal that the task is Finished).
    """

def Review_timeline(timeline, ):
    """
    Review Execution by raising this exception (Signal that the task is Finished).
    """

def Review_audio_video_alignment(alignment, ):
    """
    Review Execution by raising this exception (Signal that the task is Finished).
    """

class SupervisorAgent:
    def __init__(self, video_db_path, video_caption_path, video_summary_path, audio_caption_path, output_path, max_iterations, video_path=None, frame_folder_path=None):
        self.tools = [Review_Proposal, , finish]
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

        # Logging state only (visualization removed)
        self._agent_log = []
        self.current_section_idx = None
        self.current_shot_idx = None

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
Inspect the script, timestamps, and provided cues, then iteratively call the available tools to locate and refine clips that satisfy the creative brief.

[Tools]
• `get_video_clip_frame`: retrieve candidate video clips from the database for contextual exploration.
• `trim_video_clip`: analyze a candidate clip using dense captions and frame files so you can pinpoint the exact in/out moments; this tool returns descriptive timestamps but does not change the clip unless you provide a new, narrower time range on the next call.
• `finish`: present the final timestamped editing plan once all required clips are selected and refined.

[Workflow]
1. Review the global brief and initial observations about the video
2. Use `get_video_clip_frame` to surface promising segments aligned with the target theme, narrative logic, and emotion, filtering out clips whose narrative jumps conflict with the current storyline.
3. For each promising segment, call `trim_video_clip` to confirm visuals, capture precise beats, interpret the returned timestamped captions, and then decide whether to issue a refined time range.
4. Repeat 2–3 until the desired runtime and storytelling flow are covered.
5. Conclude with `finish`, summarizing the final ordered shot list with exact timestamps.

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
- Maximize the usefulness of `trim_video_clip` to ensure every chosen moment matches the intended pacing, emotion, and visual storytelling beats.
- After every `trim_video_clip` call, inspect the returned caption to compute updated start/end times; re-invoke the tool with those refined boundaries instead of assuming the duration changed automatically.
- Continuity check: discard retrieved clips that introduce abrupt jumps or contradict the evolving narrative before moving forward.
- Maintain consistent formatting for timestamps (HH:MM:SS or MM:SS) and keep the final plan aligned with the requested length and narrative arc.
- Ensure every selected shot lasts at least 3 seconds; extend or replace any sub-3-second candidate before calling `finish`.
- When a clear story beat is not found, expand the search window progressively (e.g., ±10–20s, then ±30–45s) and attempt multiple expansions before moving on.
- If you find a related scene but some shots/beats are missing, perform a neighborhood search centered on that segment by expanding on both sides (e.g., ±10–20s, then ±30s) to capture the surrounding context.
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
        
        if "topk" in args:
            if config.OVERWRITE_CLIP_SEARCH_TOPK > 0:
                args["topk"] = config.OVERWRITE_CLIP_SEARCH_TOPK
        
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
    
    # video_db = init_single_video_db(video_caption_path, video_db_path, config.AOAI_EMBEDDING_LARGE_DIM)
    # a = get_video_clip_frame(video_db, "A woman is sitting in a car with the man and express do.", 16)
    # Test trim_video_clip function
    result = trim_video_clip("00:10:00 to 00:10:29", frame_folder_path)
    # print(result)
    
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

