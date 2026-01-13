import vca.config as config
import os
import argparse
# from dvd.dvd_core import DVDCoreAgent
# from dvd.video_utils import load_video, decode_video_to_frames, download_srt_subtitle
# from dvd.frame_caption import process_video, process_video_lite
# from dvd.utils import extract_answer

# from dvd.frame_caption import process_video, process_video_lite
# from vca.utils import load_video, decode_video_to_frames, download_srt_subtitle
from vca.video_utils import decode_video_to_frames
from vca.asr import run_asr
from vca.build_database.video_caption import process_video
from vca.audio.audio_caption_madmom import caption_audio_with_madmom_segments
from vca.build_database.get_character import analyze_subtitles
from vca.build_database.scene_merge import OptimizedSceneSegmenter, load_shots, save_scenes
from vca.build_database.scene_analysis_video import SceneVideoAnalyzer
import vca.config as config

def parse_config_overrides(unknown_args):
    """
    Parse config override arguments in the format --config.PARAM_NAME value

    Args:
        unknown_args: List of unknown arguments from argparse

    Returns:
        None (modifies config module in place)
    """
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if arg.startswith('--config.'):
            param_name = arg[9:]  # Remove '--config.' prefix
            if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith('--'):
                value_str = unknown_args[i + 1]

                # Auto-detect type based on existing config value or infer from string
                if hasattr(config, param_name):
                    original_value = getattr(config, param_name)
                    # Preserve original type
                    if isinstance(original_value, bool):
                        value = value_str.lower() in ('true', '1', 'yes')
                    elif isinstance(original_value, int):
                        value = int(value_str)
                    elif isinstance(original_value, float):
                        value = float(value_str)
                    else:
                        value = value_str
                else:
                    # Infer type from string
                    try:
                        if '.' in value_str:
                            value = float(value_str)
                        else:
                            value = int(value_str)
                    except ValueError:
                        if value_str.lower() in ('true', 'false'):
                            value = value_str.lower() == 'true'
                        else:
                            value = value_str

                setattr(config, param_name, value)
                print(f"✓ Config override: {param_name} = {value} (type: {type(value).__name__})")
                i += 2
            else:
                print(f"⚠ Warning: --config.{param_name} specified but no value provided")
                i += 1
        else:
            print(f"⚠ Warning: Unknown argument '{arg}' ignored")
            i += 1

def main():
    parser = argparse.ArgumentParser(description="Run VideoCaptioningAgent on a video.")
    parser.add_argument("--Video_Path", help="The URL of the video to process.", default="/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/Dataset/Video/Batman.Begins.2005.1080p.BluRay.x264.YIFY.mp4")
    # parser.add_argument("--Video_Path", help="The URL of the video to process.", default="/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/Dataset/Video/VLOG_Lisbon.mp4")
    parser.add_argument("--Audio_Path", help="The URL of the video to process.", default="/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/Dataset/Audio/Way_Down_We_Go.mp3")
    parser.add_argument("--Instruction", help="The Instruction to cutting the video.", default="Create a video montage based on the background music")
    parser.add_argument("--instruction_type", help="Type of instruction: 'object' for Object-centric or 'narrative' for Narrative-driven", default="object", choices=["object", "narrative"])
    parser.add_argument("--type", help="film or vlog", default="film")

    # Parse known args and capture unknown args for config overrides
    args, unknown = parser.parse_known_args()

    # Apply config overrides
    parse_config_overrides(unknown)

    config.VIDEO_TYPE = args.type

    Video_Path = args.Video_Path
    Audio_Path = args.Audio_Path
    Instruction = args.Instruction
    instruction_type = args.instruction_type

    video_id = os.path.splitext(os.path.basename(Video_Path))[0].replace('.', '_').replace(' ', '_')
    audio_id = os.path.splitext(os.path.basename(Audio_Path))[0].replace('.', '_').replace(' ', '_')

    # Generate a safe filename from instruction
    import re
    import hashlib
    # Create a short hash of the instruction for uniqueness
    instruction_hash = hashlib.md5(Instruction.encode('utf-8')).hexdigest()[:8]
    # Create a more readable version (up to 50 characters, sanitized)
    instruction_safe = re.sub(r'[^\w\s-]', '', Instruction)[:50].strip().replace(' ', '_')
    # If instruction is too long or empty, use a more informative format
    if len(instruction_safe) > 0:
        instruction_id = f"{instruction_safe}_{instruction_hash}"
    else:
        instruction_id = f"instruction_{instruction_hash}"

    # ===== All Path Definitions =====
    # Raw video output
    output_path = os.path.join(config.VIDEO_DATABASE_FOLDER, "raw", f"{video_id}.mp4")

    # Video-related paths
    frames_dir = os.path.join(config.VIDEO_DATABASE_FOLDER, 'Video', video_id, "frames")
    video_captions_dir = os.path.join(config.VIDEO_DATABASE_FOLDER, 'Video', video_id, "captions")
    video_db_path = os.path.join(config.VIDEO_DATABASE_FOLDER, 'Video', video_id, "database.json")
    srt_path = os.path.join(config.VIDEO_DATABASE_FOLDER, 'Video', video_id, "subtitles.srt")
    srt_with_characters = os.path.join(config.VIDEO_DATABASE_FOLDER, 'Video', video_id, "subtitles_with_characters.srt")
    character_info_path = os.path.join(config.VIDEO_DATABASE_FOLDER, 'Video', video_id, "character_info.json")

    # Shot and scene paths
    shot_scenes_file = os.path.join(frames_dir, "shot_scenes.txt")
    caption_file = os.path.join(video_captions_dir, "captions.json")
    shots_dir = os.path.join(config.VIDEO_DATABASE_FOLDER, 'Video', video_id, "captions", "ckpt")
    scenes_dir = os.path.join(config.VIDEO_DATABASE_FOLDER, 'Video', video_id, "captions", "scenes")
    scenes_output = os.path.join(scenes_dir, "scene_0.json")
    scene_summaries_dir = os.path.join(config.VIDEO_DATABASE_FOLDER, 'Video', video_id, "captions", "scene_summaries_video")

    # Audio-related paths
    audio_captions_dir = os.path.join(config.VIDEO_DATABASE_FOLDER, 'Audio', audio_id, "captions")
    audio_caption_file = os.path.join(audio_captions_dir, "captions.json")

    # Output paths (include instruction type and instruction ID for different editing tasks)
    shot_plan_output_path = os.path.join(
        config.VIDEO_DATABASE_FOLDER,
        'Output',
        f"{video_id}_{audio_id}",
        instruction_type,  # Add instruction type subfolder
        f"shot_plan_{instruction_id}.json"
    )



    # # Step 1: Decode video to frames and perform shot detection
    # print(f"Processing video frames in {frames_dir}...")
    # decode_video_to_frames(
    #     Video_Path,
    #     frames_dir,
    #     config.VIDEO_FPS,
    #     config.VIDEO_RESOLUTION,
    #     max_frames=getattr(config, 'VIDEO_MAX_FRAMES', None),
    #     use_batch_processing=config.USE_BATCH_PROCESSING,
    #     shot_detection=True,
    #     shot_detection_fps=config.SHOT_DETECTION_FPS,
    #     shot_detection_threshold=config.SHOT_DETECTION_THRESHOLD,
    #     shot_detection_min_scene_len=config.SHOT_DETECTION_MIN_SCENE_LEN,
    #     shot_predictions_path=os.path.join(frames_dir, "shot_predictions.txt"),
    #     shot_scenes_path=os.path.join(frames_dir, "shot_scenes.txt"),
    #     shot_detection_model=config.SHOT_DETECTION_MODEL,
    # )
    # print("Frame extraction and shot detection completed.")

    # # Step 2: Run ASR to generate subtitles (skip for vlog)
    # if args.type != "vlog":
    #     print(f"Running ASR to generate subtitles...")
    #     run_asr(
    #         video_path=Video_Path,
    #         output_dir=frames_dir,
    #         srt_path=srt_path,
    #         asr_model=config.ASR_MODEL,
    #         asr_device="cuda:0",
    #     )
    #     print("ASR completed.")
    # else:
    #     print("Skipping ASR for vlog type.")

    # # Step 3: Identify characters from subtitles (skip for vlog), need vllm server running
    # if args.type != "vlog":
    #     if os.path.exists(srt_path) and not os.path.exists(character_info_path):
    #         print("Analyzing subtitles to identify characters...")
    #         # Extract movie name from video_id for better context
    #         movie_name = video_id.replace('_', ' ')
    #         speaker_mapping, _character_info = analyze_subtitles(
    #             srt_path=srt_path,
    #             movie_name=movie_name,
    #             output_dir=os.path.join(config.VIDEO_DATABASE_FOLDER, 'Video', video_id),
    #             use_full_subtitles=True  # Use full subtitles for better accuracy
    #         )
    #         print(f"Character identification completed. Found {len(speaker_mapping)} characters.")
    #     elif os.path.exists(character_info_path):
    #         print(f"Character info already exists at {character_info_path}.")
    #     else:
    #         print(f"Subtitle file not found at {srt_path}, skipping character identification.")
    # else:
    #     print("Skipping character identification for vlog type.")



    # # Get video captions, need vllm server running
    # if not os.path.exists(caption_file):
    #     print("Processing video to get captions...")

    #     # For vlog, don't use subtitles; for film, use subtitle with character names if available
    #     if args.type == "vlog":
    #         subtitle_to_use = None
    #         print("Processing vlog without subtitles.")
    #     else:
    #         subtitle_to_use = srt_with_characters if os.path.exists(srt_with_characters) else srt_path
    #         print(f"Using subtitle file: {subtitle_to_use}")

    #     process_video(
    #         frame_folder=frames_dir,
    #         output_caption_folder=video_captions_dir,
    #         subtitle_file_path=subtitle_to_use,
    #         long_shots_path=shot_scenes_file if os.path.exists(shot_scenes_file) else None,
    #         video_type=args.type,  # Pass video type to prevent subtitle auto-search for vlog
    #     )
    #     print("Captions generated.")
    # else:
    #     print(f"Captions already exist at {caption_file}.")

    # # Step 4: Merge shots into scenes and perform scene analysis
    # # Step 4.1: Merge shots into scenes
    # if os.path.exists(shots_dir) and not os.path.exists(scenes_output):
    #     print("Merging shots into scenes...")

    #     # Load shots
    #     shots = load_shots(shots_dir)
    #     print(f"Loaded {len(shots)} shots")

    #     if shots:
    #         # Initialize segmenter
    #         segmenter = OptimizedSceneSegmenter()

    #         # Merge shots into scenes
    #         merged_scenes = segmenter.segment(
    #             shots,
    #             threshold=config.SCENE_SIMILARITY_THRESHOLD if hasattr(config, 'SCENE_SIMILARITY_THRESHOLD') else 0.5,
    #             max_scene_duration_secs=config.MAX_SCENE_DURATION_SECS if hasattr(config, 'MAX_SCENE_DURATION_SECS') else 300
    #         )

    #         print(f"Merged {len(shots)} shots into {len(merged_scenes)} scenes")

    #         # Save scenes
    #         save_scenes(merged_scenes, scenes_dir)
    #         print(f"Scenes saved to {scenes_dir}")
    #     else:
    #         print("No shots found to merge")
    # elif os.path.exists(scenes_output):
    #     print(f"Scenes already exist at {scenes_dir}")
    # else:
    #     print(f"Shots directory not found at {shots_dir}, skipping scene merge")

    # # Step 4.2: Analyze scenes with video analysis
    # if os.path.exists(scenes_dir) and os.path.exists(scenes_output):
    #     # Check if scene summaries already exist
    #     scene_files = [f for f in os.listdir(scenes_dir) if f.endswith('.json')]
    #     summary_files = []
    #     if os.path.exists(scene_summaries_dir):
    #         summary_files = [f for f in os.listdir(scene_summaries_dir) if f.endswith('.json')]

    #     if len(summary_files) < len(scene_files):
    #         print(f"Analyzing scenes with video analysis (found {len(scene_files)} scenes, {len(summary_files)} already analyzed)...")

    #         # For vlog, don't use subtitles; for film, use subtitle with character names if available
    #         if args.type == "vlog":
    #             subtitle_to_use = None
    #             print("Analyzing scenes without subtitles for vlog.")
    #         else:
    #             srt_with_characters = os.path.join(config.VIDEO_DATABASE_FOLDER, 'Video', video_id, "subtitles_with_characters.srt")
    #             subtitle_to_use = srt_with_characters if os.path.exists(srt_with_characters) else srt_path

    #         # Initialize scene video analyzer
    #         analyzer = SceneVideoAnalyzer(
    #             frames_dir=frames_dir,
    #             subtitle_file=subtitle_to_use
    #         )

    #         # Create output directory
    #         os.makedirs(scene_summaries_dir, exist_ok=True)

    #         # Process each scene file
    #         import concurrent.futures
    #         from tqdm import tqdm

    #         tasks = [
    #             (os.path.join(scenes_dir, f), os.path.join(scene_summaries_dir, f))
    #             for f in scene_files
    #         ]

    #         max_workers = getattr(config, 'SCENE_ANALYSIS_MAX_WORKERS', 8)

    #         with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    #             results = list(tqdm(
    #                 executor.map(lambda t: analyzer.process_file(t[0], t[1]), tasks),
    #                 total=len(tasks),
    #                 desc="Analyzing scenes"
    #             ))

    #         success_count = results.count('Success')
    #         skipped_count = results.count('Skipped')
    #         print(f"Scene analysis completed: {success_count} success, {skipped_count} skipped")

    #         errors = [r for r in results if r.startswith("Error")]
    #         if errors:
    #             print(f"Errors: {len(errors)}")
    #             for e in errors[:3]:
    #                 print(f"  {e}")
    #     else:
    #         print(f"Scene summaries already exist at {scene_summaries_dir} ({len(summary_files)} files)")
    # else:
    #     print(f"Scenes directory not found or empty at {scenes_dir}, skipping scene analysis")



    # # Analyze music
    # if not os.path.exists(audio_caption_file):
    #     print("Processing audio to get captions...")
    #     caption_audio_with_madmom_segments(
    #         audio_path=Audio_Path,
    #         output_path=audio_caption_file,
    #         # AI model parameters
    #         max_tokens=config.AUDIO_ANALYSIS_MODEL_MAX_TOKEN,
    #         temperature=config.AUDIO_KEYPOINT_TEMPERATURE,
    #         top_p=config.AUDIO_KEYPOINT_TOP_P,
    #         batch_size=config.AUDIO_BATCH_SIZE,
    #         # Detection method selection (NEW: supports multiple methods)
    #         detection_methods=config.AUDIO_DETECTION_METHODS,
    #         # Downbeat detection parameters
    #         beats_per_bar=[config.AUDIO_BEATS_PER_BAR],
    #         min_bpm=config.AUDIO_MIN_BPM,
    #         max_bpm=config.AUDIO_MAX_BPM,
    #         # Pitch detection parameters
    #         pitch_tolerance=config.AUDIO_PITCH_TOLERANCE,
    #         pitch_threshold=config.AUDIO_PITCH_THRESHOLD,
    #         pitch_min_distance=config.AUDIO_PITCH_MIN_DISTANCE,
    #         pitch_nms_method=config.AUDIO_PITCH_NMS_METHOD,
    #         pitch_max_points=config.AUDIO_PITCH_MAX_POINTS,
    #         # Mel energy detection parameters
    #         mel_win_s=config.AUDIO_MEL_WIN_S,
    #         mel_n_filters=config.AUDIO_MEL_N_FILTERS,
    #         mel_threshold_ratio=config.AUDIO_MEL_THRESHOLD_RATIO,
    #         mel_min_distance=config.AUDIO_MEL_MIN_DISTANCE,
    #         mel_nms_method=config.AUDIO_MEL_NMS_METHOD,
    #         mel_max_points=config.AUDIO_MEL_MAX_POINTS,
    #         # Post-processing / Rule-based filtering parameters
    #         merge_close=config.AUDIO_MERGE_CLOSE,
    #         min_interval=config.AUDIO_MIN_INTERVAL,
    #         top_k_keypoints=config.AUDIO_TOP_K,
    #         energy_percentile=config.AUDIO_ENERGY_PERCENTILE,
    #         # Segment filtering parameters
    #         min_segment_duration=config.AUDIO_MIN_SEGMENT_DURATION,
    #         max_segment_duration=config.AUDIO_MAX_SEGMENT_DURATION,
    #         # Section-based filtering parameters
    #         use_stage1_sections=config.AUDIO_USE_STAGE1_SECTIONS,
    #         section_min_interval=config.AUDIO_SECTION_MIN_INTERVAL,
    #     )
    #     print("Captions generated.")
    # else:
    #     print(f"Captions already exist at {audio_caption_file}.")



    # Step 5: Run Screenwriter to generate shot plan
    if os.path.exists(scene_summaries_dir) and os.path.exists(audio_caption_file):
        print("\n" + "="*80)
        print("Running Screenwriter to generate shot plan...")
        print("="*80)

        from vca.Screenwriter_scene_short import Screenwriter

        # Create output directory
        os.makedirs(os.path.dirname(shot_plan_output_path), exist_ok=True)

        # Initialize Screenwriter agent
        screenwriter = Screenwriter(
            video_scene_path=scene_summaries_dir,
            audio_caption_path=audio_caption_file,
            output_path=shot_plan_output_path,
            max_iterations=20,
            video_path=Video_Path,
            frame_folder_path=frames_dir,

        )

        # Run the screenwriter with the Instruction
        print(f"Running screenwriter with instruction: '{Instruction}'")
        _shot_plan = screenwriter.run(Instruction)

        print(f"\n{'='*80}")
        print(f"Shot plan generated successfully!")
        print(f"Output saved to: {shot_plan_output_path}")
        print(f"{'='*80}\n")
    else:
        if not os.path.exists(scene_summaries_dir):
            print(f"Warning: Scene summaries directory not found at {scene_summaries_dir}")
            print("Skipping screenwriter execution.")
        if not os.path.exists(audio_caption_file):
            print(f"Warning: Audio caption file not found at {audio_caption_file}")
            print("Skipping screenwriter execution.")


    # # Step 6: Run EditorCoreAgent to select video clips based on shot plan
    # # Final output path for shot points
    # shot_point_output_path = os.path.join(
    #     config.VIDEO_DATABASE_FOLDER,
    #     'Output',
    #     f"{video_id}_{audio_id}",
    #     instruction_type,
    #     f"shot_point_{instruction_id}.json"
    # )

    # # Check if we have all required files for core agent
    # if os.path.exists(scene_summaries_dir) and os.path.exists(audio_caption_file) and os.path.exists(shot_plan_output_path):
    #     print("\n" + "="*80)
    #     print("Running EditorCoreAgent to select video clips...")
    #     print("="*80)

    #     from vca.core import EditorCoreAgent

    #     # Create output directory
    #     os.makedirs(os.path.dirname(shot_point_output_path), exist_ok=True)

    #     # Initialize EditorCoreAgent (note: video_db_path is no longer needed)
    #     editor_agent = EditorCoreAgent(
    #         video_caption_path=caption_file,
    #         video_scene_path=scene_summaries_dir,
    #         audio_caption_path=audio_caption_file,
    #         output_path=shot_point_output_path,
    #         max_iterations=config.AGENT_MAX_ITERATIONS if hasattr(config, 'AGENT_MAX_ITERATIONS') else 30,
    #         video_path=Video_Path,
    #         frame_folder_path=frames_dir,
    #         transcript_path=srt_with_characters if os.path.exists(srt_with_characters) else srt_path
    #     )

    #     # Run the editor agent with the instruction and shot plan
    #     print(f"Running editor agent with instruction: '{Instruction}'")
    #     print(f"Using shot plan from: {shot_plan_output_path}")
    #     _messages = editor_agent.run(Instruction, shot_plan_path=shot_plan_output_path)

    #     print(f"\n{'='*80}")
    #     print(f"Video clip selection completed!")
    #     print(f"Output saved to: {shot_point_output_path}")
    #     print(f"{'='*80}\n")
    # else:
    #     print("\n" + "="*80)
    #     print("Cannot run EditorCoreAgent - missing required files:")
    #     if not os.path.exists(scene_summaries_dir):
    #         print(f"  ❌ Scene summaries directory not found at {scene_summaries_dir}")
    #     if not os.path.exists(audio_caption_file):
    #         print(f"  ❌ Audio caption file not found at {audio_caption_file}")
    #     if not os.path.exists(shot_plan_output_path):
    #         print(f"  ❌ Shot plan file not found at {shot_plan_output_path}")
    #     print("="*80 + "\n")


if __name__ == "__main__":
    main()
