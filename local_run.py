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
from vca.build_database.video_caption import process_video
from vca.build_database.audio_caption import caption_audio_with_segments
import vca.config as config

def main():
    parser = argparse.ArgumentParser(description="Run VideoCaptioningAgent on a video.")
    parser.add_argument("Video_Path", help="The URL of the video to process.")
    parser.add_argument("Audio_Path", help="The URL of the video to process.")
    parser.add_argument("Text_Prompt", help="The prompt to cutting the video.")
    args = parser.parse_args()

    Video_Path = args.Video_Path
    Audio_Path = args.Audio_Path
    Text_Prompt = args.Text_Prompt
    
    video_id = os.path.splitext(os.path.basename(Video_Path))[0].replace('.', '_').replace(' ', '_')
    audio_id = os.path.splitext(os.path.basename(Audio_Path))[0].replace('.', '_').replace(' ', '_')

    output_path = os.path.join(config.VIDEO_DATABASE_FOLDER, "raw", f"{video_id}.mp4")
    frames_dir = os.path.join(config.VIDEO_DATABASE_FOLDER, 'Database', video_id, "frames")
    video_captions_dir = os.path.join(config.VIDEO_DATABASE_FOLDER, 'Database', video_id, "captions")
    audio_captions_dir = os.path.join(config.VIDEO_DATABASE_FOLDER, 'Database', audio_id, "captions")
    video_db_path = os.path.join(config.VIDEO_DATABASE_FOLDER, 'Database',video_id, "database.json")
    srt_path = os.path.join(config.VIDEO_DATABASE_FOLDER, 'Database', video_id, "subtitles.srt")



    # Decode video to frames and get subtitles
    if not os.path.exists(frames_dir) or not os.listdir(frames_dir):
        print(f"Decoding video to frames in {frames_dir}...")
        decode_video_to_frames(
            Video_Path,
            frames_dir,
            config.VIDEO_FPS,
            config.VIDEO_RESOLUTION,
            max_frames=config.VIDEO_MAX_FRAMES,
            asr_to_srt=True,
            srt_path=srt_path,
            asr_model=config.ASR_MODEL,
            asr_device="cuda:0",
            batch_size=config.VIDEO_BATCH_SIZE,
            use_batch_processing=config.USE_BATCH_PROCESSING,
            shot_detection=True,
            shot_detection_fps=config.SHOT_DETECTION_FPS,
            shot_detection_threshold=config.SHOT_DETECTION_THRESHOLD,
            shot_predictions_path=os.path.join(frames_dir, "shot_predictions.txt"),
            shot_scenes_path=os.path.join(frames_dir, "shot_scenes.txt"),
            shot_detection_model=config.SHOT_DETECTION_MODEL,

        )
        print("Video decoded.")
    else:
        print(f"Frames already exist in {frames_dir}.")


    
    # # Get video captions
    # caption_file = os.path.join(video_captions_dir, "captions.json")
    # if not os.path.exists(caption_file):
    #     print("Processing video to get captions...")
    #     shot_scenes_file = os.path.join(frames_dir, "shot_scenes.txt")
    #     process_video(
    #         frame_folder=frames_dir,
    #         output_caption_folder=video_captions_dir,
    #         subtitle_file_path=srt_path,
    #         shot_scenes_path=shot_scenes_file if os.path.exists(shot_scenes_file) else None,
    #     )
    #     print("Captions generated.")
    # else:
    #     print(f"Captions already exist at {caption_file}.")

        # Get video captions
    caption_file = os.path.join(video_captions_dir, "captions.json")
    print("Processing video to get captions...")
    shot_scenes_file = os.path.join(frames_dir, "shot_scenes.txt")
    process_video(
        frame_folder=frames_dir,
        output_caption_folder=video_captions_dir,
        subtitle_file_path=srt_path,
        shot_scenes_path=shot_scenes_file if os.path.exists(shot_scenes_file) else None,
    )


    # # Analyze music
    # audio_caption_file = os.path.join(audio_captions_dir, "captions.json")
    # if not os.path.exists(audio_caption_file):
    #     print("Processing audio to get captions...")
    #     caption_audio_with_segments(
    #         audio_path=Audio_Path,
    #         max_tokens=config.AUDIO_ANALYSIS_MODEL_MAX_TOKEN,
    #         output_path=audio_caption_file,
    #         batch_size=4  # Process 4 segments in parallel for better efficiency
    #     )
    #     print("Captions generated.")
    # else:
    #     print(f"Captions already exist at {audio_caption_file}.")

    # Initialize DVDCoreAgent
    # print("Initializing DVDCoreAgent...")
    # agent = DVDCoreAgent(video_db_path, caption_file, config.MAX_ITERATIONS)
    # print("Agent initialized.")

    # # Run with question
    # print(f"Running agent with question: '{question}'")
    # msgs = agent.run(question)
    # print(extract_answer(msgs[-1]))

if __name__ == "__main__":
    main()
