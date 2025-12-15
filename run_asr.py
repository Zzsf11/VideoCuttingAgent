#!/usr/bin/env python3
"""
ASR + Speaker Diarization Script
Uses Whisper for speech recognition and pyannote for speaker diarization.
Outputs SRT subtitle files with speaker labels.
"""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import torch
import soundfile as sf
import numpy as np
import whisper
from pyannote.audio import Pipeline
from pyannote.core import Segment

# Video file extensions that need audio extraction
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v'}


def extract_audio_from_video(video_path: str, output_path: str = None) -> str:
    """
    Extract audio from video file using ffmpeg.

    Args:
        video_path: Path to the video file
        output_path: Optional output path for the extracted audio.
                     If None, creates a temp file.

    Returns:
        Path to the extracted audio file (WAV format)
    """
    if output_path is None:
        # Create a temporary WAV file
        fd, output_path = tempfile.mkstemp(suffix='.wav')
        os.close(fd)

    print(f"Extracting audio from video: {video_path}")

    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vn',  # No video
        '-acodec', 'pcm_s16le',  # PCM 16-bit little-endian
        '-ar', '16000',  # Sample rate 16kHz (good for speech)
        '-ac', '1',  # Mono
        '-y',  # Overwrite output
        output_path
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Audio extracted to: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to extract audio from video: {e.stderr.decode()}")


def is_video_file(file_path: str) -> bool:
    """Check if the file is a video file based on extension."""
    ext = Path(file_path).suffix.lower()
    return ext in VIDEO_EXTENSIONS


def load_audio_for_pyannote(audio_path: str) -> dict:
    """
    Pre-load audio file to bypass torchcodec compatibility issues.
    Returns a dict with waveform and sample_rate that pyannote can use directly.
    Uses soundfile as backend to avoid torchcodec dependency.
    """
    # Load audio using soundfile (no torchcodec dependency)
    waveform_np, sample_rate = sf.read(audio_path, dtype='float32')

    # Convert to torch tensor with shape (channels, samples)
    waveform = torch.from_numpy(waveform_np)
    if waveform.ndim == 1:
        # Mono audio: add channel dimension
        waveform = waveform.unsqueeze(0)
    else:
        # Multi-channel: transpose from (samples, channels) to (channels, samples)
        waveform = waveform.T

    return {"waveform": waveform, "sample_rate": sample_rate}


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def get_speaker_at_time(diarization_tracks: list, start: float, end: float) -> str:
    """
    Get the dominant speaker in a given time segment.

    Args:
        diarization_tracks: List of (segment, track, speaker) tuples
        start: Start time in seconds
        end: End time in seconds
    """
    segment = Segment(start, end)
    speakers = {}

    for turn, _, speaker in diarization_tracks:
        # Calculate overlap duration
        overlap = segment & turn
        if overlap:
            duration = overlap.duration
            speakers[speaker] = speakers.get(speaker, 0) + duration

    if not speakers:
        return "UNKNOWN"

    # Return the speaker with the largest overlap
    return max(speakers, key=speakers.get)


def transcribe_with_diarization(
    audio_path: str,
    whisper_model: str = "large-v3",
    hf_token: str = None,
    device: str = None,
    language: str = None,
) -> list[dict]:
    """
    Perform ASR + Speaker Diarization

    Args:
        audio_path: Path to the audio/video file
        whisper_model: Whisper model name
        hf_token: Hugging Face token
        device: Device to run on
        language: Language code (optional)

    Returns:
        List of dicts containing timestamp, speaker, and text
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # Handle video files: extract audio first
    temp_audio_path = None
    if is_video_file(audio_path):
        temp_audio_path = extract_audio_from_video(audio_path)
        working_audio_path = temp_audio_path
    else:
        working_audio_path = audio_path

    try:
        print(f"Loading Whisper model: {whisper_model}")

        # Load Whisper model
        model = whisper.load_model(whisper_model, device=device)

        # Perform ASR
        print("Transcribing audio...")
        transcribe_options = {"word_timestamps": True}
        if language:
            transcribe_options["language"] = language

        result = model.transcribe(working_audio_path, **transcribe_options)

        # Load pyannote diarization pipeline
        print("Loading pyannote diarization pipeline...")
        # Try to get token from environment variable
        diarization_pipeline = Pipeline.from_pretrained(
            "../HF/hub/models--pyannote--speaker-diarization-community-1/snapshots/3533c8cf8e369892e6b79ff1bf80f7b0286a54ee/"
        )

        diarization_pipeline = diarization_pipeline.to(torch.device(device))

        # Perform speaker diarization
        # Pre-load audio to bypass torchcodec compatibility issues with PyTorch 2.9
        print("Loading audio for diarization...")
        audio_data = load_audio_for_pyannote(working_audio_path)

        print("Performing speaker diarization...")
        diarization = diarization_pipeline(audio_data)

        # Convert diarization output to a list of tracks for reuse
        # Handles both Annotation objects (with itertracks) and DiarizeOutput wrapper
        if hasattr(diarization, 'itertracks'):
            # Direct Annotation object
            diarization_tracks = list(diarization.itertracks(yield_label=True))
        elif hasattr(diarization, 'speaker_diarization'):
            # DiarizeOutput wrapper - use speaker_diarization attribute
            diarization_tracks = list(diarization.speaker_diarization.itertracks(yield_label=True))
        else:
            raise TypeError(f"Unknown diarization output type: {type(diarization)}")

        # Merge results
        print("Merging results...")
        segments_with_speakers = []

        for segment in result["segments"]:
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip()

            if not text:
                continue

            speaker = get_speaker_at_time(diarization_tracks, start, end)

            segments_with_speakers.append({
                "start": start,
                "end": end,
                "speaker": speaker,
                "text": text
            })

        return segments_with_speakers

    finally:
        # Clean up temporary audio file if created
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            print(f"Cleaned up temporary audio file: {temp_audio_path}")


def merge_same_speaker_segments(segments: list[dict], max_gap: float = 1.0) -> list[dict]:
    """
    Merge consecutive segments from the same speaker

    Args:
        segments: Original segment list
        max_gap: Maximum time gap allowed for merging (seconds)
    """
    if not segments:
        return []

    merged = [segments[0].copy()]

    for seg in segments[1:]:
        last = merged[-1]

        # Merge if same speaker and time gap is within threshold
        if (seg["speaker"] == last["speaker"] and
            seg["start"] - last["end"] <= max_gap):
            last["end"] = seg["end"]
            last["text"] = last["text"] + " " + seg["text"]
        else:
            merged.append(seg.copy())

    return merged


def write_srt(segments: list[dict], output_path: str, include_speaker: bool = True):
    """
    Write SRT subtitle file

    Args:
        segments: Segment list
        output_path: Output file path
        include_speaker: Whether to include speaker labels in subtitles
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start_time = format_timestamp(seg["start"])
            end_time = format_timestamp(seg["end"])

            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")

            if include_speaker:
                f.write(f"[{seg['speaker']}] {seg['text']}\n")
            else:
                f.write(f"{seg['text']}\n")

            f.write("\n")

    print(f"SRT file saved to: {output_path}")


def write_txt(segments: list[dict], output_path: str):
    """Write plain text file with speaker labels"""
    with open(output_path, "w", encoding="utf-8") as f:
        current_speaker = None
        for seg in segments:
            if seg["speaker"] != current_speaker:
                current_speaker = seg["speaker"]
                f.write(f"\n[{current_speaker}]\n")
            f.write(f"{seg['text']}\n")

    print(f"TXT file saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="ASR + Speaker Diarization: Whisper + pyannote"
    )
    parser.add_argument(
        "audio",
        type=str,
        help="Input audio/video file path"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output SRT file path (default: input_filename.srt)"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model size (default: large-v3)"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token (or set HF_TOKEN environment variable)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (cuda/cpu, default: auto-detect)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code (e.g., zh, en, ja)"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge consecutive segments from the same speaker"
    )
    parser.add_argument(
        "--merge-gap",
        type=float,
        default=1.0,
        help="Maximum time gap for merging (seconds, default: 1.0)"
    )
    parser.add_argument(
        "--no-speaker",
        action="store_true",
        help="Do not include speaker labels in SRT"
    )
    parser.add_argument(
        "--txt",
        action="store_true",
        help="Also output TXT format"
    )

    args = parser.parse_args()

    # Check input file exists
    if not os.path.exists(args.audio):
        print(f"Error: File not found: {args.audio}")
        return 1

    # Set output path
    if args.output is None:
        audio_path = Path(args.audio)
        args.output = str(audio_path.with_suffix(".srt"))

    # Get HF token
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # Perform transcription and diarization
    segments = transcribe_with_diarization(
        audio_path=args.audio,
        whisper_model=args.model,
        hf_token=hf_token,
        device=args.device,
        language=args.language,
    )

    # Merge consecutive segments from the same speaker
    if args.merge:
        segments = merge_same_speaker_segments(segments, max_gap=args.merge_gap)

    # Write SRT file
    write_srt(segments, args.output, include_speaker=not args.no_speaker)

    # Optional: write TXT file
    if args.txt:
        txt_path = str(Path(args.output).with_suffix(".txt"))
        write_txt(segments, txt_path)

    print("Done!")
    return 0


if __name__ == "__main__":
    exit(main())