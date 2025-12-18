"""
ASR (Automatic Speech Recognition) module with speaker diarization.

This module provides audio transcription using Whisper and speaker diarization
using pyannote.audio. It supports SRT subtitle generation with speaker labels.
"""

import os
import subprocess
from typing import Optional, List, Dict, Any

import torch
import soundfile as sf
import whisper
from pyannote.audio import Pipeline
from pyannote.core import Segment

from vca import config


def _ensure_dir(path: str) -> None:
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)


def format_srt_timestamp(milliseconds: int) -> str:
    """Convert milliseconds (int) to SRT timestamp HH:MM:SS,mmm."""
    ms = int(milliseconds)
    tail = ms % 1000
    s = ms // 1000
    mi = s // 60
    s = s % 60
    h = mi // 60
    mi = mi % 60
    h = f"{h:02d}"
    mi = f"{mi:02d}"
    s = f"{s:02d}"
    tail = f"{tail:03d}"
    return f"{h}:{mi}:{s},{tail}"


def write_srt_from_sentence_info(
    sentence_info: List[Dict[str, Any]],
    srt_path: str,
    include_speaker: bool = True
) -> None:
    """
    Write SRT file from sentence_info structure.
    Each sentence_info item should have 'text', 'timestamp', and optionally 'speaker' fields.
    timestamp is a list of [word, start_ms, end_ms] for each word.

    Args:
        sentence_info: List of sentence dicts with 'text', 'timestamp', and optionally 'speaker'
        srt_path: Output path for the SRT file
        include_speaker: Whether to include speaker labels in the subtitles
    """
    _ensure_dir(os.path.dirname(srt_path) or ".")
    with open(srt_path, "w", encoding="utf-8") as f:
        for idx, sent in enumerate(sentence_info):
            text = sent.get('text', '')
            timestamp = sent.get('timestamp', [])
            speaker = sent.get('speaker', None)

            if not timestamp:
                continue

            # Get start and end time from word-level timestamps
            start_ms = int(timestamp[0][1]) if len(timestamp[0]) >= 2 else 0
            end_ms = int(timestamp[-1][2]) if len(timestamp[-1]) >= 3 else start_ms

            # Clean up text (remove trailing punctuation)
            text = text.rstrip("")

            # Write SRT entry
            f.write(f"{idx + 1}\n")
            f.write(f"{format_srt_timestamp(start_ms)} --> {format_srt_timestamp(end_ms)}\n")

            # Include speaker label if available and requested
            if include_speaker and speaker:
                f.write(f"[{speaker}] {text}\n\n")
            else:
                f.write(f"{text}\n\n")


def extract_audio_wav_16k(
    video_path: str,
    audio_path: str,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None
) -> None:
    """Extract mono 16k PCM WAV from video using ffmpeg."""
    cmd: List[str] = ["ffmpeg", "-y"]
    if start_sec is not None:
        cmd += ["-ss", str(float(start_sec))]
    cmd += ["-i", video_path]
    if end_sec is not None:
        cmd += ["-to", str(float(end_sec))]
    cmd += ["-vn", "-ac", "1", "-ar", "16000", "-f", "wav", audio_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def load_audio_for_pyannote(audio_path: str) -> dict:
    """
    Pre-load audio file to bypass torchcodec compatibility issues.
    Returns a dict with waveform and sample_rate that pyannote can use directly.
    Uses soundfile as backend to avoid torchcodec dependency.
    """
    waveform_np, sample_rate = sf.read(audio_path, dtype='float32')
    waveform = torch.from_numpy(waveform_np)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.T
    return {"waveform": waveform, "sample_rate": sample_rate}


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
        overlap = segment & turn
        if overlap:
            duration = overlap.duration
            speakers[speaker] = speakers.get(speaker, 0) + duration

    if not speakers:
        return "UNKNOWN"

    return max(speakers, key=speakers.get)


def merge_same_speaker_segments(
    segments: List[Dict[str, Any]],
    max_gap: float = 1.0
) -> List[Dict[str, Any]]:
    """
    Merge consecutive segments from the same speaker.

    Args:
        segments: Original segment list
        max_gap: Maximum time gap allowed for merging (seconds)
    """
    if not segments:
        return []

    merged = [segments[0].copy()]

    for seg in segments[1:]:
        last = merged[-1]
        if (seg.get("speaker") == last.get("speaker") and
            seg["start"] - last["end"] <= max_gap):
            last["end"] = seg["end"]
            last["text"] = last["text"] + " " + seg["text"]
        else:
            merged.append(seg.copy())

    return merged


def transcribe_audio_with_diarization(
    audio_path: str,
    model_name: str,
    device: str,
    language: Optional[str] = None,
    asr_kwargs: Optional[Dict[str, Any]] = None,
    merge_segments: Optional[bool] = None,
    merge_gap: Optional[float] = None,
    diarization_model_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run Whisper ASR + pyannote speaker diarization on a single audio file.

    Returns a dict with keys:
      - text: full transcription
      - sentence_info: list of sentence dicts with 'text', 'timestamp', and 'speaker' fields
      - segments: raw segments with speaker information

    This uses native Whisper for ASR and pyannote for speaker diarization.
    Parameters default to values from config if not specified.
    """
    # Use config defaults if not specified
    if merge_segments is None:
        merge_segments = getattr(config, 'ASR_MERGE_SAME_SPEAKER', True)
    if merge_gap is None:
        merge_gap = getattr(config, 'ASR_MERGE_GAP', 1.0)
    if diarization_model_path is None:
        diarization_model_path = getattr(config, 'ASR_DIARIZATION_MODEL_PATH',
            "../HF/hub/models--pyannote--speaker-diarization-community-1/snapshots/3533c8cf8e369892e6b79ff1bf80f7b0286a54ee/")

    print(f"[ASR] Using device: {device}")

    # Load Whisper model
    print(f"[ASR] Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name, device=device)

    # Base transcription kwargs with anti-hallucination settings from config
    transcribe_options: Dict[str, Any] = {
        "word_timestamps": True,
        # Anti-hallucination parameters from config
        "no_speech_threshold": getattr(config, 'ASR_NO_SPEECH_THRESHOLD', 0.6),
        "logprob_threshold": getattr(config, 'ASR_LOGPROB_THRESHOLD', -1.0),
        "compression_ratio_threshold": getattr(config, 'ASR_COMPRESSION_RATIO_THRESHOLD', 2.4),
        "condition_on_previous_text": getattr(config, 'ASR_CONDITION_ON_PREVIOUS_TEXT', True),
    }
    if language:
        transcribe_options["language"] = language

    # Merge user-provided kwargs (can override defaults)
    if asr_kwargs:
        transcribe_options.update(asr_kwargs)

    print(f"[ASR] Transcribe options: no_speech_threshold={transcribe_options.get('no_speech_threshold')}, "
          f"logprob_threshold={transcribe_options.get('logprob_threshold')}, "
          f"compression_ratio_threshold={transcribe_options.get('compression_ratio_threshold')}")

    # Perform ASR
    print("[ASR] Transcribing audio...")
    result = model.transcribe(audio_path, **transcribe_options)

    # Load pyannote diarization pipeline
    print("[ASR] Loading pyannote diarization pipeline...")
    diarization_pipeline = Pipeline.from_pretrained(diarization_model_path)
    diarization_pipeline = diarization_pipeline.to(torch.device(device))

    # Pre-load audio to bypass torchcodec compatibility issues
    print("[ASR] Loading audio for diarization...")
    audio_data = load_audio_for_pyannote(audio_path)

    # Perform speaker diarization
    print("[ASR] Performing speaker diarization...")
    diarization = diarization_pipeline(audio_data)

    # Convert diarization output to a list of tracks
    if hasattr(diarization, 'itertracks'):
        diarization_tracks = list(diarization.itertracks(yield_label=True))
    elif hasattr(diarization, 'speaker_diarization'):
        diarization_tracks = list(diarization.speaker_diarization.itertracks(yield_label=True))
    else:
        raise TypeError(f"Unknown diarization output type: {type(diarization)}")

    # Merge results: ASR segments + speaker information
    print("[ASR] Merging ASR and diarization results...")
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

    # Optionally merge consecutive segments from the same speaker
    if merge_segments:
        segments_with_speakers = merge_same_speaker_segments(segments_with_speakers, max_gap=merge_gap)

    # Extract full text
    full_text = result.get("text", "")

    # Convert to sentence_info format for compatibility
    sentence_info = []
    for seg in segments_with_speakers:
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)

        sentence_info.append({
            "text": seg["text"],
            "speaker": seg.get("speaker", "UNKNOWN"),
            "timestamp": [[seg["text"], start_ms, end_ms]]
        })

    return {
        "text": full_text,
        "sentence_info": sentence_info,
        "segments": segments_with_speakers
    }


def run_asr(
    video_path: str,
    output_dir: str,
    srt_path: Optional[str] = None,
    asr_model: str = "base",
    asr_device: str = "cuda:0",
    asr_language: Optional[str] = None,
    asr_kwargs: Optional[Dict[str, Any]] = None,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    keep_extracted_audio: bool = False,
    include_speaker: bool = True
) -> Dict[str, Any]:
    """
    High-level ASR function: extract audio, transcribe, and generate SRT file.

    Args:
        video_path: Path to the video file.
        output_dir: Directory to save intermediate files (audio).
        srt_path: Optional output path for the SRT file. Defaults to video basename with .srt.
        asr_model: Whisper model name: "tiny", "base", "small", "medium", "large", "large-v2", "large-v3".
        asr_device: Device for ASR model, e.g., "cuda:0" or "cpu".
        asr_language: Language code for ASR (e.g., "zh", "en"). If None, auto-detect.
        asr_kwargs: Extra kwargs forwarded to whisper's transcribe() method.
        start_sec: Optional start time in seconds.
        end_sec: Optional end time in seconds.
        keep_extracted_audio: If True, keep the temporary extracted WAV file on disk.
        include_speaker: Whether to include speaker labels in the subtitles.

    Returns:
        dict with keys: {"srt_path", "sentence_info", "segments", "text"}
    """
    _ensure_dir(output_dir)

    # Determine output SRT path
    final_srt_path = srt_path or (os.path.splitext(video_path)[0] + ".srt")

    result: Dict[str, Any] = {}

    # Check if SRT file already exists
    if os.path.exists(final_srt_path):
        print(f"[Skip] Found existing SRT file at {final_srt_path}, skipping ASR")
        result["srt_path"] = final_srt_path
        return result

    # Extract audio to temporary WAV inside output_dir for locality
    audio_wav_path = os.path.join(output_dir, "audio_16k_mono.wav")
    print(f"[ASR] Extracting audio from {video_path} to {audio_wav_path}")
    extract_audio_wav_16k(video_path, audio_wav_path, start_sec, end_sec)

    # Run Whisper ASR + pyannote speaker diarization
    asr_output = transcribe_audio_with_diarization(
        audio_wav_path, asr_model, asr_device, asr_language, asr_kwargs
    )

    sentence_info = asr_output.get("sentence_info", [])

    # Adjust timestamps if we extracted a clip (add start_sec offset)
    if start_sec is not None and start_sec > 0:
        offset_ms = int(float(start_sec) * 1000)
        adjusted_sentence_info = []
        for sent in sentence_info:
            adjusted_sent = sent.copy()
            # Adjust word-level timestamps in the timestamp array
            if 'timestamp' in sent:
                adjusted_timestamps = []
                for ts in sent['timestamp']:
                    # Format: [word, start_ms, end_ms]
                    if isinstance(ts, (list, tuple)) and len(ts) >= 3:
                        adjusted_timestamps.append([
                            ts[0],  # word
                            ts[1] + offset_ms,  # start_ms + offset
                            ts[2] + offset_ms   # end_ms + offset
                        ])
                    else:
                        adjusted_timestamps.append(ts)
                adjusted_sent['timestamp'] = adjusted_timestamps
            adjusted_sentence_info.append(adjusted_sent)
        sentence_info = adjusted_sentence_info

    # Write SRT file using sentence_info
    write_srt_from_sentence_info(sentence_info, final_srt_path, include_speaker=include_speaker)
    print(f"[ASR] SRT file saved to {final_srt_path}")

    result["srt_path"] = final_srt_path
    result["sentence_info"] = sentence_info
    result["text"] = asr_output.get("text", "")

    # Also save raw segments with word-level timestamps
    if "segments" in asr_output:
        result["segments"] = asr_output["segments"]

    if not keep_extracted_audio:
        try:
            os.remove(audio_wav_path)
        except OSError:
            pass

    return result
