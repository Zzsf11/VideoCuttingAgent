import os
import subprocess
import argparse
from typing import Optional, List, Dict, Any
import uuid

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _format_srt_timestamp(milliseconds: int) -> str:
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

def _write_srt_from_sentence_info(sentence_info: List[Dict[str, Any]], srt_path: str) -> None:
    """
    Write SRT file from sentence_info structure.
    Each sentence_info item should have 'text' and 'timestamp' fields.
    timestamp is a list of [word, start_ms, end_ms] for each word.
    """
    _ensure_dir(os.path.dirname(srt_path) or ".")
    with open(srt_path, "w", encoding="utf-8") as f:
        for idx, sent in enumerate(sentence_info):
            text = sent.get('text', '')
            timestamp = sent.get('timestamp', [])
            
            if not timestamp:
                continue
            
            # Get start and end time from word-level timestamps
            start_ms = int(timestamp[0][1]) if len(timestamp[0]) >= 2 else 0
            end_ms = int(timestamp[-1][2]) if len(timestamp[-1]) >= 3 else start_ms
            
            # Clean up text (remove trailing punctuation)
            text = text.rstrip("、。，")
            
            # Write SRT entry
            f.write(f"{idx + 1}\n")
            f.write(f"{_format_srt_timestamp(start_ms)} --> {_format_srt_timestamp(end_ms)}\n")
            f.write(f"{text}\n\n")

def _extract_audio_wav_16k(video_path: str, audio_path: str, start_sec: Optional[float] = None, end_sec: Optional[float] = None) -> None:
    """Extract mono 16k PCM WAV from video using ffmpeg."""
    cmd: List[str] = ["ffmpeg", "-y"]
    
    # Input seeking (fast)
    if start_sec is not None:
        cmd += ["-ss", str(float(start_sec))]
    
    cmd += ["-i", video_path]
    
    # Duration limiting
    if end_sec is not None:
        if start_sec is not None:
            # When using input seeking (-ss before -i), timestamps are reset to 0.
            # So -to would refer to the timestamp relative to the new start.
            # If we want to stop at absolute time end_sec, we need to calculate duration.
            duration = float(end_sec) - float(start_sec)
            if duration < 0:
                raise ValueError(f"End time ({end_sec}) must be greater than start time ({start_sec})")
            cmd += ["-t", str(duration)]
        else:
            # No input seeking, timestamps are absolute (or start at 0).
            cmd += ["-to", str(float(end_sec))]
            
    cmd += ["-vn", "-ac", "1", "-ar", "16000", "-f", "wav", audio_path]
    
    print(f"Running ffmpeg command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def _get_custom_vad_segments(
    audio_path: str,
    method: str = "silero:6.0",
    min_speech_duration: float = 0.1,
    min_silence_duration: float = 1.0,
    dilatation: float = 0.5,
) -> List[tuple]:
    """
    Get custom VAD segments with adjustable parameters.
    
    Args:
        audio_path: Path to audio file
        method: VAD method (e.g., "silero:6.0", "silero:3.1", "auditok")
        min_speech_duration: Minimum duration (sec) of speech segments (lower = more sensitive)
        min_silence_duration: Minimum duration (sec) of silence to split (lower = more splits)
        dilatation: How much (sec) to expand each speech segment (lower = tighter bounds)
    
    Returns:
        List of (start, end) tuples in seconds
    """
    try:
        import whisper_timestamped as whisper
        from whisper_timestamped.transcribe import get_vad_segments, get_audio_tensor
    except ImportError:
        raise ImportError(
            "whisper-timestamped is not installed. "
            "Install it with: pip install whisper-timestamped"
        )
    
    print(f"Running VAD with: method={method}, min_speech={min_speech_duration}s, "
          f"min_silence={min_silence_duration}s, dilatation={dilatation}s")
    
    audio = whisper.load_audio(audio_path)
    audio_tensor = get_audio_tensor(audio)
    
    vad_segments = get_vad_segments(
        audio_tensor,
        output_sample=False,  # Return in seconds
        min_speech_duration=min_speech_duration,
        min_silence_duration=min_silence_duration,
        dilatation=dilatation,
        method=method,
    )
    
    # Convert to list of tuples
    segments = [(seg["start"], seg["end"]) for seg in vad_segments]
    print(f"VAD detected {len(segments)} speech segments")
    
    return segments


def _transcribe_audio_with_whisper_timestamped(
    audio_path: str,
    model_name: str,
    device: str,
    language: Optional[str] = None,
    asr_kwargs: Optional[Dict[str, Any]] = None,
    vad_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run whisper-timestamped on a single audio file.
    
    Args:
        audio_path: Path to audio file
        model_name: Whisper model name
        device: Device to use
        language: Language code (optional)
        asr_kwargs: Additional kwargs for transcription
        vad_params: VAD parameters (min_speech_duration, min_silence_duration, dilatation)
    """
    try:
        import whisper_timestamped as whisper
    except ImportError:
        raise ImportError(
            "whisper-timestamped is not installed. "
            "Install it with: pip install whisper-timestamped"
        )
    
    print(f"Loading model {model_name} on {device}...")
    model = whisper.load_model(model_name, device=device)
    
    audio = whisper.load_audio(audio_path)
    
    transcribe_kwargs: Dict[str, Any] = {
        "language": language,
        "task": "transcribe",
        # "vad": "silero:6.0",  # Default: no VAD
        "vad": False,
        "detect_disfluencies": False,
        "compute_word_confidence": True,
        "condition_on_previous_text": False,
        "no_speech_threshold": 0.6,
        "logprob_threshold": -1.0,
        "compression_ratio_threshold": 2.4,
    }
    
    # Handle custom VAD parameters
    if vad_params is not None:
        vad_method = vad_params.get("method", "silero:6.0")
        custom_segments = _get_custom_vad_segments(
            audio_path,
            method=vad_method,
            min_speech_duration=vad_params.get("min_speech_duration", 0.1),
            min_silence_duration=vad_params.get("min_silence_duration", 1.0),
            dilatation=vad_params.get("dilatation", 0.5),
        )
        # Pass custom VAD segments as a list of timestamps
        transcribe_kwargs["vad"] = custom_segments
    
    if asr_kwargs:
        transcribe_kwargs.update(asr_kwargs)
    
    print(f"Transcribing (kwargs={transcribe_kwargs})...")
    result = whisper.transcribe(model, audio, **transcribe_kwargs)
    
    full_text = result.get("text", "")
    sentence_info = []
    segments = result.get("segments", [])
    
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        
        words = seg.get("words", [])
        
        timestamp = []
        for word_info in words:
            word_text = word_info.get("text", "")
            word_start_ms = int(word_info.get("start", 0) * 1000)
            word_end_ms = int(word_info.get("end", 0) * 1000)
            timestamp.append([word_text, word_start_ms, word_end_ms])
        
        if not timestamp:
            seg_start_ms = int(seg.get("start", 0) * 1000)
            seg_end_ms = int(seg.get("end", 0) * 1000)
            timestamp = [[text, seg_start_ms, seg_end_ms]]
        
        sentence_info.append({
            "text": text,
            "timestamp": timestamp
        })
    
    return {
        "text": full_text,
        "sentence_info": sentence_info,
        "segments": segments
    }

def parse_time_string(time_str: Optional[str]) -> Optional[float]:
    """Parse time string (HH:MM:SS, MM:SS, or SS) to seconds."""
    if not time_str:
        return None
    try:
        parts = list(map(float, time_str.split(':')))
        if len(parts) == 1:
            return parts[0]
        elif len(parts) == 2:
            return parts[0] * 60 + parts[1]
        elif len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        else:
            raise ValueError(f"Invalid time format: {time_str}")
    except ValueError:
         raise ValueError(f"Invalid time format: {time_str}")

def run_asr(
    video_path: str,
    srt_path: Optional[str] = None,
    asr_model: str = "base",
    asr_device: str = "cuda:0",
    asr_language: Optional[str] = None,
    asr_kwargs: Optional[Dict[str, Any]] = None,
    keep_extracted_audio: bool = False,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    vad_params: Optional[Dict[str, Any]] = None,
):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    final_srt_path = srt_path or (os.path.splitext(video_path)[0] + ".srt")

    # Use a temporary audio file in the same directory as the video or a temp dir
    video_dir = os.path.dirname(os.path.abspath(video_path))
    # Use a unique filename to avoid conflicts when running multiple jobs
    unique_suffix = uuid.uuid4().hex
    audio_wav_path = os.path.join(video_dir, f"temp_audio_16k_mono_{unique_suffix}.wav")
    
    try:
        print(f"Extracting audio from {video_path}...")
        _extract_audio_wav_16k(video_path, audio_wav_path, start_sec=start_sec, end_sec=end_sec)
        
        asr_output = _transcribe_audio_with_whisper_timestamped(
            audio_wav_path, asr_model, asr_device, asr_language, asr_kwargs, vad_params
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
        
        print(f"Writing SRT to {final_srt_path}...")
        _write_srt_from_sentence_info(sentence_info, final_srt_path)
        print("Done.")
        
    finally:
        if not keep_extracted_audio and os.path.exists(audio_wav_path):
            os.remove(audio_wav_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract audio from video and generate SRT using Whisper.")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--output", "-o", help="Path to the output SRT file (optional)")
    parser.add_argument("--asr_model", default="base", help="Whisper model size (default: base)")
    parser.add_argument("--asr_device", default="cuda:0", help="Device to use (default: cuda:0)")
    parser.add_argument("--asr_language", help="Language code (optional, auto-detect if not set)")
    parser.add_argument("--keep_extracted_audio", action="store_true", help="Keep the extracted temporary WAV file")
    parser.add_argument("--start", type=str, help="Start time (seconds or HH:MM:SS)")
    parser.add_argument("--end", type=str, help="End time (seconds or HH:MM:SS)")
    
    # VAD options
    parser.add_argument("--vad", action="store_true", help="Enable Voice Activity Detection (VAD) to reduce hallucinations in silence/music")
    parser.add_argument("--vad_min_speech", type=float, default=0.1,
                        help="VAD: Minimum speech duration in seconds (lower = more sensitive, default: 0.1)")
    parser.add_argument("--vad_min_silence", type=float, default=1.0,
                        help="VAD: Minimum silence duration to split in seconds (lower = more splits, default: 1.0)")
    parser.add_argument("--vad_dilatation", type=float, default=0.5,
                        help="VAD: Expand speech segments by this amount in seconds (default: 0.5)")
    parser.add_argument("--vad_method", type=str, default="silero:6.0",
                        help="VAD method: silero:6.0, silero:3.1, auditok (default: silero:6.0)")

    args = parser.parse_args()
    
    asr_kwargs = {}
    vad_params = None
    
    if args.vad:
        vad_params = {
            "method": args.vad_method,
            "min_speech_duration": args.vad_min_speech,
            "min_silence_duration": args.vad_min_silence,
            "dilatation": args.vad_dilatation,
        }

    run_asr(
        video_path=args.video_path,
        srt_path=args.output,
        asr_model=args.asr_model,
        asr_device=args.asr_device,
        asr_language=args.asr_language,
        asr_kwargs=asr_kwargs,
        keep_extracted_audio=args.keep_extracted_audio,
        start_sec=parse_time_string(args.start),
        end_sec=parse_time_string(args.end),
        vad_params=vad_params,
    )
