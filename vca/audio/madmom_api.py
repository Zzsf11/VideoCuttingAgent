"""Lightweight, non-Gradio API for Madmom-based audio keypoint detection.

This module exists so you can call Madmom detection from scripts (e.g. local_run.py)
without importing any UI code.

Important:
- Do NOT import `madmom` directly from Python 3.10+/NumPy 1.24+ environments.
  Use `vca.audio.audio_Madmom` which applies compatibility monkey-patches before
  importing madmom.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _ensure_wav_for_aubio(audio_path: str) -> str:
    """Ensure the given audio path is a WAV file readable by aubio.

    Aubio's `source_wavread` requires a RIFF/WAV container. Gradio uploads are
    often mp3; in that case we convert to a sibling WAV file and reuse it.

    Returns the path to a WAV file (may be the original path).
    """

    p = Path(audio_path)
    if p.suffix.lower() in {".wav", ".wave"}:
        return audio_path

    # Write next to the input (Gradio's /tmp directory is writable and scoped)
    out_path = p.with_name(f"{p.stem}__vca.wav")

    try:
        src_mtime = p.stat().st_mtime
        if out_path.exists() and out_path.stat().st_mtime >= src_mtime:
            return str(out_path)
    except Exception:
        # If stat fails for any reason, just try converting.
        pass

    # Prefer ffmpeg if available (preserves sample rate; robust for mp3/aac/etc.)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(p),
            "-vn",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            str(out_path),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0 or (not out_path.exists()):
            raise RuntimeError(
                "ffmpeg conversion to wav failed. "
                f"input={audio_path} output={out_path}.\n{proc.stderr.strip()}"
            )
        return str(out_path)

    # Fallback: decode via our pure-python loader and write a PCM wav.
    try:
        import soundfile as sf  # noqa: WPS433 (runtime import)
        from vca.audio.audio_utils import load_audio_no_librosa  # noqa: WPS433 (runtime import)

        target_sr = 16000
        audio = load_audio_no_librosa(audio_path, sr=target_sr)
        if audio is None:
            raise RuntimeError("decoded audio is None")
        sf.write(str(out_path), audio, target_sr, subtype="PCM_16")
        return str(out_path)
    except Exception as e:
        raise RuntimeError(
            "Input audio is not WAV and ffmpeg is not available to convert it. "
            "Please install ffmpeg or provide a .wav file. "
            f"audio={audio_path} error={e}"
        )


def detect_keypoints_madmom(
    audio_path: str,
    *,
    # Detection method selection (new interface)
    detection_method: str = "downbeat",  # "downbeat", "pitch", "mel_energy"

    # DEPRECATED: The following parameters are ignored and kept only for backward compatibility.
    # They will be removed in a future version. Do not use them.
    detect_onset: bool = True,
    detect_downbeat: bool = True,
    onset_threshold: float = 0.6,  # DEPRECATED

    # Downbeat-specific parameters
    beats_per_bar: int = 4,
    dbn_threshold: float = 0.05,
    min_bpm: float = 55.0,
    max_bpm: float = 215.0,
    num_tempi: int = 60,
    transition_lambda: float = 100.0,
    observation_lambda: int = 16,
    correct_beats: bool = True,
    fps: int = 100,
    
    # Pitch-specific parameters
    pitch_tolerance: float = 0.8,
    pitch_threshold: float = 0.8,
    pitch_min_distance: float = 0.5,
    pitch_nms_method: str = "basic",
    pitch_max_points: int = 20,
    
    # Mel energy-specific parameters
    mel_threshold_ratio: float = 0.3,
    mel_min_distance: float = 0.5,
    mel_nms_method: str = "basic",
    mel_win_s: int = 512,
    mel_n_filters: int = 40,
    mel_max_points: int = 20,
    
    # Post filtering (applies to all methods)
    min_interval: float = 0.0,
    top_k: int = 0,
    energy_percentile: float = 0.0,
    
    # Silence gating (prevents points in silent regions)
    silence_filter: bool = True,
    silence_threshold_db: float = -45.0,
    silence_ref: str = "max",
    silence_pad: float = 0.05,
    silence_min_silence_len: float = 0.15,
    silence_min_sound_len: float = 0.05,
    
    # Misc
    return_python_types: bool = True,
) -> Dict[str, Any]:
    """Run Madmom keypoint detection and return a result dict.

    Returns a dict compatible with the interactive UI detector outputs.

    Core params exposed:
    - detection_method: "downbeat", "pitch", or "mel_energy"
    - For downbeat: onset_threshold, beats_per_bar, dbn_threshold
    - For pitch: pitch_tolerance, pitch_threshold, pitch_min_distance
    - For mel_energy: mel_threshold_ratio, mel_min_distance
    - silence_*: suppress keypoints in silent regions (recommended)
    """

    # Ensure aubio can read the audio. (mp3 uploads are common in Gradio.)
    analysis_audio_path = _ensure_wav_for_aubio(audio_path)

    # Import from the patched module (this applies Python/NumPy compatibility fixes
    # *before* importing madmom).
    from vca.audio.audio_Madmom import (  # noqa: WPS433 (runtime import)
        SensoryKeypointDetector,
        filter_significant_keypoints,
    )

    # Create detector with method-specific parameters
    detector = SensoryKeypointDetector(
        detection_method=detection_method,
        # Downbeat parameters (always passed, used only if method="downbeat")
        beats_per_bar=[int(beats_per_bar)],
        dbn_threshold=float(dbn_threshold),
        min_bpm=float(min_bpm),
        max_bpm=float(max_bpm),
        num_tempi=int(num_tempi),
        transition_lambda=float(transition_lambda),
        observation_lambda=int(observation_lambda),
        correct_beats=bool(correct_beats),
        fps=int(fps),
        # Pitch parameters (used only if method="pitch")
        pitch_tolerance=float(pitch_tolerance),
        pitch_threshold=float(pitch_threshold),
        pitch_min_distance=float(pitch_min_distance),
        pitch_nms_method=str(pitch_nms_method),
        pitch_max_points=int(pitch_max_points),
        # Mel parameters (used only if method="mel_energy")
        mel_win_s=int(mel_win_s),
        mel_n_filters=int(mel_n_filters),
        mel_threshold_ratio=float(mel_threshold_ratio),
        mel_min_distance=float(mel_min_distance),
        mel_nms_method=str(mel_nms_method),
        mel_max_points=int(mel_max_points),
    )

    result = detector.analyze(analysis_audio_path)

    # Apply silence filtering if requested
    if silence_filter and result.get("keypoints"):
        intervals = _compute_non_silent_intervals(
            analysis_audio_path,
            threshold_db=silence_threshold_db,
            ref=silence_ref,
            pad=silence_pad,
            min_silence_len=silence_min_silence_len,
            min_sound_len=silence_min_sound_len,
        )
        if intervals:
            result["keypoints"] = _filter_events_by_intervals(result.get("keypoints", []), intervals)
            for key in ("downbeats", "onsets", "timestamps"):
                if result.get(key) is not None:
                    result[key] = _filter_event_times_by_intervals(result.get(key), intervals)

    # Apply post-filtering (only if needed)
    if min_interval > 0 or top_k > 0 or energy_percentile > 0:
        result["keypoints"] = filter_significant_keypoints(
            result.get("keypoints", []),
            min_interval=min_interval,
            top_k=int(top_k),
            energy_percentile=energy_percentile,
            use_normalized_intensity=True,
        )

    # Normalize outputs to plain Python types (useful for JSON serialization)
    if return_python_types:
        normalized: Dict[str, Any] = {
            "success": True,
            "keypoints": result.get("keypoints", []),
            "meta": result.get("meta", {}),
        }

        def _to_list(value: Any) -> Any:
            if value is None:
                return []
            if hasattr(value, "tolist"):
                return value.tolist()
            return list(value) if isinstance(value, (tuple, set)) else value

        for key in ("downbeats", "onsets"):
            normalized[key] = _to_list(result.get(key))

        return normalized

    return result


def detect_keypoints_madmom_from_params(
    audio_path: str,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convenience wrapper: accept a params dict (e.g. loaded from UI settings)."""
    params = params or {}
    return detect_keypoints_madmom(audio_path, **params)


def _compute_non_silent_intervals(
    audio_path: str,
    *,
    sr: int = 16000,
    frame_length: int = 2048,
    hop_length: int = 512,
    threshold_db: float = -45.0,
    ref: str = "max",
    pad: float = 0.05,
    min_silence_len: float = 0.15,
    min_sound_len: float = 0.05,
) -> List[Tuple[float, float]]:
    """Compute non-silent intervals in seconds.

    This is a lightweight guardrail to prevent keypoints from appearing
    in silent regions where onset models may still fire due to noise.
    """

    try:
        from vca.audio.audio_utils import load_audio_no_librosa  # noqa: WPS433 (runtime import)

        audio = load_audio_no_librosa(audio_path, sr=sr)
    except Exception:
        return []

    if audio is None or len(audio) == 0:
        return []

    import math
    import numpy as np

    eps = 1e-12
    if len(audio) < frame_length:
        rms = float((audio.astype("float32") ** 2).mean() ** 0.5)
        db = 20.0 * float(math.log10(rms + eps))
        return [(0.0, len(audio) / float(sr))] if db > -120.0 else []

    frame_starts = range(0, len(audio) - frame_length + 1, hop_length)
    rms_values = [float((audio[i : i + frame_length] * audio[i : i + frame_length]).mean() ** 0.5) for i in frame_starts]
    rms_arr = np.asarray(rms_values, dtype=np.float32)
    db_arr = 20.0 * np.log10(np.maximum(rms_arr, eps))

    if ref == "absolute":
        thr = float(threshold_db)
    else:
        thr = float(np.max(db_arr)) + float(threshold_db)

    mask = db_arr >= thr
    if not np.any(mask):
        return []

    times = (np.arange(len(mask), dtype=np.float32) * hop_length) / float(sr)
    intervals: List[Tuple[float, float]] = []
    start: Optional[float] = None

    for t, keep in zip(times, mask):
        if keep and start is None:
            start = float(t)
        if (not keep) and start is not None:
            end = float(t) + (frame_length / float(sr))
            intervals.append((start, end))
            start = None

    if start is not None:
        end = float(times[-1]) + (frame_length / float(sr))
        intervals.append((start, end))

    # Merge close gaps and apply min lengths
    merged: List[Tuple[float, float]] = []
    for s, e in intervals:
        if not merged:
            merged.append((s, e))
            continue
        ps, pe = merged[-1]
        if s - pe <= float(min_silence_len):
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))

    cleaned: List[Tuple[float, float]] = []
    for s, e in merged:
        if (e - s) >= float(min_sound_len):
            cleaned.append((max(0.0, s - pad), e + pad))

    return cleaned


def _filter_events_by_intervals(events: List[Dict[str, Any]], intervals: List[Tuple[float, float]]) -> List[Dict[str, Any]]:
    if not events or not intervals:
        return events

    def _in_any(t: float) -> bool:
        for s, e in intervals:
            if s <= t <= e:
                return True
        return False

    return [ev for ev in events if _in_any(float(ev.get("time", 0.0)))]


def _filter_event_times_by_intervals(times: Any, intervals: List[Tuple[float, float]]):
    if times is None:
        return times
    try:
        seq = times.tolist() if hasattr(times, "tolist") else list(times)
    except Exception:
        return times

    def _in_any(t: float) -> bool:
        for s, e in intervals:
            if s <= t <= e:
                return True
        return False

    return [t for t in seq if _in_any(float(t))]
