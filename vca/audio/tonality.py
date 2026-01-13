"""Tonality / key detection utilities.

Implements two lightweight key-finding methods:
- K-S (Krumhansl-Schmuckler) template matching
- Spiral Array (geometric model; simplified, audio-based approximation)

Designed to avoid adding new heavy dependencies (no librosa).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

from vca.audio.audio_utils import load_audio_no_librosa


KeyMethod = Literal["none", "ks", "spiral"]


_PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl major/minor key profiles (classic)
_KS_MAJOR = np.asarray([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float32)
_KS_MINOR = np.asarray([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float32)


@dataclass(frozen=True)
class KeyResult:
    key: str
    score: float
    method: str


def estimate_key(
    audio_path: str,
    *,
    method: KeyMethod = "ks",
    sr: int = 16000,
    window_s: float = 0.0,
    hop_s: Optional[float] = None,
    stft_nfft: int = 4096,
) -> Dict:
    """Estimate global key or key changes.

    Returns dict:
      - key: global key label (e.g. "C Major")
      - segments: optional list of {start, end, key, score}
    """

    if method == "none":
        return {"key": "", "segments": []}

    audio = load_audio_no_librosa(audio_path, sr=sr)
    if audio is None or len(audio) == 0:
        return {"key": "", "segments": []}

    audio = np.asarray(audio, dtype=np.float32)

    if window_s and window_s > 0:
        hop_s = float(hop_s if hop_s is not None else window_s / 2.0)
        segments = _estimate_key_segments(audio, sr, method=method, window_s=float(window_s), hop_s=hop_s, nfft=stft_nfft)
        if not segments:
            return {"key": "", "segments": []}
        # pick the longest segment as global
        longest = max(segments, key=lambda s: (s[1] - s[0]))
        return {
            "key": longest[2].key,
            "segments": [
                {"start": float(s0), "end": float(s1), "key": res.key, "score": float(res.score), "method": res.method}
                for (s0, s1, res) in segments
            ],
        }

    chroma = _compute_chroma(audio, sr=sr, nfft=stft_nfft)
    res = _estimate_key_from_chroma(chroma, method=method)
    return {"key": res.key, "segments": []}


def _estimate_key_segments(
    audio: np.ndarray,
    sr: int,
    *,
    method: KeyMethod,
    window_s: float,
    hop_s: float,
    nfft: int,
) -> List[Tuple[float, float, KeyResult]]:
    win = max(1, int(window_s * sr))
    hop = max(1, int(hop_s * sr))

    keys: List[Tuple[float, float, KeyResult]] = []
    for start in range(0, max(1, len(audio) - win + 1), hop):
        end = start + win
        chunk = audio[start:end]
        chroma = _compute_chroma(chunk, sr=sr, nfft=nfft)
        res = _estimate_key_from_chroma(chroma, method=method)
        keys.append((start / sr, end / sr, res))

    if not keys:
        return []

    # compress consecutive identical keys
    merged: List[Tuple[float, float, KeyResult]] = []
    cur_s, cur_e, cur_res = keys[0]
    for s, e, res in keys[1:]:
        if res.key == cur_res.key:
            cur_e = e
            cur_res = KeyResult(cur_res.key, score=max(cur_res.score, res.score), method=cur_res.method)
        else:
            merged.append((cur_s, cur_e, cur_res))
            cur_s, cur_e, cur_res = s, e, res
    merged.append((cur_s, cur_e, cur_res))
    return merged


def _estimate_key_from_chroma(chroma: np.ndarray, *, method: KeyMethod) -> KeyResult:
    chroma = np.asarray(chroma, dtype=np.float32)
    if chroma.shape != (12,):
        chroma = chroma.reshape(-1)[:12]
    if float(np.sum(chroma)) <= 0:
        return KeyResult("", 0.0, method)

    v = chroma / float(np.sum(chroma))

    if method == "ks":
        return _estimate_key_ks(v)
    if method == "spiral":
        return _estimate_key_spiral(v)

    return KeyResult("", 0.0, method)


def _estimate_key_ks(chroma: np.ndarray) -> KeyResult:
    # cosine similarity against rotated templates
    v = chroma / (np.linalg.norm(chroma) + 1e-12)
    maj = _KS_MAJOR / (np.linalg.norm(_KS_MAJOR) + 1e-12)
    minr = _KS_MINOR / (np.linalg.norm(_KS_MINOR) + 1e-12)

    best_score = -1e9
    best_key = ""
    best_method = "ks"

    for root in range(12):
        maj_r = np.roll(maj, root)
        min_r = np.roll(minr, root)

        s_maj = float(np.dot(v, maj_r))
        s_min = float(np.dot(v, min_r))

        if s_maj > best_score:
            best_score = s_maj
            best_key = f"{_PITCH_CLASS_NAMES[root]} Major"
        if s_min > best_score:
            best_score = s_min
            best_key = f"{_PITCH_CLASS_NAMES[root]} Minor"

    return KeyResult(best_key, best_score, best_method)


def _estimate_key_spiral(chroma: np.ndarray) -> KeyResult:
    # Simplified spiral array approximation:
    # - build 3D positions for pitch classes using circle-of-fifths order
    # - build key vectors as weighted triads
    # - pick max cosine similarity between chroma centroid and key vectors

    v = chroma / (np.linalg.norm(chroma) + 1e-12)

    radius = 1.0
    height = 0.35
    w_root, w_third, w_fifth = 1.0, 0.8, 0.6

    fifth_order = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]  # C G D A E B F# C# G# D# A# F
    pc_to_k = {pc: k for k, pc in enumerate(fifth_order)}

    pos = np.zeros((12, 3), dtype=np.float32)
    for pc in range(12):
        k = pc_to_k[pc]
        theta = 2.0 * np.pi * (k / 12.0)
        pos[pc, 0] = radius * np.cos(theta)
        pos[pc, 1] = radius * np.sin(theta)
        pos[pc, 2] = height * k

    centroid = np.sum(pos * v[:, None], axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

    best_score = -1e9
    best_key = ""

    for root in range(12):
        major_vec = (
            w_root * pos[root]
            + w_third * pos[(root + 4) % 12]
            + w_fifth * pos[(root + 7) % 12]
        )
        major_vec = major_vec / (np.linalg.norm(major_vec) + 1e-12)
        s_maj = float(np.dot(centroid, major_vec))

        minor_vec = (
            w_root * pos[root]
            + w_third * pos[(root + 3) % 12]
            + w_fifth * pos[(root + 7) % 12]
        )
        minor_vec = minor_vec / (np.linalg.norm(minor_vec) + 1e-12)
        s_min = float(np.dot(centroid, minor_vec))

        if s_maj > best_score:
            best_score = s_maj
            best_key = f"{_PITCH_CLASS_NAMES[root]} Major"
        if s_min > best_score:
            best_score = s_min
            best_key = f"{_PITCH_CLASS_NAMES[root]} Minor"

    return KeyResult(best_key, best_score, "spiral")


def _compute_chroma(audio: np.ndarray, *, sr: int, nfft: int = 4096) -> np.ndarray:
    # STFT magnitude -> pitch-class energy mapping
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) == 0:
        return np.zeros(12, dtype=np.float32)

    # Pre-emphasis / DC removal (lightweight)
    audio = audio - float(np.mean(audio))

    from scipy.signal import stft

    # Keep it light: hop ~ 10ms
    hop = max(64, int(sr * 0.01))
    win = min(nfft, max(256, int(sr * 0.08)))

    _, freqs, Zxx = stft(audio, fs=sr, nperseg=win, noverlap=win - hop, nfft=nfft, padded=False, boundary=None)
    mag = np.abs(Zxx).astype(np.float32)

    # avoid DC and very low freqs
    valid = freqs >= 50.0
    freqs = freqs[valid]
    mag = mag[valid, :]

    if mag.size == 0:
        return np.zeros(12, dtype=np.float32)

    power = mag * mag

    # map each frequency bin to pitch class (nearest MIDI)
    midi = 69.0 + 12.0 * np.log2(np.maximum(freqs, 1e-6) / 440.0)
    pc = np.mod(np.rint(midi).astype(np.int32), 12)

    chroma = np.zeros(12, dtype=np.float32)
    bin_energy = np.mean(power, axis=1)
    for idx, p in enumerate(pc):
        chroma[int(p)] += float(bin_energy[idx])

    # Normalize
    if float(np.sum(chroma)) > 0:
        chroma = chroma / float(np.sum(chroma))

    return chroma
