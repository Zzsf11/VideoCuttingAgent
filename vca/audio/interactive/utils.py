import os
import sys
import base64
import mimetypes
import numpy as np
from typing import Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

def _patch_gradio_checkboxgroup_none_payload() -> None:
    """Work around a Gradio CheckboxGroup edge-case.

    Some Gradio versions may send `null` for an empty CheckboxGroup selection,
    which becomes `None` in Python and crashes in preprocess.
    """
    try:
        from gradio.components.checkboxgroup import CheckboxGroup  # type: ignore

        if getattr(CheckboxGroup.preprocess, "_vca_none_payload_patched", False):
            return

        _orig_preprocess = CheckboxGroup.preprocess

        def _preprocess_allow_none(self, payload):
            if payload is None:
                payload = []
            return _orig_preprocess(self, payload)

        _preprocess_allow_none._vca_none_payload_patched = True  # type: ignore
        CheckboxGroup.preprocess = _preprocess_allow_none  # type: ignore
    except Exception:
        # If gradio internals change, skip patch.
        return

# Mocking the import for standalone usage if the vca package isn't present
# In your real environment, this try/except block handles the import normally.
try:
    from vca.audio.audio_utils import load_audio_no_librosa
except ImportError:
    # Fallback for demonstration if vca is missing
    import librosa
    import soundfile as sf
    def load_audio_no_librosa(file, sr=16000):
        y, _ = librosa.load(file, sr=sr)
        return y, sr

# ============================================================================ #
#                          Global Audio Cache                                   #
# ============================================================================ #

_audio_cache = {}
_audio_b64_cache = {}


def get_audio_data(audio_path: str, sr: int = 16000) -> Tuple[np.ndarray, float]:
    """Load audio data with caching"""
    cache_key = (audio_path, sr)
    if cache_key not in _audio_cache:
        loaded = load_audio_no_librosa(audio_path, sr=sr)
        # `vca.audio.audio_utils.load_audio_no_librosa` returns a numpy array.
        # Some legacy fallbacks may return (audio, sr).
        if isinstance(loaded, tuple) and len(loaded) >= 1:
            audio = loaded[0]
        else:
            audio = loaded
        duration = len(audio) / sr
        _audio_cache[cache_key] = (audio, duration)
    return _audio_cache[cache_key]


def get_audio_base64(audio_path: str) -> str:
    """Convert audio file to base64 data URL for browser playback"""
    if audio_path in _audio_b64_cache:
        return _audio_b64_cache[audio_path]

    mime_type, _ = mimetypes.guess_type(audio_path)
    if mime_type is None:
        mime_type = 'audio/mpeg'

    with open(audio_path, 'rb') as f:
        audio_data = f.read()

    b64_data = base64.b64encode(audio_data).decode('utf-8')
    data_url = f"data:{mime_type};base64,{b64_data}"
    _audio_b64_cache[audio_path] = data_url
    return data_url
