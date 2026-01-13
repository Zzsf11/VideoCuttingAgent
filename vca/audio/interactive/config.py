import json
from pathlib import Path
from typing import Dict, Any

# Default config file path (in user's home directory or current directory)
_CONFIG_DIR = Path.home() / ".vca_audio_segmenter"
_CONFIG_FILE = _CONFIG_DIR / "params_config.json"

# Default parameter values
# Note: These are optimized for the interactive app default UX: fewer, more stable
# keypoints that better match perceived musical structure.
DEFAULT_PARAMS = {
    # Config version (used for light migrations)
    "config_version": 2,
    # Madmom parameters
    "beats_per_bar": 4,
    "dbn_threshold": 0.05,
    # Post-processing defaults (key to avoiding overly dense, noisy cut points)
    "merge_close": 0.15,
    "min_interval": 0.4,
    "top_k": 0,
    "energy_percentile": 75.0,
    # Silence gating (recommended)
    "silence_threshold_db": -45.0,
    # Structure analysis parameters (Omni for Level 1 sections)
    "structure_temperature": 0.7,
    "structure_top_p": 0.95,
    "structure_max_tokens": 2048,
    # Structure-based filtering parameters
    "filter_total_shots": 20,            # Total shots for proportional allocation across sections
    "filter_min_interval": 3.0,          # Min interval between filtered keypoints (global)
    "filter_min_segment": 3.0,           # Min segment duration (merge threshold)
    "filter_max_segment": 30.0,          # Max segment duration (split threshold)
    # Composite score weights (for multi-metric ranking)
    "weight_downbeat": 1.0,              # Weight for Downbeat intensity
    "weight_pitch": 1.0,                 # Weight for Pitch intensity
    "weight_mel_energy": 1.0,            # Weight for Mel Energy intensity
    # Keypoint analysis parameters (Omni for Level 2 captions)
    "keypoint_batch_size": 4,            # Batch size for AI inference
    "keypoint_temperature": 0.7,         # AI temperature for caption generation
    "keypoint_top_p": 0.95,              # AI top_p for caption generation
    "keypoint_max_tokens": 4096,         # Max tokens for caption generation
}



def load_saved_params() -> Dict[str, Any]:
    """Load saved parameters from config file"""
    try:
        if _CONFIG_FILE.exists():
            with open(_CONFIG_FILE, 'r', encoding='utf-8') as f:
                saved = json.load(f)
                # Merge with defaults (in case new params were added)
                result = DEFAULT_PARAMS.copy()
                result.update(saved)

                # Light migration: if the user is still on the old defaults for the
                # dense-output knobs, upgrade them to the new recommended defaults.
                # If they changed any of these values intentionally, we respect it.
                saved_version = saved.get("config_version", 1)
                if int(saved_version) < int(DEFAULT_PARAMS.get("config_version", 2)):
                    old_default_triplet = (
                        float(saved.get("min_interval", 0.0)) == 0.0
                        and float(saved.get("merge_close", 0.1)) == 0.1
                        and float(saved.get("energy_percentile", 0.0)) == 0.0
                        and int(saved.get("top_k", 0)) == 0
                    )
                    if old_default_triplet:
                        result["merge_close"] = DEFAULT_PARAMS["merge_close"]
                        result["min_interval"] = DEFAULT_PARAMS["min_interval"]
                        result["energy_percentile"] = DEFAULT_PARAMS["energy_percentile"]
                        result["top_k"] = DEFAULT_PARAMS["top_k"]

                    # Always bump version in memory (user can save via UI)
                    result["config_version"] = DEFAULT_PARAMS.get("config_version", 2)
                return result
    except Exception as e:
        print(f"Warning: Failed to load saved params: {e}")
    return DEFAULT_PARAMS.copy()


def save_params_to_file(params: Dict[str, Any]) -> str:
    """Save parameters to config file"""
    try:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=2, ensure_ascii=False)
        return f"✅ Parameters saved to {_CONFIG_FILE}"
    except Exception as e:
        return f"❌ Failed to save parameters: {e}"


def reset_params_to_default() -> Dict[str, Any]:
    """Reset parameters to default values"""
    return DEFAULT_PARAMS.copy()
