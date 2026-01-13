import os

# ------------------ video download and segmentation configuration ------------------ #
VIDEO_DATABASE_FOLDER = "./video_database/"
VIDEO_RESOLUTION = 360 # denotes the height of the video 
VIDEO_FPS = 2 # frames per second
VIDEO_MAX_MINUTES = 200 # maximum video duration to process in minutes
VIDEO_MAX_FRAMES = VIDEO_MAX_MINUTES * 60 * VIDEO_FPS  # automatically calculated based on max minutes and fps
# ------------------ Shot detection configuration ------------------ #
SHOT_DETECTION_FPS = 2.0 # frames per second
VIDEO_TYPE = "film"  # "film" or "vlog"
if VIDEO_TYPE == "film":
    SHOT_DETECTION_THRESHOLD = 3.0
    SHOT_DETECTION_MIN_SCENE_LEN = 3
elif VIDEO_TYPE == "vlog":
    SHOT_DETECTION_THRESHOLD = 1.5
    SHOT_DETECTION_MIN_SCENE_LEN = 45
else:
    # Film: 3.0, 3, Vlog: 1.5, 45
    SHOT_DETECTION_THRESHOLD = 3.0 # threshold for shot detection (for scenedetect: 3.0, for transnetv2/autoshot: 0.5)
    SHOT_DETECTION_MIN_SCENE_LEN = 3  # Minimum shot length in frames (only used for scenedetect)

SHOT_DETECTION_PREDICTIONS_PATH = "shot_predictions.txt"
SHOT_DETECTION_SCENES_PATH = "shot_scenes.txt"
SHOT_DETECTION_MODEL = "scenedetect" # "autoshot", "transnetv2", "Qwen3VL", "scenedetect"
# SHOT_DETECTION_MIN_SCENE_LEN = 45  # Minimum shot length in frames (only used for scenedetect)



CLIP_SECS = 30 # max clip seconds
MERGE_SHORT_SCENES = True  # if True, merge consecutive short scenes into longer clips
SCENE_MERGE_METHOD = "min_length"  # "max_length" or "min_length" - how to merge scenes
SCENE_MIN_LENGTH_SECS = 3  # minimum scene length in seconds (used when SCENE_MERGE_METHOD="min_length")
WHOLE_VIDEO_SUMMARY_BATCH_SIZE = 50  # number of clip captions per summary batch
USE_BATCH_PROCESSING = True  # If True, use ffmpeg direct extraction to avoid loading entire video into memory


# ------------------ ASR (Speech Recognition) configuration ------------------ #
ASR_MODEL = "large-v3-turbo"  # tiny, base, small, medium, large, large-v2, large-v3, large-v3-turbo
ASR_DEVICE = "cuda:0"  # Device for ASR model
ASR_LANGUAGE = "en"  # Language code (e.g., "zh", "en", "ja"). None for auto-detect

# Anti-hallucination parameters for Whisper ASR
ASR_NO_SPEECH_THRESHOLD = 0.7  # Higher = more conservative, filters non-speech segments (default 0.6)
ASR_LOGPROB_THRESHOLD = -0.8  # Higher = filters low-confidence outputs (default -1.0)
ASR_COMPRESSION_RATIO_THRESHOLD = 2.0  # Lower = detect repetitive/hallucinated outputs (default 2.4)
ASR_CONDITION_ON_PREVIOUS_TEXT = False  # False = prevent context carryover hallucinations

# Speaker diarization parameters
ASR_MERGE_SAME_SPEAKER = True  # Merge consecutive segments from the same speaker
ASR_MERGE_GAP = 1.0  # Maximum time gap for merging same-speaker segments (seconds)
ASR_DIARIZATION_MODEL_PATH = "../HF/hub/models--pyannote--speaker-diarization-community-1/snapshots/3533c8cf8e369892e6b79ff1bf80f7b0286a54ee/"

# ------------------ Video model configuration ------------------ #
# Scene Analysis Prompt Type
# Options: "film" (character-centric, for movies/TV shows) or "vlog" (journey-centric, for travel vlogs)
SCENE_PROMPT_TYPE = VIDEO_TYPE

# Video Analysis Model
VIDEO_ANALYSIS_MODEL = "/public_hw/home/cit_shifangzhao/zsf/HF/models/Qwen/Qwen3-VL-30B-A3B-Instruct"
VIDEO_ANALYSIS_MODEL_MAX_TOKEN = 16384  # Max tokens to generate (not total context length)
VLLM_ENDPOINT = "http://localhost:8888/v1/chat/completions"  # vLLM base URL (will append /v1/chat/completions)
# vllm serve /public_hw/home/cit_shifangzhao/zsf/HF/models/Qwen/Qwen3-VL-30B-A3B-Instruct \
#     --host 0.0.0.0 \
#     --port 8888 \
#     --max-model-len 65536 \
#     --max-num-seqs 256 \
#     --gpu-memory-utilization 0.9 \
#     --trust-remote-code \
#     --tensor-parallel-size 2
# CUDA_VISIBLE_DEVICES=0 vllm serve /public_hw/home/cit_shifangzhao/zsf/HF/models/Qwen/Qwen3-VL-30B-A3B-Instruct     --host 0.0.0.0     --port 8888     --max-model-len 65535     --max-num-seqs 256     --gpu-memory-utilization 0.9     --trust-remote-code 

# ------------------ Audio model configuration ------------------ #
AUDIO_ANALYSIS_MODEL = "/public_hw/home/cit_shifangzhao/zsf/HF/models/Qwen/Qwen3-Omni-30B-A3B-Instruct"
AUDIO_ANALYSIS_MODEL_MAX_TOKEN = 32768  # Max tokens to generate (not total context length)
VLLM_AUDIO_ENDPOINT = "http://localhost:8890/v1/chat/completions"  # vLLM endpoint for Qwen3-Omni audio model
# 启动audio服务的命令示例:
# vllm serve /public_hw/home/cit_shifangzhao/zsf/HF/models/Qwen/Qwen3-Omni-30B-A3B-Instruct \
#     --host 0.0.0.0 \
#     --port 8890 \
#     --max-model-len 32768 \
#     --gpu-memory-utilization 0.9 \
#     --trust-remote-code \
#     --tensor-parallel-size 2

# ------------------ Audio analysis with Madmom configuration ------------------ #
# Detection method selection (NEW: supports multiple methods like interactive interface)
# Options: "downbeat", "pitch", "mel_energy"
# You can use a single method or combine multiple methods (e.g., ["downbeat", "pitch", "mel_energy"])
# Default: ["downbeat", "pitch", "mel_energy"] to match interactive interface behavior
AUDIO_DETECTION_METHODS = ["downbeat", "pitch", "mel_energy"]

# Madmom detection parameters (downbeat)
AUDIO_BEATS_PER_BAR = 4  # Beats per bar (e.g., 4 for 4/4 time signature)
AUDIO_DBN_THRESHOLD = 0.05  # DBN activation threshold for beat tracking
# NOTE: The following onset parameters are DEPRECATED and not used in the current implementation
AUDIO_MIN_BPM = 55.0  # Minimum BPM for beat detection
AUDIO_MAX_BPM = 215.0  # Maximum BPM for beat detection

# Pitch detection parameters
AUDIO_PITCH_TOLERANCE = 0.8  # Pitch detection tolerance
AUDIO_PITCH_THRESHOLD = 0.8  # Pitch confidence threshold
AUDIO_PITCH_MIN_DISTANCE = 0.3  # Minimum distance between pitch points (seconds)
AUDIO_PITCH_NMS_METHOD = "basic"  # NMS method: "basic", "adaptive", "window"
AUDIO_PITCH_MAX_POINTS = 50  # Maximum number of pitch points to keep

# Mel energy detection parameters
AUDIO_MEL_WIN_S = 512  # FFT window size
AUDIO_MEL_N_FILTERS = 40  # Number of mel filters
AUDIO_MEL_THRESHOLD_RATIO = 0.3  # Energy threshold ratio
AUDIO_MEL_MIN_DISTANCE = 0.3  # Minimum distance between mel energy points (seconds)
AUDIO_MEL_NMS_METHOD = "basic"  # NMS method: "basic", "adaptive", "window"
AUDIO_MEL_MAX_POINTS = 50  # Maximum number of mel energy points to keep

# Post-processing / Rule-based filtering parameters (key to avoiding overly dense, noisy cut points)
# NOTE: AUDIO_MERGE_CLOSE is accepted by the API but not currently used in filter_significant_keypoints()
# It may be used in future implementations or in interactive mode
AUDIO_MERGE_CLOSE = 0  # [NOT USED] Merge keypoints closer than this threshold (seconds)
AUDIO_MIN_INTERVAL = 0  # Minimum interval between keypoints (seconds)
AUDIO_TOP_K = 0  # Keep only top K keypoints by intensity (0 = no limit)
AUDIO_ENERGY_PERCENTILE = 0  # Keep only keypoints above this energy percentile (0-100)

# Silence gating (recommended)
AUDIO_SILENCE_THRESHOLD_DB = -45.0  # Silence threshold in dB (segments below this are filtered)

# Segment filtering parameters
AUDIO_MIN_SEGMENT_DURATION = 2.0  # Minimum segment duration in seconds (merge threshold)
AUDIO_MAX_SEGMENT_DURATION = 4.0  # Maximum segment duration in seconds (split threshold)

# Structure analysis parameters (AI model for Level 1 sections)
AUDIO_STRUCTURE_TEMPERATURE = 0.7  # Temperature for AI structure analysis
AUDIO_STRUCTURE_TOP_P = 0.95  # Top-p sampling for AI structure analysis
AUDIO_STRUCTURE_MAX_TOKENS = 4096  # Max tokens for AI structure generation

# Section-based filtering parameters
AUDIO_USE_STAGE1_SECTIONS = True  # Whether to use AI-identified sections for filtering
AUDIO_SECTION_MIN_INTERVAL = 2.0  # Minimum interval between keypoints (global, across all sections)
AUDIO_TOTAL_SHOTS = 200  # Total number of shots to allocate across all sections (proportional allocation)

# Composite score weights (for multi-metric ranking when combining different keypoint types)
AUDIO_WEIGHT_DOWNBEAT = 1.0  # Weight for Downbeat intensity
AUDIO_WEIGHT_PITCH = 1.0  # Weight for Pitch intensity
AUDIO_WEIGHT_MEL_ENERGY = 1.0  # Weight for Mel Energy intensity

# Keypoint analysis parameters (AI model for Level 2 captions)
AUDIO_BATCH_SIZE = 8  # Batch size for AI inference (parallel processing of audio segments)
AUDIO_KEYPOINT_TEMPERATURE = 0.7  # Temperature for AI caption generation
AUDIO_KEYPOINT_TOP_P = 0.95  # Top-p sampling for AI caption generation
AUDIO_KEYPOINT_MAX_TOKENS = 4096  # Max tokens for AI caption generation


# ------------------ Text embedding model configuration ------------------ #
EMBEDDING_MODEL = "/public_hw/home/cit_shifangzhao/zsf/HF/models/Qwen/Qwen3-Embedding-8B"
EMBEDDING_MODEL_DIM = 4096
VLLM_EMBEDDING_ENDPOINT = "http://localhost:8001/v1/"  # vLLM embedding base URL (will append /v1/embeddings)
# 启动embedding服务的命令示例:
# CUDA_VISIBLE_DEVICES=1 vllm serve /public_hw/home/cit_shifangzhao/zsf/HF/models/Qwen/Qwen3-Embedding-8B     --host 0.0.0.0     --port 8001     --max-model-len 8192     --trust-remote-code



# ------------------ Agent model configuration ------------------ #
# AGENT_MODEL = "/public_hw/home/cit_shifangzhao/zsf/HF/models/Qwen/Qwen3-30B-A3B/"
# AGENT_MODEL = "/public_hw/home/cit_shifangzhao/zsf/HF/models/Qwen/Qwen3-VL-30B-A3B-Instruct"
# VLLM_AGENT_ENDPOINT = "http://localhost:8000"  # vLLM embedding服务端点
AGENT_MODEL_MAX_TOKEN = 8192  # Max tokens to generate (not total context length)
ENABLE_TRIM_SHOT_CHARACTER_ANALYSIS = True  # Enable VLM character analysis in trim_shot
TRIM_SHOT_CHARACTER_SAMPLE_FPS = 5        # Sampling FPS for character analysis (lower is faster, 0.5 = 1 frame per 2 seconds)

# v3-api
# AGENT_MODEL = "gemini-3-flash-preview"
# VLLM_AGENT_ENDPOINT = "https://api.gpt.ge/v1/chat/completions"
# UNIFIED_API_KEY = "sk-sXTpMrSOGfEsdmEr906135Bf32D24cEa8944C1B879E61822"

# minimax
AGENT_MODEL = "MiniMax-M2.1"
VLLM_AGENT_ENDPOINT = "https://api.minimaxi.com/v1"
UNIFIED_API_KEY = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiJTaGlmYW5nIFpoYW8iLCJVc2VyTmFtZSI6IlNoaWZhbmcgWmhhbyIsIkFjY291bnQiOiIiLCJTdWJqZWN0SUQiOiIyMDAzNzc1NDg0Mzg2NjE1MzkzIiwiUGhvbmUiOiIxNTgwMTIzNTY4MyIsIkdyb3VwSUQiOiIyMDAzNzc1NDg0MzgyNDIxMDg5IiwiUGFnZU5hbWUiOiIiLCJNYWlsIjoiIiwiQ3JlYXRlVGltZSI6IjIwMjUtMTItMjQgMjA6MjI6NTYiLCJUb2tlblR5cGUiOjEsImlzcyI6Im1pbmltYXgifQ.mEOsOZhw1djLCVWnVtlUoXodedf9ppGBSBC30Y_5aCAWKJOxJ_7qq8yfZfV7A0dC-fjC8Z7nu5hx3CEVLmk80uRFy4taUKdGRh0-wvD59-o1Scvye-bK9YsPPMEbcCfTe2nvY580lp4Z-zd3LgZJG6QBB3RqfWiyEnD_dc_idaHDbfvJLFlQaOYnrwN-TUO-hx4cU_gALW5RVfVjWqkDj0s7P0MsL2BXOjR8s85p1RMcdG_WRxoEYWwvBCyF-kJRJ4x2_9ZHDQAUhCA3eKw57EOJmAHlP48xgFT0cS5EPBbHgxNySK9MUlpd1aimkw5YfPWEDuofnzNEgclfAK8kfQ"

# 启动agent服务的命令示例:
# CUDA_VISIBLE_DEVICES=0 vllm serve /public_hw/home/cit_shifangzhao/zsf/HF/models/Qwen/Qwen3-VL-30B-A3B-Instruct     --host 0.0.0.0     --port 8000     --max-model-len 65535     --max-num-seqs 256     --gpu-memory-utilization 0.9     --trust-remote-code --enable-auto-tool-choice --tool-call-parser hermes





# ------------------ Reviewer / Face quality check ------------------ #
# Used by ReviewerAgent.check_face_quality() and check_face_quality_vlm() (vca/Reviewer.py)
# Enable or disable face quality check before finishing a shot selection
ENABLE_FACE_QUALITY_CHECK = True  # Set to True to enable face quality validation

# Face quality check method selection
# Options: "traditional" (face_recognition library) or "vlm" (Vision Language Model)
FACE_QUALITY_CHECK_METHOD = "vlm"  # "traditional" or "vlm"

# ------------------ Traditional face quality check parameters ------------------ #
# Used when FACE_QUALITY_CHECK_METHOD = "traditional"
# - max_break_ratio: fraction of sampled frames that are "break" (no face / face too small)
# - min_face_size: minimum pixel size of detected face (min(height, width))
# - sample_fps: sampling rate for running face detection (lower is faster but less strict)
FACE_QUALITY_MAX_BREAK_RATIO = 0.5  # 30% of frames can have no face or small face
FACE_QUALITY_MIN_FACE_SIZE = 50     # Minimum face size in pixels
FACE_QUALITY_SAMPLE_FPS = 3.0       # Sampling FPS for traditional face detection

# ------------------ VLM protagonist presence check parameters ------------------ #
# Used when FACE_QUALITY_CHECK_METHOD = "vlm"
# VLM method uses VIDEO_ANALYSIS_MODEL for frame-by-frame protagonist detection
# NOTE: This checks for MAIN CHARACTER presence, not strict face detection
# The system will REJECT minor characters/extras to avoid selecting shots with "小喽啰"
MAIN_CHARACTER_NAME = "Joker"  # Name of the main character/protagonist
MIN_PROTAGONIST_RATIO = 0.5         # Minimum ratio of frames where protagonist is main focus (0.0-1.0)
                                      # RELAXED from 0.7 to 0.5 to allow more shot candidates
VLM_FACE_CHECK_SAMPLE_FPS = 3.0      # Sampling FPS for VLM frame-by-frame detection (lower is faster)
VLM_MIN_BOX_SIZE = 50                # Minimum bounding box size in pixels for protagonist detection
                                      # Lowered from 50 to allow distant shots where protagonist is still the visual focus
                                      # Actual minimum will be relaxed to max(30, VLM_MIN_BOX_SIZE // 2)

# ------------------ Shot selection constraints (Plan A: Relaxed Constraints) ------------------ #
# Minimum acceptable shot duration for final selection
MIN_ACCEPTABLE_SHOT_DURATION = 2.0   # Minimum shot duration in seconds (reduced from 3.0)
                                      # Allows shorter shots when perfect matches are not available

# Fallback tolerance for imperfect matches
ALLOW_DURATION_TOLERANCE = 1.0       # Allow shots within ±1.0s of target duration
ALLOW_CONTENT_MISMATCH = True        # Allow shots with similar (not exact) content match
ENABLE_FALLBACK_STRATEGY = True      # Enable multi-tier fallback when perfect match not found

# Scene exploration constraints
SCENE_EXPLORATION_RANGE = 3          # Allow exploring ±N scenes from recommended scenes
                                      # e.g., if recommended scene is 8, can explore scenes 5-11
                                      # Set to 0 to strictly limit to recommended scenes only




