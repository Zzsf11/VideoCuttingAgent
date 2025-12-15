import os

# ------------------ video download and segmentation configuration ------------------ #
VIDEO_DATABASE_FOLDER = "./video_database/"
VIDEO_RESOLUTION = 360 # denotes the height of the video 
VIDEO_FPS = 2 # frames per second
VIDEO_MAX_MINUTES = 140 # maximum video duration to process in minutes
VIDEO_MAX_FRAMES = VIDEO_MAX_MINUTES * 60 * VIDEO_FPS  # automatically calculated based on max minutes and fps
# ------------------ Shot detection configuration ------------------ #
SHOT_DETECTION_FPS = 2.0 # frames per second
SHOT_DETECTION_THRESHOLD = 0.3 # threshold for shot detection
SHOT_DETECTION_PREDICTIONS_PATH = "shot_predictions.txt"
SHOT_DETECTION_SCENES_PATH = "shot_scenes.txt"
SHOT_DETECTION_MODEL = "Qwen3VL" # "autoshot", "transnetv2", "Qwen3VL"
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
# Video Analysis Model
VIDEO_ANALYSIS_MODEL = "/public_hw/home/cit_shifangzhao/zsf/HF/models/Qwen/Qwen3-VL-30B-A3B-Instruct"
VIDEO_ANALYSIS_MODEL_MAX_TOKEN = 16384  # Max tokens to generate (not total context length)
VLLM_ENDPOINT = "http://localhost:8888/v1/chat/completions"  # vLLM base URL (will append /v1/chat/completions)
# vllm serve /public_hw/home/cit_shifangzhao/zsf/HF/models/Qwen/Qwen3-VL-30B-A3B-Instruct \
#     --host 0.0.0.0 \
#     --port 8000 \
#     --max-model-len 65536 \
#     --max-num-seqs 256 \
#     --gpu-memory-utilization 0.9 \
#     --trust-remote-code \
#     --tensor-parallel-size 2
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
# Madmom detection parameters
AUDIO_ONSET_THRESHOLD = 0.6  # Onset detection threshold (higher = fewer onsets)
AUDIO_ONSET_COMBINE = 3.0  # Time window for combining nearby onsets (seconds)
AUDIO_MIN_BPM = 55.0  # Minimum BPM for beat detection
AUDIO_MAX_BPM = 215.0  # Maximum BPM for beat detection

# Segment filtering parameters
AUDIO_MIN_SEGMENT_DURATION = 3.0  # Minimum segment duration in seconds
AUDIO_MAX_SEGMENT_DURATION = 15.0  # Maximum segment duration in seconds
AUDIO_MAX_SEGMENTS = 20  # Maximum number of segments to create
AUDIO_MERGE_CLOSE = 0.1  # Merge keypoints closer than this threshold (seconds)

# Section-based filtering parameters
AUDIO_USE_STAGE1_SECTIONS = True  # Whether to use AI-identified sections for filtering
AUDIO_SECTION_TOP_K = 0  # Number of keypoints to keep per section, 0 means using energy percentile filtering
AUDIO_SECTION_MIN_INTERVAL = 0.0  # Minimum interval between keypoints within each section
AUDIO_SECTION_ENERGY_PERCENTILE = 70.0  # Energy percentile threshold within each section

# Type-based filtering parameters (以重音和音量为主)
# Available types: "Downbeat" (重拍), "Onset"/"Attack" (冲击点), "Energy" (能量变化),
#                  "Spectral" (频谱变化), "Timbre" (音色变化)
# 设置为 None 可以禁用类型过滤，保留所有类型
AUDIO_PREFERRED_TYPES = ["Downbeat", "Energy", "Onset"]  # 优先保留的关键点类型（保留所有类型）
AUDIO_TYPE_FILTER_MODE = "only"  # "only" = 只保留指定类型, "boost" = 增强指定类型权重, "exclude" = 排除指定类型
AUDIO_TYPE_BOOST_FACTOR = 1.5  # 当 mode="boost" 时的权重增强因子

# Batch processing
AUDIO_BATCH_SIZE = 16  # Number of audio segments to process in parallel


# ------------------ Text embedding model configuration ------------------ #
EMBEDDING_MODEL = "/public_hw/home/cit_shifangzhao/zsf/HF/models/Qwen/Qwen3-Embedding-8B"
EMBEDDING_MODEL_DIM = 4096
VLLM_EMBEDDING_ENDPOINT = "http://localhost:8889/v1/"  # vLLM embedding base URL (will append /v1/embeddings)
# 启动embedding服务的命令示例:
# CUDA_VISIBLE_DEVICES=1 vllm serve /public_hw/home/cit_shifangzhao/zsf/HF/models/Qwen/Qwen3-Embedding-8B     --host 0.0.0.0     --port 8001     --max-model-len 8192     --trust-remote-code



# ------------------ Agent model configuration ------------------ #
# AGENT_MODEL = "/public_hw/home/cit_shifangzhao/zsf/HF/models/Qwen/Qwen3-30B-A3B"
AGENT_MODEL = "/public_hw/home/cit_shifangzhao/zsf/HF/models/Qwen/Qwen3-VL-30B-A3B-Instruct"
AGENT_MODEL_MAX_TOKEN = 4096  # Max tokens to generate (not total context length)
VLLM_AGENT_ENDPOINT = "http://localhost:8888"  # vLLM embedding服务端点
# 启动agent服务的命令示例:
# CUDA_VISIBLE_DEVICES=0 vllm serve /public_hw/home/cit_shifangzhao/zsf/HF/models/Qwen/Qwen3-VL-30B-A3B-Instruct     --host 0.0.0.0     --port 8000     --max-model-len 65535     --max-num-seqs 256     --gpu-memory-utilization 0.9     --trust-remote-code --enable-auto-tool-choice --tool-call-parser hermes



OPENAI_API_KEY = os.environ.get("Osk-KLzUGaBXb0aRjUhMCd9f4dC321F943Aa83Dc415a4b4e4cB1", None) # will overwrite Azure OpenAI setting

AOAI_CAPTION_VLM_ENDPOINT_LIST = [""]
AOAI_CAPTION_VLM_MODEL_NAME = "gpt-4.1-mini"

AOAI_ORCHESTRATOR_LLM_ENDPOINT_LIST = [""]
AOAI_ORCHESTRATOR_LLM_MODEL_NAME = "o3"

AOAI_TOOL_VLM_ENDPOINT_LIST = [""]
AOAI_TOOL_VLM_MODEL_NAME = "gpt-4.1-mini"
AOAI_TOOL_VLM_MAX_FRAME_NUM = 50

AOAI_EMBEDDING_RESOURCE_LIST = [""]
AOAI_EMBEDDING_LARGE_MODEL_NAME = "text-embedding-3-large"
AOAI_EMBEDDING_LARGE_DIM = 4096

# ------------------ agent and tool setting ------------------ #
LITE_MODE = True # if True, only leverage srt subtitle, no pixel downloaded or pixel captioning
GLOBAL_BROWSE_TOPK = 300
OVERWRITE_CLIP_SEARCH_TOPK = 0 # 0 means no overwrite and let agent decide

SINGLE_CHOICE_QA = True  # Design for benchmark test. If True, the agent will only return options for single-choice questions.
MAX_ITERATIONS = 3  # Maximum number of iterations for the agent to run