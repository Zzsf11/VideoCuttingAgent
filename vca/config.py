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
SHOT_DETECTION_MODEL = "autoshot"
CLIP_SECS = 60 # max clip seconds
WHOLE_VIDEO_SUMMARY_BATCH_SIZE = 50  # number of clip captions per summary batch


# Memory optimization settings
USE_BATCH_PROCESSING = True  # If True, use ffmpeg direct extraction to avoid loading entire video into memory
VIDEO_BATCH_SIZE = 500  # Number of frames to process at once (only used with legacy processing mode)

# ------------------ Video model configuration ------------------ #
# Video Analysis Model
ASR_MODEL = "large-v3" # tiny, base, small, medium, large, large-v2, large-v3 from https://github.com/linto-ai/whisper-timestamped
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