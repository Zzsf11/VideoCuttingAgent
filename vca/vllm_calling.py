import base64
import copy
import json
import os
import random
import subprocess
import tempfile
import time
from io import BytesIO
from mimetypes import guess_type
from pathlib import Path

import cv2
import numpy as np
import requests


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 8,
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)
            # Raise exceptions for any errors not specified
            except Exception as e:
                if "rate limit" in str(e).lower() or "timed out" in str(e) \
                                    or "Too Many Requests" in str(e) or "Forbidden for url" in str(e) \
                                    or "internal" in str(e).lower():
                    # Increment retries
                    num_retries += 1

                    # Check if max retries has been reached
                    if num_retries > max_retries:
                        print("Max retries reached. Exiting.")
                        return None

                    # Increment the delay
                    delay *= exponential_base * (1 + jitter * random.random())
                    print(f"Retrying in {delay} seconds for {str(e)}...")
                    # Sleep for the delay
                    time.sleep(delay)
                else:
                    print(str(e))
                    return None

    return wrapper


# Function to encode a local image into data URL
def local_image_to_data_url(image_path):
    """Encode a local image into data URL format."""
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"  # Default MIME type if none is found

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from path: {image_path}")
    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


def local_video_to_data_url(video_path):
    """Encode a local video into data URL format."""
    # Guess the MIME type of the video based on the file extension
    mime_type, _ = guess_type(video_path)
    if mime_type is None:
        # Default to mp4 if cannot determine
        mime_type = "video/mp4"
    
    # Read and encode the video file
    with open(video_path, "rb") as video_file:
        base64_encoded_data = base64.b64encode(video_file.read()).decode("utf-8")
    
    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


def _check_ffmpeg_available() -> bool:
    """Check if ffmpeg is available on the system."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def encode_frames_to_video(image_paths: list, fps: float, output_path: str = None, use_ffmpeg: bool = None) -> str:
    """
    Encode a sequence of image frames into a video file with specified FPS.
    
    This function creates a video where:
    - The video FPS equals the sampling rate used during frame extraction
    - vLLM will read this FPS from the video metadata
    - Combined with do_sample_frames=False, vLLM will use all frames sequentially
    - Timestamps are calculated as: frame_index / fps
    
    Tries to use ffmpeg (H.264) for maximum compatibility, falls back to OpenCV if unavailable.
    
    Example:
        If you extracted 60 frames at 2 FPS from a 30-second video:
        - Create video with fps=2.0
        - vLLM decodes: 60 frames at 2 FPS = 30 seconds duration ✓
        - Frame 0 -> 0.0s, Frame 1 -> 0.5s, ..., Frame 59 -> 29.5s ✓
    
    Args:
        image_paths: List of paths to image frames (in temporal order)
        fps: The FPS used when extracting these frames (critical for timestamp calculation)
        output_path: Optional path for output video. If None, creates a temp file.
        use_ffmpeg: Whether to use ffmpeg (None=auto-detect, True=force, False=use OpenCV)
        
    Returns:
        str: Path to the encoded video file
    """
    if not image_paths:
        raise ValueError("image_paths cannot be empty")
    
    # Create output path if not provided
    if output_path is None:
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        output_path = temp_file.name
        temp_file.close()
    
    # Determine whether to use ffmpeg
    if use_ffmpeg is None:
        use_ffmpeg = _check_ffmpeg_available()
    
    if use_ffmpeg:
        return _encode_with_ffmpeg(image_paths, fps, output_path)
    else:
        return _encode_with_opencv(image_paths, fps, output_path)


def _encode_with_ffmpeg(image_paths: list, fps: float, output_path: str) -> str:
    """Encode video using ffmpeg (H.264 codec for maximum compatibility)."""
    # Create a temporary directory for symlinks
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create symlinks with sequential naming for ffmpeg
        for i, img_path in enumerate(image_paths):
            # Get file extension
            ext = os.path.splitext(img_path)[1]
            link_path = os.path.join(temp_dir, f"frame_{i:06d}{ext}")
            
            # Create symlink or copy
            try:
                os.symlink(os.path.abspath(img_path), link_path)
            except (OSError, NotImplementedError):
                # Fallback to copy if symlink not supported
                import shutil
                shutil.copy2(img_path, link_path)
        
        # Get the pattern and extension
        first_link = os.path.join(temp_dir, f"frame_%06d{ext}")
        
        # Use ffmpeg to create video with H.264 encoding
        # Optimized for vLLM compatibility:
        # -framerate: input framerate
        # -i: input pattern
        # -c:v libx264: use H.264 codec (most compatible)
        # -preset fast: fast encoding preset
        # -crf 18: high quality (0-51, lower is better)
        # -pix_fmt yuv420p: pixel format for maximum compatibility
        # -profile:v baseline: baseline profile for maximum compatibility
        # -level 3.0: compatibility level
        # -r: output framerate (same as input)
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-framerate', str(fps),  # Input FPS
            '-i', first_link,  # Input pattern
            '-c:v', 'libx264',  # H.264 codec
            '-preset', 'fast',  # Fast encoding
            '-crf', '18',  # High quality
            '-pix_fmt', 'yuv420p',  # Compatible pixel format
            '-profile:v', 'baseline',  # Baseline profile for compatibility
            '-level', '3.0',  # Compatibility level
            '-r', str(fps),  # Output FPS
            '-movflags', '+faststart',  # Enable fast start for streaming
            '-loglevel', 'error',  # Only show errors
            output_path
        ]
        
        # Run ffmpeg
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr}")
        
        # Verify the output file exists and has content
        if not os.path.exists(output_path):
            raise RuntimeError(f"Output file not created: {output_path}")
        
        file_size = os.path.getsize(output_path)
        if file_size == 0:
            raise RuntimeError(f"Output file is empty: {output_path}")
        
        # Verify the encoded video with OpenCV
        cap = cv2.VideoCapture(output_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open encoded video with OpenCV: {output_path}")
        
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # print(f"[ffmpeg] Encoded video: {frame_count} frames at {actual_fps:.2f} FPS = {frame_count/actual_fps:.2f}s duration (size: {file_size/1024:.1f} KB)")
        
        return output_path
        
    finally:
        # Clean up temporary directory
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Failed to clean up temp directory {temp_dir}: {e}")


def _encode_with_opencv(image_paths: list, fps: float, output_path: str) -> str:
    """
    Encode video using OpenCV (fallback when ffmpeg is unavailable).
    
    Note: Uses H.264 codec (avc1) which should be more compatible than mp4v.
    """
    # Read first frame to get dimensions
    first_frame = cv2.imread(image_paths[0])
    if first_frame is None:
        raise ValueError(f"Could not read first frame: {image_paths[0]}")
    
    height, width = first_frame.shape[:2]
    
    # Try different H.264 codecs in order of preference
    codecs_to_try = [
        ('avc1', 'H.264/AVC1'),  # H.264 - most compatible
        ('H264', 'H.264'),        # Alternative H.264
        ('X264', 'X264'),         # Another H.264 variant
        ('mp4v', 'MPEG-4'),       # Fallback
    ]
    
    writer = None
    for codec_str, codec_name in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec_str)
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if writer.isOpened():
                print(f"[OpenCV] Using {codec_name} codec")
                break
            writer.release()
            writer = None
        except Exception as e:
            print(f"[OpenCV] Failed to use {codec_name} codec: {e}")
            continue
    
    if writer is None or not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer with any codec")
    
    try:
        # Write all frames sequentially
        for img_path in image_paths:
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"Warning: Could not read frame {img_path}, skipping...")
                continue
            
            # Resize if dimensions don't match
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            
            writer.write(frame)
    finally:
        writer.release()
    
    # Verify the encoded video
    cap = cv2.VideoCapture(output_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open encoded video: {output_path}")
    
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # print(f"[OpenCV] Encoded video: {frame_count} frames at {actual_fps:.2f} FPS = {frame_count/actual_fps:.2f}s duration")
    
    return output_path


@retry_with_exponential_backoff
def call_vllm_model(
    messages,
    endpoint: str,
    model_name: str,
    api_key: str = "EMPTY",
    tools: list = [],  # List of tool definitions
    image_paths: list = [],
    video_path: str = None,  # Path to video file (alternative to image_paths)
    max_tokens: int = 4096,
    temperature: float = 0.0,
    tool_choice: str = "auto",  # Can be "auto", "none", or a specific tool
    return_json: bool = False,
    video_fps: float = None,  # FPS for video frames (important for temporal grounding)
    video_start_time: float = None,  # Start time in seconds for video clip
    video_end_time: float = None,  # End time in seconds for video clip
    do_sample_frames: bool = False,  # Whether to sample frames in the multimodal processor
    auto_encode_frames: bool = True,  # Auto-encode image_paths to video for proper temporal handling
    use_extra_body: bool = False,  # Whether to use extra_body for mm_processor_kwargs (disable for older vLLM)
) -> dict:
    """
    Call vLLM model with OpenAI-compatible API.
    
    ═══════════════════════════════════════════════════════════════════════════
    IMPORTANT: How vLLM Handles Video Temporal Information
    ═══════════════════════════════════════════════════════════════════════════
    
    vLLM calculates frame timestamps using: timestamp = frame_index / video_fps
    
    Where:
    - frame_index: Sequential frame number (0, 1, 2, 3, ...)
    - video_fps: Read from video file metadata
    
    For pre-extracted frame sequences, we need to ensure vLLM gets the correct fps:
    
    ✓ CORRECT WAY (auto_encode_frames=True, default):
      1. Encode frames into video with fps=sampling_rate (e.g., 2.0)
      2. vLLM decodes video and reads fps=2.0 from metadata
      3. With do_sample_frames=False, uses all 60 frames sequentially
      4. Calculates timestamps: 0/2.0=0s, 1/2.0=0.5s, ..., 59/2.0=29.5s ✓
    
    ✗ WRONG WAY (using image_paths directly):
      1. Send 60 images via OpenAI API
      2. vLLM treats them as independent images (no fps metadata)
      3. Cannot calculate correct timestamps ✗
    
    This approach leverages video file format as a metadata container, which is
    the standard way OpenAI API handles temporal video information.
    
    ═══════════════════════════════════════════════════════════════════════════
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        endpoint: vLLM server endpoint URL (e.g., "http://localhost:8000")
        model_name: Name of the model to use
        api_key: API key for authentication (default: "EMPTY" for local vLLM)
        tools: List of tool definitions for function calling
        image_paths: List of paths to images/frames to include in the request
        video_path: Path to video file (if provided, video will be encoded and sent directly)
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        tool_choice: Tool choice strategy ("auto", "none", or specific tool)
        return_json: Whether to return JSON formatted response
        video_fps: FPS (frames per second) for video. 
                   - For image_paths with auto_encode_frames=True: The FPS used during extraction (REQUIRED)
                   - For video_path: The FPS for resampling (optional, uses video's native FPS if not set)
        video_start_time: Start time in seconds for video clip (optional)
        video_end_time: End time in seconds for video clip (optional)
        do_sample_frames: Whether to sample frames in the multimodal processor (default: False)
        auto_encode_frames: If True (default), automatically encode image_paths to video for 
                            proper temporal handling. Set to False only if images are truly 
                            independent images, not video frames.
        use_extra_body: Whether to use extra_body for mm_processor_kwargs (default: True).
                        Set to False for older vLLM versions that don't support this parameter.
        
    Returns:
        dict: Response containing 'content' and optionally 'tool_calls'
    """
    
    # Auto-encode frame sequences to video for proper temporal handling
    temp_video_path = None
    if image_paths and not video_path and auto_encode_frames and video_fps is not None:
        # When we have pre-extracted frames, we need to encode them into a video
        # so that vLLM can properly handle temporal information
        # print(f"Auto-encoding {len(image_paths)} frames to video at {video_fps} FPS for proper temporal grounding...")
        temp_video_path = encode_frames_to_video(image_paths, video_fps)
        video_path = temp_video_path
        image_paths = []  # Clear image_paths since we're now using video_path
        # For encoded videos from frames, we don't want vLLM to resample
        do_sample_frames = False
    
    try:
        headers = {
            "Content-Type": "application/json",
            'Authorization': 'Bearer ' + api_key
        }
        
        # vLLM uses OpenAI-compatible API format
        # Remove trailing slash and ensure proper path construction
        endpoint = endpoint.rstrip('/')
        if '/v1/chat/completions' in endpoint:
            url = endpoint  # Already has the full path
        elif endpoint.endswith('/v1'):
            url = f"{endpoint}/chat/completions"
        else:
            url = f"{endpoint}/v1/chat/completions"

        payload = {
            "model": model_name,
            "messages": copy.deepcopy(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        if return_json:
            payload["response_format"] = {"type": "json_object"}

        # Add tools to the payload if provided
        # Only set tool_choice if tools are actually provided and non-empty
        if tools and len(tools) > 0:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        
        # Add mm_processor_kwargs for video FPS if specified
        # This is passed via extra_body to the vLLM multimodal processor
        # 
        # IMPORTANT: For pre-extracted frame sequences (image_paths), we need to:
        # 1. Set do_sample_frames=False (frames are already sampled)
        # 2. Pass the fps used during frame extraction (critical for timestamp calculation)
        # 
        # For direct video files (video_path), the user can control do_sample_frames
        mm_processor_kwargs = {}
        
        if video_fps is not None:
            mm_processor_kwargs["fps"] = video_fps
        if video_start_time is not None:
            mm_processor_kwargs["video_start"] = video_start_time
        if video_end_time is not None:
            mm_processor_kwargs["video_end"] = video_end_time
        
        # Determine do_sample_frames based on input type
        if image_paths and not video_path:
            # For pre-extracted frames, always set do_sample_frames=False
            # The frames are already sampled, and we need processor to use the provided fps
            # for correct timestamp calculation
            mm_processor_kwargs["do_sample_frames"] = False
        elif video_path:
            # For direct video input, use the user-specified do_sample_frames
            mm_processor_kwargs["do_sample_frames"] = do_sample_frames
        
        # Only add extra_body if we have any mm_processor_kwargs to pass AND use_extra_body is True
        if mm_processor_kwargs and use_extra_body:
            payload["extra_body"] = {"mm_processor_kwargs": mm_processor_kwargs}

        # Add video if provided (direct video encoding)
        if video_path:
            video_data_url = local_video_to_data_url(video_path)
            # Check if last message is from user, if not create new user message
            if not payload['messages'] or payload['messages'][-1]['role'] != 'user':
                payload['messages'].append({"role": "user", "content": []})
            else:
                # Convert string content to list if needed
                if isinstance(payload['messages'][-1]['content'], str):
                    text_content = payload['messages'][-1]['content']
                    payload['messages'][-1]['content'] = [{"type": "text", "text": text_content}]
                elif not isinstance(payload['messages'][-1]['content'], list):
                    payload['messages'][-1]['content'] = []
            
            # Add video to the content
            # IMPORTANT: Videos must use "video_url" type, not "image_url" type
            # This is the correct OpenAI API format for Qwen3-VL
            payload['messages'][-1]['content'].append({
                "type": "video_url",
                "video_url": {"url": video_data_url},
            })
        # Add images if provided (fallback for frame-based approach)
        elif image_paths:
            image_data_list = [local_image_to_data_url(image_path) for image_path in image_paths]
            # Check if last message is from user, if not create new user message
            if not payload['messages'] or payload['messages'][-1]['role'] != 'user':
                payload['messages'].append({"role": "user", "content": []})
            else:
                # Convert string content to list if needed
                if isinstance(payload['messages'][-1]['content'], str):
                    text_content = payload['messages'][-1]['content']
                    payload['messages'][-1]['content'] = [{"type": "text", "text": text_content}]
                elif not isinstance(payload['messages'][-1]['content'], list):
                    payload['messages'][-1]['content'] = []
                    
            # Add images to the content
            for image_data in image_data_list:
                payload['messages'][-1]['content'].append({
                    "type": "image_url", 
                    "image_url": {"url": image_data}
                })

        response = requests.post(url, headers=headers, json=payload, timeout=600)

        if response.status_code != 200:
            error_text = response.text
            raise Exception(f"vLLM API returned status {response.status_code}: {error_text}")

        response_data = response.json()
        
        # Get the message from the response
        message = response_data['choices'][0]['message']
        
        # Check if there's a tool call in the response
        if "tool_calls" in message and message["tool_calls"]:
            # Return the entire message object when tools are being used
            return message
        else:
            # If there's no tool call, just return the text content
            return {"content": message.get('content', '').strip(), "tool_calls": None}
    
    finally:
        # Clean up temporary video file if created
        if temp_video_path and Path(temp_video_path).exists():
            try:
                Path(temp_video_path).unlink()
            except Exception as e:
                print(f"Warning: Failed to delete temporary video file {temp_video_path}: {e}")


@retry_with_exponential_backoff
def get_vllm_embeddings(
    input_text: str | list,
    endpoint: str,
    model_name: str = None,
    api_key: str = "EMPTY",
) -> list:
    """
    Call vLLM embedding service and get embeddings for the input text.
    
    Args:
        input_text: The text or list of texts for which to generate embeddings
        endpoint: vLLM server endpoint URL (e.g., "http://localhost:8001/v1")
        model_name: Name of the embedding model (optional, vLLM will use loaded model)
        api_key: API key for authentication (default: "EMPTY" for local vLLM)
        
    Returns:
        list: The embeddings data, each item contains 'embedding' and 'index'
    """
    headers = {
        "Content-Type": "application/json",
        'Authorization': 'Bearer ' + api_key
    }
    
    # vLLM uses OpenAI-compatible embeddings API
    # Remove trailing slash and ensure proper path construction
    endpoint = endpoint.rstrip('/')
    if '/embeddings' in endpoint:
        url = endpoint  # Already has the full path
    elif endpoint.endswith('/v1'):
        url = f"{endpoint}/embeddings"
    else:
        url = f"{endpoint}/v1/embeddings"
    
    # Set up the payload for the request
    payload = {
        "input": input_text,
    }
    
    # Add model name if provided
    if model_name:
        payload["model"] = model_name
    
    # Make the request to the vLLM service
    response = requests.post(url, headers=headers, json=payload, timeout=600)
    
    # Check if the request was successful
    if response.status_code == 200:
        return response.json()['data']
    else:
        error_text = response.text
        raise Exception(f"vLLM Embedding API returned status {response.status_code}: {error_text}")


def extract_answer(message: dict) -> str | None:
    """
    Extract the plain-text answer from an assistant message that may include
    tool calls.

    The function first checks the normal `content` field (for responses that
    are not using tools). If the assistant responded via a tool call, it
    attempts to parse the JSON string stored in
    `message["tool_calls"][i]["function"]["arguments"]` and returns the value
    associated with the key `"answer"`.

    Parameters
    ----------
    message : dict
        The assistant message returned by `call_vllm_model`.

    Returns
    -------
    str | None
        The extracted answer, or ``None`` if no answer could be found.
    """
    # Direct text response
    if (content := message.get("content")):
        return content.strip()

    # Tool-based response
    for call in message.get("tool_calls", []):
        args_json = call["function"]["arguments"]
        args = json.loads(args_json)
        if (answer := args.get("answer")):
            return answer
    return None


if __name__ == "__main__":
    # Example 1: Chat completion
    # You can use any of these formats for endpoint:
    # - "http://localhost:8000" (base URL)
    # - "http://localhost:8000/v1" (with version)
    # - "http://localhost:8000/v1/chat/completions" (full path)
    chat_endpoint = "http://localhost:8000"  # Recommended: use base URL
    chat_model_name = "/public_hw/home/cit_shifangzhao/zsf/HF/models/Qwen/Qwen3-30B-A3B"
    
    # Simple text query (without tools - no need for tool_choice)
    response = call_vllm_model(
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        endpoint=chat_endpoint,
        model_name=chat_model_name,
        api_key="EMPTY",
        max_tokens=4096,
        temperature=0.0,
    )
    print("Chat response:", response)
    
    # Example 2: Embeddings
    embedding_endpoint = "http://localhost:8001"  # Recommended: use base URL
    embedding_model_name = "/public_hw/home/cit_shifangzhao/zsf/HF/models/Qwen/Qwen3-Embedding-8B"
    
    embeddings = get_vllm_embeddings(
        input_text="This is a test sentence for embedding.",
        endpoint=embedding_endpoint,
        model_name=embedding_model_name,
        api_key="EMPTY",
    )
    print(f"Embedding dimension: {len(embeddings[0]['embedding'])}")
    print(f"First few values: {embeddings[0]['embedding'][:5]}")
    
    # Example 3: Batch embeddings
    # batch_embeddings = get_vllm_embeddings(
    #     input_text=["First sentence", "Second sentence", "Third sentence"],
    #     endpoint=embedding_endpoint,
    #     model_name=embedding_model_name,
    # )
    # for i, emb_data in enumerate(batch_embeddings):
    #     print(f"Text {i}: dimension = {len(emb_data['embedding'])}")
    
    # Example 4: Function calling / Tool use
    # IMPORTANT: vLLM must be started with these flags for function calling:
    #   --enable-auto-tool-choice --tool-call-parser hermes
    # 
    # The --tool-call-parser tells vLLM how to parse the model's tool call output:
    #   - 'hermes': for Hermes/Nous models and similar formats
    #   - 'mistral': for Mistral models
    #   - 'functionary': for Functionary models
    #   - 'granite': for IBM Granite models
    # 
    # example_tools = [
    #     {
    #         "type": "function",
    #         "function": {
    #             "name": "get_weather",
    #             "description": "Get the current weather in a location",
    #             "parameters": {
    #                 "type": "object",
    #                 "properties": {
    #                     "location": {
    #                         "type": "string",
    #                         "description": "City name, e.g., Beijing, Shanghai"
    #                     }
    #                 },
    #                 "required": ["location"]
    #             }
    #         }
    #     }
    # ]
    # response = call_vllm_model(
    #     messages=[{"role": "user", "content": "What's the weather in Beijing?"}],
    #     endpoint=chat_endpoint,
    #     model_name=chat_model_name,
    #     api_key="EMPTY",
    #     tools=example_tools,
    #     tool_choice="auto",  # Let the model decide whether to use tools
    #     max_tokens=4096,
    #     temperature=0.0,
    # )
    # print(response)
    
    # Example 5: With single image
    # response = call_vllm_model(
    #     messages=[{"role": "user", "content": "What's in this image?"}],
    #     endpoint=chat_endpoint,
    #     model_name=chat_model_name,
    #     api_key="EMPTY",
    #     image_paths=["path/to/image.jpg"],
    #     max_tokens=4096,
    #     temperature=0.0,
    # )
    # print(extract_answer(response))
    
    # Example 6: With video file (RECOMMENDED - direct video encoding)
    # IMPORTANT: For video understanding, this is the recommended approach
    # The video is directly encoded and sent to the model
    # 
    # video_path = "/path/to/video.mp4"
    # fps = 2.0  # The FPS for model processing
    # 
    # response = call_vllm_model(
    #     messages=[{
    #         "role": "user",
    #         "content": "Describe what happens in this video. At what timestamp does the person start speaking?"
    #     }],
    #     endpoint=chat_endpoint,
    #     model_name=chat_model_name,
    #     api_key="EMPTY",
    #     video_path=video_path,  # Direct video file
    #     video_fps=fps,  # Critical for temporal grounding!
    #     video_start_time=10.0,  # Optional: start from 10 seconds
    #     video_end_time=30.0,    # Optional: end at 30 seconds
    #     do_sample_frames=True,
    #     max_tokens=2048,
    #     temperature=0.0,
    # )
    # print(extract_answer(response))
    # 
    # The above will send the request with:
    # - Video encoded as base64 and sent as "image_url" type (model recognizes it as video by MIME type)
    # - extra_body={"mm_processor_kwargs": {"fps": 2.0, "video_start": 10.0, "video_end": 30.0, "do_sample_frames": True}}
    
    # Example 7: With pre-extracted video frames (auto-encoding approach - RECOMMENDED)
    # Use this if you already have extracted frames
    # 
    # IMPORTANT: When using pre-extracted frames (image_paths):
    # - You MUST provide video_fps parameter (the FPS used during frame extraction)
    # - By default (auto_encode_frames=True), frames will be automatically encoded 
    #   into a temporary video file for proper temporal grounding
    # - This ensures vLLM correctly understands the video duration and timestamps
    # 
    # frame_paths = ["frame_00.jpg", "frame_01.jpg", "frame_02.jpg", ...]
    # fps = 2.0  # The FPS used when extracting frames (e.g., 1 frame every 0.5 seconds)
    # 
    # response = call_vllm_model(
    #     messages=[{
    #         "role": "user",
    #         "content": "At what timestamp does the person start speaking? Describe what happens at each second."
    #     }],
    #     endpoint=chat_endpoint,
    #     model_name=chat_model_name,
    #     api_key="EMPTY",
    #     image_paths=frame_paths,
    #     video_fps=fps,  # CRITICAL: Must match the fps used during extraction!
    #     auto_encode_frames=True,  # Default: True - frames will be encoded to video
    #     max_tokens=2048,
    #     temperature=0.0,
    # )
    # print(extract_answer(response))
    # 
    # Example: If you extracted 60 frames at 2 FPS from a 30-second video:
    # - Frame 0 -> 0.0 seconds
    # - Frame 1 -> 0.5 seconds  
    # - Frame 2 -> 1.0 seconds
    # - ...
    # - Frame 59 -> 29.5 seconds
    # 
    # With auto_encode_frames=True, these frames are encoded to a temporary video
    # at 2 FPS, creating a 30-second video that vLLM can properly understand.
    # The temp file is automatically cleaned up after the API call.
    # 
    # If you set auto_encode_frames=False, frames are sent as individual images
    # (not recommended for temporal tasks - timestamps will be incorrect)

