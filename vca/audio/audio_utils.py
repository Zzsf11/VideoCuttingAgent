"""
Audio processing utilities without librosa dependency.

This module provides audio processing functions using soundfile and scipy
to replace librosa, avoiding the numba JIT compilation issues.
"""

import base64
from io import BytesIO
import numpy as np
import soundfile as sf
from scipy import signal
import audioread
import av


SAMPLE_RATE = 16000


def load_audio_no_librosa(path_or_buffer, sr=16000, offset=0.0, duration=None):
    """
    Load audio file without using librosa.
    
    Args:
        path_or_buffer: File path, URL, or buffer
        sr: Target sample rate (default: 16000)
        offset: Start time in seconds (default: 0.0)
        duration: Duration in seconds (default: None, load entire file)
    
    Returns:
        audio: numpy array of audio samples (mono)
    """
    try:
        # Try using soundfile first (fastest for standard formats)
        audio, orig_sr = sf.read(path_or_buffer, dtype='float32')
        
        # Convert stereo to mono if needed
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        # Apply offset
        if offset > 0:
            start_sample = int(offset * orig_sr)
            audio = audio[start_sample:]
        
        # Apply duration
        if duration is not None:
            duration_samples = int(duration * orig_sr)
            audio = audio[:duration_samples]
        
        # Resample if needed
        if orig_sr != sr:
            audio = resample_audio(audio, orig_sr, sr)
        
        return audio
        
    except Exception as e:
        # Fallback to audioread for non-standard formats
        print(f"soundfile failed, using audioread fallback: {e}")
        return load_audio_with_audioread(path_or_buffer, sr, offset, duration)


def resample_audio(audio, orig_sr, target_sr):
    """
    Resample audio to target sample rate using scipy.
    
    Args:
        audio: numpy array of audio samples
        orig_sr: Original sample rate
        target_sr: Target sample rate
    
    Returns:
        Resampled audio array
    """
    if orig_sr == target_sr:
        return audio
    
    # Calculate new length
    num_samples = int(len(audio) * target_sr / orig_sr)
    
    # Use scipy's resample for high-quality resampling
    resampled = signal.resample(audio, num_samples)
    
    return resampled.astype(np.float32)


def load_audio_with_audioread(path_or_buffer, sr=16000, offset=0.0, duration=None):
    """
    Load audio using audioread (ffmpeg backend) as fallback.
    
    Args:
        path_or_buffer: File path, URL, or buffer
        sr: Target sample rate
        offset: Start time in seconds
        duration: Duration in seconds
    
    Returns:
        audio: numpy array of audio samples
    """
    with audioread.audio_open(path_or_buffer) as audio_file:
        orig_sr = audio_file.samplerate
        channels = audio_file.channels
        
        # Calculate offset and duration in samples
        offset_samples = int(offset * orig_sr)
        if duration is not None:
            duration_samples = int(duration * orig_sr)
        else:
            duration_samples = None
        
        # Read all audio data
        audio_data = []
        for buf in audio_file:
            audio_data.append(buf)
        
        # Convert to numpy array
        audio_bytes = b''.join(audio_data)
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Reshape for multi-channel audio
        if channels > 1:
            audio = audio.reshape(-1, channels)
            # Convert to mono
            audio = audio.mean(axis=1)
        
        # Apply offset
        if offset > 0:
            audio = audio[offset_samples:]
        
        # Apply duration
        if duration_samples is not None:
            audio = audio[:duration_samples]
        
        # Resample if needed
        if orig_sr != sr:
            audio = resample_audio(audio, orig_sr, sr)
        
        return audio


def _check_if_video_has_audio(video_path):
    """Check if video file has audio track."""
    container = av.open(video_path)
    audio_streams = [stream for stream in container.streams if stream.type == "audio"]
    if not audio_streams:
        return False
    return True


def process_audio_info(conversations, use_audio_in_video):
    """
    Read and process audio info without using librosa.

    Support dict keys:

    type = audio
    - audio
    - audio_start
    - audio_end

    type = video
    - video
    - video_start
    - video_end
    """
    audios = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    
    for conversation in conversations:
        for message in conversation:
            if not isinstance(message["content"], list):
                continue
            
            for ele in message["content"]:
                if ele["type"] == "audio":
                    if "audio" in ele or "audio_url" in ele:
                        path = ele.get("audio", ele.get("audio_url"))
                        audio_start = ele.get("audio_start", 0.0)
                        audio_end = ele.get("audio_end", None)
                        
                        # Handle numpy array input
                        if isinstance(path, np.ndarray):
                            if path.ndim > 1:
                                raise ValueError("Support only mono audio")
                            audios.append(
                                path[int(SAMPLE_RATE * audio_start) : None if audio_end is None else int(SAMPLE_RATE * audio_end)]
                            )
                            continue
                        
                        # Handle base64 encoded audio
                        elif path.startswith("data:audio"):
                            _, base64_data = path.split("base64,", 1)
                            data = BytesIO(base64.b64decode(base64_data))
                        
                        # Handle HTTP(S) URLs
                        elif path.startswith("http://") or path.startswith("https://"):
                            data = path
                        
                        # Handle file:// protocol
                        elif path.startswith("file://"):
                            data = path[len("file://") :]
                        
                        # Handle regular file paths
                        else:
                            data = path
                    else:
                        raise ValueError("Unknown audio {}".format(ele))
                
                elif use_audio_in_video and ele["type"] == "video":
                    if "video" in ele or "video_url" in ele:
                        path = ele.get("video", ele.get("video_url"))
                        audio_start = ele.get("video_start", 0.0)
                        audio_end = ele.get("video_end", None)
                        
                        assert _check_if_video_has_audio(
                            path
                        ), "Video must has audio track when use_audio_in_video=True"
                        
                        if path.startswith("http://") or path.startswith("https://"):
                            data = path
                        elif path.startswith("file://"):
                            data = path[len("file://") :]
                        else:
                            data = path
                    else:
                        raise ValueError("Unknown video {}".format(ele))
                else:
                    continue
                
                # Load audio using our custom function (no librosa!)
                duration = (audio_end - audio_start) if audio_end is not None else None
                audio = load_audio_no_librosa(
                    data,
                    sr=SAMPLE_RATE,
                    offset=audio_start,
                    duration=duration
                )
                audios.append(audio)
    
    if len(audios) == 0:
        audios = None
    
    return audios


def process_mm_info_no_librosa(conversations, use_audio_in_video):
    """
    Process multimodal information without using librosa.
    
    This is a replacement for qwen_omni_utils.process_mm_info that doesn't
    depend on librosa, avoiding numba JIT compilation issues.
    
    Args:
        conversations: List of conversation messages
        use_audio_in_video: Whether to extract audio from video files
    
    Returns:
        Tuple of (audios, images, videos)
    """
    # Import vision processing from qwen_omni_utils (doesn't use librosa)
    try:
        from qwen_omni_utils.v2_5.vision_process import process_vision_info
    except ImportError:
        # Fallback: return None for images and videos
        print("Warning: qwen_omni_utils not found, returning None for vision data")
        audios = process_audio_info(conversations, use_audio_in_video)
        return audios, None, None
    
    # Process audio without librosa
    audios = process_audio_info(conversations, use_audio_in_video)
    
    # Process vision using qwen_omni_utils (doesn't depend on librosa)
    vision = process_vision_info(conversations, return_video_kwargs=False)
    
    return (audios,) + vision
