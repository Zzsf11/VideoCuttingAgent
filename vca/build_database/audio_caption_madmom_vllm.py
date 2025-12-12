"""
Audio Caption Module using Madmom-based Segmentation (vLLM Version)

This module provides audio captioning functionality using Qwen3-Omni model
via vLLM server with Madmom keypoint detection for intelligent audio segmentation.

Features:
    - Madmom integration: use music keypoints (beats, onsets) for segmentation
    - vLLM server integration: call Qwen3-Omni via HTTP API
    - Batch processing for efficient parallel analysis of multiple segments
    - Configurable parameters with config file defaults

Requirements:
    - requests
    - soundfile
    - madmom
    - numpy

Usage:
    from vca.build_database.audio_caption_madmom_vllm import caption_audio_with_madmom_segments

    result = caption_audio_with_madmom_segments(
        audio_path="/path/to/audio.mp3",
        output_path="output.json",
    )
"""

import base64
import json
import os
import re
import sys
import tempfile
from mimetypes import guess_type
from typing import Dict, List, Optional

import requests
import soundfile as sf
import numpy as np

from .. import config

# --------------------------------------------------------------------------- #
#                              Prompt templates                               #
# --------------------------------------------------------------------------- #

AUDIO_OVERALL_PROMPT = """
Describe the main sounds, speech, background noises, and mood of this audio clip. Summarize the overall style, genre, and feeling. If there are lyrics, briefly mention their meaning. If the audio has distinct sections (like intro, verse, chorus, outro), give a short description of each.

Output format (example):
{
  "summary": "A brief overview of the audio's style, genre, and mood.",
  "sections": [
    {
        "name": "Section 1",
        "description": "Describe the section 1.",
        "Start_Time": "MM:SS",
        "End_Time": "MM:SS"
    },
    ...
  ]
}
"""

AUDIO_SEG_KEYPOINT_PROMPT = """
Analyze this audio segment and provide a detailed description for video editing purposes.

Focus on the music's energy, emotion, rhythm, and any notable musical elements (instruments, vocals, beats, melody changes, etc.).

Provide specific suggestions for how video clips should be edited to match this audio segment's characteristics.

Output format (JSON):
{
  "summary": "A concise overview of this audio segment's style, genre, and overall mood.",
  "emotional_tone": "Describe the emotional quality and mood (e.g., energetic, melancholic, uplifting, tense).",
  "energy_level": "Describe the energy level (e.g., high, medium, low, building, fading).",
  "musical_description": "Detailed description of what's happening musically - instruments, rhythm, melody, harmony, vocals, etc.",
  "rhythm_and_tempo": "Describe the rhythmic characteristics and tempo (e.g., fast 4/4 beat, slow waltz, syncopated rhythm).",
  "video_pacing_suggestion": "Specific suggestions for video editing pace - shot duration, cut frequency, transition style, etc.",
  "key_moments": "Any significant musical moments or transitions within this segment (e.g., drop, build-up, vocal entry)."
}
"""

# --------------------------------------------------------------------------- #
#                           vLLM API Configuration                            #
# --------------------------------------------------------------------------- #
# Default vLLM endpoint for Qwen3-Omni audio model
# Can be overridden via config.VLLM_AUDIO_ENDPOINT
VLLM_AUDIO_ENDPOINT = getattr(config, 'VLLM_AUDIO_ENDPOINT', 'http://localhost:8890')
VLLM_AUDIO_MODEL = getattr(config, 'AUDIO_ANALYSIS_MODEL', '/public_hw/home/cit_shifangzhao/zsf/HF/models/Qwen/Qwen3-Omni-30B-A3B-Instruct')


# --------------------------------------------------------------------------- #
#                           Audio Encoding Function                           #
# --------------------------------------------------------------------------- #
def local_audio_to_data_url(audio_path: str) -> str:
    """
    Encode a local audio file into data URL format.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Data URL string (data:audio/xxx;base64,...)
    """
    # Guess the MIME type of the audio based on the file extension
    mime_type, _ = guess_type(audio_path)
    if mime_type is None:
        # Default to wav if cannot determine
        mime_type = "audio/wav"
    
    # Read and encode the audio file
    with open(audio_path, "rb") as audio_file:
        base64_encoded_data = base64.b64encode(audio_file.read()).decode("utf-8")
    
    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


# --------------------------------------------------------------------------- #
#                           vLLM API Call Function                            #
# --------------------------------------------------------------------------- #
def call_vllm_audio_model(
    messages: List[Dict],
    endpoint: str = None,
    model_name: str = None,
    api_key: str = "EMPTY",
    audio_paths: List[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    top_p: float = 0.95,
    return_json: bool = False,
) -> Dict:
    """
    Call vLLM Qwen3-Omni model with audio input via OpenAI-compatible API.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        endpoint: vLLM server endpoint URL (e.g., "http://localhost:8890")
        model_name: Name of the model to use
        api_key: API key for authentication (default: "EMPTY" for local vLLM)
        audio_paths: List of paths to audio files to include in the request
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        return_json: Whether to return JSON formatted response
        
    Returns:
        dict: Response containing 'content'
    """
    import copy
    
    # Use defaults from config if not specified
    if endpoint is None:
        endpoint = VLLM_AUDIO_ENDPOINT
    if model_name is None:
        model_name = VLLM_AUDIO_MODEL
    
    headers = {
        "Content-Type": "application/json",
        'Authorization': 'Bearer ' + api_key
    }
    
    # Construct URL
    endpoint = endpoint.rstrip('/')
    if '/v1/chat/completions' in endpoint:
        url = endpoint
    elif endpoint.endswith('/v1'):
        url = f"{endpoint}/chat/completions"
    else:
        url = f"{endpoint}/v1/chat/completions"
    
    payload = {
        "model": model_name,
        "messages": copy.deepcopy(messages),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    
    if return_json:
        payload["response_format"] = {"type": "json_object"}
    
    # Add audio files to the message content
    if audio_paths:
        # Check if last message is from user
        if not payload['messages'] or payload['messages'][-1]['role'] != 'user':
            payload['messages'].append({"role": "user", "content": []})
        else:
            # Convert string content to list if needed
            if isinstance(payload['messages'][-1]['content'], str):
                text_content = payload['messages'][-1]['content']
                payload['messages'][-1]['content'] = [{"type": "text", "text": text_content}]
            elif not isinstance(payload['messages'][-1]['content'], list):
                payload['messages'][-1]['content'] = []
        
        # Add audio files to content
        for audio_path in audio_paths:
            audio_data_url = local_audio_to_data_url(audio_path)
            # Use audio_url type for vLLM/Qwen3-Omni (similar to video_url format)
            payload['messages'][-1]['content'].insert(0, {
                "type": "audio_url",
                "audio_url": {"url": audio_data_url},
            })
    
    # Make request with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Debug: show payload size
            import json as json_module
            payload_str = json_module.dumps(payload)
            payload_size_mb = len(payload_str) / (1024 * 1024)
            print(f"  [DEBUG] Sending request to {url} (payload size: {payload_size_mb:.2f} MB)...")
            
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            
            if response.status_code != 200:
                error_text = response.text
                if attempt < max_retries - 1:
                    print(f"Retry {attempt + 1}/{max_retries}: vLLM API returned status {response.status_code}")
                    import time
                    time.sleep(2 ** attempt)
                    continue
                raise Exception(f"vLLM API returned status {response.status_code}: {error_text}")
            
            response_data = response.json()
            message = response_data['choices'][0]['message']
            return {"content": message.get('content', '').strip()}
            
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                print(f"Retry {attempt + 1}/{max_retries}: Request timed out")
                import time
                time.sleep(2 ** attempt)
                continue
            raise
        except Exception as e:
            if attempt < max_retries - 1 and "rate limit" in str(e).lower():
                print(f"Retry {attempt + 1}/{max_retries}: {e}")
                import time
                time.sleep(2 ** attempt)
                continue
            raise
    
    return {"content": ""}


# --------------------------------------------------------------------------- #
#                           Time Format Helper                                #
# --------------------------------------------------------------------------- #
def seconds_to_mmss(seconds: float) -> str:
    """
    Convert seconds to MM:SS.f format (with one decimal place).

    Args:
        seconds: Time in seconds

    Returns:
        Time string in MM:SS.f format
    """
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:04.1f}"


# --------------------------------------------------------------------------- #
#                           JSON Parsing Helper                               #
# --------------------------------------------------------------------------- #
def extract_json_from_text(text: str) -> Optional[Dict]:
    """
    Extract and parse JSON from text that may contain additional content.

    Args:
        text: Text that may contain JSON

    Returns:
        Parsed JSON dict or None if no valid JSON found
    """
    # Try to find JSON block in the text
    # Look for content between { and }
    json_match = re.search(r'\{[\s\S]*\}', text)

    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Warning: Found JSON-like structure but failed to parse: {e}")
            return None

    # If no JSON found, return None
    return None


# --------------------------------------------------------------------------- #
#                           Audio Segmentation Helper                         #
# --------------------------------------------------------------------------- #
def segment_audio_file(
    audio_path: str,
    start_time: float,
    end_time: float,
    output_path: str = None
) -> str:
    """
    Extract a segment from an audio file.

    Args:
        audio_path: Path to the source audio file
        start_time: Start time in seconds
        end_time: End time in seconds
        output_path: Optional output path (if None, creates a temp file)

    Returns:
        Path to the segmented audio file
    """
    # Read the audio file
    audio_data, sample_rate = sf.read(audio_path)

    # Calculate start and end samples
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)

    # Extract the segment
    segment = audio_data[start_sample:end_sample]

    # Create output path if not specified
    if output_path is None:
        # Create a temporary file
        temp_fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(temp_fd)  # Close the file descriptor

    # Write the segment to file
    sf.write(output_path, segment, sample_rate)

    return output_path


# --------------------------------------------------------------------------- #
#                    Batch Caption Function for Multiple Segments             #
# --------------------------------------------------------------------------- #
def generate_audio_captions_batch(
    audio_paths: List[str],
    prompt: str,
    endpoint: str = None,
    model_name: str = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 4096,
) -> List[str]:
    """
    Generate audio captions for multiple audio files via vLLM API.
    
    Note: vLLM doesn't support true batch inference for audio via OpenAI API,
    so this function processes each audio file sequentially.

    Args:
        audio_paths: List of paths to audio files
        prompt: User prompt for caption generation
        endpoint: vLLM server endpoint URL
        model_name: Name of the model to use
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum tokens to generate

    Returns:
        List of generated caption strings
    """
    if not audio_paths:
        return []

    responses = []
    
    for audio_path in audio_paths:
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        try:
            response = call_vllm_audio_model(
                messages=messages,
                endpoint=endpoint,
                model_name=model_name,
                audio_paths=[audio_path],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            responses.append(response.get('content', ''))
        except Exception as e:
            print(f"Error processing audio {audio_path}: {e}")
            responses.append('')
    
    return responses


# --------------------------------------------------------------------------- #
#                    Generate Overall Audio Analysis                          #
# --------------------------------------------------------------------------- #
def generate_overall_analysis(
    audio_path: str,
    endpoint: str = None,
    model_name: str = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 4096,
) -> str:
    """
    Generate overall analysis for the entire audio file via vLLM API.

    Args:
        audio_path: Path to the audio file
        endpoint: vLLM server endpoint URL
        model_name: Name of the model to use
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum tokens to generate

    Returns:
        Generated overall analysis text
    """
    messages = [
        {
            "role": "user",
            "content": AUDIO_OVERALL_PROMPT
        }
    ]

    response = call_vllm_audio_model(
        messages=messages,
        endpoint=endpoint,
        model_name=model_name,
        audio_paths=[audio_path],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    return response.get('content', '')


# --------------------------------------------------------------------------- #
#                    Madmom-based Segmentation Function                       #
# --------------------------------------------------------------------------- #
def caption_audio_with_madmom_segments(
    audio_path: str,
    output_path: Optional[str] = None,
    endpoint: str = None,
    model_name: str = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    top_p: float = 0.95,
    batch_size: int = None,
    # Madmom detection parameters
    onset_threshold: float = None,
    onset_smooth: float = 0.5,
    onset_pre_avg: float = 0.5,
    onset_post_avg: float = 0.5,
    onset_pre_max: float = 0.5,
    onset_post_max: float = 0.5,
    onset_combine: float = None,
    beats_per_bar: list = None,
    min_bpm: float = None,
    max_bpm: float = None,
    # Filtering parameters
    min_segment_duration: float = None,
    max_segment_duration: float = None,
    max_segments: int = None,
    merge_close: float = None,
    min_interval: float = 0.0,
    top_k_keypoints: int = 0,
    energy_percentile: float = 0.0,
    # Section-based filtering (if using stage1 sections)
    use_stage1_sections: bool = None,
    section_top_k: int = None,
    section_min_interval: float = None,
    section_energy_percentile: float = None,
) -> Dict:
    """
    Generate caption for an audio file using Madmom keypoints for segmentation.
    
    This version uses vLLM server via HTTP API instead of loading model locally.

    This function performs a two-stage analysis:
    1. Detect audio keypoints using Madmom (beats, onsets, spectral changes)
    2. Segment audio based on keypoints and analyze each segment via vLLM API

    Args:
        audio_path: Path to the audio file
        output_path: Optional path to save the caption as JSON
        endpoint: vLLM server endpoint URL (default: from config.VLLM_AUDIO_ENDPOINT)
        model_name: Model name (default: from config.AUDIO_ANALYSIS_MODEL)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        batch_size: Number of audio segments to process in parallel

        # Madmom detection parameters
        onset_threshold: Onset detection threshold (higher = fewer onsets)
        onset_smooth: Smoothing window size for onset activation
        onset_pre_avg: Pre-averaging window for onset detection
        onset_post_avg: Post-averaging window for onset detection
        onset_pre_max: Pre-max window for onset peak picking
        onset_post_max: Post-max window for onset peak picking
        onset_combine: Time window for combining nearby onsets
        beats_per_bar: Beats per bar for rhythm detection (default: [4])
        min_bpm: Minimum BPM for beat detection
        max_bpm: Maximum BPM for beat detection

        # Filtering parameters
        min_segment_duration: Minimum segment duration in seconds
        max_segment_duration: Maximum segment duration in seconds
        max_segments: Maximum number of segments to create
        merge_close: Merge keypoints closer than this threshold
        min_interval: Minimum interval between keypoints
        top_k_keypoints: Keep only top K keypoints by intensity (0 = no limit)
        energy_percentile: Keep only keypoints above this energy percentile

        # Section-based filtering (alternative to simple filtering)
        use_stage1_sections: If True, first identify sections then find keypoints per section
        section_top_k: Number of keypoints to keep per section
        section_min_interval: Minimum interval between keypoints within each section
        section_energy_percentile: Energy percentile threshold within each section

    Returns:
        Dictionary containing the complete caption with segment details
    """
    # Import the madmom detector
    madmom_module_path = os.path.join(os.path.dirname(__file__))
    if madmom_module_path not in sys.path:
        sys.path.insert(0, madmom_module_path)

    from audio_Madmom import (
        SensoryKeypointDetector,
        filter_significant_keypoints,
        filter_by_sections,
    )

    # Load default values from config if not specified
    if batch_size is None:
        batch_size = getattr(config, 'AUDIO_BATCH_SIZE', 4)
    if onset_threshold is None:
        onset_threshold = getattr(config, 'AUDIO_ONSET_THRESHOLD', 0.6)
    if onset_combine is None:
        onset_combine = getattr(config, 'AUDIO_ONSET_COMBINE', 3.0)
    if min_bpm is None:
        min_bpm = getattr(config, 'AUDIO_MIN_BPM', 55.0)
    if max_bpm is None:
        max_bpm = getattr(config, 'AUDIO_MAX_BPM', 215.0)
    if min_segment_duration is None:
        min_segment_duration = getattr(config, 'AUDIO_MIN_SEGMENT_DURATION', 3.0)
    if max_segment_duration is None:
        max_segment_duration = getattr(config, 'AUDIO_MAX_SEGMENT_DURATION', 30.0)
    if max_segments is None:
        max_segments = getattr(config, 'AUDIO_MAX_SEGMENTS', 30)
    if merge_close is None:
        merge_close = getattr(config, 'AUDIO_MERGE_CLOSE', 0.1)
    if use_stage1_sections is None:
        use_stage1_sections = getattr(config, 'AUDIO_USE_STAGE1_SECTIONS', False)
    if section_top_k is None:
        section_top_k = getattr(config, 'AUDIO_SECTION_TOP_K', 3)
    if section_min_interval is None:
        section_min_interval = getattr(config, 'AUDIO_SECTION_MIN_INTERVAL', 0.0)
    if section_energy_percentile is None:
        section_energy_percentile = getattr(config, 'AUDIO_SECTION_ENERGY_PERCENTILE', 70.0)

    # Check if audio file exists
    if not audio_path.startswith("http://") and not audio_path.startswith("https://"):
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print("\n" + "="*80)
    print("MADMOM-BASED SEGMENTATION ANALYSIS")
    print("="*80)

    # Get audio duration
    audio_duration = None
    try:
        if not audio_path.startswith("http://") and not audio_path.startswith("https://"):
            info = sf.info(audio_path)
            audio_duration = info.duration
            duration_str = f"{int(audio_duration // 60):02d}:{int(audio_duration % 60):02d}"
            print(f"\n✓ Audio duration: {duration_str} ({audio_duration:.2f} seconds)")
    except Exception as e:
        print(f"⚠ Warning: Could not determine audio duration: {e}")

    # Stage 1: Detect keypoints using Madmom
    print("\n" + "="*80)
    print("STAGE 1: Detecting audio keypoints with Madmom")
    print("="*80)

    if beats_per_bar is None:
        beats_per_bar = [4]

    detector = SensoryKeypointDetector(
        onset_threshold=onset_threshold,
        onset_smooth=onset_smooth,
        onset_pre_avg=onset_pre_avg,
        onset_post_avg=onset_post_avg,
        onset_pre_max=onset_pre_max,
        onset_post_max=onset_post_max,
        onset_combine=onset_combine,
        beats_per_bar=beats_per_bar,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
    )

    result = detector.analyze(audio_path)
    keypoints = result['keypoints']

    print(f"\n✓ Detected {len(keypoints)} keypoints")
    print(f"  - {len(result['downbeats'])} downbeats")
    print(f"  - {len(result['onsets'])} onsets")
    if result.get('spectral_flux_peaks'):
        print(f"  - {len(result['spectral_flux_peaks'])} spectral changes")
    if result.get('energy_change_peaks'):
        print(f"  - {len(result['energy_change_peaks'])} energy changes")
    if result.get('centroid_change_peaks'):
        print(f"  - {len(result['centroid_change_peaks'])} timbre changes")

    # Optional: Use stage1 sections for filtering
    sections_for_filtering = None
    if use_stage1_sections:
        print("\n" + "-"*80)
        print("Note: use_stage1_sections=True, but stage1 analysis is disabled in this version")
        print("      Using simple keypoint filtering instead")
        print("-"*80)

    # Filter keypoints
    print("\n" + "-"*80)
    print("Filtering keypoints...")
    print("-"*80)

    filtered_keypoints = keypoints

    if sections_for_filtering:
        # Use section-based filtering
        filtered_keypoints = filter_by_sections(
            keypoints=filtered_keypoints,
            sections=sections_for_filtering,
            section_top_k=section_top_k,
            section_min_interval=section_min_interval,
            section_energy_percentile=section_energy_percentile
        )
    else:
        # Use simple filtering
        filtered_keypoints = filter_significant_keypoints(
            keypoints=filtered_keypoints,
            min_interval=min_interval,
            top_k=top_k_keypoints,
            energy_percentile=energy_percentile,
            merge_close=merge_close
        )

    # Further limit by max_segments
    if max_segments > 0 and len(filtered_keypoints) > max_segments:
        print(f"\n  Limiting to top {max_segments} keypoints by intensity...")
        filtered_keypoints.sort(key=lambda x: x['intensity'], reverse=True)
        filtered_keypoints = filtered_keypoints[:max_segments]
        filtered_keypoints.sort(key=lambda x: x['time'])

    print(f"\n✓ Filtered to {len(filtered_keypoints)} keypoints")

    # Convert keypoints to time segments
    print("\n" + "-"*80)
    print("Converting keypoints to time segments...")
    print("-"*80)

    keypoint_times = sorted([kp['time'] for kp in filtered_keypoints])

    # Create initial segments between consecutive keypoints
    initial_segments = []

    # Add first segment from start to first keypoint
    if keypoint_times and keypoint_times[0] > 0:
        initial_segments.append({
            "start_time": 0.0,
            "end_time": keypoint_times[0],
            "duration": keypoint_times[0]
        })

    # Add segments between consecutive keypoints
    for i in range(len(keypoint_times) - 1):
        start = keypoint_times[i]
        end = keypoint_times[i + 1]
        initial_segments.append({
            "start_time": start,
            "end_time": end,
            "duration": end - start
        })

    # Add last segment from last keypoint to end
    if keypoint_times and audio_duration:
        last_time = keypoint_times[-1]
        if last_time < audio_duration:
            initial_segments.append({
                "start_time": last_time,
                "end_time": audio_duration,
                "duration": audio_duration - last_time
            })

    print(f"✓ Created {len(initial_segments)} initial segments")

    # Helper function to find nearest keypoint
    def find_nearest_keypoint_in_range(target_time, start_time, end_time, keypoint_list, search_radius=5.0):
        """
        Find the nearest keypoint to target_time within [start_time, end_time]
        Search within ±search_radius of target_time

        Args:
            target_time: Target time to find keypoint near
            start_time: Start time of the search range
            end_time: End time of the search range
            keypoint_list: List of keypoints to search from
            search_radius: Search radius around target_time

        Returns: keypoint time or None if no suitable keypoint found
        """
        search_start = max(start_time, target_time - search_radius)
        search_end = min(end_time, target_time + search_radius)

        # Find all keypoints within the search range
        candidates = [kp['time'] for kp in keypoint_list
                     if search_start <= kp['time'] <= search_end]

        if not candidates:
            return None

        # Return the closest one to target_time
        return min(candidates, key=lambda t: abs(t - target_time))

    # Helper function to recursively split long segments at keypoints
    def split_long_segment(seg_start, seg_end, segments_list, all_keypoints, depth=0, max_depth=10):
        """
        Recursively split a long segment at keypoints near the midpoint

        Args:
            seg_start: Segment start time
            seg_end: Segment end time
            segments_list: List to append resulting segments
            all_keypoints: List of all original keypoints (before filtering)
            depth: Current recursion depth
            max_depth: Maximum recursion depth to prevent infinite loops
        """
        duration = seg_end - seg_start

        # Base case: segment is within acceptable range
        if duration <= max_segment_duration:
            segments_list.append({
                "start_time": seg_start,
                "end_time": seg_end,
                "duration": duration
            })
            return

        # Prevent infinite recursion
        if depth >= max_depth:
            print(f"  ⚠ Warning: Max recursion depth reached for segment [{seg_start:.2f}-{seg_end:.2f}]")
            segments_list.append({
                "start_time": seg_start,
                "end_time": seg_end,
                "duration": duration
            })
            return

        # Find midpoint
        midpoint = (seg_start + seg_end) / 2.0
        search_radius = min(5.0, duration * 0.2)  # Search within 5s or 20% of duration

        # Try to find a keypoint near the midpoint
        # First, try filtered keypoints
        split_point = find_nearest_keypoint_in_range(
            target_time=midpoint,
            start_time=seg_start,
            end_time=seg_end,
            keypoint_list=filtered_keypoints,
            search_radius=search_radius
        )

        if split_point is not None:
            print(f"  Splitting at filtered keypoint: [{seg_start:.2f}-{seg_end:.2f}] -> [{seg_start:.2f}-{split_point:.2f}] + [{split_point:.2f}-{seg_end:.2f}] (filtered keypoint at {split_point:.2f}s)")
        else:
            # If no filtered keypoint found, try original keypoints
            split_point = find_nearest_keypoint_in_range(
                target_time=midpoint,
                start_time=seg_start,
                end_time=seg_end,
                keypoint_list=all_keypoints,
                search_radius=search_radius
            )

            if split_point is not None:
                print(f"  Splitting at original keypoint: [{seg_start:.2f}-{seg_end:.2f}] -> [{seg_start:.2f}-{split_point:.2f}] + [{split_point:.2f}-{seg_end:.2f}] (original keypoint at {split_point:.2f}s)")
            else:
                # No keypoint found at all, split at midpoint
                split_point = midpoint
                print(f"  Splitting at midpoint: [{seg_start:.2f}-{seg_end:.2f}] -> [{seg_start:.2f}-{split_point:.2f}] + [{split_point:.2f}-{seg_end:.2f}] (no keypoint found)")

        # Recursively split both parts
        split_long_segment(seg_start, split_point, segments_list, all_keypoints, depth + 1, max_depth)
        split_long_segment(split_point, seg_end, segments_list, all_keypoints, depth + 1, max_depth)

    # Now merge short segments and split long segments
    print("\n" + "-"*80)
    print("Merging short segments and splitting long segments...")
    print("-"*80)

    segments = []
    i = 0

    while i < len(initial_segments):
        current_seg = initial_segments[i]

        # If segment is too short, try to merge with adjacent segments
        if current_seg['duration'] < min_segment_duration:
            merged = False

            # Try to merge with next segment first
            if i + 1 < len(initial_segments):
                next_seg = initial_segments[i + 1]
                merged_duration = next_seg['end_time'] - current_seg['start_time']

                # Check if merged segment is within acceptable range
                if merged_duration <= max_segment_duration:
                    merged_seg = {
                        "start_time": current_seg['start_time'],
                        "end_time": next_seg['end_time'],
                        "duration": merged_duration
                    }
                    print(f"  Merging with next: [{current_seg['start_time']:.2f}-{current_seg['end_time']:.2f}] + [{next_seg['start_time']:.2f}-{next_seg['end_time']:.2f}] -> [{merged_seg['start_time']:.2f}-{merged_seg['end_time']:.2f}] ({merged_duration:.2f}s)")
                    segments.append(merged_seg)
                    i += 2  # Skip both merged segments
                    merged = True

            # If not merged with next, try to merge with previous
            if not merged and segments:
                prev_seg = segments[-1]
                merged_duration = current_seg['end_time'] - prev_seg['start_time']

                if merged_duration <= max_segment_duration:
                    # Replace previous segment with merged one
                    segments[-1] = {
                        "start_time": prev_seg['start_time'],
                        "end_time": current_seg['end_time'],
                        "duration": merged_duration
                    }
                    print(f"  Merging with previous: [{prev_seg['start_time']:.2f}-{prev_seg['end_time']:.2f}] + [{current_seg['start_time']:.2f}-{current_seg['end_time']:.2f}] -> [{segments[-1]['start_time']:.2f}-{segments[-1]['end_time']:.2f}] ({merged_duration:.2f}s)")
                    i += 1
                    merged = True

            # If still not merged, keep the short segment (better than losing data)
            if not merged:
                print(f"  ⚠ Warning: Cannot merge short segment (would exceed max duration): [{current_seg['start_time']:.2f}-{current_seg['end_time']:.2f}] ({current_seg['duration']:.2f}s) - keeping as-is")
                segments.append(current_seg)
                i += 1

            continue

        # If segment is too long, split it at keypoints
        elif current_seg['duration'] > max_segment_duration:
            print(f"  Long segment detected: [{current_seg['start_time']:.2f}-{current_seg['end_time']:.2f}] ({current_seg['duration']:.2f}s)")
            split_long_segment(current_seg['start_time'], current_seg['end_time'], segments, keypoints)
            i += 1
            continue

        # Segment is within acceptable range
        else:
            segments.append(current_seg)
            i += 1

    print(f"✓ Created {len(segments)} segments from keypoints")
    if segments:
        print(f"  - Min segment duration: {min(s['duration'] for s in segments):.2f}s")
        print(f"  - Max segment duration: {max(s['duration'] for s in segments):.2f}s")
        print(f"  - Avg segment duration: {np.mean([s['duration'] for s in segments]):.2f}s")

    # Stage 2: Generate overall analysis
    print("\n" + "="*80)
    print("STAGE 2: Generating overall audio analysis")
    print("="*80)

    # Use vLLM endpoint defaults if not specified
    if endpoint is None:
        endpoint = VLLM_AUDIO_ENDPOINT
    if model_name is None:
        model_name = VLLM_AUDIO_MODEL
    
    print(f"\nUsing vLLM endpoint: {endpoint}")
    print(f"Model: {model_name}")

    print("\nGenerating overall analysis for the entire audio...")
    overall_analysis_text = generate_overall_analysis(
        audio_path=audio_path,
        endpoint=endpoint,
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    # Try to parse JSON from overall analysis
    overall_json = extract_json_from_text(overall_analysis_text)
    if overall_json and isinstance(overall_json, dict):
        overall_summary = overall_json.get("summary", "")
        print(f"✓ Overall analysis generated successfully")
    else:
        overall_summary = overall_analysis_text
        print(f"⚠ Overall analysis generated but JSON parsing failed")

    # Stage 3: Analyze each segment
    print("\n" + "="*80)
    print(f"STAGE 3: Analyzing {len(segments)} segments in detail")
    print("="*80)

    # Store temporary audio files for cleanup
    temp_files = []

    # Step 1: Extract all audio segments
    print(f"\n{'-'*80}")
    print("Step 1: Extracting audio segments...")
    print(f"{'-'*80}")

    segment_info_list = []  # Store (segment_index, segment_dict, segment_path)

    for i, seg in enumerate(segments):
        start_time = seg['start_time']
        end_time = seg['end_time']
        duration = seg['duration']

        try:
            # Segment the audio
            segment_path = segment_audio_file(audio_path, start_time, end_time)
            temp_files.append(segment_path)

            print(f"✓ Segment {i+1}/{len(segments)}: {start_time:.2f}s - {end_time:.2f}s ({duration:.1f}s)")

            segment_info_list.append((i, seg, segment_path))

        except Exception as e:
            print(f"✗ Segment {i+1}: Error extracting segment - {e}")
            segment_info_list.append((i, seg, None))

    # Step 2: Process segments in batches
    print(f"\n{'-'*80}")
    print(f"Step 2: Processing {len([s for s in segment_info_list if s[2] is not None])} segments in batches of {batch_size}...")
    print(f"{'-'*80}")

    # Create a mapping from segment_path to segment info
    path_to_info = {}
    valid_segment_paths = []

    for idx, seg, segment_path in segment_info_list:
        if segment_path is not None:
            path_to_info[segment_path] = (idx, seg)
            valid_segment_paths.append(segment_path)

    # Process in batches
    segment_captions = {}  # Maps segment_path to caption text

    for batch_start in range(0, len(valid_segment_paths), batch_size):
        batch_end = min(batch_start + batch_size, len(valid_segment_paths))
        batch_paths = valid_segment_paths[batch_start:batch_end]

        batch_num = batch_start // batch_size + 1
        total_batches = (len(valid_segment_paths) + batch_size - 1) // batch_size

        print(f"\n{'·'*80}")
        print(f"Processing Batch {batch_num}/{total_batches} ({len(batch_paths)} segments)")
        print(f"{'·'*80}")

        for path in batch_paths:
            idx, seg = path_to_info[path]
            print(f"  - Segment {idx+1}: {seg['start_time']:.2f}s - {seg['end_time']:.2f}s")

        # Batch generate captions via vLLM API
        batch_caption_texts = generate_audio_captions_batch(
            audio_paths=batch_paths,
            prompt=AUDIO_SEG_KEYPOINT_PROMPT,
            endpoint=endpoint,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        # Store results
        for path, caption_text in zip(batch_paths, batch_caption_texts):
            segment_captions[path] = caption_text

        print(f"✓ Batch {batch_num} completed successfully")

    # Step 3: Merge results
    print(f"\n{'-'*80}")
    print("Step 3: Merging results...")
    print(f"{'-'*80}")

    sections = []

    for idx, seg, segment_path in segment_info_list:
        # Create section with proper format
        section = {
            "name": f"Section {idx + 1}",
            "description": "",
            "Start_Time": seconds_to_mmss(seg['start_time']),
            "End_Time": seconds_to_mmss(seg['end_time'])
        }

        if segment_path is None:
            print(f"  Segment {idx+1}: No audio (skipped)")
            sections.append(section)
            continue

        # Get caption for this segment
        caption_text = segment_captions.get(segment_path)

        if caption_text is None:
            print(f"  Segment {idx+1}: Caption generation failed")
            sections.append(section)
            continue

        # Try to parse JSON from caption
        segment_json = extract_json_from_text(caption_text)

        if segment_json and isinstance(segment_json, dict):
            # Extract description from summary if available
            if "summary" in segment_json:
                section["description"] = segment_json["summary"]

            # Add detailed_analysis
            section["detailed_analysis"] = segment_json
            print(f"✓ Segment {idx+1}: Detailed analysis added")
        else:
            section["description"] = caption_text
            section["detailed_analysis_raw"] = caption_text
            print(f"⚠ Segment {idx+1}: Raw text added (JSON parsing failed)")

        sections.append(section)

    # Cleanup temporary files
    print(f"\n{'-'*80}")
    print("Cleaning up temporary files...")
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception as e:
            print(f"⚠ Warning: Could not remove temp file {temp_file}: {e}")

    # Prepare final result in the target format
    result_data = {
        "audio_path": audio_path,
        "overall_analysis": {
            "prompt": AUDIO_OVERALL_PROMPT.strip(),
            "summary": overall_summary
        },
        "sections": sections
    }

    # Save to file if output_path is specified
    if output_path:
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Save to file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*80}")
        print(f"✓ Complete caption saved to: {output_path}")
        print(f"{'='*80}")

    return result_data


# --------------------------------------------------------------------------- #
#                                    main                                     #
# --------------------------------------------------------------------------- #
def main():
    """Example usage of audio caption function with Madmom-based segmentation via vLLM."""
    # Example audio file path
    audio_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/Dataset/Audio/Way_down_we_go/Way Down We Go-Kaleo#1NrOG.mp3"

    # Generate caption with Madmom-based segmentation
    # Uses vLLM server via HTTP API
    # All parameters will use config defaults if not specified
    result = caption_audio_with_madmom_segments(
        audio_path=audio_path,
        output_path="./Ascend_caption_madmom_output.json",
        # endpoint: vLLM server endpoint (default: config.VLLM_AUDIO_ENDPOINT or http://localhost:8890)
        # model_name: model name (default: config.AUDIO_ANALYSIS_MODEL)
        max_tokens=config.AUDIO_ANALYSIS_MODEL_MAX_TOKEN,
        # All other parameters will be loaded from config automatically
    )

    print(f"\n{'='*80}")
    print("Processing Complete!")
    print(f"Total sections analyzed: {len(result.get('sections', []))}")
    print(f"{'='*80}")



if __name__ == "__main__":
    main()

