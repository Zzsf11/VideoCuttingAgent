"""
Audio Caption Module using Madmom-based Segmentation (vLLM Offline Version)

This module provides audio captioning functionality using Qwen3-Omni model
via vLLM offline inference with Madmom keypoint detection for intelligent audio segmentation.

Features:
    - Madmom integration: use music keypoints (beats, onsets) for segmentation
    - vLLM offline inference: directly load and run Qwen3-Omni model
    - Batch processing for efficient parallel analysis of multiple segments
    - Configurable parameters with config file defaults

Requirements:
    - vllm
    - soundfile
    - madmom
    - numpy

Usage:
    from vca.build_database.audio_caption_madmom_offline import caption_audio_with_madmom_segments

    result = caption_audio_with_madmom_segments(
        audio_path="/path/to/audio.mp3",
        output_path="output.json",
    )
"""

import json
import os
import re
import sys
import tempfile
import subprocess
from typing import Dict, List, Optional

import numpy as np
from vllm import LLM, SamplingParams

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
#                           vLLM Model Configuration                          #
# --------------------------------------------------------------------------- #
VLLM_AUDIO_MODEL = getattr(config, 'AUDIO_ANALYSIS_MODEL',
    '/public_hw/home/cit_shifangzhao/zsf/HF/models/Qwen/Qwen3-Omni-30B-A3B-Instruct')

# System prompt for Qwen3-Omni
DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
    "Group, capable of perceiving auditory and visual inputs, as well as "
    "generating text and speech."
)


# --------------------------------------------------------------------------- #
#                           vLLM Offline Inference Functions                  #
# --------------------------------------------------------------------------- #

# Global LLM instance (initialized once and reused)
_llm_instance = None

def get_llm_instance(
    model_path: str = None,
    max_model_len: int = 32768,
    tensor_parallel_size: int = 2,
    gpu_memory_utilization: float = 0.9,
) -> LLM:
    """
    Get or create the global LLM instance.

    Args:
        model_path: Path to the model
        max_model_len: Maximum model length
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: GPU memory utilization

    Returns:
        LLM instance
    """
    global _llm_instance

    if _llm_instance is None:
        if model_path is None:
            model_path = VLLM_AUDIO_MODEL

        print(f"\n{'='*80}")
        print("Initializing vLLM model (this may take a few minutes)...")
        print(f"Model: {model_path}")
        print(f"Max model length: {max_model_len}")
        print(f"Tensor parallel size: {tensor_parallel_size}")
        print(f"GPU memory utilization: {gpu_memory_utilization}")
        print(f"{'='*80}\n")

        _llm_instance = LLM(
            model=model_path,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            limit_mm_per_prompt={"audio": 1},  # Allow 1 audio per prompt
        )

        print(f"✓ Model loaded successfully\n")

    return _llm_instance


def preprocess_audio_fast(
    audio_path: str,
    target_sr: int = 16000,
) -> tuple:
    """
    Preprocess audio using ffmpeg + soundfile hybrid approach (fast and compatible).

    Strategy:
    1. Use ffmpeg to quickly convert to 16kHz mono WAV (fast!)
    2. Use soundfile to load the preprocessed WAV (no resampling needed, compatible with vLLM)

    Args:
        audio_path: Path to the audio file
        target_sr: Target sample rate (default: 16000 Hz for Qwen3-Omni)

    Returns:
        Tuple of (audio_data, sample_rate) in the same format as vLLM expects
    """
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("soundfile is required for audio processing. Install with: pip install soundfile")

    # Create temporary preprocessed file
    temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
    os.close(temp_fd)

    try:
        # Step 1: Use ffmpeg to quickly convert to target format (very fast!)
        cmd = [
            'ffmpeg',
            '-i', audio_path,
            '-ar', str(target_sr),  # Resample to 16kHz
            '-ac', '1',  # Convert to mono
            '-f', 'wav',  # WAV format
            '-y',  # Overwrite
            temp_path
        ]

        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

        # Step 2: Use soundfile to load (no resampling, just loading)
        # This ensures the data format is compatible with vLLM
        audio_data, sample_rate = sf.read(temp_path, dtype='float32')

        return audio_data, sample_rate

    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


def call_vllm_audio_model_offline(
    user_prompt: str,
    audio_path: str,
    model_path: str = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    top_p: float = 0.95,
    system_prompt: str = None,
) -> Dict:
    """
    Call vLLM Qwen3-Omni model with audio input via offline inference.

    Args:
        user_prompt: User prompt text
        audio_path: Path to the audio file
        model_path: Path to the model (default: from config)
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        system_prompt: System prompt (default: Qwen3-Omni default)

    Returns:
        dict: Response containing 'content'
    """
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    # Preprocess audio using ffmpeg + soundfile hybrid (fast and compatible)
    print(f"  Preprocessing audio (ffmpeg + soundfile)...")
    audio_data, sample_rate = preprocess_audio_fast(audio_path, target_sr=16000)
    print(f"  ✓ Audio preprocessed: {len(audio_data)/sample_rate:.2f}s @ {sample_rate}Hz")
    print(f"  ✓ Audio shape: {audio_data.shape}, dtype: {audio_data.dtype}, range: [{audio_data.min():.3f}, {audio_data.max():.3f}]")

    # Construct prompt with Qwen3-Omni format
    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>"
        f"{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    # Get LLM instance
    llm = get_llm_instance(model_path=model_path)

    # Prepare inputs
    inputs = {
        "prompt": prompt,
        "multi_modal_data": {
            "audio": (audio_data, sample_rate),
        },
    }

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    # Generate
    print(f"  Generating response...")
    outputs = llm.generate(inputs, sampling_params=sampling_params)

    # Extract generated text
    generated_text = outputs[0].outputs[0].text

    return {"content": generated_text.strip()}


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
    Extract a segment from an audio file using ffmpeg.

    Args:
        audio_path: Path to the source audio file
        start_time: Start time in seconds
        end_time: End time in seconds
        output_path: Optional output path (if None, creates a temp file)

    Returns:
        Path to the segmented audio file
    """
    # Create output path if not specified
    if output_path is None:
        temp_fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(temp_fd)

    # Use ffmpeg to extract segment (much faster than loading entire file)
    duration = end_time - start_time
    cmd = [
        'ffmpeg',
        '-ss', str(start_time),  # Start time
        '-i', audio_path,  # Input file
        '-t', str(duration),  # Duration
        '-ar', '16000',  # Resample to 16kHz
        '-ac', '1',  # Convert to mono
        '-f', 'wav',  # Output format
        '-y',  # Overwrite
        output_path
    ]

    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )

    return output_path


# --------------------------------------------------------------------------- #
#                    Batch Caption Function for Multiple Segments             #
# --------------------------------------------------------------------------- #
def generate_audio_captions_batch(
    audio_paths: List[str],
    prompt: str,
    model_path: str = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 4096,
) -> List[str]:
    """
    Generate audio captions for multiple audio files via vLLM offline inference.

    Note: Processes each audio file sequentially.

    Args:
        audio_paths: List of paths to audio files
        prompt: User prompt for caption generation
        model_path: Path to the model
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
        try:
            response = call_vllm_audio_model_offline(
                user_prompt=prompt,
                audio_path=audio_path,
                model_path=model_path,
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
    model_path: str = None,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 4096,
) -> str:
    """
    Generate overall analysis for the entire audio file via vLLM offline inference.

    Args:
        audio_path: Path to the audio file
        model_path: Path to the model
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum tokens to generate

    Returns:
        Generated overall analysis text
    """
    response = call_vllm_audio_model_offline(
        user_prompt=AUDIO_OVERALL_PROMPT,
        audio_path=audio_path,
        model_path=model_path,
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
    model_path: str = None,
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

    This version uses vLLM offline inference instead of HTTP API.

    This function performs a two-stage analysis:
    1. Detect audio keypoints using Madmom (beats, onsets, spectral changes)
    2. Segment audio based on keypoints and analyze each segment via vLLM offline

    Args:
        audio_path: Path to the audio file
        output_path: Optional path to save the caption as JSON
        model_path: Path to the model (default: from config)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        batch_size: Number of audio segments to process in parallel (ignored for now)

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
    print("MADMOM-BASED SEGMENTATION ANALYSIS (Offline vLLM)")
    print("="*80)

    # Get audio duration using ffprobe (ffmpeg's companion tool)
    audio_duration = None
    try:
        if not audio_path.startswith("http://") and not audio_path.startswith("https://"):
            # Use ffprobe to get duration
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                audio_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            audio_duration = float(result.stdout.strip())
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
        """Find the nearest keypoint to target_time within [start_time, end_time]"""
        search_start = max(start_time, target_time - search_radius)
        search_end = min(end_time, target_time + search_radius)

        candidates = [kp['time'] for kp in keypoint_list
                     if search_start <= kp['time'] <= search_end]

        if not candidates:
            return None

        return min(candidates, key=lambda t: abs(t - target_time))

    # Helper function to recursively split long segments at keypoints
    def split_long_segment(seg_start, seg_end, segments_list, all_keypoints, depth=0, max_depth=10):
        """Recursively split a long segment at keypoints near the midpoint"""
        duration = seg_end - seg_start

        if duration <= max_segment_duration:
            segments_list.append({
                "start_time": seg_start,
                "end_time": seg_end,
                "duration": duration
            })
            return

        if depth >= max_depth:
            print(f"  ⚠ Warning: Max recursion depth reached for segment [{seg_start:.2f}-{seg_end:.2f}]")
            segments_list.append({
                "start_time": seg_start,
                "end_time": seg_end,
                "duration": duration
            })
            return

        midpoint = (seg_start + seg_end) / 2.0
        search_radius = min(5.0, duration * 0.2)

        split_point = find_nearest_keypoint_in_range(
            target_time=midpoint,
            start_time=seg_start,
            end_time=seg_end,
            keypoint_list=filtered_keypoints,
            search_radius=search_radius
        )

        if split_point is not None:
            print(f"  Splitting at filtered keypoint: [{seg_start:.2f}-{seg_end:.2f}] -> [{seg_start:.2f}-{split_point:.2f}] + [{split_point:.2f}-{seg_end:.2f}]")
        else:
            split_point = find_nearest_keypoint_in_range(
                target_time=midpoint,
                start_time=seg_start,
                end_time=seg_end,
                keypoint_list=all_keypoints,
                search_radius=search_radius
            )

            if split_point is not None:
                print(f"  Splitting at original keypoint: [{seg_start:.2f}-{seg_end:.2f}] -> [{seg_start:.2f}-{split_point:.2f}] + [{split_point:.2f}-{seg_end:.2f}]")
            else:
                split_point = midpoint
                print(f"  Splitting at midpoint: [{seg_start:.2f}-{seg_end:.2f}] -> [{seg_start:.2f}-{split_point:.2f}] + [{split_point:.2f}-{seg_end:.2f}]")

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

        if current_seg['duration'] < min_segment_duration:
            merged = False

            if i + 1 < len(initial_segments):
                next_seg = initial_segments[i + 1]
                merged_duration = next_seg['end_time'] - current_seg['start_time']

                if merged_duration <= max_segment_duration:
                    merged_seg = {
                        "start_time": current_seg['start_time'],
                        "end_time": next_seg['end_time'],
                        "duration": merged_duration
                    }
                    print(f"  Merging with next: [{current_seg['start_time']:.2f}-{current_seg['end_time']:.2f}] + [{next_seg['start_time']:.2f}-{next_seg['end_time']:.2f}]")
                    segments.append(merged_seg)
                    i += 2
                    merged = True

            if not merged and segments:
                prev_seg = segments[-1]
                merged_duration = current_seg['end_time'] - prev_seg['start_time']

                if merged_duration <= max_segment_duration:
                    segments[-1] = {
                        "start_time": prev_seg['start_time'],
                        "end_time": current_seg['end_time'],
                        "duration": merged_duration
                    }
                    print(f"  Merging with previous: [{prev_seg['start_time']:.2f}-{prev_seg['end_time']:.2f}] + [{current_seg['start_time']:.2f}-{current_seg['end_time']:.2f}]")
                    i += 1
                    merged = True

            if not merged:
                print(f"  ⚠ Warning: Cannot merge short segment: [{current_seg['start_time']:.2f}-{current_seg['end_time']:.2f}] - keeping as-is")
                segments.append(current_seg)
                i += 1

            continue

        elif current_seg['duration'] > max_segment_duration:
            print(f"  Long segment detected: [{current_seg['start_time']:.2f}-{current_seg['end_time']:.2f}] ({current_seg['duration']:.2f}s)")
            split_long_segment(current_seg['start_time'], current_seg['end_time'], segments, keypoints)
            i += 1
            continue

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

    if model_path is None:
        model_path = VLLM_AUDIO_MODEL

    print(f"\nModel: {model_path}")

    print("\nGenerating overall analysis for the entire audio...")
    overall_analysis_text = generate_overall_analysis(
        audio_path=audio_path,
        model_path=model_path,
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

    segment_info_list = []

    for i, seg in enumerate(segments):
        start_time = seg['start_time']
        end_time = seg['end_time']
        duration = seg['duration']

        try:
            segment_path = segment_audio_file(audio_path, start_time, end_time)
            temp_files.append(segment_path)

            print(f"✓ Segment {i+1}/{len(segments)}: {start_time:.2f}s - {end_time:.2f}s ({duration:.1f}s)")

            segment_info_list.append((i, seg, segment_path))

        except Exception as e:
            print(f"✗ Segment {i+1}: Error extracting segment - {e}")
            segment_info_list.append((i, seg, None))

    # Step 2: Process segments sequentially
    print(f"\n{'-'*80}")
    print(f"Step 2: Processing {len([s for s in segment_info_list if s[2] is not None])} segments...")
    print(f"{'-'*80}")

    # Create a mapping from segment_path to segment info
    path_to_info = {}
    valid_segment_paths = []

    for idx, seg, segment_path in segment_info_list:
        if segment_path is not None:
            path_to_info[segment_path] = (idx, seg)
            valid_segment_paths.append(segment_path)

    # Process segments
    segment_captions = {}

    for i, segment_path in enumerate(valid_segment_paths):
        idx, seg = path_to_info[segment_path]

        print(f"\n{'·'*80}")
        print(f"Processing Segment {i+1}/{len(valid_segment_paths)}")
        print(f"  - Segment {idx+1}: {seg['start_time']:.2f}s - {seg['end_time']:.2f}s")
        print(f"{'·'*80}")

        try:
            caption_text = call_vllm_audio_model_offline(
                user_prompt=AUDIO_SEG_KEYPOINT_PROMPT,
                audio_path=segment_path,
                model_path=model_path,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )['content']

            segment_captions[segment_path] = caption_text
            print(f"✓ Segment {idx+1} completed successfully")

        except Exception as e:
            print(f"✗ Segment {idx+1} failed: {e}")
            segment_captions[segment_path] = ''

    # Step 3: Merge results
    print(f"\n{'-'*80}")
    print("Step 3: Merging results...")
    print(f"{'-'*80}")

    sections = []

    for idx, seg, segment_path in segment_info_list:
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

        caption_text = segment_captions.get(segment_path)

        if caption_text is None:
            print(f"  Segment {idx+1}: Caption generation failed")
            sections.append(section)
            continue

        segment_json = extract_json_from_text(caption_text)

        if segment_json and isinstance(segment_json, dict):
            if "summary" in segment_json:
                section["description"] = segment_json["summary"]

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

    # Prepare final result
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
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

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
    """Example usage of audio caption function with Madmom-based segmentation via vLLM offline."""
    # Example audio file path
    audio_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/Dataset/Audio/Way_down_we_go/Way Down We Go-Kaleo#1NrOG.mp3"

    # Generate caption with Madmom-based segmentation
    # Uses vLLM offline inference
    result = caption_audio_with_madmom_segments(
        audio_path=audio_path,
        output_path="./Ascend_caption_madmom_offline_output.json",
        max_tokens=config.AUDIO_ANALYSIS_MODEL_MAX_TOKEN,
    )

    print(f"\n{'='*80}")
    print("Processing Complete!")
    print(f"Total sections analyzed: {len(result.get('sections', []))}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
