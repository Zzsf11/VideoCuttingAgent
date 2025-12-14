"""
Audio Caption Module using Madmom-based Segmentation

This module provides audio captioning functionality using Qwen3-Omni model
with Madmom keypoint detection for intelligent audio segmentation.

Features:
    - Madmom integration: use music keypoints (beats, onsets) for segmentation
    - Batch processing for efficient parallel analysis of multiple segments
    - Configurable parameters with config file defaults

Requirements:
    - transformers
    - torch
    - soundfile
    - madmom
    - numpy

Usage:
    from vca.build_database.audio_caption_madmom import caption_audio_with_madmom_segments

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
from typing import Dict, List, Optional

import soundfile as sf
import numpy as np
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

# Use our custom audio processing without librosa
from ..audio_utils import process_mm_info_no_librosa
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

AUDIO_SEG_KEYPOINT_PROMPT = """You are a professional music analyst for video editing. Analyze this audio segment and describe its characteristics for matching with video footage.

Focus on:
- Musical style and instrumentation
- Emotional atmosphere and mood
- Energy dynamics and intensity changes
- Rhythmic patterns and tempo feel

Output ONLY valid JSON (no markdown, no explanation):
{
  "summary": "Brief description of genre, instrumentation, and overall mood (1-2 sentences)",
  "emotion": "Primary emotional tone (e.g., energetic, melancholic, uplifting, tense, romantic, triumphant, mysterious, nostalgic)",
  "energy": "Energy level 1-10 with trend (e.g., '7, building intensity', '3, calm and steady', '9, explosive climax', '5, gradually fading')",
  "rhythm": "Tempo and rhythmic feel (e.g., '128 BPM, driving electronic beat', '85 BPM, relaxed groove', '60 BPM, slow ambient pulse', 'free tempo, atmospheric')"
}"""

# Global variables to store loaded model and processor
_MODEL = None
_PROCESSOR = None


# --------------------------------------------------------------------------- #
#                           Model Loading Function                            #
# --------------------------------------------------------------------------- #
def load_model_and_processor(
    model_path: str = None,
    device_map: str = "auto",
    dtype: str = "auto",
    use_flash_attn2: bool = False,
):
    """
    Load Qwen3-Omni model and processor.

    Args:
        model_path: Path to the model checkpoint
        device_map: Device map for model loading
        dtype: Data type for model loading
        use_flash_attn2: Whether to use flash attention 2

    Returns:
        Tuple of (model, processor)
    """
    global _MODEL, _PROCESSOR

    # Return cached model if already loaded
    if _MODEL is not None and _PROCESSOR is not None:
        return _MODEL, _PROCESSOR

    # Use config defaults if not specified
    if model_path is None:
        model_path = getattr(config, 'AUDIO_ANALYSIS_MODEL', config.VIDEO_ANALYSIS_MODEL)

    print(f"Loading model from: {model_path}")

    # Load model with optional flash attention
    if use_flash_attn2:
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path,
            dtype=dtype,
            attn_implementation='flash_attention_2',
            device_map=device_map
        )
    else:
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path,
            device_map=device_map,
            dtype=dtype
        )

    # Load processor
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)

    # Cache the model and processor
    _MODEL = model
    _PROCESSOR = processor

    print("Model and processor loaded successfully!")
    return model, processor


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
    model,
    processor,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 20,
    max_tokens: int = 4096,
) -> List[str]:
    """
    Generate audio captions for multiple audio files in a batch.

    Args:
        audio_paths: List of paths to audio files
        prompt: User prompt for caption generation
        model: Loaded Qwen3-Omni model
        processor: Loaded Qwen3-Omni processor
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        max_tokens: Maximum tokens to generate

    Returns:
        List of generated caption strings
    """
    if not audio_paths:
        return []

    # Prepare batch messages
    batch_messages = []
    for audio_path in audio_paths:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": prompt}
                ],
            }
        ]
        batch_messages.append(messages)

    USE_AUDIO_IN_VIDEO = True

    # Process each message and prepare batch inputs
    batch_texts = []
    batch_audios = []

    for messages in batch_messages:
        # Apply chat template
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        batch_texts.append(text)

        # Process multimodal information
        audios, _, _ = process_mm_info_no_librosa(messages, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        # audios is a list, we take the first one for each message
        if audios:
            batch_audios.append(audios[0])

    # Prepare batch inputs
    inputs = processor(
        text=batch_texts,
        audio=batch_audios if batch_audios else None,
        images=None,
        videos=None,
        return_tensors="pt",
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
        padding=True,  # Enable padding for batch processing
        audio_kwargs={
            "max_length": 4800000,
            "return_attention_mask": True,
        }
    )

    # Move inputs to model device
    inputs = inputs.to(model.device).to(model.dtype)

    # Generate response for batch
    # Note: Batch inference doesn't support audio output, so we set thinker_return_dict_in_generate=False
    text_ids, _ = model.generate(
        **inputs,
        thinker_return_dict_in_generate=False,  # No audio output in batch mode
        thinker_max_new_tokens=max_tokens,
        thinker_do_sample=True,
        thinker_temperature=temperature,
        thinker_top_p=top_p,
        thinker_top_k=top_k,
        return_audio = False,
        use_audio_in_video=USE_AUDIO_IN_VIDEO
    )

    # Decode responses
    # When thinker_return_dict_in_generate=False, text_ids is the output directly
    responses = processor.batch_decode(
        text_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return responses


# --------------------------------------------------------------------------- #
#                    Generate Overall Audio Analysis                          #
# --------------------------------------------------------------------------- #
def generate_overall_analysis(
    audio_path: str,
    model,
    processor,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 20,
    max_tokens: int = 4096,
) -> str:
    """
    Generate overall analysis for the entire audio file.

    Args:
        audio_path: Path to the audio file
        model: Loaded Qwen3-Omni model
        processor: Loaded Qwen3-Omni processor
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        max_tokens: Maximum tokens to generate

    Returns:
        Generated overall analysis text
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": AUDIO_OVERALL_PROMPT}
            ],
        }
    ]

    USE_AUDIO_IN_VIDEO = True

    # Apply chat template
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    # Process multimodal information
    audios, _, _ = process_mm_info_no_librosa(messages, use_audio_in_video=USE_AUDIO_IN_VIDEO)

    # Prepare inputs
    inputs = processor(
        text=text,
        audio=audios if audios else None,
        images=None,
        videos=None,
        return_tensors="pt",
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
        audio_kwargs={
            "max_length": 4800000,
            "return_attention_mask": True,
        }
    )

    # Move inputs to model device
    inputs = inputs.to(model.device).to(model.dtype)

    # Generate response
    text_ids, _ = model.generate(
        **inputs,
        thinker_return_dict_in_generate=False,
        thinker_max_new_tokens=max_tokens,
        thinker_do_sample=True,
        thinker_temperature=temperature,
        thinker_top_p=top_p,
        thinker_top_k=top_k,
        return_audio=False,
        use_audio_in_video=USE_AUDIO_IN_VIDEO
    )

    # Decode response
    response = processor.batch_decode(
        text_ids[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return response


# --------------------------------------------------------------------------- #
#                    Time Parsing Helper                                      #
# --------------------------------------------------------------------------- #
def mmss_to_seconds(mmss: str) -> float:
    """
    Convert MM:SS or MM:SS.f format to seconds.

    Args:
        mmss: Time string in MM:SS or MM:SS.f format

    Returns:
        Time in seconds
    """
    try:
        parts = mmss.split(':')
        if len(parts) == 2:
            minutes = int(parts[0])
            secs = float(parts[1])
            return minutes * 60 + secs
        else:
            return float(mmss)
    except (ValueError, AttributeError):
        return 0.0


# --------------------------------------------------------------------------- #
#                    Split Point Search Helper                                #
# --------------------------------------------------------------------------- #
def _find_split_points_near_midpoints(
    start_time: float,
    end_time: float,
    num_parts: int,
    all_keypoints: List[Dict],
    search_radius: float = 3.0
) -> List[float]:
    """
    在均分点附近搜索最强的关键点作为分割点。

    Args:
        start_time: 片段开始时间
        end_time: 片段结束时间
        num_parts: 需要分成的部分数
        all_keypoints: 所有原始关键点列表
        search_radius: 搜索半径（秒），在均分点 ± search_radius 范围内搜索

    Returns:
        分割点时间列表（包含 start_time 和 end_time）
    """
    duration = end_time - start_time
    part_duration = duration / num_parts

    # 计算理想的均分点
    ideal_midpoints = []
    for i in range(1, num_parts):
        ideal_midpoints.append(start_time + i * part_duration)

    # 对每个均分点，在附近搜索最强的关键点
    actual_split_points = []
    for midpoint in ideal_midpoints:
        # 在 midpoint ± search_radius 范围内搜索
        candidates = [
            kp for kp in all_keypoints
            if midpoint - search_radius <= kp['time'] <= midpoint + search_radius
        ]

        if candidates:
            # 找到强度最高的关键点
            best = max(candidates, key=lambda x: x.get('normalized_intensity', x.get('intensity', 0)))
            actual_split_points.append(best['time'])
            offset = best['time'] - midpoint
            offset_str = f"+{offset:.2f}s" if offset >= 0 else f"{offset:.2f}s"
            print(f"      ✓ midpoint {midpoint:.2f}s → keypoint {best['time']:.2f}s ({offset_str}, "
                  f"type={best.get('type', 'Unknown')[:20]}, intensity={best.get('intensity', 0):.3f})")
        else:
            # 没找到，使用均分点
            actual_split_points.append(midpoint)
            print(f"      ✗ midpoint {midpoint:.2f}s → no keypoint in ±{search_radius}s, using midpoint")

    # 去重并排序
    actual_split_points = sorted(set(actual_split_points))

    # 构建完整的分割点列表
    return [start_time] + actual_split_points + [end_time]


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
    top_k: int = 20,
    use_flash_attn2: bool = True,
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
    # Type-based filtering parameters
    preferred_types: List[str] = None,
    type_filter_mode: str = None,
    type_boost_factor: float = None,
) -> Dict:
    """
    Generate caption for an audio file using Madmom keypoints for segmentation.

    This function produces a two-level hierarchical structure:
    - Level 1: High-level sections from overall audio analysis (Intro, Verse, Chorus, etc.)
    - Level 2: Fine-grained sub-segments within each section based on Madmom keypoints

    The processing pipeline:
    1. Detect audio keypoints using Madmom (beats, onsets, spectral changes) + Rule-based filtering
       - Apply merge_close, min_interval, top_k, energy_percentile, max_segments filters
    2. Use AI model to generate overall analysis and identify Level 1 sections
    3. For each Level 1 section, create Level 2 sub-segments using filtered Madmom keypoints
    4. Use AI model to analyze each sub-segment in detail (caption generation)
    5. Merge into two-level output format

    Args:
        audio_path: Path to the audio file
        output_path: Optional path to save the caption as JSON
        model_path: Model checkpoint path (default: from config)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        use_flash_attn2: Whether to use flash attention 2
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

        # Type-based filtering
        preferred_types: List of keypoint types to prioritize (e.g., ["Downbeat", "Energy", "Onset"])
        type_filter_mode: "only" (keep only specified types), "boost" (enhance weights), "exclude" (remove specified types)
        type_boost_factor: Weight multiplier when type_filter_mode="boost" (default 1.5)

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
        filter_by_type,
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
    if preferred_types is None:
        preferred_types = getattr(config, 'AUDIO_PREFERRED_TYPES', None)
    if type_filter_mode is None:
        type_filter_mode = getattr(config, 'AUDIO_TYPE_FILTER_MODE', 'boost')
    if type_boost_factor is None:
        type_boost_factor = getattr(config, 'AUDIO_TYPE_BOOST_FACTOR', 1.5)

    # Check if audio file exists
    if not audio_path.startswith("http://") and not audio_path.startswith("https://"):
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print("\n" + "="*80)
    print("MADMOM-BASED SEGMENTATION ANALYSIS")
    print("="*80)
    print("\nProcessing Pipeline:")
    print("  1. Madmom keypoint detection + Rule-based filtering")
    print("  2. AI-based Level 1 section segmentation")
    print("  3. AI-based caption for each sub-segment")
    print("  4. Merge into two-level output format")

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
    print("STAGE 1: Madmom keypoint detection + Rule-based filtering")
    print("="*80)
    print("\n[Step 1.1] Detecting audio keypoints with Madmom...")

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

    # Stage 1.5: 规则过滤 - 按照配置参数过滤分割点
    print("\n[Step 1.2] Applying rule-based filtering...")
    print(f"  Parameters: merge_close={merge_close}s, min_interval={min_interval}s, ")
    print(f"              top_k={top_k_keypoints}, energy_percentile={energy_percentile}, max_segments={max_segments}")

    # 应用所有规则过滤
    filtered_keypoints = filter_significant_keypoints(
        keypoints=keypoints,
        merge_close=merge_close,
        min_interval=min_interval,
        top_k=top_k_keypoints,
        energy_percentile=energy_percentile,
        segment_duration=0,  # 不使用时间段分割，使用sections分割
        segment_top_k=0,
        use_normalized_intensity=True
    )

    print(f"✓ After rule-based filtering: {len(filtered_keypoints)} keypoints")

    # Step 1.3: 按类型过滤/增强关键点
    if preferred_types:
        print("\n[Step 1.3] Applying type-based filtering...")
        filtered_keypoints = filter_by_type(
            keypoints=filtered_keypoints,
            preferred_types=preferred_types,
            mode=type_filter_mode,
            boost_factor=type_boost_factor
        )
        print(f"✓ After type-based filtering: {len(filtered_keypoints)} keypoints")

    # 注意: max_segments 限制移到 section-based filtering 之后执行
    # 这样可以确保每个 section 都有机会保留关键点

    # Stage 2: Generate overall analysis (Level 1 sections) using AI model
    print("\n" + "="*80)
    print("STAGE 2: AI-based Level 1 section segmentation")
    print("="*80)
    print("\nUsing AI model to identify high-level sections (Intro, Verse, Chorus, etc.)...")

    # Load model and processor
    model, processor = load_model_and_processor(
        model_path=model_path,
        use_flash_attn2=use_flash_attn2
    )

    print("\nGenerating overall analysis for the entire audio...")
    overall_analysis_text = generate_overall_analysis(
        audio_path=audio_path,
        model=model,
        processor=processor,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
    )

    # Try to parse JSON from overall analysis
    overall_json = extract_json_from_text(overall_analysis_text)
    if overall_json and isinstance(overall_json, dict):
        overall_summary = overall_json.get("summary", "")
        stage1_sections = overall_json.get("sections", [])
        print(f"✓ Overall analysis generated successfully")
        print(f"  Found {len(stage1_sections)} Level 1 sections from overall analysis")
    else:
        overall_summary = overall_analysis_text
        stage1_sections = []
        print(f"⚠ Overall analysis generated but JSON parsing failed")
        print(f"  Will create default sections based on audio duration")

    # If no sections from stage1, create default sections based on audio duration
    if not stage1_sections:
        print("\n  Creating default Level 1 sections based on audio duration...")
        # 每30秒为一个默认段落
        default_section_duration = 30.0
        section_start = 0.0
        section_idx = 1
        while section_start < audio_duration:
            section_end = min(section_start + default_section_duration, audio_duration)
            stage1_sections.append({
                "name": f"Section {section_idx}",
                "description": "",
                "Start_Time": seconds_to_mmss(section_start),
                "End_Time": seconds_to_mmss(section_end)
            })
            section_start = section_end
            section_idx += 1
        print(f"  Created {len(stage1_sections)} default sections (每{default_section_duration}s一段)")

    # Stage 2.5: 基于 sections 的分割点过滤
    # 如果启用 use_stage1_sections，使用基于段落的过滤方法
    if use_stage1_sections and stage1_sections:
        print("\n" + "-"*80)
        print("[Step 2.5] Applying section-based keypoint filtering...")
        print("-"*80)

        # 将 stage1_sections 转换为 filter_by_sections 需要的格式
        sections_for_filter = []
        for sec in stage1_sections:
            try:
                start_time = mmss_to_seconds(sec.get("Start_Time", "00:00"))
                end_time = mmss_to_seconds(sec.get("End_Time", "00:00"))
                if end_time > start_time:
                    sections_for_filter.append({
                        'name': sec.get('name', 'Unknown'),
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': end_time - start_time
                    })
            except Exception as e:
                print(f"⚠ Warning: Failed to parse section: {sec.get('name', 'Unknown')} - {e}")

        if sections_for_filter:
            # 如果 section_min_interval 未设置（0），使用 min_segment_duration
            # 这样确保选择的关键点之间间隔足够，不会产生太短的片段
            effective_min_interval = section_min_interval if section_min_interval > 0 else min_segment_duration

            print(f"  Filtering based on {len(sections_for_filter)} sections")
            print(f"  (dynamic_top_k enabled: min top_k={section_top_k}, max_segment_duration={max_segment_duration}s)")
            print(f"  (section_min_interval={effective_min_interval}s to avoid short segments)")
            filtered_keypoints = filter_by_sections(
                keypoints=filtered_keypoints,
                sections=sections_for_filter,
                section_top_k=section_top_k,
                section_min_interval=effective_min_interval,
                section_energy_percentile=section_energy_percentile,
                use_normalized_intensity=True,
                dynamic_top_k=True,
                max_segment_duration=max_segment_duration,
            )
            print(f"✓ After section-based filtering: {len(filtered_keypoints)} keypoints")
        else:
            print("⚠ No valid sections found, skipping section-based filtering")

    # 最后限制最大 segment 数量（在所有过滤之后）
    if max_segments > 0 and len(filtered_keypoints) > max_segments:
        # 按强度排序，保留最强的 max_segments 个
        intensity_key = 'normalized_intensity' if filtered_keypoints and 'normalized_intensity' in filtered_keypoints[0] else 'intensity'
        filtered_keypoints.sort(key=lambda x: x.get(intensity_key, x.get('intensity', 0)), reverse=True)
        filtered_keypoints = filtered_keypoints[:max_segments]
        filtered_keypoints.sort(key=lambda x: x['time'])
        print(f"✓ Limited to max {max_segments} segments: {len(filtered_keypoints)} keypoints")

    print(f"\n✓ Using {len(filtered_keypoints)} filtered keypoints for segmentation")

    # Debug: 打印所有分割点的详细信息
    print("\n" + "-"*80)
    print("[DEBUG] Final keypoints for segmentation:")
    print("-"*80)
    print(f"{'#':>3} | {'Time':>8} | {'Type':<30} | {'Intensity':>9} | {'Norm_Int':>9} | {'Section':<15}")
    print("-"*100)
    for idx, kp in enumerate(filtered_keypoints):
        time_str = f"{kp['time']:.2f}s"
        kp_type = kp.get('type', 'Unknown')[:30]
        intensity = kp.get('intensity', 0)
        norm_intensity = kp.get('normalized_intensity', intensity)
        section = kp.get('section', '-')[:15]
        boosted = " *" if kp.get('type_boosted', False) else ""
        print(f"{idx+1:>3} | {time_str:>8} | {kp_type:<30} | {intensity:>9.4f} | {norm_intensity:>9.4f} | {section:<15}{boosted}")
    print("-"*100)
    if any(kp.get('type_boosted', False) for kp in filtered_keypoints):
        print("  * = type_boosted (权重已增强)")
    print()

    # Stage 2.6: 将 Level 1 sections 的边界吸附到最近的分割点
    print("\n" + "-"*80)
    print("[Step 2.6] Snapping Level 1 section boundaries to nearest keypoints...")
    print("-"*80)

    # 获取所有分割点时间（包括音频开头和结尾）
    keypoint_times = sorted([kp['time'] for kp in filtered_keypoints])
    snap_points = [0.0] + keypoint_times + [audio_duration] if audio_duration else [0.0] + keypoint_times
    snap_points = sorted(set(snap_points))  # 去重并排序

    def find_nearest_snap_point(t, snap_points, exclude_exact=False):
        """找到最近的吸附点"""
        if not snap_points:
            return t
        # 找到最近的点
        min_dist = float('inf')
        nearest = t
        for sp in snap_points:
            if exclude_exact and abs(sp - t) < 0.01:
                continue
            dist = abs(sp - t)
            if dist < min_dist:
                min_dist = dist
                nearest = sp
        return nearest

    # 对每个 section 的边界进行吸附
    snapped_sections = []
    for section_idx, stage1_sec in enumerate(stage1_sections):
        original_start = mmss_to_seconds(stage1_sec.get("Start_Time", "00:00"))
        original_end = mmss_to_seconds(stage1_sec.get("End_Time", "00:00"))

        # 第一个 section 的开始固定为 0
        if section_idx == 0:
            snapped_start = 0.0
        else:
            # 使用上一个 section 的结束时间作为这个 section 的开始时间
            snapped_start = snapped_sections[-1]['snapped_end']

        # 最后一个 section 的结束固定为音频时长
        if section_idx == len(stage1_sections) - 1 and audio_duration:
            snapped_end = audio_duration
        else:
            # 找到最近的分割点（必须大于 snapped_start）
            valid_snap_points = [sp for sp in snap_points if sp > snapped_start + 1.0]  # 至少 1 秒的 section
            if valid_snap_points:
                snapped_end = find_nearest_snap_point(original_end, valid_snap_points)
            else:
                snapped_end = original_end

        snapped_sections.append({
            'original': stage1_sec,
            'original_start': original_start,
            'original_end': original_end,
            'snapped_start': snapped_start,
            'snapped_end': snapped_end
        })

        print(f"  [{stage1_sec.get('name', 'Section')}] "
              f"{seconds_to_mmss(original_start)}-{seconds_to_mmss(original_end)} -> "
              f"{seconds_to_mmss(snapped_start)}-{seconds_to_mmss(snapped_end)}")

    # 更新 stage1_sections 中的时间
    for i, snapped in enumerate(snapped_sections):
        stage1_sections[i]['Start_Time'] = seconds_to_mmss(snapped['snapped_start'])
        stage1_sections[i]['End_Time'] = seconds_to_mmss(snapped['snapped_end'])

    print(f"✓ Section boundaries snapped to {len(keypoint_times)} keypoints")

    # Stage 3: For each Level 1 section, create Level 2 sub-segments using filtered keypoints
    print("\n" + "="*80)
    print("STAGE 3: Creating Level 2 sub-segments based on filtered keypoints")
    print("="*80)
    print("\nMapping filtered keypoints to Level 1 sections...")

    # Store temporary audio files for cleanup
    temp_files = []

    # Build the final sections with two-level structure
    final_sections = []

    # Collect all sub-segments to process in batches
    all_subsegments = []  # List of (section_idx, subseg_idx, start, end)

    for section_idx, stage1_sec in enumerate(stage1_sections):
        # Parse section times
        sec_start = mmss_to_seconds(stage1_sec.get("Start_Time", "00:00"))
        sec_end = mmss_to_seconds(stage1_sec.get("End_Time", "00:00"))

        # Validate times
        if sec_end <= sec_start:
            if audio_duration:
                sec_end = audio_duration
            else:
                sec_end = sec_start + 30.0  # Default 30s section

        print(f"\n{'-'*80}")
        print(f"Level 1 Section {section_idx + 1}: {stage1_sec.get('name', 'Section')} [{sec_start:.2f}s - {sec_end:.2f}s]")
        print(f"{'-'*80}")

        # Find filtered keypoints within this section's time range
        section_keypoints = [kp for kp in filtered_keypoints
                           if sec_start <= kp['time'] < sec_end]

        print(f"  Found {len(section_keypoints)} filtered keypoints within this section")

        # Create sub-segments based on keypoints within this section
        subsegments = []
        keypoint_times_in_section = sorted([kp['time'] for kp in section_keypoints])

        # Add section boundaries
        all_times = [sec_start] + keypoint_times_in_section + [sec_end]
        all_times = sorted(set(all_times))  # Remove duplicates and sort

        # Create sub-segments between consecutive times
        for i in range(len(all_times) - 1):
            sub_start = all_times[i]
            sub_end = all_times[i + 1]
            sub_duration = sub_end - sub_start

            # Skip very short segments
            if sub_duration < 1.0:
                continue

            subsegments.append({
                "start_time": sub_start,
                "end_time": sub_end,
                "duration": sub_duration,
                # Store relative times for the detailed_analysis output
                "relative_start": sub_start - sec_start,
                "relative_end": sub_end - sec_start
            })

        # Merge short sub-segments
        merged_subsegments = []
        i = 0
        while i < len(subsegments):
            current = subsegments[i]

            if current['duration'] < min_segment_duration and merged_subsegments:
                # Merge with previous
                prev = merged_subsegments[-1]
                merged_subsegments[-1] = {
                    "start_time": prev['start_time'],
                    "end_time": current['end_time'],
                    "duration": current['end_time'] - prev['start_time'],
                    "relative_start": prev['relative_start'],
                    "relative_end": current['relative_end']
                }
            elif current['duration'] < min_segment_duration and i + 1 < len(subsegments):
                # Merge with next
                next_seg = subsegments[i + 1]
                subsegments[i + 1] = {
                    "start_time": current['start_time'],
                    "end_time": next_seg['end_time'],
                    "duration": next_seg['end_time'] - current['start_time'],
                    "relative_start": current['relative_start'],
                    "relative_end": next_seg['relative_end']
                }
            else:
                merged_subsegments.append(current)
            i += 1

        # Split long sub-segments that exceed max_segment_duration
        # Use strongest keypoints in the interval instead of equal division
        if max_segment_duration > 0:
            split_subsegments = []
            for subseg in merged_subsegments:
                if subseg['duration'] > max_segment_duration:
                    # Need to split this segment - find strongest keypoints in the interval
                    num_splits = int(np.ceil(subseg['duration'] / max_segment_duration))
                    num_split_points_needed = num_splits - 1  # N splits need N-1 split points

                    # Get all original keypoints within this subsegment's time range
                    subseg_all_keypoints = [kp for kp in keypoints
                                           if subseg['start_time'] < kp['time'] < subseg['end_time']]

                    if subseg_all_keypoints and num_split_points_needed > 0:
                        # Sort by intensity (strongest first) and take top N-1
                        subseg_all_keypoints.sort(
                            key=lambda x: x.get('normalized_intensity', x.get('intensity', 0)),
                            reverse=True
                        )

                        # Select strongest keypoints, ensuring minimum interval between them
                        selected_split_times = []
                        for kp in subseg_all_keypoints:
                            kp_time = kp['time']
                            # Check minimum interval from start, end, and other selected points
                            valid = True
                            if kp_time - subseg['start_time'] < min_segment_duration:
                                valid = False
                            if subseg['end_time'] - kp_time < min_segment_duration:
                                valid = False
                            for existing_time in selected_split_times:
                                if abs(kp_time - existing_time) < min_segment_duration:
                                    valid = False
                                    break
                            if valid:
                                selected_split_times.append(kp_time)
                                if len(selected_split_times) >= num_split_points_needed:
                                    break

                        # Build split points list
                        split_points = [subseg['start_time']] + sorted(selected_split_times) + [subseg['end_time']]

                        # Create sub-segments from split points
                        for i in range(len(split_points) - 1):
                            sp_start = split_points[i]
                            sp_end = split_points[i + 1]
                            sp_duration = sp_end - sp_start
                            if sp_duration >= 1.0:  # Skip very short segments
                                split_subsegments.append({
                                    "start_time": sp_start,
                                    "end_time": sp_end,
                                    "duration": sp_duration,
                                    "relative_start": sp_start - sec_start,
                                    "relative_end": sp_end - sec_start
                                })
                    else:
                        # No strong keypoints available, search near midpoints
                        print(f"    ⚠ No strong keypoints in [{subseg['start_time']:.1f}s-{subseg['end_time']:.1f}s], searching near midpoints")
                        split_points = _find_split_points_near_midpoints(
                            subseg['start_time'], subseg['end_time'],
                            num_splits, keypoints, search_radius=3.0
                        )
                        for i in range(len(split_points) - 1):
                            sp_start = split_points[i]
                            sp_end = split_points[i + 1]
                            split_subsegments.append({
                                "start_time": sp_start,
                                "end_time": sp_end,
                                "duration": sp_end - sp_start,
                                "relative_start": sp_start - sec_start,
                                "relative_end": sp_end - sec_start
                            })
                else:
                    split_subsegments.append(subseg)

            # 二次检查：确保没有超长片段，如有则在中间点附近找最强分割点
            final_split_subsegments = []
            for subseg in split_subsegments:
                if subseg['duration'] > max_segment_duration:
                    # 仍然超长，在中间点附近搜索最强分割点
                    num_parts = int(np.ceil(subseg['duration'] / max_segment_duration))
                    print(f"    ⚠ Segment [{subseg['start_time']:.1f}s-{subseg['end_time']:.1f}s] still too long ({subseg['duration']:.1f}s), searching for {num_parts}-way split points")
                    split_points = _find_split_points_near_midpoints(
                        subseg['start_time'], subseg['end_time'],
                        num_parts, keypoints, search_radius=3.0
                    )
                    for i in range(len(split_points) - 1):
                        p_start = split_points[i]
                        p_end = split_points[i + 1]
                        final_split_subsegments.append({
                            "start_time": p_start,
                            "end_time": p_end,
                            "duration": p_end - p_start,
                            "relative_start": p_start - sec_start,
                            "relative_end": p_end - sec_start
                        })
                else:
                    final_split_subsegments.append(subseg)
            merged_subsegments = final_split_subsegments

        # If no sub-segments, create one for the entire section
        if not merged_subsegments:
            merged_subsegments = [{
                "start_time": sec_start,
                "end_time": sec_end,
                "duration": sec_end - sec_start,
                "relative_start": 0.0,
                "relative_end": sec_end - sec_start
            }]

        print(f"  Created {len(merged_subsegments)} Level 2 sub-segments")

        # Store for batch processing
        for subseg_idx, subseg in enumerate(merged_subsegments):
            all_subsegments.append((section_idx, subseg_idx, subseg))

        # Initialize the section entry
        final_sections.append({
            "name": stage1_sec.get("name", f"Section {section_idx + 1}"),
            "description": stage1_sec.get("description", ""),
            "Start_Time": stage1_sec.get("Start_Time", seconds_to_mmss(sec_start)),
            "End_Time": stage1_sec.get("End_Time", seconds_to_mmss(sec_end)),
            "detailed_analysis": {
                "summary": "",
                "sections": []
            },
            "_subsegments": merged_subsegments  # Temporary storage
        })

    # Stage 4: Extract and analyze all sub-segments in batches using AI model
    print("\n" + "="*80)
    print("STAGE 4: AI-based caption generation for sub-segments")
    print("="*80)
    print(f"\nUsing AI model to caption {len(all_subsegments)} sub-segments (between keypoints)...")

    # Step 1: Extract all audio sub-segments
    print(f"\n{'-'*80}")
    print("Step 1: Extracting audio sub-segments...")
    print(f"{'-'*80}")

    subsegment_info_list = []  # Store (section_idx, subseg_idx, subseg_dict, segment_path)

    for section_idx, subseg_idx, subseg in all_subsegments:
        start_time = subseg['start_time']
        end_time = subseg['end_time']
        duration = subseg['duration']

        try:
            # Segment the audio
            segment_path = segment_audio_file(audio_path, start_time, end_time)
            temp_files.append(segment_path)

            print(f"✓ Section {section_idx + 1}, Sub-segment {subseg_idx + 1}: {start_time:.2f}s - {end_time:.2f}s ({duration:.1f}s)")

            subsegment_info_list.append((section_idx, subseg_idx, subseg, segment_path))

        except Exception as e:
            print(f"✗ Section {section_idx + 1}, Sub-segment {subseg_idx + 1}: Error extracting - {e}")
            subsegment_info_list.append((section_idx, subseg_idx, subseg, None))

    # Step 2: Process sub-segments in batches
    print(f"\n{'-'*80}")
    valid_count = len([s for s in subsegment_info_list if s[3] is not None])
    print(f"Step 2: Processing {valid_count} sub-segments in batches of {batch_size}...")
    print(f"{'-'*80}")

    # Create a mapping from segment_path to subsegment info
    path_to_info = {}
    valid_segment_paths = []

    for section_idx, subseg_idx, subseg, segment_path in subsegment_info_list:
        if segment_path is not None:
            path_to_info[segment_path] = (section_idx, subseg_idx, subseg)
            valid_segment_paths.append(segment_path)

    # Process in batches
    subsegment_captions = {}  # Maps segment_path to caption text

    for batch_start in range(0, len(valid_segment_paths), batch_size):
        batch_end = min(batch_start + batch_size, len(valid_segment_paths))
        batch_paths = valid_segment_paths[batch_start:batch_end]

        batch_num = batch_start // batch_size + 1
        total_batches = (len(valid_segment_paths) + batch_size - 1) // batch_size

        print(f"\n{'·'*80}")
        print(f"Processing Batch {batch_num}/{total_batches} ({len(batch_paths)} sub-segments)")
        print(f"{'·'*80}")

        for path in batch_paths:
            section_idx, subseg_idx, subseg = path_to_info[path]
            print(f"  - Section {section_idx + 1}, Sub {subseg_idx + 1}: {subseg['start_time']:.2f}s - {subseg['end_time']:.2f}s")

        # Batch generate captions
        batch_caption_texts = generate_audio_captions_batch(
            audio_paths=batch_paths,
            prompt=AUDIO_SEG_KEYPOINT_PROMPT,
            model=model,
            processor=processor,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
        )

        # Store results
        for path, caption_text in zip(batch_paths, batch_caption_texts):
            subsegment_captions[path] = caption_text

        print(f"✓ Batch {batch_num} completed successfully")

    # Step 3: Build the two-level structure
    print(f"\n{'-'*80}")
    print("Step 3: Building two-level structure...")
    print(f"{'-'*80}")

    for section_idx, subseg_idx, subseg, segment_path in subsegment_info_list:
        # Create sub-section entry with relative times
        sub_section = {
            "name": f"Section {subseg_idx + 1}",
            "description": "",
            "Start_Time": seconds_to_mmss(subseg['relative_start']),
            "End_Time": seconds_to_mmss(subseg['relative_end'])
        }

        if segment_path is None:
            print(f"  Section {section_idx + 1}, Sub {subseg_idx + 1}: No audio (skipped)")
            final_sections[section_idx]["detailed_analysis"]["sections"].append(sub_section)
            continue

        # Get caption for this sub-segment
        caption_text = subsegment_captions.get(segment_path)

        if caption_text is None:
            print(f"  Section {section_idx + 1}, Sub {subseg_idx + 1}: Caption generation failed")
            final_sections[section_idx]["detailed_analysis"]["sections"].append(sub_section)
            continue

        # Try to parse JSON from caption
        segment_json = extract_json_from_text(caption_text)

        if segment_json and isinstance(segment_json, dict):
            # Merge the analysis fields into sub_section
            if "summary" in segment_json:
                sub_section["description"] = segment_json["summary"]
            if "emotion" in segment_json:
                sub_section["Emotional_Tone"] = segment_json["emotion"]
            if "energy" in segment_json:
                sub_section["energy"] = segment_json["energy"]
            if "rhythm" in segment_json:
                sub_section["rhythm"] = segment_json["rhythm"]

            # Also store the raw detailed analysis if there are extra fields
            for key, value in segment_json.items():
                if key not in ["summary", "emotion", "energy", "rhythm"]:
                    sub_section[key] = value

            print(f"✓ Section {section_idx + 1}, Sub {subseg_idx + 1}: Detailed analysis added")
        else:
            sub_section["description"] = caption_text
            print(f"⚠ Section {section_idx + 1}, Sub {subseg_idx + 1}: Raw text added (JSON parsing failed)")

        final_sections[section_idx]["detailed_analysis"]["sections"].append(sub_section)

    # Stage 5: Merge into two-level output format
    print(f"\n{'='*80}")
    print("STAGE 5: Merging into two-level output format")
    print("="*80)

    # Remove temporary storage
    for section in final_sections:
        if "_subsegments" in section:
            del section["_subsegments"]

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
        "sections": final_sections,
        # Debug: 保存分割点详细信息
        "_debug_keypoints": [
            {
                "time": kp['time'],
                "time_mmss": seconds_to_mmss(kp['time']),
                "type": kp.get('type', 'Unknown'),
                "intensity": round(kp.get('intensity', 0), 4),
                "normalized_intensity": round(kp.get('normalized_intensity', kp.get('intensity', 0)), 4),
                "section": kp.get('section', None),
                "type_boosted": kp.get('type_boosted', False)
            }
            for kp in filtered_keypoints
        ],
        "_debug_config": {
            "preferred_types": preferred_types,
            "type_filter_mode": type_filter_mode,
            "type_boost_factor": type_boost_factor,
            "section_top_k": section_top_k,
            "section_energy_percentile": section_energy_percentile,
            "max_segments": max_segments,
            "min_segment_duration": min_segment_duration,
            "max_segment_duration": max_segment_duration,
        }
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
        print("✓ Processing Complete!")
        print(f"  - Level 1 sections: {len(final_sections)}")
        total_subsections = sum(len(sec.get('detailed_analysis', {}).get('sections', [])) for sec in final_sections)
        print(f"  - Level 2 sub-segments: {total_subsections}")
        print(f"  - Keypoints used: {len(filtered_keypoints)}")
        print(f"  - Output saved to: {output_path}")
        print(f"  - Debug info: _debug_keypoints, _debug_config (in JSON)")
        print(f"{'='*80}")

    return result_data


# --------------------------------------------------------------------------- #
#                                    main                                     #
# --------------------------------------------------------------------------- #
def main():
    """Example usage of audio caption function with Madmom-based segmentation."""
    # Example audio file path
    audio_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/Dataset/Audio/Way_down_we_go/Way Down We Go-Kaleo#1NrOG.mp3"

    # Generate caption with Madmom-based segmentation
    # All parameters will use config defaults if not specified
    result = caption_audio_with_madmom_segments(
        audio_path=audio_path,
        output_path="./captioner_Way_down_we_go_caption_madmom_output.json",
        max_tokens=config.AUDIO_ANALYSIS_MODEL_MAX_TOKEN,
        # All other parameters will be loaded from config automatically
    )

    print(f"\n{'='*80}")
    print("Processing Complete!")
    print(f"Total sections analyzed: {len(result.get('sections', []))}")
    print(f"{'='*80}")



if __name__ == "__main__":
    main()

# 0.11.0rc2.dev113+gf9e714813