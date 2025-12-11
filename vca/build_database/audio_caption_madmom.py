"""
Audio Caption Module using Transformers with Madmom Integration

This module provides audio captioning functionality using Qwen3-Omni model
through the Transformers library, with custom audio processing that doesn't
rely on librosa (avoiding numba JIT compilation issues).

Features:
    - Basic audio captioning with structure and section identification
    - Segment-wise detailed analysis with audio cutting and re-captioning
    - Two-stage analysis: overall structure + detailed keypoint analysis per section
    - Batch processing for efficient parallel analysis of multiple segments
    - Madmom integration: use music keypoints (beats, onsets) for segmentation

Requirements:
    - transformers (install from GitHub: pip install git+https://github.com/huggingface/transformers)
    - torch
    - soundfile (pip install soundfile)
    - scipy (pip install scipy)
    - audioread (pip install audioread)
    - av (pip install av)
    - madmom (pip install madmom)
    - flash-attn (optional, for flash attention 2: pip install -U flash-attn --no-build-isolation)

Usage:
    # Basic caption
    from vca.build_database.audio_caption_madmom import caption_audio

    caption = caption_audio(
        audio_path="/path/to/audio.mp3",
        model_path="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        use_flash_attn2=True  # Optional, requires flash-attn installed
    )

    # Segment-wise detailed caption with batch processing
    from vca.build_database.audio_caption_madmom import caption_audio_with_segments

    result = caption_audio_with_segments(
        audio_path="/path/to/audio.mp3",
        output_path="output.json",
        use_flash_attn2=True,
        batch_size=4  # Process 4 segments in parallel
    )

    # Use Madmom keypoints for segmentation
    from vca.build_database.audio_caption_madmom import caption_audio_with_madmom_segments

    result = caption_audio_with_madmom_segments(
        audio_path="/path/to/audio.mp3",
        output_path="output.json",
        use_flash_attn2=True,
        batch_size=4,
        # Madmom parameters
        onset_threshold=0.6,
        min_segment_duration=3.0,
        max_segments=30
    )
"""

import json
import os
import re
import tempfile
from typing import Dict, List, Optional, Tuple

# ============================================================================ #
# Audio processing without librosa dependency
# We use soundfile + scipy instead of librosa to avoid numba JIT issues
# ============================================================================ #

import torch
import soundfile as sf
import numpy as np
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
# Use our custom audio processing without librosa
from ..audio_utils import process_mm_info_no_librosa
# Use relative import since this file is inside vca package
from .. import config

# --------------------------------------------------------------------------- #
#                              Prompt templates                               #
# --------------------------------------------------------------------------- #

AUDIO_STRUCTURE_SEG_PROMPT = """
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
Analyze the provided audio track to generate a detailed editing guide for a [describe the theme of your video, e.g., fast-paced travel vlog, a cinematic short film, a sentimental wedding video].

My goal is to align the video edits perfectly with the music's energy and emotion. Please focus on [mention your specific focus, e.g., the driving drum beat, the underlying synth melody, the emotional arc of the vocals].

Please be as detailed and prescriptive as possible.

Output format (example):
{
  "summary": "A brief overview of the audio's style, genre, and mood.",
  "sections": [
    {
        "name": "Section 1", 
        "description": "Describe the section 1.", 
        "Emotional_Tone":"What is the mood?", 
        "Start_Time": "MM:SS", 
        "End_Time": "MM:SS",
        "Musical_Description": "What is happening musically?",
        "Video_Pacing_Suggestion": "Based on the music, what should the video editing pace be like here?",
    },
    ...
  ]
}
"""

SYSTEM_PROMPT = "You are a helpful assistant specialized in analyzing and describing audio content."

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
#                           Audio Caption with Transformers                   #
# --------------------------------------------------------------------------- #
def generate_audio_caption_with_transformers(
    audio_path: str,
    prompt: str,
    model,
    processor,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 20,
    max_tokens: int = 4096,
) -> str:
    """
    Generate audio caption using transformers model.
    
    Args:
        audio_path: Path to the audio file
        prompt: User prompt for caption generation
        model: Loaded Qwen3-Omni model
        processor: Loaded Qwen3-Omni processor
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        max_tokens: Maximum tokens to generate
    
    Returns:
        Generated caption string
    """
    # Prepare messages in Qwen3-Omni format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": prompt}
            ],
        }
    ]
    
    # Use audio in video setting
    USE_AUDIO_IN_VIDEO = True
    
    # Apply chat template
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
    # Process multimodal information (without librosa)
    audios, images, videos = process_mm_info_no_librosa(messages, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    
    # Prepare inputs
    # Note: WhisperFeatureExtractor has a default n_samples of 480000 (30s * 16000)
    # We need to set max_length to support longer audio
    # Based on preprocessor_config.json: sampling_rate=16000
    # To support 5 minutes: max_length = 300 * 16000 = 4800000
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
        # Pass audio processing parameters via audio_kwargs
        audio_kwargs={
            "max_length": 4800000,  # 300 seconds * 16000 Hz = support up to 5 minutes
            "return_attention_mask": True,
        }
    )
    
    # Move inputs to model device
    inputs = inputs.to(model.device).to(model.dtype)
    
    # Generate response
    # When thinker_return_dict_in_generate=True, model.generate() returns (text_ids, audio)
    text_ids, _ = model.generate(
        **inputs,
        thinker_return_dict_in_generate=True,
        thinker_max_new_tokens=max_tokens,
        thinker_do_sample=True,
        thinker_temperature=temperature,
        thinker_top_p=top_p,
        thinker_top_k=top_k,
        return_audio = False,
        use_audio_in_video=USE_AUDIO_IN_VIDEO
    )
    
    # Decode response
    # Need to access .sequences attribute from text_ids
    response = processor.batch_decode(
        text_ids.sequences[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    
    return response


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


def validate_audio_caption_json(data: Dict, required_keys: List[str] = None, check_time_fields: bool = False) -> bool:
    """
    Validate if the parsed JSON contains all required keys.
    
    Args:
        data: Parsed JSON dictionary
        required_keys: List of required keys to check (default: ["summary", "sections"])
        check_time_fields: Whether to check for time fields (start_time/Start_Time, end_time/End_Time)
    
    Returns:
        True if all required keys are present and valid, False otherwise
    """
    if required_keys is None:
        required_keys = ["summary", "sections"]
    
    # Check if all required keys exist
    for key in required_keys:
        if key not in data:
            print(f"Warning: Missing required key '{key}' in parsed JSON")
            return False
    
    # Additional validation for sections
    if "sections" in required_keys:
        sections = data.get("sections")
        if not isinstance(sections, list):
            print(f"Warning: 'sections' should be a list, got {type(sections)}")
            return False
        
        if len(sections) == 0:
            print(f"Warning: 'sections' list is empty")
            return False
        
        # Check each section has required fields
        section_required_keys = ["name", "description"]
        for i, section in enumerate(sections):
            if not isinstance(section, dict):
                print(f"Warning: Section {i} is not a dictionary")
                return False
            
            for sec_key in section_required_keys:
                if sec_key not in section:
                    print(f"Warning: Section {i} missing required key '{sec_key}'")
                    return False
            
            # Check for time fields if required
            if check_time_fields:
                has_start_time = "start_time" in section or "Start_Time" in section
                has_end_time = "end_time" in section or "End_Time" in section
                
                if not has_start_time:
                    print(f"Warning: Section {i} missing start time field (start_time or Start_Time)")
                    return False
                
                if not has_end_time:
                    print(f"Warning: Section {i} missing end time field (end_time or End_Time)")
                    return False
    
    return True


def validate_section_coverage(sections: List[Dict], audio_duration: float, tolerance_seconds: float = 5.0) -> bool:
    """
    Validate if sections cover the entire audio duration.
    
    Args:
        sections: List of section dictionaries
        audio_duration: Total audio duration in seconds
        tolerance_seconds: Acceptable difference in seconds (default: 5.0)
    
    Returns:
        True if sections adequately cover the audio, False otherwise
    """
    if not sections or audio_duration is None:
        return True  # Can't validate without data
    
    try:
        # Get the last section's end time
        last_section = sections[-1]
        last_end_str = last_section.get("end_time", last_section.get("End_Time", ""))
        
        if not last_end_str:
            print(f"Warning: Last section has no end time")
            return False
        
        # Parse end time
        last_end_seconds = parse_time_to_seconds(last_end_str)
        
        # Calculate difference
        diff_seconds = abs(audio_duration - last_end_seconds)
        coverage_percent = (last_end_seconds / audio_duration) * 100
        
        # Check if coverage is adequate
        if diff_seconds > tolerance_seconds:
            print(f"Warning: Section coverage mismatch!")
            print(f"  Last section ends at: {last_end_str} ({last_end_seconds:.1f}s)")
            print(f"  Audio duration: {int(audio_duration//60):02d}:{int(audio_duration%60):02d} ({audio_duration:.1f}s)")
            print(f"  Difference: {diff_seconds:.1f} seconds ({100 - coverage_percent:.1f}% missing)")
            return False
        
        return True
        
    except Exception as e:
        print(f"Warning: Could not validate section coverage: {e}")
        return False


# --------------------------------------------------------------------------- #
#                           Audio Segmentation Helper                         #
# --------------------------------------------------------------------------- #
def parse_time_to_seconds(time_str: str) -> float:
    """
    Parse time string in format "MM:SS" or "HH:MM:SS" to seconds.
    
    Args:
        time_str: Time string like "01:30" or "1:30:45"
    
    Returns:
        Time in seconds as float
    """
    parts = time_str.strip().split(":")
    if len(parts) == 2:
        # MM:SS format
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    elif len(parts) == 3:
        # HH:MM:SS format
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    else:
        raise ValueError(f"Invalid time format: {time_str}")


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
#                           Main Caption Function                             #
# --------------------------------------------------------------------------- #
def caption_audio(
    audio_path: str,
    prompt: str = AUDIO_STRUCTURE_SEG_PROMPT,
    output_path: Optional[str] = None,
    model_path: str = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 20,
    use_flash_attn2: bool = True,
    max_retries: int = 3,
    required_keys: List[str] = None,
) -> str:
    """
    Generate caption for an audio file using transformers.
    
    Args:
        audio_path: Path to the audio file
        prompt: Custom prompt for audio captioning (default: AUDIO_CAPTION_PROMPT)
        output_path: Optional path to save the caption as JSON
        model_path: Model checkpoint path (default: from config)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        use_flash_attn2: Whether to use flash attention 2
        max_retries: Maximum number of retries if JSON parsing or validation fails (default: 3)
        required_keys: List of required keys to validate in the parsed JSON (default: ["summary", "sections"])
    
    Returns:
        String containing the audio caption
    """
    # Check if audio file exists
    if not audio_path.startswith("http://") and not audio_path.startswith("https://"):
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Load model and processor
    model, processor = load_model_and_processor(
        model_path=model_path,
        use_flash_attn2=use_flash_attn2
    )
    
    # Set default required keys if not specified
    if required_keys is None:
        required_keys = ["summary", "sections"]
    
    # Retry loop for generating valid JSON
    caption = None
    parsed_json = None
    
    for attempt in range(max_retries):
        print(f"\n{'='*60}")
        print(f"Attempt {attempt + 1}/{max_retries}: Processing audio: {audio_path}")
        print(f"{'='*60}")
        
        # Generate caption
        caption = generate_audio_caption_with_transformers(
            audio_path=audio_path,
            prompt=prompt,
            model=model,
            processor=processor,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
        )
        
        if not caption:
            print(f"Attempt {attempt + 1} failed: No caption generated")
            continue
        
        # Fix: Only print the generated caption if it is not None or empty
        if caption:
            print(f"\nGenerated caption:\n{caption}")
        
        # Try to extract and validate JSON from caption
        parsed_json = extract_json_from_text(caption)
        
        if parsed_json is None:
            print(f"Attempt {attempt + 1} failed: Could not extract JSON from caption")
            if attempt < max_retries - 1:
                print("Retrying...")
            continue
        
        # Validate the parsed JSON
        if validate_audio_caption_json(parsed_json, required_keys):
            print(f"✓ Successfully generated and validated JSON on attempt {attempt + 1}")
            break
        else:
            print(f"Attempt {attempt + 1} failed: JSON validation failed (missing required keys)")
            if attempt < max_retries - 1:
                print("Retrying...")
            parsed_json = None
            continue
    
    # After all attempts, save the result
    if caption:
        # Save to file if output_path is specified
        if output_path:
            if parsed_json:
                # Save the parsed JSON structure directly
                result = {
                    "audio_path": audio_path,
                    "prompt": prompt,
                    "caption": parsed_json
                }
                print(f"\n✓ Successfully parsed and validated JSON")
            else:
                # If no valid JSON found after all retries, save the raw caption
                result = {
                    "audio_path": audio_path,
                    "prompt": prompt,
                    "caption": caption
                }
                print(f"\n⚠ Warning: Could not extract valid JSON after {max_retries} attempts, saving raw text")
            
            # Create output directory if needed
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Save to file
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"✓ Caption saved to: {output_path}")
    else:
        print(f"\n✗ Failed to generate caption after {max_retries} attempts")
    
    return caption


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
        audios, images, videos = process_mm_info_no_librosa(messages, use_audio_in_video=USE_AUDIO_IN_VIDEO)
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
#                    Segment-wise Caption Function                            #
# --------------------------------------------------------------------------- #
def caption_audio_with_segments(
    audio_path: str,
    output_path: Optional[str] = None,
    model_path: str = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 20,
    use_flash_attn2: bool = True,
    max_retries: int = 3,
    batch_size: int = 4,
) -> Dict:
    """
    Generate caption for an audio file with segment-wise detailed analysis.
    
    This function performs a two-stage analysis:
    1. First, analyze the entire audio to identify sections
    2. Then, analyze each section individually for detailed keypoint information
    
    Args:
        audio_path: Path to the audio file
        output_path: Optional path to save the caption as JSON
        model_path: Model checkpoint path (default: from config)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        use_flash_attn2: Whether to use flash attention 2
        max_retries: Maximum number of retries if JSON parsing or validation fails
        batch_size: Number of audio segments to process in parallel (default: 4)
    
    Returns:
        Dictionary containing the complete caption with segment details
    """
    # Check if audio file exists
    if not audio_path.startswith("http://") and not audio_path.startswith("https://"):
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Get audio duration to help model identify all sections
    audio_duration = None
    try:
        if not audio_path.startswith("http://") and not audio_path.startswith("https://"):
            info = sf.info(audio_path)
            audio_duration = info.duration
            duration_str = f"{int(audio_duration // 60):02d}:{int(audio_duration % 60):02d}"
            print(f"\n✓ Audio duration: {duration_str} ({audio_duration:.2f} seconds)")
    except Exception as e:
        print(f"⚠ Warning: Could not determine audio duration: {e}")
    
    print("\n" + "="*80)
    print("STAGE 1: Analyzing entire audio to identify sections")
    print("="*80)
    
    # Enhance prompt with duration info if available
    stage1_prompt = AUDIO_STRUCTURE_SEG_PROMPT
    if audio_duration is not None:
        duration_str = f"{int(audio_duration // 60):02d}:{int(audio_duration % 60):02d}"
        stage1_prompt = f"Note: This audio file is {duration_str} long. Please analyze the ENTIRE audio from 00:00 to {duration_str}.\n\n" + AUDIO_STRUCTURE_SEG_PROMPT
    
    # Retry loop for Stage 1 with coverage validation
    max_stage1_retries = max_retries
    sections = None
    summary = None
    
    for stage1_attempt in range(max_stage1_retries):
        if stage1_attempt > 0:
            print(f"\n{'='*80}")
            print(f"STAGE 1 RETRY {stage1_attempt}/{max_stage1_retries - 1}: Re-analyzing entire audio")
            print(f"{'='*80}")
        
        # Stage 1: Get overall structure and sections
        caption_text = caption_audio(
            audio_path=audio_path,
            prompt=stage1_prompt,
            output_path=None,  # Don't save intermediate result
            model_path=model_path,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            use_flash_attn2=use_flash_attn2,
            max_retries=1,  # Use 1 retry per attempt, we handle retries at this level
            required_keys=["summary", "sections"]
        )
        
        # Parse the JSON from caption
        parsed_json = extract_json_from_text(caption_text)
        
        if not parsed_json:
            print(f"✗ Attempt {stage1_attempt + 1}: Failed to extract JSON from caption")
            if stage1_attempt < max_stage1_retries - 1:
                print("Retrying...")
                continue
            else:
                raise ValueError("Failed to extract JSON from initial caption after all retries")
        
        # Validate basic structure
        if not validate_audio_caption_json(parsed_json, required_keys=["summary", "sections"]):
            print(f"✗ Attempt {stage1_attempt + 1}: Invalid section structure")
            if stage1_attempt < max_stage1_retries - 1:
                print("Retrying...")
                continue
            else:
                raise ValueError("Failed to get valid section information from initial caption after all retries")
        
        # Extract sections
        temp_sections = parsed_json.get("sections", [])
        temp_summary = parsed_json.get("summary", "")
        
        print(f"\n✓ Found {len(temp_sections)} sections")
        
        # Validate that sections have time fields
        print("Validating section time fields...")
        if not validate_audio_caption_json(parsed_json, required_keys=["sections"], check_time_fields=True):
            print(f"✗ Attempt {stage1_attempt + 1}: Sections missing required time fields")
            if stage1_attempt < max_stage1_retries - 1:
                print("Retrying...")
                continue
            else:
                raise ValueError("Sections are missing required time fields after all retries")
        
        print("✓ All sections have time fields")
        
        # Validate section coverage matches audio duration
        coverage_valid = True
        if audio_duration is not None:
            print(f"\nValidating audio coverage...")
            if validate_section_coverage(temp_sections, audio_duration, tolerance_seconds=5.0):
                last_section = temp_sections[-1]
                last_end_str = last_section.get("end_time", last_section.get("End_Time", ""))
                print(f"✓ Sections cover full audio duration (ends at {last_end_str})")
                coverage_valid = True
            else:
                # Coverage validation failed
                print(f"✗ Attempt {stage1_attempt + 1}: Audio coverage validation failed")
                coverage_valid = False
        
        # Check if this attempt succeeded
        if coverage_valid:
            sections = temp_sections
            summary = temp_summary
            print(f"\n✓ Stage 1 completed successfully on attempt {stage1_attempt + 1}")
            break
        else:
            # Coverage failed - retry if attempts remain
            if stage1_attempt < max_stage1_retries - 1:
                print(f"\nRetrying Stage 1 to get better audio coverage...")
            else:
                # Last attempt failed - use what we got but warn the user
                sections = temp_sections
                summary = temp_summary
                print(f"\n⚠ WARNING: Using sections from last attempt despite coverage issues")
    
    if sections is None or summary is None:
        raise ValueError("Failed to get valid sections after all retries")
    
    # Stage 2: Analyze each section in detail
    print("\n" + "="*80)
    print("STAGE 2: Analyzing each section in detail")
    print("="*80)
    
    # Load model and processor (reuse if already loaded)
    model, processor = load_model_and_processor(
        model_path=model_path,
        use_flash_attn2=use_flash_attn2
    )
    
    # Store temporary audio files for cleanup
    temp_files = []
    
    # Step 1: Extract all audio segments first
    print(f"\n{'-'*80}")
    print("Step 1: Extracting audio segments...")
    print(f"{'-'*80}")
    
    segment_info_list = []  # Store (section_index, section, segment_path)
    
    for i, section in enumerate(sections):
        section_name = section.get("name", f"Section {i+1}")
        
        # Get time range
        start_time_str = section.get("start_time", section.get("Start_Time", ""))
        end_time_str = section.get("end_time", section.get("End_Time", ""))
        
        if not start_time_str or not end_time_str:
            print(f"⚠ Section {i+1} '{section_name}': missing time information, will skip")
            segment_info_list.append((i, section, None))
            continue
        
        try:
            # Parse time strings to seconds
            start_seconds = parse_time_to_seconds(start_time_str)
            end_seconds = parse_time_to_seconds(end_time_str)
            
            # Segment the audio
            segment_path = segment_audio_file(audio_path, start_seconds, end_seconds)
            temp_files.append(segment_path)
            
            duration = end_seconds - start_seconds
            print(f"✓ Section {i+1} '{section_name}': {start_time_str} - {end_time_str} ({duration:.1f}s)")
            
            segment_info_list.append((i, section, segment_path))
            
        except Exception as e:
            print(f"✗ Section {i+1} '{section_name}': Error extracting segment - {e}")
            segment_info_list.append((i, section, None))
    
    
    # Step 2: Process segments in batches
    print(f"\n{'-'*80}")
    print(f"Step 2: Processing {len([s for s in segment_info_list if s[2] is not None])} segments in batches of {batch_size}...")
    print(f"{'-'*80}")
    
    # Create a mapping from segment_path to section info
    path_to_info = {}
    valid_segment_paths = []
    
    for idx, section, segment_path in segment_info_list:
        if segment_path is not None:
            path_to_info[segment_path] = (idx, section)
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
            idx, section = path_to_info[path]
            section_name = section.get("name", f"Section {idx+1}")
            print(f"  - Section {idx+1}: {section_name}")
        
        # try:
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
            segment_captions[path] = caption_text
        
        print(f"✓ Batch {batch_num} completed successfully")
            
        # except Exception as e:
        #     print(f"✗ Batch {batch_num} failed: {e}")
        #     # Mark all segments in this batch as failed
        #     for path in batch_paths:
        #         segment_captions[path] = None
    
    # Step 3: Merge results back to sections
    print(f"\n{'-'*80}")
    print("Step 3: Merging results...")
    print(f"{'-'*80}")
    
    detailed_sections = []
    
    for idx, section, segment_path in segment_info_list:
        section_name = section.get("name", f"Section {idx+1}")
        detailed_section = section.copy()
        
        if segment_path is None:
            # No segment was extracted
            print(f"  Section {idx+1} '{section_name}': No segment (skipped)")
            detailed_sections.append(detailed_section)
            continue
        
        # Get caption for this segment
        caption_text = segment_captions.get(segment_path)
        
        if caption_text is None:
            # Caption generation failed
            print(f"  Section {idx+1} '{section_name}': Caption generation failed")
            detailed_sections.append(detailed_section)
            continue
        
        # Try to parse JSON from caption
        segment_json = extract_json_from_text(caption_text)
        
        if segment_json and isinstance(segment_json, dict):
            detailed_section["detailed_analysis"] = segment_json
            print(f"✓ Section {idx+1} '{section_name}': Detailed analysis added")
        else:
            detailed_section["detailed_analysis_raw"] = caption_text
            print(f"⚠ Section {idx+1} '{section_name}': Raw text added (JSON parsing failed)")
        
        detailed_sections.append(detailed_section)
    
    # Cleanup temporary files
    print(f"\n{'-'*80}")
    print("Cleaning up temporary files...")
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"✓ Removed: {temp_file}")
        except Exception as e:
            print(f"⚠ Warning: Could not remove temp file {temp_file}: {e}")
    
    # Prepare final result
    result = {
        "audio_path": audio_path,
        "overall_analysis": {
            "prompt": AUDIO_STRUCTURE_SEG_PROMPT,
            "summary": summary,
        },
        "sections": detailed_sections
    }
    
    # Save to file if output_path is specified
    if output_path:
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Save to file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*80}")
        print(f"✓ Complete caption saved to: {output_path}")
        print(f"{'='*80}")
    
    return result


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
    max_retries: int = 3,
    batch_size: int = 4,
    # Madmom detection parameters
    onset_threshold: float = 0.6,
    onset_smooth: float = 0.5,
    onset_pre_avg: float = 0.5,
    onset_post_avg: float = 0.5,
    onset_pre_max: float = 0.5,
    onset_post_max: float = 0.5,
    onset_combine: float = 3.0,
    beats_per_bar: list = None,
    min_bpm: float = 55.0,
    max_bpm: float = 215.0,
    # Filtering parameters
    min_segment_duration: float = 3.0,
    max_segment_duration: float = 30.0,
    max_segments: int = 30,
    merge_close: float = 0.1,
    min_interval: float = 0.0,
    top_k_keypoints: int = 0,
    energy_percentile: float = 0.0,
    # Section-based filtering (if using stage1 sections)
    use_stage1_sections: bool = False,
    section_top_k: int = 3,
    section_min_interval: float = 0.0,
    section_energy_percentile: float = 70.0,
) -> Dict:
    """
    Generate caption for an audio file using Madmom keypoints for segmentation.

    This function performs a two-stage analysis:
    1. Detect audio keypoints using Madmom (beats, onsets, spectral changes)
    2. Segment audio based on keypoints and analyze each segment

    Args:
        audio_path: Path to the audio file
        output_path: Optional path to save the caption as JSON
        model_path: Model checkpoint path (default: from config)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        use_flash_attn2: Whether to use flash attention 2
        max_retries: Maximum number of retries if JSON parsing or validation fails
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
    import sys
    import os
    madmom_module_path = os.path.join(os.path.dirname(__file__))
    if madmom_module_path not in sys.path:
        sys.path.insert(0, madmom_module_path)

    from audio_Madmom import (
        SensoryKeypointDetector,
        filter_significant_keypoints,
        filter_by_sections,
        load_sections_from_caption
    )

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
        print("Optional: Getting sections from Stage 1 audio analysis")
        print("-"*80)

        # Run stage1 to get sections
        stage1_prompt = AUDIO_STRUCTURE_SEG_PROMPT
        if audio_duration is not None:
            duration_str = f"{int(audio_duration // 60):02d}:{int(audio_duration % 60):02d}"
            stage1_prompt = f"Note: This audio file is {duration_str} long. Please analyze the ENTIRE audio from 00:00 to {duration_str}.\n\n" + AUDIO_STRUCTURE_SEG_PROMPT

        # caption_text = caption_audio(
        #     audio_path=audio_path,
        #     prompt=stage1_prompt,
        #     output_path=None,
        #     model_path=model_path,
        #     max_tokens=max_tokens,
        #     temperature=temperature,
        #     top_p=top_p,
        #     top_k=top_k,
        #     use_flash_attn2=use_flash_attn2,
        #     max_retries=1,
        #     required_keys=["summary", "sections"]
        # )
        caption_text = '''Generated caption:
{
  "summary": "This is a cinematic and emotional piece of music that blends orchestral elements with a modern electronic beat. It features a powerful female vocalist and has a strong, anthemic quality, creating a mood that is both melancholic and uplifting.",
  "sections": [
    {
      "name": "Intro",
      "description": "The track begins with a gentle, atmospheric melody played on a keyboard, accompanied by a sustained string pad. The mood is pensive and anticipatory.",
      "Start_Time": "00:00",
      "End_Time": "00:36"
    },
    {
      "name": "Verse 1",
      "description": "A soft, steady electronic beat and a simple bassline enter, supporting the female vocalist. The lyrics express regret and a need for reconciliation, creating an intimate and emotional atmosphere.",
      "Start_Time": "00:36",
      "End_Time": "01:12"
    },
    {
      "name": "Instrumental Bridge",
      "description": "The vocals drop out, and the music becomes more instrumental and dramatic. A violin-like synth takes the lead, soaring over the electronic beat and atmospheric pads, building tension.",
      "Start_Time": "01:12",
      "End_Time": "01:42"
    },
    {
      "name": "Chorus",
      "description": "The full arrangement returns with a powerful and uplifting chorus. The beat is strong and driving, the strings are sweeping, and the vocals are passionate and empowering. The lyrics speak of rebirth, resilience, and finding one's path.",
      "Start_Time": "01:42",
      "End_Time": "02:19"
    },
    {
      "name": "Outro",
      "description": "The music begins to fade. The powerful beat and strings gradually diminish, leaving the gentle keyboard melody from the intro to conclude the piece on a reflective note.",
      "Start_Time": "02:19",
      "End_Time": "02:58"
    }
  ]
}'''

        parsed_json = extract_json_from_text(caption_text)
        if parsed_json and validate_audio_caption_json(parsed_json, required_keys=["summary", "sections"]):
            sections_for_filtering = parsed_json.get("sections", [])
            print(f"✓ Got {len(sections_for_filtering)} sections from Stage 1")

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

    # Create segments between consecutive keypoints
    segments = []

    # Add first segment from start to first keypoint
    if keypoint_times and keypoint_times[0] > 0:
        duration = keypoint_times[0]
        if min_segment_duration <= duration <= max_segment_duration:
            segments.append({
                "start_time": 0.0,
                "end_time": keypoint_times[0],
                "duration": duration
            })

    # Add segments between consecutive keypoints
    for i in range(len(keypoint_times) - 1):
        start = keypoint_times[i]
        end = keypoint_times[i + 1]
        duration = end - start

        # Check duration constraints
        if min_segment_duration <= duration <= max_segment_duration:
            segments.append({
                "start_time": start,
                "end_time": end,
                "duration": duration
            })
        elif duration > max_segment_duration:
            # Split long segment into multiple parts
            num_parts = int(np.ceil(duration / max_segment_duration))
            part_duration = duration / num_parts
            for j in range(num_parts):
                part_start = start + j * part_duration
                part_end = start + (j + 1) * part_duration
                if part_end - part_start >= min_segment_duration:
                    segments.append({
                        "start_time": part_start,
                        "end_time": part_end,
                        "duration": part_end - part_start
                    })

    # Add last segment from last keypoint to end
    if keypoint_times and audio_duration:
        last_time = keypoint_times[-1]
        if last_time < audio_duration:
            duration = audio_duration - last_time
            if min_segment_duration <= duration <= max_segment_duration:
                segments.append({
                    "start_time": last_time,
                    "end_time": audio_duration,
                    "duration": duration
                })
            elif duration > max_segment_duration:
                # Split last segment if too long
                num_parts = int(np.ceil(duration / max_segment_duration))
                part_duration = duration / num_parts
                for j in range(num_parts):
                    part_start = last_time + j * part_duration
                    part_end = min(last_time + (j + 1) * part_duration, audio_duration)
                    if part_end - part_start >= min_segment_duration:
                        segments.append({
                            "start_time": part_start,
                            "end_time": part_end,
                            "duration": part_end - part_start
                        })

    print(f"✓ Created {len(segments)} segments from keypoints")
    print(f"  - Min segment duration: {min(s['duration'] for s in segments):.2f}s")
    print(f"  - Max segment duration: {max(s['duration'] for s in segments):.2f}s")
    print(f"  - Avg segment duration: {np.mean([s['duration'] for s in segments]):.2f}s")

    # Stage 2: Analyze each segment
    print("\n" + "="*80)
    print(f"STAGE 2: Analyzing {len(segments)} segments in detail")
    print("="*80)

    # Load model and processor
    model, processor = load_model_and_processor(
        model_path=model_path,
        use_flash_attn2=use_flash_attn2
    )

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
            segment_captions[path] = caption_text

        print(f"✓ Batch {batch_num} completed successfully")

    # Step 3: Merge results
    print(f"\n{'-'*80}")
    print("Step 3: Merging results...")
    print(f"{'-'*80}")

    detailed_segments = []

    for idx, seg, segment_path in segment_info_list:
        detailed_segment = seg.copy()
        detailed_segment['segment_id'] = idx + 1

        if segment_path is None:
            print(f"  Segment {idx+1}: No audio (skipped)")
            detailed_segments.append(detailed_segment)
            continue

        # Get caption for this segment
        caption_text = segment_captions.get(segment_path)

        if caption_text is None:
            print(f"  Segment {idx+1}: Caption generation failed")
            detailed_segments.append(detailed_segment)
            continue

        # Try to parse JSON from caption
        segment_json = extract_json_from_text(caption_text)

        if segment_json and isinstance(segment_json, dict):
            detailed_segment["detailed_analysis"] = segment_json
            print(f"✓ Segment {idx+1}: Detailed analysis added")
        else:
            detailed_segment["detailed_analysis_raw"] = caption_text
            print(f"⚠ Segment {idx+1}: Raw text added (JSON parsing failed)")

        detailed_segments.append(detailed_segment)

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
        "madmom_analysis": {
            "total_keypoints_detected": len(keypoints),
            "filtered_keypoints": len(filtered_keypoints),
            "keypoint_types": {
                "downbeats": len(result['downbeats']),
                "onsets": len(result['onsets']),
                "spectral_flux": len(result.get('spectral_flux_peaks', [])),
                "energy_changes": len(result.get('energy_change_peaks', [])),
                "timbre_changes": len(result.get('centroid_change_peaks', [])),
            },
            "filtering_params": {
                "onset_threshold": onset_threshold,
                "min_segment_duration": min_segment_duration,
                "max_segment_duration": max_segment_duration,
                "max_segments": max_segments,
                "merge_close": merge_close,
            }
        },
        "segments": detailed_segments
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
    """Example usage of audio caption function with Madmom-based segmentation."""
    # Example audio file path
    audio_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/Dataset/Audio/Call_of_Slience/CallofSilence.mp3"

    # Generate caption with Madmom-based segmentation
    result = caption_audio_with_madmom_segments(
        audio_path=audio_path,
        output_path="./audio_caption_madmom_output.json",
        max_tokens=config.AUDIO_ANALYSIS_MODEL_MAX_TOKEN,
        batch_size=4,
        # Madmom parameters
        onset_threshold=0.6,
        onset_combine=3.0,
        min_segment_duration=3.0,
        max_segment_duration=30.0,
        max_segments=30,
        merge_close=0.1,
        # Optional: use stage1 sections for filtering
        use_stage1_sections=True,
        section_top_k=3,
        section_energy_percentile=70.0,
    )

    print(f"\n{'='*80}")
    print("Processing Complete!")
    print(f"Total segments analyzed: {len(result.get('segments', []))}")
    print(f"{'='*80}")



if __name__ == "__main__":
    main()

