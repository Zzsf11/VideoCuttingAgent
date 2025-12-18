"""
Character identification and subtitle enhancement script.
This script analyzes subtitles to identify characters and replace speaker labels with actual character names.
"""

import os
import sys
import re
import json
import requests
from typing import Dict, List, Tuple, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vca.config import VLLM_ENDPOINT, VIDEO_ANALYSIS_MODEL_MAX_TOKEN, VIDEO_ANALYSIS_MODEL


def parse_srt(srt_path: str) -> List[Dict]:
    """
    Parse an SRT subtitle file into a list of subtitle entries.

    Args:
        srt_path: Path to the SRT file

    Returns:
        List of dictionaries containing subtitle information
    """
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by double newlines to get individual subtitle blocks
    blocks = re.split(r'\n\n+', content.strip())

    subtitles = []
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue

        try:
            index = int(lines[0])
            timestamp = lines[1]
            text = '\n'.join(lines[2:])

            # Extract speaker label if present
            speaker_match = re.match(r'\[([^\]]+)\]\s*(.*)', text, re.DOTALL)
            if speaker_match:
                speaker = speaker_match.group(1)
                dialogue = speaker_match.group(2).strip()
            else:
                speaker = "UNKNOWN"
                dialogue = text.strip()

            subtitles.append({
                'index': index,
                'timestamp': timestamp,
                'speaker': speaker,
                'dialogue': dialogue,
                'raw_text': text
            })
        except (ValueError, IndexError):
            continue

    return subtitles


def get_speaker_dialogues(subtitles: List[Dict]) -> Dict[str, List[str]]:
    """
    Group dialogues by speaker.

    Args:
        subtitles: List of parsed subtitle entries

    Returns:
        Dictionary mapping speaker labels to their dialogues
    """
    speaker_dialogues = {}
    for sub in subtitles:
        speaker = sub['speaker']
        if speaker not in speaker_dialogues:
            speaker_dialogues[speaker] = []
        if sub['dialogue']:  # Only add non-empty dialogues
            speaker_dialogues[speaker].append(sub['dialogue'])
    return speaker_dialogues


def estimate_tokens(text: str) -> int:
    """
    Estimate token count. Rough heuristic: ~3-4 chars per token for English.
    """
    return len(text) // 3


def format_dialogues_for_analysis(
    speaker_dialogues: Dict[str, List[str]],
    max_samples: int = None,  # None means all samples
    max_total_tokens: int = 50000  # Leave room for prompt template and response
) -> str:
    """
    Format speaker dialogues for LLM analysis.

    Args:
        speaker_dialogues: Dictionary of speaker to dialogues
        max_samples: Maximum samples per speaker (None = all)
        max_total_tokens: Maximum estimated tokens

    Returns:
        Formatted string for LLM prompt
    """
    formatted = []
    for speaker, dialogues in sorted(speaker_dialogues.items()):
        if speaker == "UNKNOWN":
            continue
        # Take all or limited samples
        samples = dialogues if max_samples is None else dialogues[:max_samples]
        sample_text = ' | '.join(samples)
        formatted.append(f"[{speaker}]: {sample_text}")

    result = '\n\n'.join(formatted)

    # Estimate and warn about token count
    estimated_tokens = estimate_tokens(result)
    print(f"  Dialogue text length: {len(result)} chars, ~{estimated_tokens} tokens")

    if estimated_tokens > max_total_tokens:
        print(f"  WARNING: Estimated tokens ({estimated_tokens}) exceeds limit ({max_total_tokens})")
        print(f"  Consider using max_samples parameter to limit dialogue samples")

    return result


def format_full_subtitles(subtitles: List[Dict]) -> str:
    """
    Format full subtitles for LLM analysis (preserves conversation flow).

    Args:
        subtitles: List of parsed subtitle entries

    Returns:
        Formatted string with full subtitle content
    """
    lines = []
    for sub in subtitles:
        speaker = sub['speaker']
        dialogue = sub['dialogue']
        if dialogue.strip():  # Skip empty dialogues
            lines.append(f"[{speaker}] {dialogue}")

    result = '\n'.join(lines)
    estimated_tokens = estimate_tokens(result)
    print(f"  Full subtitle length: {len(result)} chars, ~{estimated_tokens} tokens")

    return result


def query_vllm_for_characters(dialogues_text: str, movie_name: str = "") -> Dict:
    """
    Query vLLM to identify characters from their dialogues.

    Args:
        dialogues_text: Formatted dialogue text
        movie_name: Optional movie name for context

    Returns:
        Dictionary mapping speaker labels to character information
    """

    movie_context = f"This is from the movie '{movie_name}'." if movie_name else ""

    prompt = f"""You are analyzing subtitles from a movie to identify characters. {movie_context}

Below are dialogue samples grouped by speaker labels (like SPEAKER_01, SPEAKER_02, etc.).
Based on the dialogue content, context clues, and any names mentioned, identify who each speaker is.

DIALOGUE SAMPLES:
{dialogues_text}

TASK:
1. Analyze each speaker's dialogues carefully
2. Look for:
   - Names mentioned in conversation (e.g., "Rachel, let me see" suggests the listener might be Rachel)
   - Character traits, relationships, and speaking patterns
   - Context clues about their role (e.g., butler, villain, hero)
3. For this specific movie, common characters might include protagonists, villains, mentors, love interests, etc.

OUTPUT FORMAT (JSON):
Return a JSON object where keys are speaker labels and values are objects containing:
- "name": The character's name (use "Unknown" if cannot be determined)
- "confidence": "high", "medium", or "low"
- "evidence": Brief explanation of why you identified this character
- "role": Character's role in the story (e.g., "protagonist", "mentor", "villain", "supporting")

Example output:
{{
    "SPEAKER_01": {{
        "name": "Alfred",
        "confidence": "high",
        "evidence": "Addresses Bruce as 'Master Wayne', butler-like speech patterns",
        "role": "supporting"
    }},
    "SPEAKER_02": {{
        "name": "Bruce Wayne",
        "confidence": "high",
        "evidence": "Called 'Bruce' by others, main character dialogue",
        "role": "protagonist"
    }}
}}

Only output the JSON object, no other text. /no_think"""

    try:
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": VIDEO_ANALYSIS_MODEL_MAX_TOKEN,
            "temperature": 0.3
        }
        # Only add model if specified in config
        if VIDEO_ANALYSIS_MODEL:
            payload["model"] = VIDEO_ANALYSIS_MODEL

        response = requests.post(
            VLLM_ENDPOINT,
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        result = response.json()

        content = result['choices'][0]['message']['content']

        # Extract JSON from response
        # Try to find JSON in the response
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
        else:
            print(f"Warning: Could not parse JSON from response: {content[:500]}")
            return {}

    except requests.exceptions.RequestException as e:
        print(f"Error querying vLLM: {e}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        return {}


def refine_character_mapping(
    character_info: Dict,
    speaker_dialogues: Dict[str, List[str]]
) -> Dict[str, str]:
    """
    Refine character mapping and handle edge cases.

    Args:
        character_info: Raw character identification results
        speaker_dialogues: Original speaker to dialogues mapping

    Returns:
        Clean mapping of speaker labels to character names
    """
    mapping = {}

    for speaker, info in character_info.items():
        if isinstance(info, dict):
            name = info.get('name', 'Unknown')
            confidence = info.get('confidence', 'low')

            # If confidence is low and name is Unknown, keep original label
            if name == 'Unknown' and confidence == 'low':
                mapping[speaker] = speaker
            else:
                mapping[speaker] = name
        else:
            mapping[speaker] = str(info) if info else speaker

    # Add any speakers not in the response
    for speaker in speaker_dialogues.keys():
        if speaker not in mapping:
            mapping[speaker] = speaker

    return mapping


def create_new_subtitles(
    subtitles: List[Dict],
    speaker_mapping: Dict[str, str]
) -> List[Dict]:
    """
    Create new subtitle entries with character names instead of speaker labels.

    Args:
        subtitles: Original subtitle entries
        speaker_mapping: Mapping from speaker labels to character names

    Returns:
        New subtitle entries with character names
    """
    new_subtitles = []

    for sub in subtitles:
        new_sub = sub.copy()
        original_speaker = sub['speaker']
        character_name = speaker_mapping.get(original_speaker, original_speaker)

        # Update the raw text with the character name
        new_sub['speaker'] = character_name
        new_sub['raw_text'] = f"[{character_name}] {sub['dialogue']}"

        new_subtitles.append(new_sub)

    return new_subtitles


def write_srt(subtitles: List[Dict], output_path: str):
    """
    Write subtitles to an SRT file.

    Args:
        subtitles: List of subtitle entries
        output_path: Path to output SRT file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for sub in subtitles:
            f.write(f"{sub['index']}\n")
            f.write(f"{sub['timestamp']}\n")
            f.write(f"{sub['raw_text']}\n")
            f.write("\n")

    print(f"Saved new subtitles to: {output_path}")


def write_character_info(character_info: Dict, output_path: str):
    """
    Write character information to a JSON file.

    Args:
        character_info: Character identification results
        output_path: Path to output JSON file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(character_info, f, ensure_ascii=False, indent=2)

    print(f"Saved character info to: {output_path}")


def analyze_subtitles(
    srt_path: str,
    movie_name: str = "",
    output_dir: Optional[str] = None,
    use_full_subtitles: bool = True
) -> Tuple[Dict[str, str], Dict]:
    """
    Main function to analyze subtitles and identify characters.

    Args:
        srt_path: Path to the SRT subtitle file
        movie_name: Name of the movie (optional, for better context)
        output_dir: Directory to save output files (optional)
        use_full_subtitles: If True, send full subtitles (better accuracy);
                           If False, send grouped by speaker (less tokens)

    Returns:
        Tuple of (speaker_mapping, character_info)
    """
    print(f"Analyzing subtitles: {srt_path}")

    # Parse subtitles
    subtitles = parse_srt(srt_path)
    print(f"Parsed {len(subtitles)} subtitle entries")

    # Group dialogues by speaker
    speaker_dialogues = get_speaker_dialogues(subtitles)
    print(f"Found {len(speaker_dialogues)} unique speakers:")
    for speaker, dialogues in sorted(speaker_dialogues.items()):
        print(f"  {speaker}: {len(dialogues)} dialogues")

    # Format dialogues for analysis
    if use_full_subtitles:
        print("\nUsing FULL subtitles for analysis (better accuracy)...")
        dialogues_text = format_full_subtitles(subtitles)
    else:
        print("\nUsing grouped dialogues for analysis (less tokens)...")
        dialogues_text = format_dialogues_for_analysis(speaker_dialogues)

    # Query vLLM for character identification
    print("\nQuerying vLLM for character identification...")
    character_info = query_vllm_for_characters(dialogues_text, movie_name)

    if character_info:
        print("\nIdentified characters:")
        for speaker, info in sorted(character_info.items()):
            if isinstance(info, dict):
                name = info.get('name', 'Unknown')
                confidence = info.get('confidence', 'unknown')
                role = info.get('role', 'unknown')
                evidence = info.get('evidence', '')
                print(f"  {speaker} -> {name} (confidence: {confidence}, role: {role})")
                print(f"    Evidence: {evidence[:100]}...")
            else:
                print(f"  {speaker} -> {info}")

    # Create clean mapping
    speaker_mapping = refine_character_mapping(character_info, speaker_dialogues)

    # Save outputs if output_dir is specified
    if output_dir is None:
        output_dir = os.path.dirname(srt_path)

    # Create new subtitles with character names
    new_subtitles = create_new_subtitles(subtitles, speaker_mapping)

    # Write new subtitle file
    base_name = os.path.splitext(os.path.basename(srt_path))[0]
    new_srt_path = os.path.join(output_dir, f"{base_name}_with_characters.srt")
    write_srt(new_subtitles, new_srt_path)

    # Write character info JSON
    char_info_path = os.path.join(output_dir, "character_info.json")
    write_character_info(character_info, char_info_path)

    # Write speaker mapping JSON
    mapping_path = os.path.join(output_dir, "speaker_mapping.json")
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(speaker_mapping, f, ensure_ascii=False, indent=2)
    print(f"Saved speaker mapping to: {mapping_path}")

    return speaker_mapping, character_info


def main():
    """Main entry point for command line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze subtitles to identify characters and create enhanced subtitle files"
    )
    parser.add_argument(
        "srt_path",
        help="Path to the SRT subtitle file"
    )
    parser.add_argument(
        "--movie-name", "-m",
        default="",
        help="Name of the movie (optional, for better context)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory (default: same as input file)"
    )
    parser.add_argument(
        "--grouped", "-g",
        action="store_true",
        help="Use grouped dialogues instead of full subtitles (less accurate but fewer tokens)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.srt_path):
        print(f"Error: File not found: {args.srt_path}")
        sys.exit(1)

    speaker_mapping, character_info = analyze_subtitles(
        args.srt_path,
        args.movie_name,
        args.output_dir,
        use_full_subtitles=not args.grouped  # Default is full subtitles
    )

    print("\n" + "="*50)
    print("Summary - Speaker to Character Mapping:")
    print("="*50)
    for speaker, character in sorted(speaker_mapping.items()):
        print(f"  {speaker} -> {character}")


if __name__ == "__main__":
    main()
