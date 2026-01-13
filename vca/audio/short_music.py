# -*- coding: utf-8 -*-
"""
Short music climax extraction script
Select the section with highest total intensity
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple


def parse_time_to_seconds(time_str: str) -> float:
    """Convert time string to seconds, supports formats: '01:23.45' or '83.45'"""
    if ':' in time_str:
        parts = time_str.split(':')
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        elif len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return float(time_str)


def seconds_to_time_str(seconds: float) -> str:
    """Convert seconds to time string format MM:SS.xx"""
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:05.2f}"


def get_section_duration(section: dict) -> float:
    """Calculate section duration in seconds"""
    start_time = parse_time_to_seconds(section.get('Start_Time', '0'))
    end_time = parse_time_to_seconds(section.get('End_Time', '0'))
    return end_time - start_time


def extract_intensity(intensity_str: str) -> float:
    """Extract numeric intensity value from string"""
    if not intensity_str:
        return 0.0
    # Try to extract first numeric value (including decimals)
    match = re.search(r'(\d+\.?\d*)', str(intensity_str))
    return float(match.group(1)) if match else 0.0


def calculate_section_intensity(section: dict) -> Dict[str, float]:
    """
    Calculate total intensity for a section by summing all intensity values

    Returns: {
        'total_intensity': sum of all intensity values,
        'duration': section duration in seconds,
        'avg_intensity': average intensity per second
    }
    """
    detailed = section.get('detailed_analysis', {})
    segments = detailed.get('sections', [])

    if not segments:
        duration = get_section_duration(section)
        return {
            'total_intensity': 0.0,
            'duration': duration,
            'avg_intensity': 0.0,
            'num_segments': 0
        }

    # Extract intensity values from all segments
    intensities = []
    for seg in segments:
        # Try different possible field names
        intensity_str = seg.get('intensity', '') or seg.get('energy', '') or seg.get('Energy', '')
        if intensity_str:
            intensity = extract_intensity(intensity_str)
            intensities.append(intensity)

    total_intensity = sum(intensities)
    duration = get_section_duration(section)
    avg_intensity = total_intensity / duration if duration > 0 else 0

    return {
        'total_intensity': round(total_intensity, 2),
        'duration': round(duration, 2),
        'avg_intensity': round(avg_intensity, 3),
        'num_segments': len(intensities)
    }


def merge_sections(sections: List[dict], indices: List[int]) -> dict:
    """
    Merge multiple sections by index

    Args:
        sections: List of all sections
        indices: List of section indices to merge

    Returns:
        Merged section with adjusted absolute timestamps in detailed_analysis
    """
    if not indices:
        raise ValueError("No sections to merge")

    indices = sorted(indices)
    merged_sections = [sections[i] for i in indices]

    # Get time range
    start_time = parse_time_to_seconds(merged_sections[0].get('Start_Time', '0'))
    end_time = parse_time_to_seconds(merged_sections[-1].get('End_Time', '0'))

    # Merge detailed analysis with relative time adjustment
    # All times should be relative to the merged section's start time (start_time)
    all_segments = []
    for sec in merged_sections:
        # Get this section's start time
        sec_start_time = parse_time_to_seconds(sec.get('Start_Time', '0'))

        # Calculate offset relative to merged section's start
        offset = sec_start_time - start_time

        detailed = sec.get('detailed_analysis', {})
        segments = detailed.get('sections', [])

        # Adjust each segment's timestamp to be relative to merged section's start
        for seg in segments:
            adjusted_seg = seg.copy()

            # Original times are relative to this section's start
            seg_start_rel = parse_time_to_seconds(seg.get('Start_Time', '0'))
            seg_end_rel = parse_time_to_seconds(seg.get('End_Time', '0'))

            # Add offset to make them relative to merged section's start
            seg_start_new = seg_start_rel + offset
            seg_end_new = seg_end_rel + offset

            # Update with new relative timestamps
            adjusted_seg['Start_Time'] = seconds_to_time_str(seg_start_new)
            adjusted_seg['End_Time'] = seconds_to_time_str(seg_end_new)

            all_segments.append(adjusted_seg)

    # Create merged section
    section_names = [s.get('name', f'Section {i}') for i, s in zip(indices, merged_sections)]
    merged = {
        'name': ' + '.join(section_names),
        'description': f"Merged from sections: {', '.join(section_names)}",
        'Start_Time': seconds_to_time_str(start_time),
        'End_Time': seconds_to_time_str(end_time),
        'detailed_analysis': {
            'sections': all_segments
        },
        'merged_indices': indices
    }

    return merged


def find_best_section_with_min_duration(
    sections: List[dict],
    min_duration: float = 20.0
) -> Tuple[dict, Dict[str, float]]:
    """
    Find the section with highest intensity density (intensity per second), ensuring minimum duration

    Args:
        sections: List of sections
        min_duration: Minimum duration in seconds (default 20s)

    Returns:
        (best_section, intensity_info)
    """
    if not sections:
        raise ValueError("No sections provided")

    # Calculate intensity for each section
    scored_sections = []
    for i, section in enumerate(sections):
        intensity_info = calculate_section_intensity(section)
        scored_sections.append({
            'index': i,
            'section': section,
            'intensity_info': intensity_info
        })

    # Sort by intensity density (avg_intensity, highest first)
    scored_sections.sort(key=lambda x: x['intensity_info']['avg_intensity'], reverse=True)

    # Try to find best section with sufficient duration
    for item in scored_sections:
        section = item['section']
        intensity_info = item['intensity_info']
        idx = item['index']

        if intensity_info['duration'] >= min_duration:
            # Duration is sufficient
            return section, intensity_info

        # Need to merge with neighbors
        print(f"Section '{section.get('name')}' duration {intensity_info['duration']:.2f}s < {min_duration}s")
        print(f"Attempting to merge with adjacent sections...")

        # Try expanding left and right
        left_idx = idx - 1
        right_idx = idx + 1
        merged_indices = [idx]
        current_sections = [section]
        current_duration = intensity_info['duration']

        # Alternately add left and right sections until we reach min_duration
        while current_duration < min_duration:
            added = False

            # Try adding from right
            if right_idx < len(sections):
                right_section = sections[right_idx]
                right_duration = get_section_duration(right_section)
                if current_duration + right_duration <= min_duration * 1.5:  # Don't add too much
                    merged_indices.append(right_idx)
                    current_sections.append(right_section)
                    current_duration += right_duration
                    right_idx += 1
                    added = True
                    if current_duration >= min_duration:
                        break

            # Try adding from left
            if left_idx >= 0:
                left_section = sections[left_idx]
                left_duration = get_section_duration(left_section)
                if current_duration + left_duration <= min_duration * 1.5:
                    merged_indices.insert(0, left_idx)
                    current_sections.insert(0, left_section)
                    current_duration += left_duration
                    left_idx -= 1
                    added = True
                    if current_duration >= min_duration:
                        break

            if not added:
                # Can't expand anymore
                if current_duration < min_duration:
                    print(f"Warning: Could not reach minimum duration. Final duration: {current_duration:.2f}s")
                break

        # Merge sections
        merged_section = merge_sections(sections, merged_indices)
        merged_intensity = calculate_section_intensity(merged_section)

        print(f"Merged {len(merged_indices)} sections. New duration: {merged_intensity['duration']:.2f}s")

        return merged_section, merged_intensity

    # Fallback: return the best section even if it doesn't meet min duration
    best = scored_sections[0]
    return best['section'], best['intensity_info']


def filter_keypoints(keypoints: List[dict], start_time: float, end_time: float) -> List[dict]:
    """
    Filter keypoints within the specified time range

    Args:
        keypoints: List of keypoint dictionaries
        start_time: Start time in seconds
        end_time: End time in seconds

    Returns:
        Filtered list of keypoints
    """
    if not keypoints:
        return []

    filtered = []
    for kp in keypoints:
        kp_time = kp.get('time', 0)
        if start_time <= kp_time <= end_time:
            filtered.append(kp)

    return filtered


def find_climax_section(
    audio_caption_path: str,
    output_path: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    min_duration: float = 20.0,
    filter_kp: bool = False
) -> dict:
    """
    Find the section with highest intensity density, or use specified time range

    Args:
        audio_caption_path: Path to audio caption JSON file
        output_path: Output file path, auto-generated if None
        start_time: Optional start time (e.g., '01:23.45' or '83.45')
        end_time: Optional end time (e.g., '01:45.67' or '105.67')
        min_duration: Minimum duration in seconds (default 20s)
        filter_kp: Whether to filter keypoints to selected time range (default False)

    Returns:
        Information about the selected section in same format as input
    """
    # Read JSON file
    with open(audio_caption_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    sections = data.get('sections', [])
    if not sections:
        raise ValueError("No sections found in the audio caption file")

    # Get keypoints if available
    all_keypoints = data.get('_keypoints_detail', [])

    # If time range is specified, use it directly
    if start_time is not None and end_time is not None:
        start_sec = parse_time_to_seconds(start_time)
        end_sec = parse_time_to_seconds(end_time)

        # Find overlapping sections
        overlapping = []
        for i, section in enumerate(sections):
            sec_start = parse_time_to_seconds(section.get('Start_Time', '0'))
            sec_end = parse_time_to_seconds(section.get('End_Time', '0'))
            if sec_start <= end_sec and sec_end >= start_sec:
                overlapping.append(i)

        if overlapping:
            # Merge all overlapping sections
            target_section = merge_sections(sections, overlapping)

            # Filter and adjust segments to fit within user-specified time range
            # The segments in detailed_analysis have times relative to the merged section's start
            # Important: merged_start is the ORIGINAL merged section's start (before user time override)
            # This is needed to correctly convert relative times to absolute times
            original_merged_start = parse_time_to_seconds(sections[overlapping[0]].get('Start_Time', '0'))
            original_segments = target_section.get('detailed_analysis', {}).get('sections', [])

            filtered_segments = []
            for seg in original_segments:
                # Convert relative times to absolute times
                seg_start_rel = parse_time_to_seconds(seg.get('Start_Time', '0'))
                seg_end_rel = parse_time_to_seconds(seg.get('End_Time', '0'))
                seg_start_abs = original_merged_start + seg_start_rel
                seg_end_abs = original_merged_start + seg_end_rel

                # Check if segment overlaps with user-specified range
                if seg_start_abs < end_sec and seg_end_abs > start_sec:
                    # Clip to user-specified range
                    clipped_start = max(seg_start_abs, start_sec)
                    clipped_end = min(seg_end_abs, end_sec)

                    # Convert back to relative times (relative to user-specified start)
                    new_seg = seg.copy()
                    new_seg['Start_Time'] = seconds_to_time_str(clipped_start - start_sec)
                    new_seg['End_Time'] = seconds_to_time_str(clipped_end - start_sec)
                    filtered_segments.append(new_seg)

            target_section['detailed_analysis']['sections'] = filtered_segments
            target_section['Start_Time'] = start_time
            target_section['End_Time'] = end_time
        else:
            # Create custom section
            target_section = {
                'name': f'Custom Section ({start_time} - {end_time})',
                'description': 'User-specified time range',
                'Start_Time': start_time,
                'End_Time': end_time,
                'detailed_analysis': {'sections': []}
            }

        intensity_info = calculate_section_intensity(target_section)

        print(f"Using specified time range: {start_time} - {end_time}")
        print(f"Duration: {intensity_info['duration']:.2f}s")
        print(f"Total Intensity: {intensity_info['total_intensity']:.2f}")

    else:
        # Auto mode: Find best section with minimum duration
        print(f"Auto mode: Finding section with highest intensity density (min duration: {min_duration}s)")
        target_section, intensity_info = find_best_section_with_min_duration(sections, min_duration)

        print(f"\nSelected Section: {target_section.get('name')}")
        print(f"Time Range: {target_section.get('Start_Time')} - {target_section.get('End_Time')}")
        print(f"Duration: {intensity_info['duration']:.2f}s")
        print(f"Intensity Density: {intensity_info['avg_intensity']:.3f} per second (PRIMARY METRIC)")
        print(f"Total Intensity: {intensity_info['total_intensity']:.2f}")
        print(f"Number of Segments: {intensity_info['num_segments']}")

    # Filter keypoints for the selected section (if enabled)
    if filter_kp:
        section_start = parse_time_to_seconds(target_section.get('Start_Time', '0'))
        section_end = parse_time_to_seconds(target_section.get('End_Time', '0'))
        output_keypoints = filter_keypoints(all_keypoints, section_start, section_end)
        print(f"Filtered {len(output_keypoints)} keypoints from {len(all_keypoints)} total keypoints")
    else:
        output_keypoints = all_keypoints
        print(f"Keeping all {len(all_keypoints)} keypoints (filter_kp=False)")

    # Build output in same format as input
    result = {
        'audio_path': data.get('audio_path', ''),
        'overall_analysis': data.get('overall_analysis', {}),
        'sections': [target_section],
        '_keypoints_detail': output_keypoints
    }

    # Determine output path
    if output_path is None:
        input_path = Path(audio_caption_path)
        suffix = 'climax' if start_time is None else 'custom'
        output_path = input_path.parent / f"{input_path.stem}_{suffix}.json"
    else:
        output_path = Path(output_path)

    # Save result
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nResult saved to: {output_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract music climax section based on intensity')
    parser.add_argument('audio_caption', nargs='?',
                       default="/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Audio/Way_Down_We_Go/audio_caption_interactive.json",
                       help='Path to audio caption JSON file')
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('-s', '--start', help='Start time (e.g., 01:23.45 or 83.45)')
    parser.add_argument('-e', '--end', help='End time (e.g., 01:45.67 or 105.67)')
    parser.add_argument('-m', '--min-duration', type=float, default=15.0,
                       help='Minimum duration in seconds (default: 15)')
    parser.add_argument('--filter-kp', action='store_true', default=False,
                       help='Filter keypoints to selected time range (default: False)')

    args = parser.parse_args()

    find_climax_section(
        args.audio_caption,
        output_path=args.output,
        start_time=args.start,
        end_time=args.end,
        min_duration=args.min_duration,
        filter_kp=args.filter_kp
    )
