#!/usr/bin/env python3
"""
Batch render videos from video_database/Output directory structure

This script is specifically designed for the Output directory structure:
video_database/Output/{Movie}_{Audio}/{narrative|object}/shot_point_*.json
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Tuple


class Colors:
    """ANSI color codes"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    MAGENTA = '\033[0;35m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'

def find_shot_files(base_dir: str = "video_database/Output") -> List[Path]:
    """Find all shot_point*.json files in Output directory"""
    base_path = Path(base_dir)
    if not base_path.exists():
        return []

    # Pattern: video_database/Output/{Movie}_{Audio}/{type}/shot_point_*.json
    shot_files = list(base_path.glob("*/*/shot_point_*.json"))
    return sorted(shot_files)


def resolve_paths(shot_plan_path: Path) -> Tuple[Path, Path]:
    """
    Resolve video and audio paths from shot_plan.json

    Returns:
        Tuple of (video_path, audio_path)
    """
    try:
        with open(shot_plan_path, 'r', encoding='utf-8') as f:
            plan_data = json.load(f)

        metadata = plan_data.get('metadata', {})
        video_str = metadata.get('video_path', '')
        audio_str = metadata.get('audio_path', '')

        # Resolve video path
        video_path = None
        if video_str:
            video_candidate = Path(video_str)
            if video_candidate.exists():
                video_path = video_candidate

        # Resolve audio path from captions.json path
        audio_path = None
        if audio_str:
            audio_p = Path(audio_str)
            if 'Audio' in audio_p.parts:
                audio_idx = audio_p.parts.index('Audio')
                if audio_idx + 1 < len(audio_p.parts):
                    audio_name = audio_p.parts[audio_idx + 1]
                    audio_candidate = Path(f"Dataset/Audio/{audio_name}.mp3")
                    if audio_candidate.exists():
                        audio_path = audio_candidate

        return video_path, audio_path

    except Exception as e:
        print(f"{Colors.YELLOW}Warning: Could not resolve paths: {e}{Colors.NC}")
        return None, None


def render_video(
    shot_file: Path,
    output_path: Path,
    shot_plan_path: Path,
    video_path: Path,
    audio_path: Path,
    crop_ratio: str = None,
    verbose: bool = False,
    dry_run: bool = False
) -> bool:
    """Render a single video"""
    cmd = [
        'python', 'render_video.py',
        '--shot-json', str(shot_file),
        '--output', str(output_path),
        '--no-labels'
    ]

    if shot_plan_path:
        cmd.extend(['--shot-plan', str(shot_plan_path)])
    if video_path:
        cmd.extend(['--video', str(video_path)])
    if audio_path:
        cmd.extend(['--audio', str(audio_path)])
    if crop_ratio:
        cmd.extend(['--crop-ratio', crop_ratio])
    if verbose:
        cmd.append('--verbose')
    if dry_run:
        cmd.append('--dry-run')

    print(f"{Colors.CYAN}Command: {' '.join(cmd)}{Colors.NC}")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError:
        return False
    except Exception as e:
        print(f"{Colors.RED}Error: {e}{Colors.NC}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Batch render videos from video_database/Output directory'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default='video_database/Output',
        help='Base directory (default: video_database/Output)'
    )
    parser.add_argument(
        '--crop-ratio',
        type=str,
        default=None,
        help='Aspect ratio for cropping (e.g., "9:16")'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed output'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be rendered without rendering'
    )
    parser.add_argument(
        '--filter',
        type=str,
        default=None,
        help='Only process files matching this pattern'
    )

    args = parser.parse_args()

    print("=" * 60)
    print(f"{Colors.MAGENTA}Batch Video Rendering (Output Directory){Colors.NC}")
    print("=" * 60)
    print()

    # Find all shot_point files
    print(f"Searching in: {args.base_dir}")
    shot_files = find_shot_files(args.base_dir)

    if not shot_files:
        print(f"{Colors.RED}No shot_point*.json files found!{Colors.NC}")
        return 1

    # Apply filter
    if args.filter:
        shot_files = [f for f in shot_files if args.filter.lower() in str(f).lower()]
        print(f"Applied filter: '{args.filter}'")

    if not shot_files:
        print(f"{Colors.RED}No files match the filter!{Colors.NC}")
        return 1

    print(f"{Colors.GREEN}Found {len(shot_files)} file(s) to render{Colors.NC}")
    print()
