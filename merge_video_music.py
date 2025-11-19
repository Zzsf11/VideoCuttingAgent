#!/usr/bin/env python3
"""Merge output section videos and add background music using ffmpeg.

This script searches for files that match `output_section_*.mp4` (by default),
sorts them by their numeric suffix, concatenates them in order, and muxes the
video stream with a user-provided audio track.

Example:
    python merge_video_music.py \
        --input-dir /path/to/sections \
        --audio /path/to/music.mp3 \
        --output /path/to/final_video.mp4

Requires `ffmpeg` to be installed and available on the system PATH.
"""

from __future__ import annotations

import argparse
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Concatenate sequential section videos (e.g., output_section_0.mp4) "
            "and add a background music track using ffmpeg."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory containing the section video files (default: current directory).",
    )
    parser.add_argument(
        "--pattern",
        default="output_section_*.mp4",
        help="Glob pattern for section videos relative to the input directory (default: output_section_*.mp4).",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        required=True,
        help="Path to the background music file (any format supported by ffmpeg).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("merged_output.mp4"),
        help="Path to the merged output video (default: merged_output.mp4 in the working directory).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--loop-audio",
        action="store_true",
        help="Loop the background music to cover the entire video duration if needed.",
    )
    parser.add_argument(
        "--keep-list-file",
        action="store_true",
        help="Keep the temporary concat list file instead of deleting it (useful for debugging).",
    )
    return parser


def _extract_numeric_suffix(path: Path) -> int:
    match = re.search(r"(\d+)(?!.*\d)", path.stem)
    if match:
        return int(match.group(1))
    raise ValueError(f"Unable to determine numeric suffix for file: {path}")


def _collect_section_files(input_dir: Path, pattern: str) -> List[Path]:
    files = sorted(
        input_dir.glob(pattern),
        key=_extract_numeric_suffix,
    )
    return [f.resolve() for f in files if f.is_file()]


def _run_ffmpeg(cmd: Iterable[str]) -> None:
    cmd_list = list(cmd)
    print("[ffmpeg] Running:")
    print("  " + " ".join(shlex.quote(part) for part in cmd_list))
    try:
        subprocess.run(cmd_list, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"ffmpeg command failed with exit code {exc.returncode}") from exc


def main(argv: List[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    input_dir = args.input_dir.resolve()
    audio_path = args.audio.resolve()
    output_path = args.output.resolve()

    if not input_dir.is_dir():
        parser.error(f"Input directory does not exist: {input_dir}")

    if not audio_path.is_file():
        parser.error(f"Audio file not found: {audio_path}")

    if shutil.which("ffmpeg") is None:
        parser.error("ffmpeg executable not found on PATH. Please install ffmpeg.")

    if output_path.exists() and not args.overwrite:
        parser.error(
            f"Output file already exists: {output_path}. Use --overwrite to replace it."
        )

    section_files = _collect_section_files(input_dir, args.pattern)
    if not section_files:
        parser.error(
            f"No files matching pattern '{args.pattern}' were found in {input_dir}."
        )

    print(f"Found {len(section_files)} section video(s):")
    for idx, path in enumerate(section_files, start=1):
        print(f"  {idx:02d}: {path}")

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as list_file:
        for video_path in section_files:
            list_file.write(f"file '{video_path.as_posix()}'\n")
        list_file_path = Path(list_file.name)

    ffmpeg_cmd = [
        "ffmpeg",
    ]

    if args.overwrite:
        ffmpeg_cmd.append("-y")
    else:
        ffmpeg_cmd.append("-n")

    ffmpeg_cmd.extend([
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file_path),
    ])

    if args.loop_audio:
        ffmpeg_cmd.extend(["-stream_loop", "-1"])

    ffmpeg_cmd.extend([
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ])

    try:
        _run_ffmpeg(ffmpeg_cmd)
    finally:
        if args.keep_list_file:
            print(f"Keeping concat list file at {list_file_path}")
        else:
            try:
                list_file_path.unlink()
            except OSError as exc:
                print(f"Warning: failed to remove temp file {list_file_path}: {exc}")

    print(f"Merged video created at: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

