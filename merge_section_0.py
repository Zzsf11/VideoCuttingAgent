#!/usr/bin/env python3
"""Concatenate output_section_0_* videos, mute originals, and add section 0 bgm.

This script will:
- Find all videos matching 'output_section_0_*.mp4' in the project root
- Sort them by their last numeric suffix (e.g., shot index)
- Concatenate them in order via ffmpeg concat demuxer
- Replace original audio with '/reports/audio/proposal_section_0.mp3'
- Loop the audio to cover the full video duration, then stop at video end

Output: 'merged_section_0.mp4' in the project root.

Requires ffmpeg installed and on PATH.
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List


PROJECT_ROOT = Path("/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent").resolve()
AUDIO_PATH = PROJECT_ROOT / "reports/audio/proposal_section_2.mp3"
VIDEO_GLOB = "output_section_2_*.mp4"
OUTPUT_PATH = PROJECT_ROOT / "merged_section_2.mp4"


def extract_last_int_from_stem(path: Path) -> int:
    match = re.search(r"(\d+)(?!.*\d)", path.stem)
    if not match:
        raise ValueError(f"Unable to extract numeric suffix from: {path.name}")
    return int(match.group(1))


def collect_videos(directory: Path, pattern: str) -> List[Path]:
    files = sorted(directory.glob(pattern), key=extract_last_int_from_stem)
    return [f.resolve() for f in files if f.is_file()]


def run_ffmpeg(cmd: List[str]) -> None:
    print("[ffmpeg] Running:")
    print("  " + " ".join(shlex.quote(part) for part in cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    if shutil.which("ffmpeg") is None:
        print("Error: ffmpeg not found on PATH.", file=sys.stderr)
        return 2

    if not AUDIO_PATH.is_file():
        print(f"Error: audio not found: {AUDIO_PATH}", file=sys.stderr)
        return 2

    videos = collect_videos(PROJECT_ROOT, VIDEO_GLOB)
    if not videos:
        print(
            f"Error: no videos found in {PROJECT_ROOT} matching pattern '{VIDEO_GLOB}'.",
            file=sys.stderr,
        )
        return 2

    print(f"Found {len(videos)} video(s) to concatenate:")
    for idx, v in enumerate(videos, 1):
        print(f"  {idx:02d}: {v}")

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as list_file:
        for v in videos:
            list_file.write(f"file '{v.as_posix()}'\n")
        list_path = Path(list_file.name)

    # Build ffmpeg command: concat videos, loop audio, map video+new audio, mute original
    cmd = [
        "ffmpeg",
        "-y",  # overwrite output if exists
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        # loop the audio infinitely; combined with -shortest we end at video end
        "-stream_loop",
        "-1",
        "-i",
        str(AUDIO_PATH),
        # map first input's video only and second input's audio only
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        # copy video to avoid re-encoding; encode audio to AAC for mp4
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        # stop when the shortest stream ends (i.e., video), so audio looping doesn't run forever
        "-shortest",
        str(OUTPUT_PATH),
    ]

    try:
        run_ffmpeg(cmd)
    except subprocess.CalledProcessError as exc:
        print(f"ffmpeg failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode
    finally:
        try:
            list_path.unlink(missing_ok=True)
        except Exception as cleanup_exc:  # noqa: BLE001
            print(f"Warning: failed to remove temp file {list_path}: {cleanup_exc}")

    print(f"Merged video created at: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


