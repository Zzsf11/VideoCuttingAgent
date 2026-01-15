#!/bin/bash

# Batch render all videos from video_database/Output directory
# This script will find all shot_point*.json files in the deepest directories
# and render them using the video/audio paths from shot_plan*.json

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Batch Video Rendering Script (Output Database)"
echo "=========================================="
echo ""

# Base directory
BASE_DIR="/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent"
OUTPUT_DIR="${BASE_DIR}/video_database/Output"

# Check if output directory exists
if [ ! -d "$OUTPUT_DIR" ]; then
    echo -e "${RED}Error: Output directory not found: $OUTPUT_DIR${NC}"
    exit 1
fi

# Find all shot_point*.json files in the deepest directories (narrative/object subdirs)
# Store in array to avoid stdin issues with render_video.py
mapfile -t shot_files_array < <(find "$OUTPUT_DIR" -type f -name "shot_point*.json" | sort)

if [ ${#shot_files_array[@]} -eq 0 ]; then
    echo -e "${RED}No shot_point*.json files found in $OUTPUT_DIR${NC}"
    exit 1
fi

# Count total files
total_files=${#shot_files_array[@]}
echo -e "${GREEN}Found $total_files shot_point files to render${NC}"
echo ""

# Counter
current=0
success=0
failed=0

# Process each shot_point file
for shot_file in "${shot_files_array[@]}"; do
    current=$((current + 1))

    echo "=========================================="
    echo -e "${YELLOW}[$current/$total_files] Processing: $shot_file${NC}"
    echo "=========================================="

    # Get directory and base name
    dir=$(dirname "$shot_file")
    base=$(basename "$shot_file" .json)

    # Determine output video name
    output_video="${dir}/${base}_rendered.mp4"

    # Check if already rendered
    if [ -f "$output_video" ]; then
        echo -e "${BLUE}Already rendered: $output_video${NC}"
        echo -e "${BLUE}Skipping...${NC}"
        echo ""
        continue
    fi

    echo "Output will be: $output_video"

    # Find corresponding shot_plan*.json in the same directory
    shot_plan=$(find "$dir" -maxdepth 1 -name "shot_plan*.json" -type f | head -1)

    if [ -z "$shot_plan" ] || [ ! -f "$shot_plan" ]; then
        echo -e "${RED}Warning: No shot_plan*.json found in $dir${NC}"
        echo -e "${RED}Skipping...${NC}"
        failed=$((failed + 1))
        echo ""
        continue
    fi

    echo "Found shot_plan: $shot_plan"

    # Extract video and audio paths from shot_plan.json
    video_path=$(python3 << PYEOF
import json
import sys
try:
    with open('''$shot_plan''', 'r') as f:
        data = json.load(f)
    video_path = data.get('metadata', {}).get('video_path', '')
    if video_path:
        # Convert relative path to absolute
        if not video_path.startswith('/'):
            video_path = '${BASE_DIR}/' + video_path
        print(video_path)
except Exception as e:
    print('', file=sys.stderr)
PYEOF
)

    audio_path=$(python3 << PYEOF
import json
import sys
import os
try:
    with open('''$shot_plan''', 'r') as f:
        data = json.load(f)
    audio_path = data.get('metadata', {}).get('audio_path', '')
    # audio_path in shot_plan points to captions.json, we need the actual audio file
    # Extract the audio name from the path
    if 'Audio/' in audio_path:
        audio_name = audio_path.split('Audio/')[1].split('/')[0]
        # Find the actual audio file in Dataset/Audio (prefer .mp3)
        audio_dir = '${BASE_DIR}/Dataset/Audio'
        if os.path.exists(audio_dir):
            # First try exact match with .mp3
            mp3_file = os.path.join(audio_dir, audio_name + '.mp3')
            if os.path.exists(mp3_file):
                print(mp3_file)
            else:
                # Try to find any file starting with audio_name
                for file in sorted(os.listdir(audio_dir)):
                    if file.startswith(audio_name) and file.endswith('.mp3'):
                        print(os.path.join(audio_dir, file))
                        break
except Exception as e:
    print('', file=sys.stderr)
PYEOF
)

    # Build render command
    cmd="python ${BASE_DIR}/render_video.py --shot-json \"$shot_file\" --output \"$output_video\""

    if [ -n "$shot_plan" ] && [ -f "$shot_plan" ]; then
        cmd="$cmd --shot-plan \"$shot_plan\""
    fi

    if [ -n "$video_path" ] && [ -f "$video_path" ]; then
        cmd="$cmd --video \"$video_path\""
        echo "Video: $video_path"
    else
        echo -e "${RED}Warning: Video file not found or not specified: $video_path${NC}"
    fi

    if [ -n "$audio_path" ] && [ -f "$audio_path" ]; then
        cmd="$cmd --audio \"$audio_path\""
        echo "Audio: $audio_path"
    else
        echo -e "${YELLOW}Warning: Audio file not found or not specified: $audio_path${NC}"
    fi

    # Add common options
    # cmd="$cmd --no-labels"

    echo ""
    echo "Running: $cmd"
    echo ""

    # Execute render command
    if eval "$cmd"; then
        echo -e "${GREEN}✓ Successfully rendered: $output_video${NC}"
        success=$((success + 1))
    else
        echo -e "${RED}✗ Failed to render: $shot_file${NC}"
        failed=$((failed + 1))
    fi

    echo ""

done

# Summary
echo "=========================================="
echo "Batch Rendering Complete"
echo "=========================================="
echo -e "Total files: $total_files"
echo -e "${GREEN}Success: $success${NC}"
echo -e "${RED}Failed: $failed${NC}"
echo "=========================================="
