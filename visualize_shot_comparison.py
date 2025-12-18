"""
Visualize scene comparison between original shot detection and merged captions

This script compares:
- Original shot boundaries from shot_scenes.txt (RED borders)
- Merged scene boundaries from captions.json (GREEN borders)
"""

import json
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from typing import List, Tuple


def parse_shot_scenes(shot_scenes_path: str) -> List[Tuple[int, int]]:
    """Parse shot_scenes.txt to get original scene boundaries."""
    scenes = []
    if not os.path.isfile(shot_scenes_path):
        return scenes

    with open(shot_scenes_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                start_frame = int(parts[0])
                end_frame = int(parts[1])
                scenes.append((start_frame, end_frame))
    return scenes


def parse_captions_json(captions_path: str, shot_detection_fps: float = 2.0) -> List[Tuple[int, int]]:
    """Parse captions.json to get merged scene boundaries."""
    scenes = []

    with open(captions_path, "r") as f:
        captions = json.load(f)

    # Extract time ranges from keys (format: "start_end")
    for key in captions.keys():
        parts = key.split("_")
        if len(parts) >= 2:
            try:
                start_sec = float(parts[0])
                end_sec = float(parts[1])
                # Convert seconds to frame numbers
                start_frame = int(start_sec * shot_detection_fps)
                end_frame = int(end_sec * shot_detection_fps)
                scenes.append((start_frame, end_frame))
            except ValueError:
                continue

    # Sort by start frame
    scenes.sort(key=lambda x: x[0])
    return scenes


def visualize_autoshot_style(
    frames: np.ndarray,
    predictions: np.ndarray,
    width: int = 25,
    show_frame_num: bool = True,
    frame_timestamps: List[float] = None,
    color: Tuple[int, int, int] = (255, 0, 0),
) -> Image.Image:
    """Visualize frames with boundary markers (AutoShot style)."""
    ih, iw, ic = frames.shape[1:]  # 27, 48, 3
    num_predictions = 1

    # Pad frames to make divisible by width
    pad_with = width - len(frames) % width if len(frames) % width != 0 else 0
    frames = np.pad(frames, [(0, pad_with), (0, 1), (0, num_predictions), (0, 0)])
    predictions = np.pad(predictions, (0, pad_with))

    height = len(frames) // width

    # Reshape into grid
    img = frames.reshape([height, width, ih + 1, iw + num_predictions, ic])
    img_tmp = np.concatenate(np.split(
        np.concatenate(np.split(img, height), axis=2)[0], width
    ), axis=2)[0, :-1]

    img = Image.fromarray(img_tmp)
    draw = ImageDraw.Draw(img)

    # Load font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 8)
    except:
        font = ImageFont.load_default()

    # Draw frame numbers
    if show_frame_num:
        for h in range(height):
            for w in range(width):
                n = h * width + w
                if n >= len(frames) - pad_with:
                    break

                # Calculate background brightness for text color
                region = img_tmp[h * (ih + 1) + 2 : h * (ih + 1) + 10,
                               w * (iw + 1) : w * (iw + 1) + 20, :]
                avg_c = region.mean() if region.size > 0 else 128
                text_color = (255, 255, 255) if avg_c < 128 else (0, 0, 0)

                # Show timestamp
                if frame_timestamps and n < len(frame_timestamps):
                    label = f"{frame_timestamps[n]:.1f}s"
                else:
                    label = str(n)

                draw.text(
                    (w * (iw + num_predictions), h * (ih + 1) + 2),
                    label,
                    fill=text_color,
                    font=font
                )

    # Draw boundary markers
    for i in range(len(predictions) - pad_with):
        x_pos = i % width
        y_pos = i // width

        pred_val = float(predictions[i])
        if pred_val > 0:
            x_left = x_pos * (iw + num_predictions)
            x_right = x_left + iw - 1
            y_top = y_pos * (ih + 1)
            y_bottom = y_top + ih - 1

            draw.rectangle(
                [(x_left, y_top), (x_right, y_bottom)],
                outline=color,
                width=2
            )

    return img


def visualize_scene_comparison(
    frame_folder: str,
    shot_scenes_path: str,
    captions_json_path: str,
    output_path: str,
    shot_detection_fps: float = 2.0,
    width: int = 25,
    max_frames: int = None,
    start_frame: int = 0,
):
    """
    Visualize comparison between original shots and merged scenes.

    Args:
        frame_folder: Path to folder containing video frames
        shot_scenes_path: Path to original shot_scenes.txt
        captions_json_path: Path to merged captions.json
        output_path: Path to save visualization
        shot_detection_fps: FPS used for shot detection
        width: Frames per row in visualization
        max_frames: Limit visualization to N frames (after start_frame)
        start_frame: Start visualization from this frame index (default: 0)
    """
    print("=" * 60)
    print("Scene Comparison Visualization")
    print("=" * 60)

    # Parse original scenes
    print(f"\n1. Parsing original scenes from: {shot_scenes_path}")
    original_scenes = parse_shot_scenes(shot_scenes_path)
    print(f"   Found {len(original_scenes)} original scenes")

    # Parse merged scenes
    print(f"\n2. Parsing merged scenes from: {captions_json_path}")
    merged_scenes = parse_captions_json(captions_json_path, shot_detection_fps)
    print(f"   Found {len(merged_scenes)} merged scenes")
    print(f"   Reduction: {len(original_scenes) - len(merged_scenes)} scenes")

    # Load frames
    print(f"\n3. Loading frames from: {frame_folder}")
    all_frame_files = sorted(
        [f for f in os.listdir(frame_folder)
         if f.startswith("frame") and (f.endswith(".jpg") or f.endswith(".png"))],
        key=lambda x: int(x.split("_")[-1].rstrip(".jpg").rstrip(".png")),
    )

    if not all_frame_files:
        print("Error: No frames found!")
        return

    # Apply start_frame and max_frames limit
    if start_frame > 0:
        frame_files = all_frame_files[start_frame:]
        print(f"   Starting from frame {start_frame}")
    else:
        frame_files = all_frame_files

    if max_frames:
        frame_files = frame_files[:max_frames]
        print(f"   Limiting to {max_frames} frames")

    num_frames = len(frame_files)
    print(f"   Loading {num_frames} frames...")

    frames_list = []
    for frame_file in tqdm(frame_files, desc="   Loading"):
        frame_path = os.path.join(frame_folder, frame_file)
        try:
            img = Image.open(frame_path).convert('RGB')
            img = img.resize((48, 27), Image.Resampling.BILINEAR)
            frames_list.append(np.array(img))
        except Exception as e:
            print(f"   Warning: Failed to load {frame_file}: {e}")
            frames_list.append(np.zeros((27, 48, 3), dtype=np.uint8))

    frames = np.array(frames_list)

    # Calculate timestamps (use actual frame indices from filenames)
    frame_indices = [int(f.split("_")[-1].rstrip(".jpg").rstrip(".png")) for f in frame_files]
    frame_timestamps = [idx / shot_detection_fps for idx in frame_indices]

    # Create prediction arrays
    print("\n4. Creating boundary markers...")
    original_predictions = np.zeros(num_frames, dtype=np.float32)
    merged_predictions = np.zeros(num_frames, dtype=np.float32)

    # Build a mapping from actual frame index to array position
    frame_idx_to_pos = {idx: pos for pos, idx in enumerate(frame_indices)}

    # Mark original scene boundaries (start of each scene except first)
    for i, (start, _) in enumerate(original_scenes):
        if i > 0 and start in frame_idx_to_pos:
            original_predictions[frame_idx_to_pos[start]] = 1.0

    # Mark merged scene boundaries
    for i, (start, _) in enumerate(merged_scenes):
        if i > 0 and start in frame_idx_to_pos:
            merged_predictions[frame_idx_to_pos[start]] = 1.0

    print(f"   Original boundaries: {int(original_predictions.sum())}")
    print(f"   Merged boundaries: {int(merged_predictions.sum())}")

    # Generate visualizations
    print("\n5. Generating visualizations...")
    print("   - Original scenes (RED borders)...")
    img_original = visualize_autoshot_style(
        frames,
        predictions=original_predictions,
        width=width,
        show_frame_num=True,
        frame_timestamps=frame_timestamps,
        color=(255, 0, 0),  # RED
    )

    print("   - Merged scenes (GREEN borders)...")
    img_merged = visualize_autoshot_style(
        frames,
        predictions=merged_predictions,
        width=width,
        show_frame_num=True,
        frame_timestamps=frame_timestamps,
        color=(0, 255, 0),  # GREEN
    )

    # Combine images vertically
    print("\n6. Combining images...")
    w1, h1 = img_original.size
    w2, h2 = img_merged.size

    label_height = 30
    combined_img = Image.new('RGB', (max(w1, w2), h1 + h2 + label_height * 3 + 20), (255, 255, 255))
    draw = ImageDraw.Draw(combined_img)

    # Load font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Draw labels and paste images
    current_y = 5
    draw.text((10, current_y),
              f"Original Shot Detection ({len(original_scenes)} scenes, {int(original_predictions.sum())} boundaries)",
              fill=(255, 0, 0), font=font)
    current_y += label_height
    combined_img.paste(img_original, (0, current_y))

    current_y += h1 + 10
    draw.text((10, current_y),
              f"Merged Scenes ({len(merged_scenes)} scenes, {int(merged_predictions.sum())} boundaries)",
              fill=(0, 255, 0), font=font)
    current_y += label_height
    combined_img.paste(img_merged, (0, current_y))

    # Add statistics
    current_y += h2 + 10
    reduction = len(original_scenes) - len(merged_scenes)
    reduction_pct = (reduction / len(original_scenes) * 100) if original_scenes else 0
    stats_text = f"Summary: {len(original_scenes)} → {len(merged_scenes)} scenes | Reduction: {reduction} ({reduction_pct:.1f}%)"
    draw.text((10, current_y), stats_text, fill=(0, 0, 0), font=font_small)

    # Save
    combined_img.save(output_path)
    print(f"\n7. Visualization saved to: {output_path}")
    print("\n" + "=" * 60)
    print("Interpretation Guide:")
    print("  🔴 RED borders  = Original shot boundaries")
    print("  🟢 GREEN borders = Merged scene boundaries")
    print("  → Fewer green borders means scenes were successfully merged")
    print("=" * 60)


def visualize_single_segmentation(
    frame_folder: str,
    segmentation_path: str,
    output_path: str,
    shot_detection_fps: float = 2.0,
    width: int = 25,
    max_frames: int = None,
    start_frame: int = 0,
    segmentation_type: str = "auto",
    color: Tuple[int, int, int] = (0, 255, 0),
    title: str = None,
):
    """
    Visualize a single segmentation result.

    Args:
        frame_folder: Path to folder containing video frames
        segmentation_path: Path to segmentation file (can be .txt or .json)
        output_path: Path to save visualization
        shot_detection_fps: FPS used for shot detection
        width: Frames per row in visualization
        max_frames: Limit visualization to N frames (after start_frame)
        start_frame: Start visualization from this frame index (default: 0)
        segmentation_type: Type of segmentation file ("shot_scenes", "captions", or "auto" to auto-detect)
        color: RGB color for boundary markers (default: bright green (0,255,0))
        title: Custom title for the visualization
    """
    print("=" * 60)
    print("Single Segmentation Visualization")
    print("=" * 60)

    # Auto-detect segmentation type
    if segmentation_type == "auto":
        if segmentation_path.endswith(".txt"):
            segmentation_type = "shot_scenes"
        elif segmentation_path.endswith(".json"):
            segmentation_type = "captions"
        else:
            print("Error: Cannot auto-detect file type. Please specify segmentation_type.")
            return

    # Parse segmentation
    print(f"\n1. Parsing segmentation from: {segmentation_path}")
    print(f"   Type: {segmentation_type}")

    if segmentation_type == "shot_scenes":
        scenes = parse_shot_scenes(segmentation_path)
    elif segmentation_type == "captions":
        scenes = parse_captions_json(segmentation_path, shot_detection_fps)
    else:
        print(f"Error: Unknown segmentation type '{segmentation_type}'")
        return

    print(f"   Found {len(scenes)} scenes")

    # Load frames
    print(f"\n2. Loading frames from: {frame_folder}")
    all_frame_files = sorted(
        [f for f in os.listdir(frame_folder)
         if f.startswith("frame") and (f.endswith(".jpg") or f.endswith(".png"))],
        key=lambda x: int(x.split("_")[-1].rstrip(".jpg").rstrip(".png")),
    )

    if not all_frame_files:
        print("Error: No frames found!")
        return

    # Apply start_frame and max_frames limit
    if start_frame > 0:
        frame_files = all_frame_files[start_frame:]
        print(f"   Starting from frame {start_frame}")
    else:
        frame_files = all_frame_files

    if max_frames:
        frame_files = frame_files[:max_frames]
        print(f"   Limiting to {max_frames} frames")

    num_frames = len(frame_files)
    print(f"   Loading {num_frames} frames...")

    frames_list = []
    for frame_file in tqdm(frame_files, desc="   Loading"):
        frame_path = os.path.join(frame_folder, frame_file)
        try:
            img = Image.open(frame_path).convert('RGB')
            img = img.resize((48, 27), Image.Resampling.BILINEAR)
            frames_list.append(np.array(img))
        except Exception as e:
            print(f"   Warning: Failed to load {frame_file}: {e}")
            frames_list.append(np.zeros((27, 48, 3), dtype=np.uint8))

    frames = np.array(frames_list)

    # Calculate timestamps (use actual frame indices from filenames)
    frame_indices = [int(f.split("_")[-1].rstrip(".jpg").rstrip(".png")) for f in frame_files]
    frame_timestamps = [idx / shot_detection_fps for idx in frame_indices]

    # Create prediction array
    print("\n3. Creating boundary markers...")
    predictions = np.zeros(num_frames, dtype=np.float32)

    # Build a mapping from actual frame index to array position
    frame_idx_to_pos = {idx: pos for pos, idx in enumerate(frame_indices)}

    # Mark scene boundaries (start of each scene except first)
    for i, (start, _) in enumerate(scenes):
        if i > 0 and start in frame_idx_to_pos:
            predictions[frame_idx_to_pos[start]] = 1.0

    print(f"   Total boundaries: {int(predictions.sum())}")

    # Generate visualization
    print("\n4. Generating visualization...")
    img_result = visualize_autoshot_style(
        frames,
        predictions=predictions,
        width=width,
        show_frame_num=True,
        frame_timestamps=frame_timestamps,
        color=color,
    )

    # Add title and statistics
    print("\n5. Adding title and statistics...")
    w, h = img_result.size
    label_height = 40
    final_img = Image.new('RGB', (w, h + label_height + 10), (255, 255, 255))
    draw = ImageDraw.Draw(final_img)

    # Load font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()

    # Draw title
    if title is None:
        title = os.path.basename(segmentation_path)
    title_text = f"{title} ({len(scenes)} scenes, {int(predictions.sum())} boundaries)"
    draw.text((10, 5), title_text, fill=color, font=font)

    # Paste visualization
    final_img.paste(img_result, (0, label_height + 5))

    # Save
    final_img.save(output_path)
    print(f"\n6. Visualization saved to: {output_path}")
    print("\n" + "=" * 60)
    print(f"Summary: {len(scenes)} scenes with {int(predictions.sum())} boundaries")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize video segmentation results")
    parser.add_argument("--mode", type=str, default="compare", choices=["single", "compare"],
                        help="Visualization mode: 'single' for one result, 'compare' for two results")
    parser.add_argument("--frame_folder", type=str, required=True,
                        help="Path to folder containing video frames")
    parser.add_argument("--segmentation", type=str,
                        help="Path to segmentation file (for single mode)")
    parser.add_argument("--shot_scenes", type=str,
                        help="Path to shot_scenes.txt (for compare mode)")
    parser.add_argument("--captions", type=str,
                        help="Path to captions.json (for compare mode)")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save visualization")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="FPS used for shot detection")
    parser.add_argument("--width", type=int, default=25,
                        help="Frames per row in visualization")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Limit to N frames (after start_frame)")
    parser.add_argument("--start_frame", type=int, default=0,
                        help="Start visualization from this frame index (default: 0)")
    parser.add_argument("--type", type=str, default="auto", choices=["auto", "shot_scenes", "captions"],
                        help="Type of segmentation file (for single mode)")
    parser.add_argument("--title", type=str, default=None,
                        help="Custom title for visualization (for single mode)")
    parser.add_argument("--color", type=str, default="green", choices=["red", "green", "blue", "yellow", "magenta", "cyan"],
                        help="Color for boundary markers (for single mode, default: green)")

    args = parser.parse_args()

    # Color mapping
    color_map = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 120, 255),
        "yellow": (255, 255, 0),
        "magenta": (255, 0, 255),
        "cyan": (0, 255, 255),
    }

    if args.mode == "single":
        if not args.segmentation:
            print("Error: --segmentation is required for single mode")
            exit(1)

        visualize_single_segmentation(
            frame_folder=args.frame_folder,
            segmentation_path=args.segmentation,
            output_path=args.output,
            shot_detection_fps=args.fps,
            width=args.width,
            max_frames=args.max_frames,
            start_frame=args.start_frame,
            segmentation_type=args.type,
            color=color_map[args.color],
            title=args.title,
        )

    elif args.mode == "compare":
        if not args.shot_scenes or not args.captions:
            print("Error: --shot_scenes and --captions are required for compare mode")
            exit(1)

        visualize_scene_comparison(
            frame_folder=args.frame_folder,
            shot_scenes_path=args.shot_scenes,
            captions_json_path=args.captions,
            output_path=args.output,
            shot_detection_fps=args.fps,
            width=args.width,
            max_frames=args.max_frames,
            start_frame=args.start_frame,
        )

#   python visualize_scene_comparison.py \
#       --mode single \
#       --frame_folder /public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Database/VLOG_Lisbon/frames \
#       --segmentation /public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Database/VLOG_Lisbon/frames/shot_scenes.txt \
#       --output shot_scenes_visualization.png \
#       --title "Original Shot Detection" \
#       --max_frames 500

# Example with start_frame (visualize frames 100-300):
#   python visualize_scene_comparison.py \
#       --mode single \
#       --frame_folder video_database/Database/Batman/frames \
#       --segmentation video_database/Database/Batman/frames/shot_scenes.txt \
#       --output visualization.png \
#       --start_frame 100 \
#       --max_frames 200

#   python visualize_scene_comparison.py \
#       --mode single \
#       --frame_folder video_database/Database/Batman_Begins_2005_1080p_BluRay_x264_YIFY/frames \
#       --segmentation video_database/Database/Batman_Begins_2005_1080p_BluRay_x264_YIFY/captions_min_length/captions.json \
#       --output merged_scenes_visualization.png \
#       --title "Merged Scenes"
