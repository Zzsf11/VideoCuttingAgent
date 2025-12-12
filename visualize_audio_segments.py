#!/usr/bin/env python3
"""
可视化音频分割结果
创建一个视频，在每个音频片段切换背景颜色并显示对应的caption
"""

import json
import numpy as np
import cv2
from pathlib import Path
import argparse
from tqdm import tqdm


def hex_to_bgr(hex_color):
    """将十六进制颜色转换为BGR格式"""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])  # BGR格式


def generate_colors(n):
    """生成n个区分度高的颜色"""
    colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
        '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#52B788',
        '#E63946', '#457B9D', '#F1C40F', '#E74C3C', '#9B59B6',
        '#3498DB', '#1ABC9C', '#E67E22', '#95A5A6', '#34495E'
    ]
    # 如果需要更多颜色，循环使用
    while len(colors) < n:
        colors.extend(colors)
    return [hex_to_bgr(c) for c in colors[:n]]


def wrap_text(text, font, font_scale, thickness, max_width):
    """将长文本分行，确保每行不超过最大宽度"""
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        test_line = ' '.join(current_line + [word])
        (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)

        if text_width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
            else:
                # 单个词太长，强制添加
                lines.append(word)

    if current_line:
        lines.append(' '.join(current_line))

    return lines


def draw_text_with_background(img, text, position, font, font_scale, text_color, bg_color, thickness, padding=10):
    """在背景上绘制文本"""
    x, y = position
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # 绘制背景矩形
    cv2.rectangle(img,
                  (x - padding, y - text_height - padding),
                  (x + text_width + padding, y + baseline + padding),
                  bg_color,
                  cv2.FILLED)

    # 绘制文本
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

    return text_height + baseline + padding * 2


def time_str_to_seconds(time_str):
    """将 MM:SS 格式的时间字符串转换为秒数"""
    if isinstance(time_str, (int, float)):
        return float(time_str)
    parts = time_str.split(':')
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    else:
        return float(time_str)


def normalize_segments(data):
    """标准化segments格式，支持多种输入格式"""
    # 检查是使用 'sections' 还是 'segments'
    segments_raw = data.get('segments') or data.get('sections', [])

    normalized = []
    for idx, seg in enumerate(segments_raw):
        # 处理时间字段 - 支持不同的字段名和格式
        start_time = seg.get('start_time') or seg.get('Start_Time', 0)
        end_time = seg.get('end_time') or seg.get('End_Time', 0)

        # 转换时间格式
        start_time = time_str_to_seconds(start_time)
        end_time = time_str_to_seconds(end_time)

        # 计算duration
        duration = end_time - start_time

        # 标准化segment
        normalized_seg = {
            'segment_id': seg.get('segment_id', idx + 1),
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'detailed_analysis': seg.get('detailed_analysis', {
                'summary': seg.get('description', 'No description'),
                'emotional_tone': seg.get('name', 'N/A'),
                'energy_level': 'N/A'
            })
        }
        normalized.append(normalized_seg)

    return normalized


def create_visualization_video(json_path, output_path, width=1920, height=1080, fps=30, audio_override=None):
    """创建可视化视频"""

    # 读取JSON文件
    print(f"Reading JSON file: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 使用指定的音频或JSON中的音频路径
    audio_path = audio_override if audio_override else data['audio_path']

    # 标准化segments格式
    segments = normalize_segments(data)

    print(f"Audio path: {audio_path}")
    print(f"Total segments: {len(segments)}")

    # 检查音频文件是否存在
    if not Path(audio_path).exists():
        print(f"Warning: Audio file not found: {audio_path}")
        print("Creating video without audio")
        audio_path = None
    else:
        print(f"✓ Audio file found, will be added to video")

    # 生成颜色
    colors = generate_colors(len(segments))

    # 计算总时长
    total_duration = segments[-1]['end_time']
    total_frames = int(total_duration * fps)

    print(f"Video params: {width}x{height} @ {fps}fps")
    print(f"Total duration: {total_duration:.2f}s ({total_frames} frames)")

    # 创建视频写入器（先不添加音频）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_video_path = output_path.replace('.mp4', '_temp.mp4')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    # 字体设置
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_font_scale = 1.2
    text_font_scale = 0.7
    small_font_scale = 0.5
    thickness = 2
    text_color = (255, 255, 255)  # 白色

    current_segment_idx = 0

    print("\nGenerating video frames...")
    for frame_idx in tqdm(range(total_frames), desc="Generating video"):
        current_time = frame_idx / fps

        # 找到当前时间对应的片段
        while (current_segment_idx < len(segments) - 1 and
               current_time >= segments[current_segment_idx + 1]['start_time']):
            current_segment_idx += 1

        segment = segments[current_segment_idx]

        # 检查当前时间是否在片段的有效范围内
        is_in_gap = current_time < segment['start_time'] or current_time > segment['end_time']

        bg_color = colors[current_segment_idx]

        # 创建背景
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        if is_in_gap:
            # 在间隙中使用灰色背景
            frame[:] = (80, 80, 80)
        else:
            frame[:] = bg_color

        # 添加半透明渐变效果
        overlay = frame.copy()
        gradient = np.linspace(0, 100, height, dtype=np.uint8).reshape(-1, 1)
        gradient = np.repeat(gradient, width, axis=1)
        for i in range(3):
            overlay[:, :, i] = np.clip(overlay[:, :, i].astype(int) + gradient - 50, 0, 255).astype(np.uint8)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # 绘制标题
        if is_in_gap:
            title = "Gap (No Segment)"
        else:
            title = f"Segment {segment['segment_id']}"
        y_offset = 80
        draw_text_with_background(frame, title, (50, y_offset), font, title_font_scale,
                                 text_color, (0, 0, 0), thickness + 1, padding=15)

        # 绘制时间信息
        y_offset += 60
        time_info = f"Time: {current_time:.1f}s / {total_duration:.1f}s"
        draw_text_with_background(frame, time_info, (50, y_offset), font, text_font_scale,
                                 text_color, (0, 0, 0), thickness, padding=10)

        # 绘制片段时间范围
        y_offset += 50
        if is_in_gap:
            # 在间隙中显示前后片段信息
            if current_time < segment['start_time']:
                # 在第一个片段之前
                segment_time = f"Before first segment (starts at {segment['start_time']:.1f}s)"
            else:
                # 在两个片段之间
                if current_segment_idx < len(segments) - 1:
                    next_segment_start = segments[current_segment_idx + 1]['start_time']
                    segment_time = f"Gap: {segment['end_time']:.1f}s - {next_segment_start:.1f}s"
                else:
                    segment_time = f"After last segment (ended at {segment['end_time']:.1f}s)"
        else:
            segment_time = f"Segment: {segment['start_time']:.1f}s - {segment['end_time']:.1f}s ({segment['duration']:.1f}s)"
        draw_text_with_background(frame, segment_time, (50, y_offset), font, small_font_scale,
                                 text_color, (0, 0, 0), thickness, padding=10)

        # 绘制进度条
        y_offset += 50
        progress_width = width - 100
        progress_height = 30
        progress_x = 50
        progress_y = y_offset

        # 绘制进度条背景
        cv2.rectangle(frame, (progress_x, progress_y),
                     (progress_x + progress_width, progress_y + progress_height),
                     (50, 50, 50), cv2.FILLED)

        # 绘制总进度
        total_progress = current_time / total_duration
        cv2.rectangle(frame, (progress_x, progress_y),
                     (progress_x + int(progress_width * total_progress), progress_y + progress_height),
                     (100, 200, 100), cv2.FILLED)

        # 绘制片段进度
        if segment['duration'] > 0:
            segment_progress = (current_time - segment['start_time']) / segment['duration']
            segment_progress_width = int(progress_width * (segment['duration'] / total_duration))
            segment_start_x = progress_x + int(progress_width * (segment['start_time'] / total_duration))
            cv2.rectangle(frame, (segment_start_x, progress_y),
                         (segment_start_x + int(segment_progress_width * segment_progress), progress_y + progress_height),
                         (255, 255, 100), cv2.FILLED)

        # 绘制边框
        cv2.rectangle(frame, (progress_x, progress_y),
                     (progress_x + progress_width, progress_y + progress_height),
                     (200, 200, 200), 2)

        # 绘制片段分隔线
        for seg in segments:
            sep_x = progress_x + int(progress_width * (seg['start_time'] / total_duration))
            cv2.line(frame, (sep_x, progress_y), (sep_x, progress_y + progress_height),
                    (255, 255, 255), 1)

        # 绘制Caption区域
        y_offset += 80
        caption_y = y_offset

        # Caption标题
        cv2.putText(frame, "Music Analysis:", (50, caption_y), font, text_font_scale,
                   (255, 255, 0), thickness, cv2.LINE_AA)

        # 绘制摘要
        y_offset += 50
        summary = segment['detailed_analysis'].get('summary', 'No description')
        max_text_width = width - 100
        lines = wrap_text(summary, font, small_font_scale, thickness - 1, max_text_width)

        for line in lines:
            y_offset += draw_text_with_background(frame, line, (50, y_offset), font, small_font_scale,
                                                 text_color, (0, 0, 0), thickness - 1, padding=8)

        # 绘制其他分析信息
        y_offset += 30
        info_items = [
            ('Emotion', segment['detailed_analysis'].get('emotional_tone', 'N/A')),
            ('Energy', segment['detailed_analysis'].get('energy_level', 'N/A'))
        ]

        for label, value in info_items:
            y_offset += 40
            text = f"{label}: {value}"
            lines = wrap_text(text, font, small_font_scale - 0.1, thickness - 1, max_text_width)
            for line in lines:
                y_offset += draw_text_with_background(frame, line, (50, y_offset), font,
                                                     small_font_scale - 0.1, text_color,
                                                     (0, 0, 0), thickness - 1, padding=8)

        # 写入帧
        out.write(frame)

    out.release()
    print(f"\nVideo frames completed: {temp_video_path}")

    # 添加音频（如果有）
    if audio_path and Path(audio_path).exists():
        try:
            print("\nAdding audio with FFmpeg...")
            print(f"Audio file: {audio_path}")
            import subprocess

            cmd = [
                'ffmpeg', '-y',
                '-i', temp_video_path,
                '-i', audio_path,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-shortest',
                output_path
            ]

            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ Successfully added audio to video")
                # 删除临时文件
                Path(temp_video_path).unlink()
                print(f"✓ Temporary file deleted")
            else:
                print(f"✗ FFmpeg failed")
                print(f"Error: {result.stderr}")
                print(f"Using video without audio: {temp_video_path}")
                Path(temp_video_path).rename(output_path)
        except FileNotFoundError:
            print("✗ FFmpeg not found, cannot add audio")
            print("Install FFmpeg: sudo apt-get install ffmpeg")
            print(f"Using video without audio: {temp_video_path}")
            Path(temp_video_path).rename(output_path)
        except Exception as e:
            print(f"✗ Error adding audio: {e}")
            print(f"Using video without audio: {temp_video_path}")
            Path(temp_video_path).rename(output_path)
    else:
        print("\nNo audio file provided, creating video without audio")
        Path(temp_video_path).rename(output_path)

    print(f"\n✓ Video generation complete: {output_path}")
    print(f"Total frames: {total_frames}")
    print(f"Segments: {len(segments)}")


def main():
    parser = argparse.ArgumentParser(description='Audio Segmentation Visualization Tool')
    parser.add_argument('--json', type=str,
                       default='audio_caption_madmom_output.json',
                       help='JSON file path')
    parser.add_argument('--output', type=str,
                       default='audio_visualization.mp4',
                       help='Output video path')
    parser.add_argument('--audio', type=str,
                       default=None,
                       help='Audio file path (optional, defaults to JSON path)')
    parser.add_argument('--width', type=int, default=480, help='Video width')
    parser.add_argument('--height', type=int, default=480, help='Video height')
    parser.add_argument('--fps', type=int, default=15, help='Frame rate')

    args = parser.parse_args()

    print("=" * 60)
    print("Audio Segmentation Visualization Tool")
    print("=" * 60)

    create_visualization_video(
        args.json,
        args.output,
        width=args.width,
        height=args.height,
        fps=args.fps,
        audio_override=args.audio
    )


if __name__ == '__main__':
    main()
