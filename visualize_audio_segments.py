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


def normalize_segments(data, flatten_sub_sections=True):
    """标准化segments格式，支持多种输入格式

    Args:
        data: JSON数据
        flatten_sub_sections: 是否将子sections展平为独立的关键点
    """
    # 检查是使用 'sections' 还是 'segments'
    segments_raw = data.get('segments') or data.get('sections', [])

    normalized = []
    global_idx = 0

    for parent_idx, seg in enumerate(segments_raw):
        # 处理时间字段 - 支持不同的字段名和格式
        parent_start = seg.get('start_time') or seg.get('Start_Time', 0)
        parent_end = seg.get('end_time') or seg.get('End_Time', 0)

        # 转换时间格式
        parent_start = time_str_to_seconds(parent_start)
        parent_end = time_str_to_seconds(parent_end)

        # 处理 detailed_analysis - 适配新的madmom格式
        detailed_analysis = seg.get('detailed_analysis', {})
        parent_name = seg.get('name', f'Segment {parent_idx + 1}')
        parent_description = seg.get('description', '')

        # 新格式：detailed_analysis 包含 summary 和 sections 数组
        if flatten_sub_sections and 'sections' in detailed_analysis and isinstance(detailed_analysis.get('sections'), list):
            sub_sections = detailed_analysis['sections']

            # 将每个子section展平为独立的关键点
            for sub_idx, sub_sec in enumerate(sub_sections):
                # 子section的时间是相对于父section的偏移
                sub_start_rel = time_str_to_seconds(sub_sec.get('Start_Time', 0))
                sub_end_rel = time_str_to_seconds(sub_sec.get('End_Time', 0))

                # 转换为绝对时间
                sub_start_abs = parent_start + sub_start_rel
                sub_end_abs = parent_start + sub_end_rel

                # 确保不超过父section的结束时间
                sub_end_abs = min(sub_end_abs, parent_end)

                duration = sub_end_abs - sub_start_abs

                # 构建名称：父section名 + 子section名
                sub_name = sub_sec.get('name', f'Sub {sub_idx + 1}')
                full_name = f"{parent_name} - {sub_name}"

                normalized_seg = {
                    'segment_id': global_idx + 1,
                    'segment_name': full_name,
                    'parent_name': parent_name,
                    'sub_name': sub_name,
                    'start_time': sub_start_abs,
                    'end_time': sub_end_abs,
                    'duration': duration,
                    'description': sub_sec.get('description', parent_description),
                    'level': 2,  # 标记为子级别
                    'detailed_analysis': {
                        'summary': sub_sec.get('description', detailed_analysis.get('summary', 'No description')),
                        'emotional_tone': sub_sec.get('Emotional_Tone', 'N/A'),
                        'energy_level': sub_sec.get('energy', 'N/A'),
                        'rhythm': sub_sec.get('rhythm', 'N/A')
                    }
                }
                normalized.append(normalized_seg)
                global_idx += 1
        else:
            # 旧格式或不展平子sections
            duration = parent_end - parent_start

            if isinstance(detailed_analysis, dict) and 'summary' in detailed_analysis:
                normalized_detailed = {
                    'summary': detailed_analysis.get('summary', parent_description),
                    'emotional_tone': detailed_analysis.get('emotional_tone', 'N/A'),
                    'energy_level': detailed_analysis.get('energy_level', 'N/A'),
                    'rhythm': detailed_analysis.get('rhythm', 'N/A')
                }
            else:
                normalized_detailed = {
                    'summary': parent_description or 'No description',
                    'emotional_tone': parent_name,
                    'energy_level': 'N/A',
                    'rhythm': 'N/A'
                }

            normalized_seg = {
                'segment_id': global_idx + 1,
                'segment_name': parent_name,
                'parent_name': parent_name,
                'sub_name': None,
                'start_time': parent_start,
                'end_time': parent_end,
                'duration': duration,
                'description': parent_description,
                'level': 1,  # 标记为父级别
                'detailed_analysis': normalized_detailed
            }
            normalized.append(normalized_seg)
            global_idx += 1

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

    # 创建视频写入器（使用无损编码保证质量）
    # 使用 AVI + MJPG 或 FFV1 作为无损中间格式
    temp_video_path = output_path.replace('.mp4', '_temp.avi')
    # 尝试使用 FFV1 无损编码，如果不可用则回退到高质量 MJPG
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')  # 无损编码
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
    if not out.isOpened():
        # 回退到 MJPG（高质量有损）
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
        print("Using MJPG encoder (high quality)")

    # 字体设置 - 根据分辨率自动缩放
    # 基准分辨率为 1920x1080，按比例缩放
    base_width = 1920
    scale_factor = width / base_width

    font = cv2.FONT_HERSHEY_SIMPLEX
    title_font_scale = max(0.4, 1.2 * scale_factor)
    text_font_scale = max(0.3, 0.7 * scale_factor)
    small_font_scale = max(0.25, 0.5 * scale_factor)
    thickness = max(1, int(2 * scale_factor))
    text_color = (255, 255, 255)  # 白色

    # 布局参数也按比例缩放
    margin = max(15, int(50 * scale_factor))
    padding_large = max(5, int(15 * scale_factor))
    padding_small = max(3, int(10 * scale_factor))

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
            level_indicator = ""
        else:
            # 优先使用 segment_name，否则使用 segment_id
            segment_name = segment.get('segment_name', f"Segment {segment['segment_id']}")
            title = f"{segment_name}"
            # 层级指示器
            level = segment.get('level', 1)
            level_indicator = f"[L{level}] " if level else ""

        y_offset = max(30, int(80 * scale_factor))
        # 显示层级指示
        if level_indicator and not is_in_gap:
            level_color = (100, 255, 100) if segment.get('level', 1) == 1 else (255, 200, 100)
            cv2.putText(frame, level_indicator, (margin, y_offset - int(30 * scale_factor)), font, small_font_scale,
                       level_color, thickness, cv2.LINE_AA)

        draw_text_with_background(frame, title, (margin, y_offset), font, title_font_scale,
                                 text_color, (0, 0, 0), thickness + 1, padding=padding_large)

        # 绘制时间信息
        y_offset += max(20, int(60 * scale_factor))
        time_info = f"Time: {current_time:.1f}s / {total_duration:.1f}s"
        draw_text_with_background(frame, time_info, (margin, y_offset), font, text_font_scale,
                                 text_color, (0, 0, 0), thickness, padding=padding_small)

        # 绘制片段时间范围
        y_offset += max(18, int(50 * scale_factor))
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
        draw_text_with_background(frame, segment_time, (margin, y_offset), font, small_font_scale,
                                 text_color, (0, 0, 0), thickness, padding=padding_small)

        # 绘制进度条
        y_offset += max(18, int(50 * scale_factor))
        progress_width = width - 2 * margin
        progress_height = max(10, int(30 * scale_factor))
        progress_x = margin
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
        y_offset += max(25, int(80 * scale_factor))
        caption_y = y_offset

        # Caption标题
        cv2.putText(frame, "Music Analysis:", (margin, caption_y), font, text_font_scale,
                   (255, 255, 0), thickness, cv2.LINE_AA)

        # 绘制摘要
        y_offset += max(15, int(50 * scale_factor))
        summary = segment['detailed_analysis'].get('summary', 'No description')
        max_text_width = width - 2 * margin
        lines = wrap_text(summary, font, small_font_scale, max(1, thickness - 1), max_text_width)

        for line in lines:
            y_offset += draw_text_with_background(frame, line, (margin, y_offset), font, small_font_scale,
                                                 text_color, (0, 0, 0), max(1, thickness - 1), padding=max(3, int(8 * scale_factor)))

        # 绘制父级section名称（如果是子级）
        if segment.get('level', 1) == 2 and segment.get('parent_name'):
            y_offset += max(8, int(20 * scale_factor))
            parent_info = f"Parent: {segment['parent_name']}"
            cv2.putText(frame, parent_info, (margin, y_offset), font, small_font_scale,
                       (150, 200, 255), max(1, thickness - 1), cv2.LINE_AA)

        # 绘制其他分析信息
        y_offset += max(10, int(30 * scale_factor))
        info_items = [
            ('Emotion', segment['detailed_analysis'].get('emotional_tone', 'N/A')),
            ('Energy', segment['detailed_analysis'].get('energy_level', 'N/A')),
            ('Rhythm', segment['detailed_analysis'].get('rhythm', 'N/A'))
        ]

        tiny_font_scale = max(0.2, small_font_scale - 0.1)
        for label, value in info_items:
            if value and value != 'N/A':  # 只显示有效值
                y_offset += max(12, int(40 * scale_factor))
                text = f"{label}: {value}"
                lines = wrap_text(text, font, tiny_font_scale, max(1, thickness - 1), max_text_width)
                for line in lines:
                    y_offset += draw_text_with_background(frame, line, (margin, y_offset), font,
                                                         tiny_font_scale, text_color,
                                                         (0, 0, 0), max(1, thickness - 1), padding=max(3, int(8 * scale_factor)))

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
                '-preset', 'veryslow',  # 最慢但质量最好
                '-crf', '0',           # 0为无损模式
                '-tune', 'stillimage',  # 优化静态图像/文字
                '-pix_fmt', 'yuv420p',
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
        # 没有音频时，也用 FFmpeg 重编码以提高质量
        print("\nNo audio file, re-encoding with FFmpeg for better quality...")
        try:
            import subprocess
            cmd = [
                'ffmpeg', '-y',
                '-i', temp_video_path,
                '-c:v', 'libx264',
                '-preset', 'veryslow',
                '-crf', '15',
                '-tune', 'stillimage',
                '-pix_fmt', 'yuv420p',
                output_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                Path(temp_video_path).unlink()
                print(f"✓ Re-encoded successfully")
            else:
                print(f"✗ FFmpeg re-encode failed, using original")
                Path(temp_video_path).rename(output_path)
        except Exception:
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
    parser.add_argument('--width', type=int, default=1920, help='Video width')
    parser.add_argument('--height', type=int, default=1080, help='Video height')
    parser.add_argument('--fps', type=int, default=10, help='Frame rate')

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
