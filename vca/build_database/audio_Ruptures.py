"""
音频变化点检测脚本
使用Ruptures库检测音频中的所有变化点（change points）
不使用librosa，使用audioread和soundfile
"""

import sys
import os
import numpy as np
import ruptures as rpt
import argparse
from typing import List, Tuple
from scipy import signal as scipy_signal
from scipy.fft import rfft, rfftfreq
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from pathlib import Path

# 配置中文字体
def setup_chinese_font():
    """配置matplotlib的中文字体"""
    import matplotlib.font_manager as fm

    # 常见的中文字体列表
    chinese_fonts = [
        'SimHei',           # 黑体
        'Microsoft YaHei',  # 微软雅黑
        'WenQuanYi Micro Hei',  # 文泉驿微米黑 (Linux常见)
        'Noto Sans CJK SC',     # 思源黑体简体
        'AR PL UMing CN',       # 文鼎PL简中明体
        'STSong',               # 华文宋体
        'STHeiti',              # 华文黑体
    ]

    # 获取系统所有字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    # 找到第一个可用的中文字体
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            return font

    # 如果找不到中文字体，使用英文标签
    print("⚠️  未找到中文字体，将使用英文标签")
    return None

setup_chinese_font()

# 添加vca模块到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from vca.audio_utils import load_audio_no_librosa


def load_audio(audio_path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    加载音频文件（使用audioread/soundfile，不使用librosa）

    Args:
        audio_path: 音频文件路径
        sr: 采样率，默认16000Hz

    Returns:
        audio: 音频数据
        sr: 采样率
    """
    audio = load_audio_no_librosa(audio_path, sr=sr)
    return audio, sr


def compute_rms(audio: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    """
    计算RMS能量

    Args:
        audio: 音频数据
        frame_length: 帧长度
        hop_length: 帧移

    Returns:
        rms: RMS能量 (1, n_frames)
    """
    # 分帧
    n_frames = 1 + (len(audio) - frame_length) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(frame_length, n_frames),
        strides=(audio.itemsize, hop_length * audio.itemsize)
    )

    # 计算RMS
    rms = np.sqrt(np.mean(frames ** 2, axis=0))
    return rms.reshape(1, -1)


def compute_zcr(audio: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    """
    计算过零率（Zero Crossing Rate）

    Args:
        audio: 音频数据
        frame_length: 帧长度
        hop_length: 帧移

    Returns:
        zcr: 过零率 (1, n_frames)
    """
    # 分帧
    n_frames = 1 + (len(audio) - frame_length) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(frame_length, n_frames),
        strides=(audio.itemsize, hop_length * audio.itemsize)
    )

    # 计算过零率
    zcr = np.mean(np.abs(np.diff(np.sign(frames), axis=0)), axis=0) / 2
    return zcr.reshape(1, -1)


def compute_spectral_features(audio: np.ndarray, sr: int = 16000,
                              frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    """
    计算频谱特征（频谱质心、频谱滚降）

    Args:
        audio: 音频数据
        sr: 采样率
        frame_length: 帧长度
        hop_length: 帧移

    Returns:
        features: 频谱特征 (2, n_frames)
    """
    # 分帧
    n_frames = 1 + (len(audio) - frame_length) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(frame_length, n_frames),
        strides=(audio.itemsize, hop_length * audio.itemsize)
    )

    # 加窗
    window = scipy_signal.get_window('hann', frame_length)
    frames = frames * window[:, np.newaxis]

    # 计算FFT
    fft = np.abs(rfft(frames, axis=0))
    freqs = rfftfreq(frame_length, 1/sr)

    # 频谱质心
    spectral_centroid = np.sum(freqs[:, np.newaxis] * fft, axis=0) / (np.sum(fft, axis=0) + 1e-10)

    # 频谱滚降（85%能量点）
    cumsum = np.cumsum(fft, axis=0)
    total = cumsum[-1, :]
    threshold = 0.85 * total
    spectral_rolloff = np.zeros(n_frames)
    for i in range(n_frames):
        idx = np.where(cumsum[:, i] >= threshold[i])[0]
        if len(idx) > 0:
            spectral_rolloff[i] = freqs[idx[0]]
        else:
            spectral_rolloff[i] = freqs[-1]

    return np.vstack([spectral_centroid, spectral_rolloff])


def compute_mfcc_simple(audio: np.ndarray, sr: int = 16000, n_mfcc: int = 5,
                       frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    """
    计算简化的MFCC特征（使用mel滤波器组）

    Args:
        audio: 音频数据
        sr: 采样率
        n_mfcc: MFCC系数数量
        frame_length: 帧长度
        hop_length: 帧移

    Returns:
        mfcc: MFCC特征 (n_mfcc, n_frames)
    """
    # 分帧
    n_frames = 1 + (len(audio) - frame_length) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(frame_length, n_frames),
        strides=(audio.itemsize, hop_length * audio.itemsize)
    )

    # 加窗
    window = scipy_signal.get_window('hann', frame_length)
    frames = frames * window[:, np.newaxis]

    # 计算功率谱
    fft = np.abs(rfft(frames, axis=0)) ** 2

    # 创建Mel滤波器组（简化版）
    n_fft = frame_length // 2 + 1
    n_mels = 40
    mel_filters = create_mel_filterbank(n_fft, n_mels, sr)

    # 应用Mel滤波器
    mel_spec = np.dot(mel_filters, fft)
    mel_spec = np.where(mel_spec == 0, np.finfo(float).eps, mel_spec)

    # 对数
    log_mel_spec = np.log(mel_spec)

    # DCT变换得到MFCC
    from scipy.fftpack import dct
    mfcc = dct(log_mel_spec, axis=0, norm='ortho')[:n_mfcc, :]

    return mfcc


def create_mel_filterbank(n_fft: int, n_mels: int = 40, sr: int = 16000) -> np.ndarray:
    """
    创建Mel滤波器组

    Args:
        n_fft: FFT点数
        n_mels: Mel滤波器数量
        sr: 采样率

    Returns:
        mel_filters: Mel滤波器组 (n_mels, n_fft)
    """
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    # Mel频率范围
    mel_min = hz_to_mel(0)
    mel_max = hz_to_mel(sr / 2)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    # 转换为FFT bin索引
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    # 创建滤波器
    filters = np.zeros((n_mels, n_fft))
    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]

        # 上升斜坡
        for j in range(left, center):
            filters[i, j] = (j - left) / (center - left)

        # 下降斜坡
        for j in range(center, right):
            filters[i, j] = (right - j) / (right - center)

    return filters


def extract_features(audio: np.ndarray, sr: int, feature_type: str = 'rms',
                     hop_length: int = 512) -> np.ndarray:
    """
    提取音频特征（不使用librosa）

    Args:
        audio: 音频数据
        sr: 采样率
        feature_type: 特征类型，可选 'rms', 'zcr', 'spectral', 'mfcc'
        hop_length: 帧移，值越大速度越快

    Returns:
        features: 特征矩阵 (n_features, n_frames)
    """
    frame_length = hop_length * 2  # 默认帧长度为hop_length的2倍

    if feature_type == 'rms':
        # RMS能量（最快）
        features = compute_rms(audio, frame_length=frame_length, hop_length=hop_length)
    elif feature_type == 'zcr':
        # 过零率
        features = compute_zcr(audio, frame_length=frame_length, hop_length=hop_length)
    elif feature_type == 'spectral':
        # 频谱特征
        features = compute_spectral_features(audio, sr=sr, frame_length=frame_length, hop_length=hop_length)
    elif feature_type == 'mfcc':
        # MFCC特征
        features = compute_mfcc_simple(audio, sr=sr, n_mfcc=5, frame_length=frame_length, hop_length=hop_length)
    elif feature_type == 'combined':
        # 组合特征（RMS + ZCR）
        rms = compute_rms(audio, frame_length=frame_length, hop_length=hop_length)
        zcr = compute_zcr(audio, frame_length=frame_length, hop_length=hop_length)
        features = np.vstack([rms, zcr])
    else:
        raise ValueError(f"不支持的特征类型: {feature_type}")

    return features


def detect_change_points(
    features: np.ndarray,
    method: str = 'Pelt',
    model: str = 'l2',
    min_size: int = 10,
    jump: int = 10,
    pen: float = None
) -> Tuple[List[int], float]:
    """
    使用Ruptures检测变化点

    Args:
        features: 特征矩阵 (n_features, n_frames)
        method: 检测方法，可选 'Pelt', 'Binseg', 'BottomUp', 'Window'
        model: 损失模型，可选 'l1', 'l2', 'rbf', 'linear', 'normal', 'ar'
               注意：'l2'最快，'rbf'最慢但可能更准确
        min_size: 两个变化点之间的最小距离
        jump: 跳跃步长，用于加速计算（值越大越快，但精度略降）
        pen: 惩罚值，如果为None则自动估计

    Returns:
        change_points: 变化点列表（帧索引）
        pen: 实际使用的惩罚值
    """
    # 转置特征矩阵，ruptures需要 (n_frames, n_features) 格式
    signal = features.T

    print(f"  信号形状: {signal.shape} (帧数 x 特征维度)")

    # 选择检测算法
    if method == 'Pelt':
        algo = rpt.Pelt(model=model, min_size=min_size, jump=jump)
    elif method == 'Binseg':
        algo = rpt.Binseg(model=model, min_size=min_size, jump=jump)
    elif method == 'BottomUp':
        algo = rpt.BottomUp(model=model, min_size=min_size, jump=jump)
    elif method == 'Window':
        algo = rpt.Window(width=40, model=model, min_size=min_size, jump=jump)
    else:
        raise ValueError(f"不支持的检测方法: {method}")

    # 拟合数据
    print(f"  正在拟合数据...")
    algo.fit(signal)

    # 如果没有指定惩罚值，自动估计
    if pen is None:
        # 使用更小的惩罚值以检测更多变化点
        # 原始公式: np.log(n_samples) * n_features
        # 调整为更敏感的版本
        base_pen = np.log(signal.shape[0]) * signal.shape[1]
        pen = base_pen * 0.3  # 降低到30%，增加敏感度

    print(f"  使用惩罚值: {pen:.2f} (值越小检测到的变化点越多)")

    # 预测变化点
    print(f"  正在预测变化点...")
    change_points = algo.predict(pen=pen)

    # 移除最后一个点（总是数据的末尾）
    if change_points and change_points[-1] == signal.shape[0]:
        change_points = change_points[:-1]

    return change_points, pen


def frames_to_time(frames: List[int], sr: int, hop_length: int = 512) -> List[float]:
    """
    将帧索引转换为时间（秒）

    Args:
        frames: 帧索引列表
        sr: 采样率
        hop_length: 帧移

    Returns:
        times: 时间列表（秒）
    """
    times = [frame * hop_length / sr for frame in frames]
    return times


def detect_audio_change_points(
    audio_path: str,
    sr: int = 16000,
    feature_type: str = 'combined',
    method: str = 'Pelt',
    model: str = 'l2',
    min_size: int = 10,
    jump: int = 10,
    pen: float = None,
    hop_length: int = 1024
) -> Tuple[List[int], List[float], float]:
    """
    检测音频文件中的所有变化点

    Args:
        audio_path: 音频文件路径
        sr: 采样率，降低可加速（如8000）
        feature_type: 特征类型
            - 'combined': RMS+ZCR组合（默认，平衡速度和准确性）
            - 'rms': 仅RMS能量（最快但可能不够敏感）
            - 'mfcc': MFCC特征（更准确但较慢）
            - 'spectral': 频谱特征
            - 'zcr': 过零率
        method: 检测方法
        model: 损失模型，'l2'最快，'rbf'最慢但可能更准确
        min_size: 最小段长度（帧数）
        jump: 跳跃步长，值越大越快
        pen: 惩罚值，值越大检测到的变化点越少（默认自动计算）
        hop_length: 帧移，值越大速度越快但时间精度降低

    Returns:
        change_points_frames: 变化点帧索引列表
        change_points_times: 变化点时间列表（秒）
        actual_pen: 实际使用的惩罚值
    """
    print(f"正在加载音频: {audio_path}")
    audio, sr = load_audio(audio_path, sr)
    print(f"  音频时长: {len(audio)/sr:.2f}秒")

    print(f"正在提取特征: {feature_type} (hop_length={hop_length})")
    features = extract_features(audio, sr, feature_type, hop_length=hop_length)
    print(f"  特征形状: {features.shape}")

    print(f"正在检测变化点: method={method}, model={model}, jump={jump}")
    change_points_frames, actual_pen = detect_change_points(
        features, method=method, model=model,
        min_size=min_size, jump=jump, pen=pen
    )

    print(f"正在转换为时间...")
    change_points_times = frames_to_time(change_points_frames, sr, hop_length)

    return change_points_frames, change_points_times, actual_pen


def generate_color_video(
    audio_path: str,
    change_points_times: List[float],
    output_path: str = None,
    fps: int = 30,
    resolution: Tuple[int, int] = (1920, 1080),
    color_palette: List[Tuple[int, int, int]] = None,
    params_info: dict = None
):
    """
    生成一个带有纯色画面的视频，当遇到分割点时切换颜色

    Args:
        audio_path: 音频文件路径
        change_points_times: 变化点时间列表（秒）
        output_path: 输出视频路径，如果为None则自动生成
        fps: 视频帧率，默认30
        resolution: 视频分辨率 (宽, 高)，默认1920x1080
        color_palette: 颜色列表，每个颜色为RGB元组。如果为None则使用默认调色板
        params_info: 参数信息字典，用于在视频中显示
    """
    try:
        import subprocess
        import tempfile
        import shutil
    except ImportError as e:
        print(f"❌ 缺少必要的库: {e}")
        return

    # 检查ffmpeg是否可用
    if shutil.which('ffmpeg') is None:
        print("❌ 未找到ffmpeg，请安装ffmpeg后重试")
        return

    # 默认颜色调色板（柔和的颜色）
    if color_palette is None:
        color_palette = [
            (70, 130, 180),   # Steel Blue
            (255, 182, 193),  # Light Pink
            (144, 238, 144),  # Light Green
            (255, 218, 185),  # Peach
            (221, 160, 221),  # Plum
            (135, 206, 250),  # Light Sky Blue
            (255, 255, 224),  # Light Yellow
            (176, 224, 230),  # Powder Blue
            (255, 228, 225),  # Misty Rose
            (240, 230, 140),  # Khaki
            (173, 216, 230),  # Light Blue
            (255, 160, 122),  # Light Salmon
            (152, 251, 152),  # Pale Green
            (238, 130, 238),  # Violet
            (250, 250, 210),  # Light Goldenrod Yellow
        ]

    # 获取音频时长
    audio, sr = load_audio(audio_path, sr=16000)
    duration = len(audio) / sr

    # 构建时间段列表
    segments = []
    prev_time = 0.0
    for cp_time in change_points_times:
        if cp_time > prev_time:
            segments.append((prev_time, cp_time))
        prev_time = cp_time
    # 添加最后一个段
    if prev_time < duration:
        segments.append((prev_time, duration))

    # 如果没有分割点，整个视频使用一个颜色
    if not segments:
        segments = [(0.0, duration)]

    print(f"\n📊 视频段落信息:")
    print(f"  总时长: {duration:.2f}秒")
    print(f"  段落数: {len(segments)}")

    # 确定输出路径
    if output_path is None:
        audio_file = Path(audio_path)
        output_path = audio_file.parent / f"{audio_file.stem}_color_video.mp4"
    output_path = str(output_path)

    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        segment_files = []

        # 为每个段落生成视频片段
        for i, (start_time, end_time) in enumerate(segments):
            color = color_palette[i % len(color_palette)]
            segment_duration = end_time - start_time

            # 转换RGB为十六进制
            color_hex = '#{:02x}{:02x}{:02x}'.format(*color)

            segment_file = os.path.join(temp_dir, f"segment_{i:04d}.mp4")
            segment_files.append(segment_file)

            print(f"  段落 {i+1}: {start_time:.2f}s - {end_time:.2f}s (颜色: {color_hex})")

            # 构建参数显示文本
            if params_info:
                # 格式化参数信息
                param_lines = []
                param_lines.append(f"Feature: {params_info.get('feature', 'N/A')}")
                param_lines.append(f"Method: {params_info.get('method', 'N/A')}")
                param_lines.append(f"Model: {params_info.get('model', 'N/A')}")
                param_lines.append(f"Hop Length: {params_info.get('hop_length', 'N/A')}")
                param_lines.append(f"Jump: {params_info.get('jump', 'N/A')}")
                param_lines.append(f"Penalty: {params_info.get('pen', 'auto')}")
                param_lines.append(f"Min Size: {params_info.get('min_size', 'N/A')}")
                param_lines.append(f"Change Points: {params_info.get('n_change_points', 'N/A')}")
                param_lines.append(f"Segment: {i+1}/{len(segments)}")
                param_lines.append(f"Time: {start_time:.2f}s - {end_time:.2f}s")

                # 根据背景颜色选择文字颜色（深色背景用白色，浅色背景用黑色）
                brightness = (color[0] * 299 + color[1] * 587 + color[2] * 114) / 1000
                text_color = 'white' if brightness < 128 else 'black'
                shadow_color = 'black@0.5' if brightness < 128 else 'white@0.5'

                # 计算字体大小（根据分辨率调整）
                font_size = max(16, resolution[1] // 30)
                line_height = font_size + 8

                # 查找可用的字体文件
                font_paths = [
                    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                    '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
                    '/usr/share/fonts/TTF/DejaVuSans.ttf',
                    '/usr/share/fonts/dejavu/DejaVuSans.ttf',
                    '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
                    '/System/Library/Fonts/Helvetica.ttc',  # macOS
                    'C:/Windows/Fonts/arial.ttf',  # Windows
                ]

                font_file = None
                for fp in font_paths:
                    if os.path.exists(fp):
                        font_file = fp
                        break

                if font_file:
                    # 构建多行文字滤镜（使用字体文件）
                    drawtext_filters = []
                    for idx, line in enumerate(param_lines):
                        # 转义特殊字符
                        escaped_line = line.replace("'", "\\'").replace(":", "\\:")
                        y_pos = 20 + idx * line_height
                        drawtext_filters.append(
                            f"drawtext=text='{escaped_line}':fontfile='{font_file}':fontsize={font_size}:"
                            f"fontcolor={text_color}:x=20:y={y_pos}:shadowcolor={shadow_color}:shadowx=2:shadowy=2"
                        )

                    filter_chain = ','.join(drawtext_filters)

                    # 使用ffmpeg生成纯色视频段（带文字）
                    cmd = [
                        'ffmpeg', '-y',
                        '-f', 'lavfi',
                        '-i', f'color=c={color_hex}:s={resolution[0]}x{resolution[1]}:r={fps}:d={segment_duration}',
                        '-vf', filter_chain,
                        '-c:v', 'libx264',
                        '-pix_fmt', 'yuv420p',
                        '-t', str(segment_duration),
                        segment_file
                    ]
                else:
                    # 没有找到字体文件，不添加文字
                    print(f"  ⚠️  未找到字体文件，视频将不显示文字")
                    cmd = [
                        'ffmpeg', '-y',
                        '-f', 'lavfi',
                        '-i', f'color=c={color_hex}:s={resolution[0]}x{resolution[1]}:r={fps}:d={segment_duration}',
                        '-c:v', 'libx264',
                        '-pix_fmt', 'yuv420p',
                        '-t', str(segment_duration),
                        segment_file
                    ]
            else:
                # 使用ffmpeg生成纯色视频段（不带文字）
                cmd = [
                    'ffmpeg', '-y',
                    '-f', 'lavfi',
                    '-i', f'color=c={color_hex}:s={resolution[0]}x{resolution[1]}:r={fps}:d={segment_duration}',
                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    '-t', str(segment_duration),
                    segment_file
                ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ 生成段落 {i+1} 失败: {result.stderr}")
                return

        # 创建文件列表用于concat
        concat_list_file = os.path.join(temp_dir, 'concat_list.txt')
        with open(concat_list_file, 'w') as f:
            for segment_file in segment_files:
                f.write(f"file '{segment_file}'\n")

        # 合并所有视频段（无音频）
        temp_video = os.path.join(temp_dir, 'temp_video.mp4')
        cmd = [
            'ffmpeg', '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_list_file,
            '-c', 'copy',
            temp_video
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ 合并视频段失败: {result.stderr}")
            return

        # 添加音频
        print(f"\n🎵 正在添加音频...")
        cmd = [
            'ffmpeg', '-y',
            '-i', temp_video,
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-shortest',
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ 添加音频失败: {result.stderr}")
            return

    print(f"\n✅ 视频已保存到: {output_path}")


def visualize_change_points(
    audio: np.ndarray,
    sr: int,
    change_points_times: List[float],
    features: np.ndarray = None,
    feature_type: str = 'rms',
    hop_length: int = 1024,
    output_path: str = None
):
    """
    可视化音频和检测到的变化点

    Args:
        audio: 音频数据
        sr: 采样率
        change_points_times: 变化点时间列表
        features: 提取的特征（可选）
        feature_type: 特征类型
        hop_length: 帧移
        output_path: 输出图片路径，如果为None则自动生成
    """
    # 标签（中英文）
    labels = {
        'waveform_title': 'Audio Waveform and Change Points',
        'amplitude': 'Amplitude',
        'spectrogram_title': 'Spectrogram',
        'frequency': 'Frequency (Hz)',
        'power': 'Power (dB)',
        'features_title': f'Audio Features ({feature_type})',
        'normalized_feature': 'Normalized Feature Value',
        'time': 'Time (s)',
        'change_point': 'Change Point',
        'no_features': 'No features provided',
        'feature': 'Feature',
    }

    # 创建图形
    fig = plt.figure(figsize=(16, 10))

    # 时间轴
    duration = len(audio) / sr
    time_audio = np.linspace(0, duration, len(audio))

    # 1. 绘制音频波形
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(time_audio, audio, color='steelblue', linewidth=0.5, alpha=0.7)
    ax1.set_ylabel(labels['amplitude'], fontsize=12)
    ax1.set_title(labels['waveform_title'], fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, duration)

    # 标记变化点
    for cp_time in change_points_times:
        ax1.axvline(x=cp_time, color='red', linestyle='--', linewidth=2, alpha=0.8,
                   label=labels['change_point'] if cp_time == change_points_times[0] else '')

    if change_points_times:
        ax1.legend(loc='upper right', fontsize=10)

    # 2. 绘制时频图（Spectrogram）
    ax2 = plt.subplot(3, 1, 2)

    # 计算STFT
    nperseg = 2048
    noverlap = nperseg // 2
    f, t, Sxx = scipy_signal.spectrogram(audio, sr, nperseg=nperseg, noverlap=noverlap)

    # 转换为dB
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    # 绘制时频图
    im = ax2.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis')
    ax2.set_ylabel(labels['frequency'], fontsize=12)
    ax2.set_title(labels['spectrogram_title'], fontsize=14, fontweight='bold')
    ax2.set_ylim(0, min(8000, sr/2))  # 只显示到8kHz

    # 添加colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label(labels['power'], fontsize=10)

    # 标记变化点
    for cp_time in change_points_times:
        ax2.axvline(x=cp_time, color='red', linestyle='--', linewidth=2, alpha=0.8)

    # 3. 绘制特征图
    ax3 = plt.subplot(3, 1, 3)

    if features is not None:
        # 特征的时间轴
        n_frames = features.shape[1]
        time_features = np.arange(n_frames) * hop_length / sr

        # 绘制每个特征维度（使用英文标签）
        feature_names = {
            'rms': ['RMS Energy'],
            'zcr': ['Zero Crossing Rate'],
            'combined': ['RMS Energy', 'Zero Crossing Rate'],
            'spectral': ['Spectral Centroid', 'Spectral Rolloff'],
            'mfcc': [f'MFCC-{i+1}' for i in range(features.shape[0])]
        }

        names = feature_names.get(feature_type, [f'Feature-{i+1}' for i in range(features.shape[0])])
        colors = plt.cm.tab10(np.linspace(0, 1, features.shape[0]))

        for i in range(features.shape[0]):
            # 归一化特征以便显示
            feat_norm = (features[i] - features[i].min()) / (features[i].max() - features[i].min() + 1e-10)
            label = names[i] if i < len(names) else f'Feature-{i+1}'
            ax3.plot(time_features, feat_norm, label=label, color=colors[i], linewidth=1.5, alpha=0.8)

        ax3.set_ylabel(labels['normalized_feature'], fontsize=12)
        ax3.set_title(labels['features_title'], fontsize=14, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=9)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, labels['no_features'], ha='center', va='center', fontsize=14, transform=ax3.transAxes)
        ax3.set_ylabel(labels['feature'], fontsize=12)
        ax3.set_title('Audio Features', fontsize=14, fontweight='bold')

    # 标记变化点
    for cp_time in change_points_times:
        ax3.axvline(x=cp_time, color='red', linestyle='--', linewidth=2, alpha=0.8)

    ax3.set_xlabel(labels['time'], fontsize=12)
    ax3.set_xlim(0, duration)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    if output_path is None:
        output_path = 'audio_change_points_visualization.png'

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='使用Ruptures检测音频变化点（不使用librosa）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
性能优化建议:
  - 快速模式: --feature rms --jump 20 --hop-length 2048 --pen 1.0
  - 平衡模式: --feature combined --jump 10 --hop-length 1024 (默认)
  - 精确模式: --feature mfcc --jump 5 --hop-length 512 --pen 1.0
  - 敏感模式: --pen 0.5 (检测更多变化点)

示例:
  # 默认检测（平衡模式，使用RMS+ZCR特征）
  python audio_Ruptures.py audio.wav

  # 快速检测
  python audio_Ruptures.py audio.wav --feature rms --jump 20

  # 精确检测（使用MFCC特征）
  python audio_Ruptures.py audio.wav --feature mfcc --jump 5

  # 检测更多变化点（降低惩罚值）
  python audio_Ruptures.py audio.wav --pen 1.0

  # 检测并生成可视化图片
  python audio_Ruptures.py audio.wav --visualize

  # 指定可视化输出路径
  python audio_Ruptures.py audio.wav -v -o result.png

  # 生成颜色变化视频（纯色画面，分割点处切换颜色）
  python audio_Ruptures.py audio.wav --video

  # 生成自定义分辨率和帧率的视频
  python audio_Ruptures.py audio.wav --video --resolution 1280x720 --fps 24

  # 同时生成图片和视频
  python audio_Ruptures.py audio.wav --visualize --video

  # 完整示例：使用MFCC特征，降低惩罚值，并生成可视化
  python audio_Ruptures.py audio.wav --feature mfcc --pen 1.0 --visualize
        """
    )
    parser.add_argument('audio_path', type=str, help='音频文件路径')
    parser.add_argument('--sr', type=int, default=16000,
                        help='采样率，默认16000（降低如8000可加速）')
    parser.add_argument('--feature', type=str, default='combined',
                        choices=['rms', 'zcr', 'spectral', 'mfcc', 'combined'],
                        help='特征类型，默认combined（RMS+ZCR，平衡速度和准确性）')
    parser.add_argument('--method', type=str, default='Pelt',
                        choices=['Pelt', 'Binseg', 'BottomUp', 'Window'],
                        help='检测方法，默认Pelt')
    parser.add_argument('--model', type=str, default='l2',
                        choices=['l1', 'l2', 'rbf', 'linear', 'normal', 'ar'],
                        help='损失模型，默认l2（最快），rbf更准确但慢得多')
    parser.add_argument('--min-size', type=int, default=10,
                        help='最小段长度（帧数），默认10')
    parser.add_argument('--jump', type=int, default=10,
                        help='跳跃步长，默认10（值越大速度越快但精度略降）')
    parser.add_argument('--pen', type=float, default=None,
                        help='惩罚值，默认自动计算（值越大检测到的变化点越少）')
    parser.add_argument('--hop-length', type=int, default=1024,
                        help='帧移，默认1024（值越大速度越快但时间精度降低）')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='生成可视化图片（包含波形、时频图和特征）')
    parser.add_argument('--video', action='store_true',
                        help='生成颜色变化视频（纯色画面配音频，分割点处切换颜色）')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='可视化图片或视频输出路径')
    parser.add_argument('--fps', type=int, default=30,
                        help='生成视频的帧率，默认30')
    parser.add_argument('--resolution', type=str, default='1920x1080',
                        help='生成视频的分辨率，默认1920x1080')

    args = parser.parse_args()

    import time
    start_time = time.time()

    # 先加载音频和提取特征（用于可视化）
    print(f"正在加载音频: {args.audio_path}")
    audio, sr = load_audio(args.audio_path, args.sr)
    audio_duration = len(audio) / sr
    print(f"  音频时长: {audio_duration:.2f}秒")

    print(f"正在提取特征: {args.feature} (hop_length={args.hop_length})")
    features = extract_features(audio, sr, args.feature, hop_length=args.hop_length)
    print(f"  特征形状: {features.shape}")

    print(f"正在检测变化点: method={args.method}, model={args.model}, jump={args.jump}")
    change_points_frames, actual_pen = detect_change_points(
        features, method=args.method, model=args.model,
        min_size=args.min_size, jump=args.jump, pen=args.pen
    )

    print(f"正在转换为时间...")
    change_points_times = frames_to_time(change_points_frames, sr, args.hop_length)

    elapsed_time = time.time() - start_time

    # 输出结果
    print(f"\n{'='*50}")
    print(f"检测完成! 耗时: {elapsed_time:.2f}秒")
    print(f"检测到 {len(change_points_frames)} 个变化点:")
    print(f"{'='*50}")

    if len(change_points_frames) == 0:
        print("\n⚠️  未检测到任何变化点！")
        print("建议:")
        print("  1. 降低惩罚值: --pen 1.0 或 --pen 0.5")
        print("  2. 使用更敏感的特征: --feature mfcc")
        print("  3. 减小最小段长度: --min-size 5")
        print("  4. 减小跳跃步长: --jump 5")
    else:
        print(f"\n{'帧索引':>8} | {'时间(秒)':>10}")
        print("-" * 30)
        for frame, time in zip(change_points_frames, change_points_times):
            print(f"{frame:8d} | {time:10.3f}")

    # 可视化
    if args.visualize:
        print(f"\n正在生成可视化图片...")

        # 构建包含参数的文件名（使用实际的pen值）
        pen_str = f"pen{actual_pen:.2f}"
        param_suffix = f"_{args.feature}_{args.method}_{args.model}_hop{args.hop_length}_jump{args.jump}_{pen_str}_cp{len(change_points_frames)}"

        # 确定输出路径
        if args.output is None:
            audio_path = Path(args.audio_path)
            output_path = audio_path.parent / f"{audio_path.stem}{param_suffix}.png"
        else:
            # 如果用户指定了输出路径，在文件名中插入参数
            output_p = Path(args.output)
            output_path = output_p.parent / f"{output_p.stem}{param_suffix}{output_p.suffix}"

        visualize_change_points(
            audio=audio,
            sr=sr,
            change_points_times=change_points_times,
            features=features,
            feature_type=args.feature,
            hop_length=args.hop_length,
            output_path=str(output_path)
        )

    # 生成颜色变化视频
    if args.video:
        print(f"\n正在生成颜色变化视频...")

        # 解析分辨率
        try:
            width, height = map(int, args.resolution.split('x'))
            resolution = (width, height)
        except ValueError:
            print(f"⚠️  无效的分辨率格式 '{args.resolution}'，使用默认值 1920x1080")
            resolution = (1920, 1080)

        # 构建包含参数的文件名（使用实际的pen值）
        pen_str = f"pen{actual_pen:.2f}"
        param_suffix = f"_{args.feature}_{args.method}_{args.model}_hop{args.hop_length}_jump{args.jump}_{pen_str}_cp{len(change_points_frames)}"

        # 确定输出路径
        if args.output is None:
            audio_file = Path(args.audio_path)
            video_output_path = audio_file.parent / f"{audio_file.stem}{param_suffix}.mp4"
        else:
            # 如果用户指定了输出路径，在文件名中插入参数
            output_p = Path(args.output)
            video_output_path = output_p.parent / f"{output_p.stem}{param_suffix}{output_p.suffix}"

        # 构建参数信息字典
        params_info = {
            'feature': args.feature,
            'method': args.method,
            'model': args.model,
            'hop_length': args.hop_length,
            'jump': args.jump,
            'pen': f"{actual_pen:.2f}",
            'min_size': args.min_size,
            'n_change_points': len(change_points_frames),
            'sr': sr,
        }

        generate_color_video(
            audio_path=args.audio_path,
            change_points_times=change_points_times,
            output_path=str(video_output_path),
            fps=args.fps,
            resolution=resolution,
            params_info=params_info
        )

    return change_points_frames, change_points_times


if __name__ == '__main__':
    main()
