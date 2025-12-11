"""
音频关键点检测脚本
使用Madmom库检测音频中的节拍、冲击点等关键感官点
支持可视化和视频生成
"""
# ============ Python 3.10+ 和 NumPy 1.24+ 兼容性修复 ============
# 必须在导入 madmom 之前执行

# 修复 collections 模块（Python 3.10+ 移除了直接从 collections 导入抽象基类）
import collections
import collections.abc
for attr in ('MutableSequence', 'Iterable', 'Mapping', 'MutableMapping', 'Callable'):
    if not hasattr(collections, attr):
        setattr(collections, attr, getattr(collections.abc, attr))

# 修复 numpy 模块（NumPy 1.24+ 移除了 np.float, np.int 等别名）
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    if not hasattr(np, 'float'):
        np.float = np.float64
    if not hasattr(np, 'int'):
        np.int = np.int64
    if not hasattr(np, 'complex'):
        np.complex = np.complex128
    if not hasattr(np, 'object'):
        np.object = np.object_
    if not hasattr(np, 'bool'):
        np.bool = np.bool_
    if not hasattr(np, 'str'):
        np.str = np.str_

# ============ 兼容性修复结束 ============

import os
import sys
import time
import argparse
import tempfile
import subprocess
import shutil
import json
from typing import List, Tuple
from scipy import signal as scipy_signal
from scipy.fft import rfft, rfftfreq
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from madmom.features.onsets import CNNOnsetProcessor, OnsetPeakPickingProcessor
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
from madmom.features.key import CNNKeyRecognitionProcessor, key_prediction_to_label
from madmom.audio.signal import Signal
import madmom.features.downbeats as _downbeats_module
import itertools as _it

# ============ NumPy 2.x 兼容性修复 for DBNDownBeatTrackingProcessor ============
# madmom 0.16.1 中的 np.asarray(results)[:, 1] 在 NumPy 2.x 中会失败
# 因为 results 中的元素 (path, log_prob) 形状不一致

def _patched_dbn_process(self, activations, **kwargs):
    """修复 NumPy 2.x 兼容性的 DBNDownBeatTrackingProcessor.process 方法"""
    first = 0
    if self.threshold:
        idx = np.nonzero(activations >= self.threshold)[0]
        if idx.any():
            first = max(first, np.min(idx))
            last = min(len(activations), np.max(idx) + 1)
        else:
            last = first
        activations = activations[first:last]
    
    if not activations.any():
        return np.empty((0, 2))
    
    results = list(self.map(_downbeats_module._process_dbn, 
                            zip(self.hmms, _it.repeat(activations))))
    
    # 修复: 使用列表推导式获取 log probabilities，而不是 np.asarray(results)[:, 1]
    log_probs = [r[1] for r in results]
    best = np.argmax(log_probs)
    
    path, _ = results[best]
    st = self.hmms[best].transition_model.state_space
    om = self.hmms[best].observation_model
    positions = st.state_positions[path]
    beat_numbers = positions.astype(int) + 1
    
    if self.correct:
        beats = np.empty(0, dtype=np.int64)  # 修复: np.int -> np.int64
        beat_range = om.pointers[path] >= 1
        idx = np.nonzero(np.diff(beat_range.astype(np.int64)))[0] + 1  # 修复
        if beat_range[0]:
            idx = np.r_[0, idx]
        if beat_range[-1]:
            idx = np.r_[idx, beat_range.size]
        if idx.any():
            for left, right in idx.reshape((-1, 2)):
                peak = np.argmax(activations[left:right]) // 2 + left
                beats = np.hstack((beats, peak))
    else:
        beats = np.nonzero(np.diff(beat_numbers))[0] + 1
    
    return np.vstack(((beats + first) / float(self.fps), beat_numbers[beats])).T

# 应用 monkey-patch
DBNDownBeatTrackingProcessor.process = _patched_dbn_process
# ============ NumPy 2.x 兼容性修复结束 ============

# 配置中文字体
def setup_chinese_font():
    """配置matplotlib的中文字体"""
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


class SensoryKeypointDetector:
    def __init__(
        self,
        # Onset 检测参数
        onset_threshold: float = 0.6,
        onset_smooth: float = 0.0,
        onset_pre_avg: float = 0.0,
        onset_post_avg: float = 0.0,
        onset_pre_max: float = 0.01,
        onset_post_max: float = 0.01,
        onset_combine: float = 0.03,
        # DBN 节拍检测参数
        beats_per_bar: list = None,
        min_bpm: float = 55.0,
        max_bpm: float = 215.0,
        num_tempi: int = 60,
        transition_lambda: float = 100,
        observation_lambda: int = 16,
        dbn_threshold: float = 0.05,
        correct_beats: bool = True,
        fps: int = 100,
        # 额外的音频特征检测参数
        detect_spectral_flux: bool = True,
        spectral_flux_threshold: float = 0.3,
        detect_energy_change: bool = True,
        energy_change_threshold: float = 0.15,
        detect_spectral_centroid: bool = True,
        centroid_change_threshold: float = 0.2,
    ):
        """
        音频感官关键点检测器
        
        Onset 检测参数:
            onset_threshold: Onset检测阈值，值越高检测到的冲击点越少 (默认0.6)
            onset_smooth: 平滑激活函数的窗口大小(秒) (默认0.0)
            onset_pre_avg: 计算移动平均时向前看的窗口大小(秒) (默认0.0)
            onset_post_avg: 计算移动平均时向后看的窗口大小(秒) (默认0.0)
            onset_pre_max: 计算局部最大值时向前看的窗口大小(秒) (默认0.01)
            onset_post_max: 计算局部最大值时向后看的窗口大小(秒) (默认0.01)
            onset_combine: 合并相近onset的时间窗口(秒) (默认0.03)
        
        DBN 节拍检测参数:
            beats_per_bar: 每小节的拍数，如[4]表示4/4拍，[3,4]同时检测3/4和4/4拍 (默认[4])
            min_bpm: 最小BPM (默认55.0)
            max_bpm: 最大BPM (默认215.0)
            num_tempi: 建模的速度数量 (默认60)
            transition_lambda: 速度变化的指数分布参数，值越大越倾向保持恒定速度 (默认100)
            observation_lambda: 将一个节拍周期分成的部分数 (默认16)
            dbn_threshold: 在Viterbi解码前对激活值进行阈值处理 (默认0.05)
            correct_beats: 是否将节拍对齐到最近的激活峰值 (默认True)
            fps: 帧率 (默认100)
        
        额外音频特征检测参数:
            detect_spectral_flux: 是否检测频谱通量变化（对人声/乐器变化敏感）(默认True)
            spectral_flux_threshold: 频谱通量变化阈值，值越低检测到的点越多 (默认0.3)
            detect_energy_change: 是否检测能量突变点 (默认True)
            energy_change_threshold: 能量变化阈值 (默认0.15)
            detect_spectral_centroid: 是否检测频谱质心变化（音色明暗变化）(默认True)
            centroid_change_threshold: 频谱质心变化阈值 (默认0.2)
        """
        # Onset 参数
        self.onset_threshold = onset_threshold
        self.onset_smooth = onset_smooth
        self.onset_pre_avg = onset_pre_avg
        self.onset_post_avg = onset_post_avg
        self.onset_pre_max = onset_pre_max
        self.onset_post_max = onset_post_max
        self.onset_combine = onset_combine
        
        # DBN 参数
        self.beats_per_bar = beats_per_bar if beats_per_bar is not None else [4]
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.num_tempi = num_tempi
        self.transition_lambda = transition_lambda
        self.observation_lambda = observation_lambda
        self.dbn_threshold = dbn_threshold
        self.correct_beats = correct_beats
        self.fps = fps
        
        # 额外特征检测参数
        self.detect_spectral_flux = detect_spectral_flux
        self.spectral_flux_threshold = spectral_flux_threshold
        self.detect_energy_change = detect_energy_change
        self.energy_change_threshold = energy_change_threshold
        self.detect_spectral_centroid = detect_spectral_centroid
        self.centroid_change_threshold = centroid_change_threshold

    def analyze(self, audio_path):
        print(f"正在分析音频: {audio_path} ...")

        # 1. 节奏分析 (Rhythm) - 获取强拍
        print(" -> 检测节奏 (Beats/Downbeats)...")
        print(f"    参数: beats_per_bar={self.beats_per_bar}, BPM范围=[{self.min_bpm}, {self.max_bpm}], "
              f"transition_lambda={self.transition_lambda}")
        beat_proc = RNNDownBeatProcessor()
        beat_act = beat_proc(audio_path)
        
        # 使用 DBNDownBeatTrackingProcessor（已通过 monkey-patch 修复 NumPy 2.x 兼容性）
        beat_tracker = DBNDownBeatTrackingProcessor(
            beats_per_bar=self.beats_per_bar,
            min_bpm=self.min_bpm,
            max_bpm=self.max_bpm,
            num_tempi=self.num_tempi,
            transition_lambda=self.transition_lambda,
            observation_lambda=self.observation_lambda,
            threshold=self.dbn_threshold,
            correct=self.correct_beats,
            fps=self.fps
        )
        beat_info = beat_tracker(beat_act)
        beat_info = np.array(beat_info)
        
        # 提取强拍
        if len(beat_info) > 0:
            downbeats = beat_info[beat_info[:, 1] == 1][:, 0]
        else:
            downbeats = np.array([])
            print("    ⚠️ 未能检测到节拍")
        
        # 2. 冲击力分析 (Impact) - 获取明显的起始点
        print(f" -> 检测冲击点 (Onsets, threshold={self.onset_threshold}, combine={self.onset_combine})...")
        onset_proc = CNNOnsetProcessor()
        onset_act = onset_proc(audio_path)
        # 使用可配置的阈值和其他参数
        onset_picker = OnsetPeakPickingProcessor(
            threshold=self.onset_threshold,
            smooth=self.onset_smooth,
            pre_avg=self.onset_pre_avg,
            post_avg=self.onset_post_avg,
            pre_max=self.onset_pre_max,
            post_max=self.onset_post_max,
            combine=self.onset_combine,
            fps=self.fps
        )
        onsets = onset_picker(onset_act)

        # 3. 能量分析 (Volume/Energy) - 计算均方根 (RMS)
        # 用于判断当前段落是激昂还是平静
        print(" -> 计算能量动态...")
        sig = Signal(audio_path)
        
        # 如果是多声道，转为单声道
        if len(sig.shape) > 1:
            sig_mono = np.mean(sig, axis=1)
        else:
            sig_mono = np.array(sig)
        
        # 简单的分帧计算 RMS
        frame_size = 2048
        hop_size = 1024
        rms = []
        for i in range(0, len(sig_mono), hop_size):
            frame = sig_mono[i:i+frame_size]
            if len(frame) > 0:
                rms_val = np.sqrt(np.mean(frame**2))
                rms.append(float(rms_val))  # 确保是标量
        rms = np.array(rms)
        avg_rms = float(np.mean(rms))
        
        # 5. 频谱通量检测 (Spectral Flux) - 对人声和乐器变化敏感
        spectral_flux_peaks = []
        if self.detect_spectral_flux:
            print(f" -> 检测频谱通量变化 (人声/乐器变化, threshold={self.spectral_flux_threshold})...")
            # 计算短时傅里叶变换
            n_fft = 2048
            hop_length = 512
            
            # 计算频谱图
            spectrogram = []
            for i in range(0, len(sig_mono) - n_fft, hop_length):
                frame = sig_mono[i:i+n_fft]
                windowed = frame * np.hanning(n_fft)
                spectrum = np.abs(np.fft.rfft(windowed))
                spectrogram.append(spectrum)
            spectrogram = np.array(spectrogram)
            
            if len(spectrogram) > 1:
                # 计算频谱通量（相邻帧之间的频谱差异）
                spectral_flux = np.zeros(len(spectrogram))
                for i in range(1, len(spectrogram)):
                    # 只考虑正向变化（能量增加）
                    diff = spectrogram[i] - spectrogram[i-1]
                    diff = np.maximum(diff, 0)  # 只保留正值
                    spectral_flux[i] = np.sum(diff)
                
                # 归一化
                if np.max(spectral_flux) > 0:
                    spectral_flux = spectral_flux / np.max(spectral_flux)
                
                # 检测峰值
                from scipy.signal import find_peaks
                peaks, properties = find_peaks(
                    spectral_flux, 
                    height=self.spectral_flux_threshold,
                    distance=int(0.1 * sig.sample_rate / hop_length)  # 最小间隔0.1秒
                )
                
                # 转换为时间
                for peak_idx in peaks:
                    t = peak_idx * hop_length / sig.sample_rate
                    height = float(properties['peak_heights'][list(peaks).index(peak_idx)])
                    spectral_flux_peaks.append({'time': t, 'intensity': height})
                
                print(f"    检测到 {len(spectral_flux_peaks)} 个频谱变化点")
        
        # 6. 能量突变检测 (Energy Change Detection)
        energy_change_peaks = []
        if self.detect_energy_change:
            print(f" -> 检测能量突变点 (threshold={self.energy_change_threshold})...")
            if len(rms) > 1:
                # 计算RMS的差分
                rms_diff = np.abs(np.diff(rms))
                if np.max(rms_diff) > 0:
                    rms_diff_norm = rms_diff / np.max(rms_diff)
                else:
                    rms_diff_norm = rms_diff
                
                # 检测峰值
                from scipy.signal import find_peaks
                peaks, properties = find_peaks(
                    rms_diff_norm, 
                    height=self.energy_change_threshold,
                    distance=int(0.2 * sig.sample_rate / hop_size)  # 最小间隔0.2秒
                )
                
                for peak_idx in peaks:
                    t = peak_idx * hop_size / sig.sample_rate
                    height = float(properties['peak_heights'][list(peaks).index(peak_idx)])
                    energy_change_peaks.append({'time': t, 'intensity': height})
                
                print(f"    检测到 {len(energy_change_peaks)} 个能量突变点")
        
        # 7. 频谱质心变化检测 (Spectral Centroid Change) - 音色明暗变化
        centroid_change_peaks = []
        if self.detect_spectral_centroid:
            print(f" -> 检测频谱质心变化 (音色变化, threshold={self.centroid_change_threshold})...")
            n_fft = 2048
            hop_length = 512
            
            centroids = []
            for i in range(0, len(sig_mono) - n_fft, hop_length):
                frame = sig_mono[i:i+n_fft]
                windowed = frame * np.hanning(n_fft)
                spectrum = np.abs(np.fft.rfft(windowed))
                
                # 计算频谱质心
                freqs = np.fft.rfftfreq(n_fft, 1/sig.sample_rate)
                if np.sum(spectrum) > 0:
                    centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
                else:
                    centroid = 0
                centroids.append(centroid)
            
            centroids = np.array(centroids)
            
            if len(centroids) > 1:
                # 平滑频谱质心
                from scipy.ndimage import uniform_filter1d
                centroids_smooth = uniform_filter1d(centroids, size=5)
                
                # 计算变化率
                centroid_diff = np.abs(np.diff(centroids_smooth))
                if np.max(centroid_diff) > 0:
                    centroid_diff_norm = centroid_diff / np.max(centroid_diff)
                else:
                    centroid_diff_norm = centroid_diff
                
                # 检测峰值
                from scipy.signal import find_peaks
                peaks, properties = find_peaks(
                    centroid_diff_norm, 
                    height=self.centroid_change_threshold,
                    distance=int(0.15 * sig.sample_rate / hop_length)  # 最小间隔0.15秒
                )
                
                for peak_idx in peaks:
                    t = peak_idx * hop_length / sig.sample_rate
                    height = float(properties['peak_heights'][list(peaks).index(peak_idx)])
                    centroid_change_peaks.append({'time': t, 'intensity': height})
                
                print(f"    检测到 {len(centroid_change_peaks)} 个音色变化点")

        # 8. 情感基调 (Emotion Context) - 调性识别
        print(" -> 识别调性与情感基调...")
        key_proc = CNNKeyRecognitionProcessor()
        key_probs = key_proc(audio_path)
        key_label = key_prediction_to_label(key_probs)
        
        # 结果整合
        timeline = []

        # 计算每个时间点的局部能量（用于强度评估）
        duration = len(sig_mono) / sig.sample_rate
        rms_times = np.linspace(0, duration, len(rms))
        rms_max = np.max(rms) if len(rms) > 0 else 1.0
        
        # 局部窗口大小（秒），用于计算局部相对能量
        local_window = 10.0  # 10秒窗口
        
        def get_local_relative_energy(t):
            """
            获取时间t处的局部相对能量
            使用局部窗口内的最大值进行归一化，这样intro部分的关键点也能有较高的相对强度
            """
            if len(rms) == 0:
                return 0.5
            
            idx = np.argmin(np.abs(rms_times - t))
            current_rms = float(rms[idx])
            
            # 找到局部窗口内的所有RMS值
            window_start = max(0, t - local_window / 2)
            window_end = min(duration, t + local_window / 2)
            
            mask = (rms_times >= window_start) & (rms_times <= window_end)
            local_rms = rms[mask]
            
            if len(local_rms) == 0:
                return 0.5
            
            local_max = np.max(local_rms)
            local_min = np.min(local_rms)
            
            # 局部相对能量：在局部范围内的相对位置
            if local_max - local_min < 1e-10:
                local_relative = 0.5
            else:
                local_relative = (current_rms - local_min) / (local_max - local_min)
            
            # 全局相对能量
            global_relative = current_rms / (rms_max + 1e-10)
            
            # 综合：70% 局部相对 + 30% 全局相对
            # 这样既保留了局部变化的敏感性，又保留了一定的全局信息
            return 0.7 * local_relative + 0.3 * global_relative
        
        # 添加强拍事件（强拍的基础强度更高）
        for t in downbeats:
            energy = get_local_relative_energy(t)
            # 强拍强度 = 基础强度(0.7) + 能量加成(0.3)
            intensity = 0.7 + 0.3 * energy
            timeline.append({'time': float(t), 'type': 'Downbeat (重拍)', 'intensity': float(intensity)})

        # 添加 Onset 事件 (为了避免和强拍重复，可以做个简单的去重或标记)
        for t in onsets:
            # 如果这个 onset 距离某个 downbeat 很近 (<0.05s)，则忽略它(视为同一点)
            if not np.any(np.abs(downbeats - t) < 0.05):
                energy = get_local_relative_energy(t)
                # 冲击点强度 = 基础强度(0.5) + 能量加成(0.5)
                intensity = 0.5 + 0.5 * energy
                timeline.append({'time': float(t), 'type': 'Strong Attack (冲击)', 'intensity': float(intensity)})
        
        # 添加频谱通量变化点（人声/乐器变化）
        existing_times = [kp['time'] for kp in timeline]
        for sf in spectral_flux_peaks:
            t = sf['time']
            # 避免与已有点重复
            if not any(abs(t - et) < 0.08 for et in existing_times):
                energy = get_local_relative_energy(t)
                intensity = 0.4 + 0.4 * sf['intensity'] + 0.2 * energy
                timeline.append({'time': float(t), 'type': 'Spectral Change (频谱变化)', 'intensity': float(intensity)})
                existing_times.append(t)
        
        # 添加能量突变点
        for ec in energy_change_peaks:
            t = ec['time']
            if not any(abs(t - et) < 0.08 for et in existing_times):
                intensity = 0.5 + 0.5 * ec['intensity']
                timeline.append({'time': float(t), 'type': 'Energy Change (能量变化)', 'intensity': float(intensity)})
                existing_times.append(t)
        
        # 添加音色变化点
        for cc in centroid_change_peaks:
            t = cc['time']
            if not any(abs(t - et) < 0.08 for et in existing_times):
                energy = get_local_relative_energy(t)
                intensity = 0.35 + 0.35 * cc['intensity'] + 0.3 * energy
                timeline.append({'time': float(t), 'type': 'Timbre Change (音色变化)', 'intensity': float(intensity)})

        # 按时间排序
        timeline.sort(key=lambda x: x['time'])

        return {
            "meta": {
                "key": key_label,
                "avg_energy": f"{avg_rms:.4f}",
                "emotion_clue": "Happy/Bright" if "Major" in key_label else "Sad/Serious"
            },
            "keypoints": timeline,
            "downbeats": downbeats,
            "onsets": onsets,
            "spectral_flux_peaks": spectral_flux_peaks,
            "energy_change_peaks": energy_change_peaks,
            "centroid_change_peaks": centroid_change_peaks,
            "beat_info": beat_info,
            "onset_activation": onset_act,
            "rms": np.array(rms),
            "audio_signal": sig,
            "sample_rate": sig.sample_rate
        }


def filter_significant_keypoints(
    keypoints: List[dict],
    min_interval: float = 0.0,
    top_k: int = 0,
    energy_percentile: float = 0.0,
    merge_close: float = 0.1,
    segment_duration: float = 0.0,
    segment_top_k: int = 0
) -> List[dict]:
    """
    过滤关键点，只保留显著的点
    
    Args:
        keypoints: 原始关键点列表
        min_interval: 最小间隔（秒），间隔内只保留最强的点
        top_k: 只保留强度最高的前K个点，0表示不限制
        energy_percentile: 只保留强度高于该百分位数的点(0-100)
        merge_close: 合并间隔小于此值的相邻点
        segment_duration: 分段时长（秒），与segment_top_k配合使用
        segment_top_k: 每个时间段内保留的最强点数量，0表示不使用分段过滤
    
    Returns:
        过滤后的关键点列表
    """
    if not keypoints:
        return []
    
    filtered = list(keypoints)
    
    # 1. 合并相近的点（保留强度最高的）
    if merge_close > 0:
        filtered.sort(key=lambda x: x['time'])
        merged = []
        i = 0
        while i < len(filtered):
            # 收集在 merge_close 范围内的所有点
            group = [filtered[i]]
            j = i + 1
            while j < len(filtered) and filtered[j]['time'] - filtered[i]['time'] < merge_close:
                group.append(filtered[j])
                j += 1
            # 保留强度最高的点
            best = max(group, key=lambda x: x['intensity'])
            merged.append(best)
            i = j
        filtered = merged
        print(f"    合并相近点后: {len(filtered)} 个关键点 (merge_close={merge_close}s)")
    
    # 2. 按强度百分位数过滤
    if energy_percentile > 0 and filtered:
        intensities = [kp['intensity'] for kp in filtered]
        threshold = np.percentile(intensities, energy_percentile)
        filtered = [kp for kp in filtered if kp['intensity'] >= threshold]
        print(f"    强度过滤后: {len(filtered)} 个关键点 (保留强度>={threshold:.3f}的点)")
    
    # 3. 按最小间隔过滤（在每个间隔内只保留最强的点）
    if min_interval > 0 and filtered:
        filtered.sort(key=lambda x: x['time'])
        interval_filtered = []
        current_interval_start = filtered[0]['time']
        current_best = filtered[0]
        
        for kp in filtered[1:]:
            if kp['time'] - current_interval_start < min_interval:
                # 在同一间隔内，保留强度更高的
                if kp['intensity'] > current_best['intensity']:
                    current_best = kp
            else:
                # 新间隔，保存之前的最佳点
                interval_filtered.append(current_best)
                current_interval_start = kp['time']
                current_best = kp
        
        # 添加最后一个
        interval_filtered.append(current_best)
        filtered = interval_filtered
        print(f"    最小间隔过滤后: {len(filtered)} 个关键点 (min_interval={min_interval}s)")
    
    # 4. 只保留 top_k 个
    if top_k > 0 and len(filtered) > top_k:
        # 按强度排序，取前k个，然后再按时间排序
        filtered.sort(key=lambda x: x['intensity'], reverse=True)
        filtered = filtered[:top_k]
        filtered.sort(key=lambda x: x['time'])
        print(f"    Top-K 过滤后: {len(filtered)} 个关键点 (top_k={top_k})")
    
    # 5. 分段过滤：每个时间段保留segment_top_k个最强的点（保证各段都有代表）
    if segment_duration > 0 and segment_top_k > 0 and filtered:
        filtered.sort(key=lambda x: x['time'])
        max_time = max(kp['time'] for kp in filtered)
        
        segment_filtered = []
        segment_start = 0
        
        while segment_start < max_time:
            segment_end = segment_start + segment_duration
            # 获取该段内的所有点
            segment_points = [kp for kp in filtered 
                            if segment_start <= kp['time'] < segment_end]
            
            if segment_points:
                # 按强度排序，取前segment_top_k个
                segment_points.sort(key=lambda x: x['intensity'], reverse=True)
                segment_filtered.extend(segment_points[:segment_top_k])
            
            segment_start = segment_end
        
        # 按时间重新排序
        segment_filtered.sort(key=lambda x: x['time'])
        filtered = segment_filtered
        print(f"    分段过滤后: {len(filtered)} 个关键点 "
              f"(每{segment_duration}s保留{segment_top_k}个最强点)")
    
    return filtered


def parse_time_str(time_str: str) -> float:
    """
    解析时间字符串为秒数
    支持格式: "MM:SS" 或 "HH:MM:SS" 或直接数字
    """
    if isinstance(time_str, (int, float)):
        return float(time_str)
    
    time_str = str(time_str).strip()
    parts = time_str.split(':')
    
    if len(parts) == 1:
        return float(parts[0])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    else:
        raise ValueError(f"无法解析时间格式: {time_str}")


def load_sections_from_caption(caption_path: str) -> List[dict]:
    """
    从 caption JSON 文件加载 sections 信息
    
    Args:
        caption_path: caption.json 文件路径
    
    Returns:
        sections 列表，每个元素包含 name, start_time, end_time
    """
    with open(caption_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sections = []
    for sec in data.get('sections', []):
        try:
            start_time = parse_time_str(sec.get('Start_Time', 0))
            end_time = parse_time_str(sec.get('End_Time', 0))
            name = sec.get('name', 'Unknown')
            
            if end_time > start_time:
                sections.append({
                    'name': name,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time
                })
        except Exception as e:
            print(f"    ⚠️ 解析 section 失败: {sec.get('name', 'Unknown')} - {e}")
    
    return sections


def filter_by_sections(
    keypoints: List[dict],
    sections: List[dict],
    section_top_k: int = 3,
    section_min_interval: float = 0.0,
    section_energy_percentile: float = 0.0
) -> List[dict]:
    """
    基于音乐段落（sections）进行关键点过滤
    确保每个段落都有代表性的关键点
    
    Args:
        keypoints: 原始关键点列表
        sections: 段落列表，每个包含 name, start_time, end_time
        section_top_k: 每个段落保留的最强点数量
        section_min_interval: 每个段落内的最小间隔
        section_energy_percentile: 每个段落内的强度百分位数阈值(0-100)，只保留高于该阈值的点
    
    Returns:
        过滤后的关键点列表
    """
    if not keypoints or not sections:
        return keypoints
    
    filtered = []
    
    print(f"\n    📂 基于 {len(sections)} 个音乐段落进行过滤:")
    
    for sec in sections:
        name = sec.get('name', 'Unknown')
        
        # 兼容不同的键名和时间格式
        start_val = sec.get('start_time', sec.get('Start_Time', 0))
        end_val = sec.get('end_time', sec.get('End_Time', 0))
        
        try:
            start = parse_time_str(start_val)
            end = parse_time_str(end_val)
        except Exception as e:
            print(f"       ⚠️ 跳过无效时间段: {name} ({start_val}-{end_val}) - {e}")
            continue
            
        duration = sec.get('duration', end - start)
        
        # 获取该段落内的所有关键点
        section_points = [kp for kp in keypoints 
                         if start <= kp['time'] < end]
        
        if not section_points:
            print(f"       [{name}] {start:.1f}s-{end:.1f}s: 无关键点")
            continue
        
        # 1. 如果设置了段落内百分位数过滤，先应用
        if section_energy_percentile > 0 and len(section_points) > 1:
            intensities = [kp['intensity'] for kp in section_points]
            threshold = np.percentile(intensities, section_energy_percentile)
            before_count = len(section_points)
            section_points = [kp for kp in section_points if kp['intensity'] >= threshold]
            if len(section_points) < before_count:
                pass  # 过滤成功
        
        # 2. 如果设置了最小间隔，在段落内应用
        if section_min_interval > 0 and section_points:
            section_points.sort(key=lambda x: x['time'])
            interval_filtered = []
            current_start = section_points[0]['time']
            current_best = section_points[0]
            
            for kp in section_points[1:]:
                if kp['time'] - current_start < section_min_interval:
                    if kp['intensity'] > current_best['intensity']:
                        current_best = kp
                else:
                    interval_filtered.append(current_best)
                    current_start = kp['time']
                    current_best = kp
            interval_filtered.append(current_best)
            section_points = interval_filtered
        
        # 3. 按强度排序，如果设置了 section_top_k 则取前 K 个
        section_points.sort(key=lambda x: x['intensity'], reverse=True)
        if section_top_k > 0:
            selected = section_points[:section_top_k]
        else:
            # section_top_k=0 表示不限制数量，保留所有经过前面过滤的点
            selected = section_points
        
        # 为选中的点添加段落信息
        for pt in selected:
            pt['section'] = name
        
        filtered.extend(selected)
        
        print(f"       [{name}] {start:.1f}s-{end:.1f}s ({duration:.1f}s): "
              f"保留 {len(selected)}/{len([kp for kp in keypoints if start <= kp['time'] < end])} 个点")
    
    # 按时间排序
    filtered.sort(key=lambda x: x['time'])
    
    print(f"    段落过滤后共: {len(filtered)} 个关键点")
    
    return filtered


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


def visualize_keypoints(
    audio_path: str,
    result: dict,
    output_path: str = None,
    show_beats: bool = True,
    show_onsets: bool = True
):
    """
    可视化音频和检测到的关键点

    Args:
        audio_path: 音频文件路径
        result: analyze()返回的结果字典
        output_path: 输出图片路径，如果为None则自动生成
        show_beats: 是否显示节拍点
        show_onsets: 是否显示冲击点
    """
    # 获取数据
    sig = result['audio_signal']
    sr = result['sample_rate']
    downbeats = result['downbeats']
    onsets = result['onsets']
    beat_info = result['beat_info']
    onset_act = result['onset_activation']
    rms = result['rms']
    
    # 如果是多声道，转为单声道
    if len(sig.shape) > 1:
        audio = np.mean(sig, axis=1)
    else:
        audio = np.array(sig)
    
    duration = len(audio) / sr
    time_audio = np.linspace(0, duration, len(audio))

    # 创建图形
    fig = plt.figure(figsize=(16, 12))

    # 1. 绘制音频波形
    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(time_audio, audio, color='steelblue', linewidth=0.5, alpha=0.7)
    ax1.set_ylabel('Amplitude', fontsize=12)
    ax1.set_title('Audio Waveform with Keypoints', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, duration)

    # 标记强拍（红色）
    if show_beats:
        for i, t in enumerate(downbeats):
            ax1.axvline(x=t, color='red', linestyle='-', linewidth=1.5, alpha=0.8,
                       label='Downbeat' if i == 0 else '')
    
    # 标记冲击点（绿色）
    if show_onsets:
        for i, t in enumerate(onsets):
            ax1.axvline(x=t, color='green', linestyle='--', linewidth=1, alpha=0.6,
                       label='Onset' if i == 0 else '')

    ax1.legend(loc='upper right', fontsize=10)

    # 2. 绘制时频图（Spectrogram）
    ax2 = plt.subplot(4, 1, 2)
    nperseg = 2048
    noverlap = nperseg // 2
    f, t, Sxx = scipy_signal.spectrogram(audio, sr, nperseg=nperseg, noverlap=noverlap)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    im = ax2.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis')
    ax2.set_ylabel('Frequency (Hz)', fontsize=12)
    ax2.set_title('Spectrogram', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, min(8000, sr/2))
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Power (dB)', fontsize=10)

    # 标记关键点
    if show_beats:
        for t_pt in downbeats:
            ax2.axvline(x=t_pt, color='red', linestyle='-', linewidth=1.5, alpha=0.8)
    if show_onsets:
        for t_pt in onsets:
            ax2.axvline(x=t_pt, color='green', linestyle='--', linewidth=1, alpha=0.6)

    # 3. 绘制节拍信息（所有拍子）
    ax3 = plt.subplot(4, 1, 3)
    
    # 绘制所有节拍
    beat_times = beat_info[:, 0]
    beat_nums = beat_info[:, 1]
    
    # 用不同颜色标记不同拍号
    colors_map = {1: 'red', 2: 'blue', 3: 'green', 4: 'orange'}
    for beat_num in np.unique(beat_nums):
        mask = beat_nums == beat_num
        times_subset = beat_times[mask]
        ax3.scatter(times_subset, [beat_num] * len(times_subset), 
                   c=colors_map.get(int(beat_num), 'gray'), 
                   s=50, alpha=0.7, label=f'Beat {int(beat_num)}')
    
    ax3.set_ylabel('Beat Number', fontsize=12)
    ax3.set_title('Beat Detection (1=Downbeat)', fontsize=14, fontweight='bold')
    ax3.set_xlim(0, duration)
    ax3.set_yticks([1, 2, 3, 4])
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', fontsize=9)

    # 4. 绘制Onset激活函数和RMS能量
    ax4 = plt.subplot(4, 1, 4)
    
    # Onset激活函数
    onset_time = np.linspace(0, duration, len(onset_act))
    ax4.plot(onset_time, onset_act / np.max(onset_act), 
             color='orange', linewidth=1, alpha=0.8, label='Onset Activation (normalized)')
    
    # RMS能量
    rms_time = np.linspace(0, duration, len(rms))
    rms_norm = rms / np.max(rms)
    ax4.plot(rms_time, rms_norm, color='purple', linewidth=1, alpha=0.8, label='RMS Energy (normalized)')
    
    ax4.set_ylabel('Normalized Value', fontsize=12)
    ax4.set_xlabel('Time (s)', fontsize=12)
    ax4.set_title('Onset Activation & RMS Energy', fontsize=14, fontweight='bold')
    ax4.set_xlim(0, duration)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right', fontsize=9)

    # 标记关键点
    if show_beats:
        for t_pt in downbeats:
            ax4.axvline(x=t_pt, color='red', linestyle='-', linewidth=1.5, alpha=0.5)
    if show_onsets:
        for t_pt in onsets:
            ax4.axvline(x=t_pt, color='green', linestyle='--', linewidth=1, alpha=0.4)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    if output_path is None:
        audio_file = Path(audio_path)
        output_path = audio_file.parent / f"{audio_file.stem}_madmom_keypoints.png"

    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    print(f"\n✅ 可视化图片已保存到: {output_path}")

    plt.close()
    return str(output_path)


def generate_color_video(
    audio_path: str,
    keypoints: List[dict],
    output_path: str = None,
    fps: int = 30,
    resolution: Tuple[int, int] = (1920, 1080),
    color_palette: List[Tuple[int, int, int]] = None,
    params_info: dict = None
):
    """
    生成一个带有纯色画面的视频，当遇到关键点时切换颜色

    Args:
        audio_path: 音频文件路径
        keypoints: 关键点列表，每个元素包含'time'字段
        output_path: 输出视频路径，如果为None则自动生成
        fps: 视频帧率，默认30
        resolution: 视频分辨率 (宽, 高)，默认1920x1080
        color_palette: 颜色列表，每个颜色为RGB元组。如果为None则使用默认调色板
        params_info: 参数信息字典，用于在视频中显示
    """
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

    # 提取关键点时间
    change_points_times = sorted([kp['time'] for kp in keypoints])

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
        output_path = audio_file.parent / f"{audio_file.stem}_madmom_color_video.mp4"
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

            if i < 10 or i >= len(segments) - 3:  # 只打印前10个和最后3个
                print(f"  段落 {i+1}: {start_time:.2f}s - {end_time:.2f}s (颜色: {color_hex})")
            elif i == 10:
                print(f"  ... (共 {len(segments)} 个段落)")

            # 构建参数显示文本
            if params_info:
                # 格式化参数信息
                param_lines = []
                param_lines.append(f"Method: Madmom")
                param_lines.append(f"Onset Threshold: {params_info.get('onset_threshold', 'N/A')}")
                param_lines.append(f"Key: {params_info.get('key', 'N/A')}")
                param_lines.append(f"Emotion: {params_info.get('emotion', 'N/A')}")
                param_lines.append(f"Keypoints: {params_info.get('n_keypoints', 'N/A')}")
                param_lines.append(f"Segment: {i+1}/{len(segments)}")
                param_lines.append(f"Time: {start_time:.2f}s - {end_time:.2f}s")

                # 根据背景颜色选择文字颜色
                brightness = (color[0] * 299 + color[1] * 587 + color[2] * 114) / 1000
                text_color = 'white' if brightness < 128 else 'black'

                # 计算字体大小
                font_size = max(16, resolution[1] // 30)

                # 查找可用的字体文件
                font_paths = [
                    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                    '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
                    '/usr/share/fonts/TTF/DejaVuSans.ttf',
                    '/usr/share/fonts/dejavu/DejaVuSans.ttf',
                    '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
                    '/System/Library/Fonts/Helvetica.ttc',
                    'C:/Windows/Fonts/arial.ttf',
                ]

                font_file = None
                for fp in font_paths:
                    if os.path.exists(fp):
                        font_file = fp
                        break

                if font_file:
                    # 构建多行文字滤镜
                    drawtext_filters = []
                    for idx, line in enumerate(param_lines):
                        y_pos = 20 + idx * (font_size + 8)
                        escaped_line = line.replace(":", r"\:").replace("'", r"\'")
                        drawtext_filters.append(
                            f"drawtext=fontfile='{font_file}':text='{escaped_line}':"
                            f"x=20:y={y_pos}:fontsize={font_size}:fontcolor={text_color}"
                        )

                    filter_chain = ','.join(drawtext_filters)

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
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='使用Madmom检测音频关键点（节拍、冲击点等）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 默认检测
  python audio_Madmom.py audio.wav

  # 检测并生成可视化图片
  python audio_Madmom.py audio.wav --visualize

  # 调整onset阈值（值越高检测到的冲击点越少）
  python audio_Madmom.py audio.wav --onset-threshold 0.8 --visualize

  # 检测3/4拍或4/4拍的音乐
  python audio_Madmom.py audio.wav --beats-per-bar 3 4

  # 指定BPM范围（适用于已知速度的音乐）
  python audio_Madmom.py audio.wav --min-bpm 80 --max-bpm 120

  # 使用高transition-lambda保持稳定节拍（适合节奏稳定的音乐）
  python audio_Madmom.py audio.wav --transition-lambda 200

  # 生成颜色变化视频
  python audio_Madmom.py audio.wav --video

  # 生成自定义分辨率和帧率的视频
  python audio_Madmom.py audio.wav --video --resolution 1280x720 --video-fps 24

  # 同时生成图片和视频
  python audio_Madmom.py audio.wav --visualize --video

  # 只使用强拍作为分割点（减少分割数量）
  python audio_Madmom.py audio.wav --video --downbeats-only

  # ===== 显著性过滤示例（减少关键点数量） =====
  
  # 只保留强度最高的20个关键点
  python audio_Madmom.py audio.wav --top-k 20 --video

  # 设置最小间隔0.5秒，每0.5秒内只保留最强的一个点
  python audio_Madmom.py audio.wav --min-interval 0.5 --video

  # 只保留强度高于50%百分位数的点（去掉一半弱点）
  python audio_Madmom.py audio.wav --energy-percentile 50 --video

  # 合并间隔小于0.2秒的相邻点（减少碎片化）
  python audio_Madmom.py audio.wav --merge-close 0.2 --video

  # 综合过滤：合并相近点 + 最小间隔 + 只取前30个最强点
  python audio_Madmom.py audio.wav --merge-close 0.15 --min-interval 0.3 --top-k 30 --video

  # ===== 分段均匀采样（保留intro等低能量段的关键点） =====
  
  # 每10秒保留3个最强点（确保各时间段都有代表）
  python audio_Madmom.py audio.wav --segment-duration 10 --segment-top-k 3 --video

  # 先合并相近点，再分段采样
  python audio_Madmom.py audio.wav --merge-close 0.2 --segment-duration 15 --segment-top-k 5 --video

  # ===== 基于 caption 段落过滤（推荐方式） =====
  
  # 根据 caption.json 中的 sections 划分，每段保留3个最强点
  python audio_Madmom.py audio.wav --caption captions.json --section-top-k 3 --video

  # 每段保留5个点，段内最小间隔0.5秒
  python audio_Madmom.py audio.wav --caption captions.json --section-top-k 5 --section-min-interval 0.5 --video

  # 每段只保留强度高于50%百分位的点，然后取前3个
  python audio_Madmom.py audio.wav --caption captions.json --section-energy-percentile 50 --section-top-k 3 --video

  # 完整参数示例
  python audio_Madmom.py audio.wav --onset-threshold 0.7 --beats-per-bar 4 \\
      --min-bpm 60 --max-bpm 180 --transition-lambda 100 --visualize --video
        """
    )
    parser.add_argument('audio_path', type=str, help='音频文件路径')
    
    # === Onset 检测参数 ===
    onset_group = parser.add_argument_group('Onset检测参数')
    onset_group.add_argument('--onset-threshold', type=float, default=0.6,
                        help='Onset检测阈值，默认0.6（值越高检测到的冲击点越少）')
    onset_group.add_argument('--onset-smooth', type=float, default=0.5,
                        help='平滑激活函数的窗口大小(秒)，默认0.0')
    onset_group.add_argument('--onset-pre-avg', type=float, default=0.5,
                        help='计算移动平均时向前看的窗口大小(秒)，默认0.0')
    onset_group.add_argument('--onset-post-avg', type=float, default=0.5,
                        help='计算移动平均时向后看的窗口大小(秒)，默认0.0')
    onset_group.add_argument('--onset-pre-max', type=float, default=0.5,
                        help='计算局部最大值时向前看的窗口大小(秒)，默认0.01')
    onset_group.add_argument('--onset-post-max', type=float, default=0.5,
                        help='计算局部最大值时向后看的窗口大小(秒)，默认0.01')
    onset_group.add_argument('--onset-combine', type=float, default=3,
                        help='合并相近onset的时间窗口(秒)，默认0.03')
    
    # === DBN 节拍检测参数 ===
    beat_group = parser.add_argument_group('DBN节拍检测参数')
    beat_group.add_argument('--beats-per-bar', type=int, nargs='+', default=[4],
                        help='每小节的拍数，可指定多个值如"3 4"同时检测3/4和4/4拍，默认[4]')
    beat_group.add_argument('--min-bpm', type=float, default=55.0,
                        help='最小BPM，默认55.0')
    beat_group.add_argument('--max-bpm', type=float, default=215.0,
                        help='最大BPM，默认215.0')
    beat_group.add_argument('--num-tempi', type=int, default=60,
                        help='建模的速度数量，默认60')
    beat_group.add_argument('--transition-lambda', type=float, default=100,
                        help='速度变化分布参数，值越大越倾向保持恒定速度，默认100')
    beat_group.add_argument('--observation-lambda', type=int, default=16,
                        help='将一个节拍周期分成的部分数，默认16')
    beat_group.add_argument('--dbn-threshold', type=float, default=0.2,
                        help='DBN激活值阈值，默认0.05')
    beat_group.add_argument('--no-correct-beats', action='store_true',
                        help='不对齐节拍到最近的激活峰值')
    beat_group.add_argument('--fps', type=int, default=100,
                        help='帧率(用于节拍检测)，默认100')
    
    # === 额外音频特征检测参数 ===
    feature_group = parser.add_argument_group('额外音频特征检测参数（人声/乐器变化）')
    feature_group.add_argument('--no-spectral-flux', action='store_true',
                        help='禁用频谱通量检测（人声/乐器变化）')
    feature_group.add_argument('--spectral-flux-threshold', type=float, default=0.3,
                        help='频谱通量变化阈值，值越低检测越敏感，默认0.3')
    feature_group.add_argument('--no-energy-change', action='store_true',
                        help='禁用能量突变检测')
    feature_group.add_argument('--energy-change-threshold', type=float, default=0.15,
                        help='能量变化阈值，值越低检测越敏感，默认0.15')
    feature_group.add_argument('--no-centroid-change', action='store_true',
                        help='禁用频谱质心变化检测（音色变化）')
    feature_group.add_argument('--centroid-change-threshold', type=float, default=0.2,
                        help='频谱质心变化阈值，值越低检测越敏感，默认0.2')
    
    # === 输出参数 ===
    output_group = parser.add_argument_group('输出参数')
    output_group.add_argument('--visualize', '-v', action='store_true',
                        help='生成可视化图片（包含波形、时频图、节拍和能量）')
    output_group.add_argument('--video', action='store_true',
                        help='生成颜色变化视频（纯色画面配音频，关键点处切换颜色）')
    output_group.add_argument('--output', '-o', type=str, default=None,
                        help='可视化图片或视频输出路径')
    output_group.add_argument('--video-fps', type=int, default=30,
                        help='生成视频的帧率，默认30')
    output_group.add_argument('--resolution', type=str, default='1920x1080',
                        help='生成视频的分辨率，默认1920x1080')
    output_group.add_argument('--downbeats-only', action='store_true',
                        help='只使用强拍作为分割点（减少分割数量）')
    output_group.add_argument('--show-beats', action='store_true', default=True,
                        help='在可视化中显示节拍点')
    output_group.add_argument('--show-onsets', action='store_true', default=True,
                        help='在可视化中显示冲击点')
    
    # === 显著性过滤参数 ===
    filter_group = parser.add_argument_group('显著性过滤参数')
    filter_group.add_argument('--min-interval', type=float, default=0.0,
                        help='关键点之间的最小间隔(秒)，间隔内只保留最强的点，默认0.0（不过滤）')
    filter_group.add_argument('--top-k', type=int, default=0,
                        help='只保留强度最高的前K个关键点，默认0（不限制）')
    filter_group.add_argument('--energy-percentile', type=float, default=0.0,
                        help='只保留能量高于该百分位数的点(0-100)，默认0（不过滤）')
    filter_group.add_argument('--merge-close', type=float, default=0.1,
                        help='合并间隔小于此值(秒)的相邻关键点，默认0.1')
    filter_group.add_argument('--segment-duration', type=float, default=0.0,
                        help='分段时长(秒)，与--segment-top-k配合使用，确保每段都有关键点，默认0（不分段）')
    filter_group.add_argument('--segment-top-k', type=int, default=0,
                        help='每个时间段内保留的最强点数量，默认0（不使用分段过滤）')
    
    # === 基于 Caption 段落过滤参数 ===
    caption_group = parser.add_argument_group('基于Caption段落过滤参数')
    caption_group.add_argument('--caption', type=str, default=None,
                        help='caption.json 文件路径，用于读取音乐段落(sections)划分')
    caption_group.add_argument('--section-top-k', type=int, default=0,
                        help='每个音乐段落内保留的最强点数量，默认0')
    caption_group.add_argument('--section-min-interval', type=float, default=0.0,
                        help='每个音乐段落内的最小间隔(秒)，默认0（不限制）')
    caption_group.add_argument('--section-energy-percentile', type=float, default=70.0,
                        help='每个音乐段落内的强度百分位数阈值(0-100)，只保留高于该阈值的点，默认0（不过滤）')

    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.audio_path):
        print(f"❌ 文件不存在: {args.audio_path}")
        return

    print(f"\n{'='*60}")
    print(f"🎵 Madmom 音频关键点检测")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        # 创建检测器并分析（传入所有参数）
        detector = SensoryKeypointDetector(
            # Onset 参数
            onset_threshold=args.onset_threshold,
            onset_smooth=args.onset_smooth,
            onset_pre_avg=args.onset_pre_avg,
            onset_post_avg=args.onset_post_avg,
            onset_pre_max=args.onset_pre_max,
            onset_post_max=args.onset_post_max,
            onset_combine=args.onset_combine,
            # DBN 参数
            beats_per_bar=args.beats_per_bar,
            min_bpm=args.min_bpm,
            max_bpm=args.max_bpm,
            num_tempi=args.num_tempi,
            transition_lambda=args.transition_lambda,
            observation_lambda=args.observation_lambda,
            dbn_threshold=args.dbn_threshold,
            correct_beats=not args.no_correct_beats,
            fps=args.fps,
            # 额外特征检测参数
            detect_spectral_flux=not args.no_spectral_flux,
            spectral_flux_threshold=args.spectral_flux_threshold,
            detect_energy_change=not args.no_energy_change,
            energy_change_threshold=args.energy_change_threshold,
            detect_spectral_centroid=not args.no_centroid_change,
            centroid_change_threshold=args.centroid_change_threshold,
        )
        result = detector.analyze(args.audio_path)

        elapsed_time = time.time() - start_time

        # 输出结果
        print(f"\n{'='*50}")
        print(f"检测完成! 耗时: {elapsed_time:.2f}秒")
        print(f"{'='*50}")
        
        print(f"\n📊 分析报告:")
        print(f"  整体基调: {result['meta']['key']} ({result['meta']['emotion_clue']})")
        print(f"  平均能量: {result['meta']['avg_energy']}")
        print(f"  检测到 {len(result['downbeats'])} 个强拍")
        print(f"  检测到 {len(result['onsets'])} 个冲击点")
        if result.get('spectral_flux_peaks'):
            print(f"  检测到 {len(result['spectral_flux_peaks'])} 个频谱变化点 (人声/乐器)")
        if result.get('energy_change_peaks'):
            print(f"  检测到 {len(result['energy_change_peaks'])} 个能量突变点")
        if result.get('centroid_change_peaks'):
            print(f"  检测到 {len(result['centroid_change_peaks'])} 个音色变化点")
        print(f"  原始关键点: {len(result['keypoints'])} 个")
        
        # 应用显著性过滤
        original_count = len(result['keypoints'])
        need_filter = (args.min_interval > 0 or args.top_k > 0 or 
                       args.energy_percentile > 0 or args.merge_close > 0 or
                       (args.segment_duration > 0 and args.segment_top_k > 0))
        
        if need_filter:
            print(f"\n🔍 应用显著性过滤...")
            filtered_keypoints = filter_significant_keypoints(
                result['keypoints'],
                min_interval=args.min_interval,
                top_k=args.top_k,
                energy_percentile=args.energy_percentile,
                merge_close=args.merge_close,
                segment_duration=args.segment_duration,
                segment_top_k=args.segment_top_k
            )
            result['keypoints_original'] = result['keypoints']
            result['keypoints'] = filtered_keypoints
            print(f"  过滤后关键点: {len(filtered_keypoints)} 个 (减少了 {original_count - len(filtered_keypoints)} 个)")
        
        # 基于 Caption 段落过滤
        if args.caption:
            if os.path.exists(args.caption):
                print(f"\n📂 加载 Caption 段落信息: {args.caption}")
                sections = load_sections_from_caption(args.caption)
                
                if sections:
                    print(f"  共解析到 {len(sections)} 个段落:")
                    for sec in sections:
                        print(f"    - {sec['name']}: {sec['start_time']:.1f}s - {sec['end_time']:.1f}s")
                    
                    # 先合并相近点（如果还没合并过）
                    keypoints_to_filter = result['keypoints']
                    if args.merge_close > 0 and 'keypoints_original' not in result:
                        # 已经在前面合并过了
                        pass
                    elif args.merge_close <= 0:
                        # 没设置合并，这里做一个默认合并
                        print(f"\n🔍 合并相近点 (默认 merge_close=0.15s)...")
                        keypoints_to_filter = filter_significant_keypoints(
                            keypoints_to_filter,
                            merge_close=0.15
                        )
                    
                    filtered_keypoints = filter_by_sections(
                        keypoints_to_filter,
                        sections,
                        section_top_k=args.section_top_k,
                        section_min_interval=args.section_min_interval,
                        section_energy_percentile=args.section_energy_percentile
                    )
                    
                    if 'keypoints_original' not in result:
                        result['keypoints_original'] = result['keypoints']
                    result['keypoints'] = filtered_keypoints
                    result['sections'] = sections
                else:
                    print(f"  ⚠️ 未能从 caption 文件解析到有效段落")
            else:
                print(f"  ⚠️ Caption 文件不存在: {args.caption}")
        
        print(f"\n前 15 个关键点:")
        print(f"{'时间(秒)':>10} | {'类型':<25} | {'强度':>6}")
        print("-" * 50)
        for pt in result['keypoints'][:15]:
            print(f"{pt['time']:10.3f} | {pt['type']:<25} | {pt['intensity']:6.2f}")
        
        if len(result['keypoints']) > 15:
            print(f"  ... (共 {len(result['keypoints'])} 个关键点)")

        # 可视化
        if args.visualize:
            print(f"\n正在生成可视化图片...")
            
            if args.output and not args.video:
                output_path = args.output
            else:
                audio_file = Path(args.audio_path)
                output_path = audio_file.parent / f"{audio_file.stem}_madmom_thr{args.onset_threshold}_kp{len(result['keypoints'])}.png"
            
            visualize_keypoints(
                audio_path=args.audio_path,
                result=result,
                output_path=str(output_path),
                show_beats=args.show_beats,
                show_onsets=args.show_onsets
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

            # 选择关键点
            if args.downbeats_only:
                keypoints = [{'time': t} for t in result['downbeats']]
                suffix = "_downbeats"
            else:
                keypoints = result['keypoints']
                suffix = ""

            # 确定输出路径
            if args.output:
                video_output_path = args.output
            else:
                audio_file = Path(args.audio_path)
                video_output_path = audio_file.parent / f"{audio_file.stem}_madmom_thr{args.onset_threshold}_kp{len(keypoints)}{suffix}.mp4"

            # 构建参数信息字典
            params_info = {
                'onset_threshold': args.onset_threshold,
                'beats_per_bar': args.beats_per_bar,
                'bpm_range': f'{args.min_bpm}-{args.max_bpm}',
                'transition_lambda': args.transition_lambda,
                'key': result['meta']['key'],
                'emotion': result['meta']['emotion_clue'],
                'n_keypoints': len(keypoints),
            }

            generate_color_video(
                audio_path=args.audio_path,
                keypoints=keypoints,
                output_path=str(video_output_path),
                fps=args.video_fps,
                resolution=resolution,
                params_info=params_info
            )

    except Exception as e:
        import traceback
        print(f"\n❌ 发生错误: {e}")
        traceback.print_exc()
        print("\n请确保已安装 madmom 和 ffmpeg")
        return

    return result


# --- 使用示例 ---
if __name__ == "__main__":
    # 如果没有命令行参数，使用默认示例
    if len(sys.argv) == 1:
        print("用法: python audio_Madmom.py <音频文件路径> [选项]")
        print("使用 --help 查看详细帮助")
        sys.exit(0)
    
    main()