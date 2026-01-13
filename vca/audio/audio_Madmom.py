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
import json
import subprocess
import tempfile
from typing import List, Tuple
from pathlib import Path

from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
from madmom.audio.signal import Signal
import madmom.features.downbeats as _downbeats_module
import itertools as _it

# ============ Lightweight caches for interactive usage ============
# Madmom's RNN/CNN processors are expensive; in the UI users often tweak
# thresholds and re-run. Cache intermediate activations per audio file.
_VCA_ACT_CACHE_MAX = 8
_vca_cache_beat_act = {}


def _vca_cache_key(audio_path: str) -> tuple:
    try:
        return (audio_path, os.path.getmtime(audio_path))
    except Exception:
        return (audio_path, None)


def _vca_cache_put(cache: dict, key: tuple, value):
    cache[key] = value
    if len(cache) > _VCA_ACT_CACHE_MAX:
        # Keep it simple: clear to bound memory.
        cache.clear()

# ============ End caches ============

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

# 添加vca模块到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from vca.audio.audio_utils import load_audio_no_librosa

# ============ NMS（非极大值抑制）函数 ============

def nms_1d(values, timestamps, min_distance, threshold=None, max_points=None,
           sort_by_values=None):
    """
    一维非极大值抑制

    Args:
        values: 检测值（用于阈值过滤，如confidence）
        timestamps: 对应的时间戳（秒）
        min_distance: 最小间隔（秒），相邻点之间至少间隔这么多秒
        threshold: 可选的阈值，低于此值的点被忽略
        max_points: 可选的最大点数限制
        sort_by_values: 用于排序的值（如pitch），如果为None则使用values排序

    Returns:
        selected_timestamps: 筛选后的时间戳
        selected_values: 筛选后的值
        selected_indices: 筛选后的原始索引
    """
    values = np.array(values, dtype=np.float64)
    timestamps = np.array(timestamps, dtype=np.float64)

    if sort_by_values is not None:
        sort_by_values = np.array(sort_by_values, dtype=np.float64)
    else:
        sort_by_values = values.copy()

    if len(values) == 0:
        return np.array([]), np.array([]), np.array([], dtype=np.int64)

    original_indices = np.arange(len(values))

    # 阈值过滤（基于values，如confidence）
    if threshold is not None:
        mask = values >= threshold
        values = values[mask]
        timestamps = timestamps[mask]
        sort_by_values = sort_by_values[mask]
        original_indices = original_indices[mask]

    if len(values) == 0:
        return np.array([]), np.array([]), np.array([], dtype=np.int64)

    # 按sort_by_values从大到小排序（优先保留值最大的点）
    sorted_order = np.argsort(sort_by_values)[::-1]
    selected_mask = np.zeros(len(values), dtype=bool)

    # 贪心选择
    for idx in sorted_order:
        t = timestamps[idx]
        already_selected_times = timestamps[selected_mask]

        if len(already_selected_times) == 0 or np.all(np.abs(already_selected_times - t) >= min_distance):
            selected_mask[idx] = True
            if max_points is not None and np.sum(selected_mask) >= max_points:
                break

    # 按时间排序返回
    selected_timestamps = timestamps[selected_mask]
    selected_values = values[selected_mask]
    selected_indices = original_indices[selected_mask]

    time_order = np.argsort(selected_timestamps)
    return (
        selected_timestamps[time_order],
        selected_values[time_order],
        selected_indices[time_order]
    )


def nms_adaptive(values, timestamps, min_distance, adaptive_ratio=0.5):
    """自适应阈值的NMS，阈值 = 最大值 * adaptive_ratio"""
    values = np.array(values, dtype=np.float64)
    if len(values) == 0:
        return np.array([]), np.array([]), np.array([], dtype=np.int64)
    threshold = np.max(values) * adaptive_ratio
    return nms_1d(values, timestamps, min_distance, threshold=threshold)


def nms_window(values, timestamps, window_size, top_k=1):
    """窗口NMS：每个时间窗口内保留top-k个点"""
    values = np.array(values, dtype=np.float64)
    timestamps = np.array(timestamps, dtype=np.float64)
    original_indices = np.arange(len(values))

    if len(values) == 0:
        return np.array([]), np.array([]), np.array([], dtype=np.int64)

    t_min, t_max = timestamps.min(), timestamps.max()
    selected_mask = np.zeros(len(values), dtype=bool)

    window_start = t_min
    while window_start <= t_max:
        window_end = window_start + window_size
        window_mask = (timestamps >= window_start) & (timestamps < window_end)
        window_indices = np.where(window_mask)[0]

        if len(window_indices) > 0:
            window_values = values[window_indices]
            top_k_indices = window_indices[np.argsort(window_values)[::-1][:top_k]]
            selected_mask[top_k_indices] = True

        window_start = window_end

    selected_timestamps = timestamps[selected_mask]
    selected_values = values[selected_mask]
    selected_indices = original_indices[selected_mask]

    time_order = np.argsort(selected_timestamps)
    return (
        selected_timestamps[time_order],
        selected_values[time_order],
        selected_indices[time_order]
    )

# ============ Pitch检测函数 ============

def detect_pitch(audio_path, samplerate=0, tolerance=0.8):
    """
    检测音频的Pitch

    Returns:
        pitches: pitch数组
        confidences: 置信度数组
        timestamps: 时间戳数组
        actual_samplerate: 实际采样率
    """
    from aubio import source, pitch
    from pathlib import Path

    # Convert MP3 to WAV if needed (aubio's source_wavread requires WAV format)
    p = Path(audio_path)
    if p.suffix.lower() not in {".wav", ".wave"}:
        # Create a temporary WAV file
        wav_path = p.with_name(f"{p.stem}__vca_pitch.wav")

        # Check if WAV already exists and is newer than source
        if not wav_path.exists() or wav_path.stat().st_mtime < p.stat().st_mtime:
            # Convert using ffmpeg
            import shutil
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg:
                cmd = [
                    ffmpeg, "-y", "-i", str(p),
                    "-vn", "-ac", "1", "-acodec", "pcm_s16le",
                    str(wav_path)
                ]
                result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if result.returncode != 0:
                    raise RuntimeError(f"ffmpeg failed to convert {audio_path} to WAV")
            else:
                raise RuntimeError("ffmpeg not found. Cannot convert MP3 to WAV for pitch detection.")

        audio_path = str(wav_path)

    downsample = 1
    win_s = 4096 // downsample
    hop_s = 512 // downsample

    s = source(audio_path, samplerate, hop_s)
    actual_samplerate = s.samplerate

    pitch_o = pitch("yin", win_s, hop_s, actual_samplerate)
    pitch_o.set_unit("midi")
    pitch_o.set_tolerance(tolerance)

    pitches = []
    confidences = []
    timestamps = []
    total_frames = 0

    while True:
        samples, read = s()
        p = pitch_o(samples)[0]
        c = pitch_o.get_confidence()

        timestamps.append(total_frames / float(actual_samplerate))
        pitches.append(p)
        confidences.append(c)

        total_frames += read
        if read < hop_s:
            break

    return np.array(pitches), np.array(confidences), np.array(timestamps), actual_samplerate


# ============ Mel能量检测函数 ============

def compute_mel_energies(audio_path, samplerate=0, win_s=512, n_filters=40):
    """
    计算mel能量

    Returns:
        timestamps: 时间戳数组
        energies: (n_frames, n_filters) 的能量矩阵
        total_energies: 每帧的总能量
        samplerate: 实际采样率
    """
    from aubio import source, pvoc, filterbank
    from pathlib import Path

    # Convert MP3 to WAV if needed (aubio's source_wavread requires WAV format)
    p = Path(audio_path)
    if p.suffix.lower() not in {".wav", ".wave"}:
        # Create a temporary WAV file
        wav_path = p.with_name(f"{p.stem}__vca_mel.wav")

        # Check if WAV already exists and is newer than source
        if not wav_path.exists() or wav_path.stat().st_mtime < p.stat().st_mtime:
            # Convert using ffmpeg
            import shutil
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg:
                cmd = [
                    ffmpeg, "-y", "-i", str(p),
                    "-vn", "-ac", "1", "-acodec", "pcm_s16le",
                    str(wav_path)
                ]
                result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if result.returncode != 0:
                    raise RuntimeError(f"ffmpeg failed to convert {audio_path} to WAV")
            else:
                raise RuntimeError("ffmpeg not found. Cannot convert MP3 to WAV for mel energy computation.")

        audio_path = str(wav_path)

    hop_s = win_s // 4

    s = source(audio_path, samplerate, hop_s)
    actual_samplerate = s.samplerate

    pv = pvoc(win_s, hop_s)
    f = filterbank(n_filters, win_s)
    f.set_mel_coeffs_slaney(actual_samplerate)

    energies_list = []
    timestamps = []
    total_frames = 0

    while True:
        samples, read = s()
        fftgrain = pv(samples)
        new_energies = f(fftgrain)

        timestamps.append(total_frames / float(actual_samplerate))
        energies_list.append(new_energies.copy())

        total_frames += read
        if read < hop_s:
            break

    energies = np.vstack(energies_list)
    timestamps = np.array(timestamps)
    total_energies = np.sum(energies, axis=1)

    return timestamps, energies, total_energies, actual_samplerate





class SensoryKeypointDetector:
    def __init__(
        self,
        # 检测方法选择
        detection_method: str = "downbeat",  # "downbeat", "pitch", "mel_energy"
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
        # Pitch检测参数
        pitch_tolerance: float = 0.8,
        pitch_threshold: float = 0.8,
        pitch_min_distance: float = 0.5,
        pitch_nms_method: str = "basic",  # "basic", "adaptive", "window"
        pitch_max_points: int = None,
        # Mel能量检测参数
        mel_win_s: int = 512,
        mel_n_filters: int = 40,
        mel_threshold_ratio: float = 0.3,
        mel_min_distance: float = 0.5,
        mel_nms_method: str = "basic",
        mel_max_points: int = None,
    ):
        """
        音频感官关键点检测器 - 支持多种检测方法
        
        Args:
            detection_method: 检测方法 ("downbeat", "pitch", "mel_energy")
            
            DBN 节拍检测参数:
                beats_per_bar: 每小节的拍数，如[4]表示4/4拍 (默认[4])
                min_bpm: 最小BPM (默认55.0)
                max_bpm: 最大BPM (默认215.0)
                ...
            
            Pitch检测参数:
                pitch_tolerance: pitch检测容差 (默认0.8)
                pitch_threshold: pitch置信度阈值 (默认0.8)
                pitch_min_distance: pitch检测的最小间隔(秒) (默认0.5)
                pitch_nms_method: NMS方法 (默认"basic")
                pitch_max_points: 最大保留点数
            
            Mel能量检测参数:
                mel_win_s: FFT窗口大小 (默认512)
                mel_n_filters: Mel滤波器数量 (默认40)
                mel_threshold_ratio: 能量阈值比例 (默认0.3)
                mel_min_distance: 最小间隔(秒) (默认0.5)
                mel_nms_method: NMS方法 (默认"basic")
                mel_max_points: 最大保留点数
        """
        self.detection_method = detection_method
        
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
        
        # Pitch参数
        self.pitch_tolerance = pitch_tolerance
        self.pitch_threshold = pitch_threshold
        self.pitch_min_distance = pitch_min_distance
        self.pitch_nms_method = pitch_nms_method
        self.pitch_max_points = pitch_max_points
        
        # Mel参数
        self.mel_win_s = mel_win_s
        self.mel_n_filters = mel_n_filters
        self.mel_threshold_ratio = mel_threshold_ratio
        self.mel_min_distance = mel_min_distance
        self.mel_nms_method = mel_nms_method
        self.mel_max_points = mel_max_points

    def analyze_pitch(self, audio_path):
        """Pitch检测 + NMS，逻辑与demo_pitch_nms.py完全一致"""
        print(f"正在进行Pitch检测: {audio_path} ...")

        pitches, confidences, timestamps, samplerate = detect_pitch(
            audio_path, tolerance=self.pitch_tolerance
        )

        print(f"检测完成: {len(pitches)} 个点, 时长 {timestamps[-1]:.2f}s")

        # ============== NMS过滤（与demo_pitch_nms.py完全一致）==============
        # 使用绝对阈值
        threshold = self.pitch_threshold
        min_distance = self.pitch_min_distance
        max_points = self.pitch_max_points
        method = self.pitch_nms_method

        print(f"\n应用NMS过滤 (method={method}, min_distance={min_distance}s, threshold={threshold})")
        if max_points is not None:
            print(f"  限制点数: {max_points}")

        if method == "adaptive":
            sel_t, sel_c, sel_i = nms_adaptive(confidences, timestamps, min_distance,
                                               adaptive_ratio=threshold)
        elif method == "window":
            sel_t, sel_c, sel_i = nms_window(confidences, timestamps, 1.0, top_k=1)
        else:  # basic
            sel_t, sel_c, sel_i = nms_1d(
                confidences, timestamps, min_distance,
                threshold=threshold, max_points=max_points,
                sort_by_values=pitches  # 按pitch排序
            )

        sel_p = pitches[sel_i]

        print(f"过滤后: {len(sel_t)} 个显著点")
        
        # 转换为关键点格式
        timeline = []
        for t, p, c in zip(sel_t, sel_p, sel_c):
            timeline.append({
                'time': float(t),
                'type': 'Pitch',
                'pitch': float(p),
                'confidence': float(c),
                'intensity': float(c)
            })
        
        return {
            "meta": {"n_pitch_points": len(timeline)},
            "keypoints": timeline,
            "pitches": sel_p,
            "confidences": sel_c,
            "timestamps": sel_t,
            "sample_rate": samplerate
        }

    def analyze_mel_energy(self, audio_path):
        """Mel能量检测 + NMS"""
        print(f"正在进行Mel能量检测: {audio_path} ...")
        
        timestamps, energies, total_energies, samplerate = compute_mel_energies(
            audio_path, win_s=self.mel_win_s, n_filters=self.mel_n_filters
        )
        
        print(f"  计算完成: {len(timestamps)} 帧, 时长 {timestamps[-1]:.2f}s")
        print(f"  采样率: {samplerate} Hz, Mel滤波器: {self.mel_n_filters} 个")
        
        # 使用全部频带
        selected_energies = total_energies
        
        # 计算阈值
        max_energy = np.max(selected_energies)
        threshold = max_energy * self.mel_threshold_ratio
        
        print(f"  应用NMS过滤 (method={self.mel_nms_method}, min_distance={self.mel_min_distance}s)")
        print(f"  阈值: {threshold:.4f} (最大值的 {self.mel_threshold_ratio*100:.0f}%)")
        
        if self.mel_nms_method == "adaptive":
            sel_t, sel_e, sel_i = nms_adaptive(selected_energies, timestamps, self.mel_min_distance, 
                                               adaptive_ratio=self.mel_threshold_ratio)
        elif self.mel_nms_method == "window":
            sel_t, sel_e, sel_i = nms_window(selected_energies, timestamps, 1.0, top_k=1)
        else:  # basic
            sel_t, sel_e, sel_i = nms_1d(
                selected_energies, timestamps, self.mel_min_distance,
                threshold=threshold, max_points=self.mel_max_points
            )
        
        print(f"  过滤后: {len(sel_t)} 个显著点")
        
        # 转换为关键点格式
        timeline = []
        for t, e in zip(sel_t, sel_e):
            relative = e / max_energy * 100
            timeline.append({
                'time': float(t),
                'type': 'MelEnergy',
                'energy': float(e),
                'relative_intensity': float(relative),
                'intensity': float(e / max_energy)
            })
        
        return {
            "meta": {"n_mel_points": len(timeline)},
            "keypoints": timeline,
            "energies": sel_e,
            "timestamps": sel_t,
            "sample_rate": samplerate
        }

    def analyze(self, audio_path):
        """根据选择的方法进行分析"""
        print(f"正在分析音频: {audio_path} ...")
        print(f"使用检测方法: {self.detection_method}")
        
        if self.detection_method == "pitch":
            return self.analyze_pitch(audio_path)
        elif self.detection_method == "mel_energy":
            return self.analyze_mel_energy(audio_path)
        else:  # downbeat (默认)
            return self.analyze_downbeat(audio_path)
    
    def analyze_downbeat(self, audio_path):

        cache_key = _vca_cache_key(audio_path)

        # 1. 节奏分析 (Rhythm) - 获取强拍
        downbeats = np.array([])
        print(" -> 检测节奏 (Beats/Downbeats)...")
        print(f"    参数: beats_per_bar={self.beats_per_bar}, BPM范围=[{self.min_bpm}, {self.max_bpm}], "
              f"transition_lambda={self.transition_lambda}")
        beat_act = _vca_cache_beat_act.get(cache_key)
        if beat_act is None:
            beat_proc = RNNDownBeatProcessor()
            beat_act = beat_proc(audio_path)
            _vca_cache_put(_vca_cache_beat_act, cache_key, beat_act)
        
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
            print("    ⚠️ 未能检测到节拍")

        # 2. 从 beat_act 中提取 downbeat 激活值作为 intensity
        # beat_act 是 (n_frames, n_classes) 数组，其中第二列是 downbeat 激活值
        # fps 是帧率，用于将时间转换为帧索引

        def get_downbeat_activation(t, beat_act, fps):
            """获取时间t处的downbeat激活值"""
            if beat_act is None or len(beat_act) == 0:
                return 0.5

            # 将时间转换为帧索引
            frame_idx = int(t * fps)
            frame_idx = max(0, min(frame_idx, len(beat_act) - 1))

            # beat_act 的第二列是 downbeat 激活值
            if beat_act.ndim == 2 and beat_act.shape[1] >= 2:
                activation = float(beat_act[frame_idx, 1])
            elif beat_act.ndim == 1:
                activation = float(beat_act[frame_idx])
            else:
                activation = 0.5

            # 确保激活值在合理范围内
            if not np.isfinite(activation):
                activation = 0.5

            return activation

        # 计算音频信号 (用于返回)
        sig = Signal(audio_path)

        # 结果整合 - 只保留downbeat
        timeline = []

        # 只保留 Downbeat (重拍)，使用 DBN 激活值作为 intensity
        for t in downbeats:
            activation = get_downbeat_activation(t, beat_act, self.fps)
            timeline.append({
                'time': float(t),
                'type': 'Downbeat',
                'activation': float(activation),
                'intensity': float(activation)
            })

        # 按时间排序
        timeline.sort(key=lambda x: x['time'])

        return {
            "meta": {
                "n_downbeats": len(downbeats)
            },
            "keypoints": timeline,
            "downbeats": downbeats,
            "beat_info": beat_info,
            "beat_activation": beat_act,
            "audio_signal": sig,
            "sample_rate": sig.sample_rate
        }


def normalize_intensity_by_type(keypoints: List[dict]) -> List[dict]:
    """
    按类型归一化关键点强度，使不同类型的关键点可以公平比较

    每种类型使用其主要特征值进行归一化：
    - Downbeat: 使用 activation (DBN激活值)

    归一化后所有类型的强度都在 0 ~ 1 范围内

    Args:
        keypoints: 原始关键点列表

    Returns:
        添加了 normalized_intensity 字段的关键点列表
    """
    if not keypoints:
        return []

    # 按类型分组
    by_type = {}
    for kp in keypoints:
        kp_type = kp.get('type', 'Unknown')
        by_type.setdefault(kp_type, []).append(kp)

    # 类型到主特征字段的映射
    type_to_feature = {
        'Downbeat': 'activation',
    }

    print(f"    按类型归一化强度:")

    # 对每种类型分别归一化
    for type_name, points in by_type.items():
        # 获取该类型的主特征字段
        feature_key = type_to_feature.get(type_name, 'activation')

        # 获取特征值，如果不存在则尝试使用 intensity（兼容旧数据）
        intensities = []
        for p in points:
            val = p.get(feature_key, p.get('intensity', 0.5))
            intensities.append(val)

        min_i = min(intensities)
        max_i = max(intensities)
        range_i = max_i - min_i

        for i, p in enumerate(points):
            if range_i > 1e-6:
                p['normalized_intensity'] = (intensities[i] - min_i) / range_i
            else:
                # 如果该类型所有点强度相同，归一化为 0.5
                p['normalized_intensity'] = 0.5
            # 同时设置 intensity 字段以保持兼容性
            p['intensity'] = intensities[i]

        print(f"      - {type_name}: {len(points)} 个点, "
              f"{feature_key} [{min_i:.3f}, {max_i:.3f}] -> 归一化 [0, 1]")

    return keypoints


def filter_significant_keypoints(
    keypoints: List[dict],
    min_interval: float = 0.0,
    top_k: int = 0,
    energy_percentile: float = 0.0,
    use_normalized_intensity: bool = True
) -> List[dict]:
    """
    过滤关键点，只保留显著的点

    Args:
        keypoints: 原始关键点列表
        min_interval: 最小间隔（秒），间隔内只保留最强的点
        top_k: 只保留强度最高的前K个点，0表示不限制
        energy_percentile: 只保留强度高于该百分位数的点(0-100)
        use_normalized_intensity: 是否使用归一化后的强度进行过滤（推荐True）

    Returns:
        过滤后的关键点列表
    """
    if not keypoints:
        return []

    filtered = list(keypoints)
    print(f"\n=== 关键点过滤流程 ===")
    print(f"0. 初始关键点: {len(filtered)} 个")

    # 0. 先按类型归一化强度
    if use_normalized_intensity:
        filtered = normalize_intensity_by_type(filtered)
        intensity_key = 'normalized_intensity'
    else:
        intensity_key = 'intensity'
        # 确保所有点都有 normalized_intensity 字段（设为原始值）
        for kp in filtered:
            kp['normalized_intensity'] = kp['intensity']

    # 1. 按强度百分位数过滤（使用归一化强度）
    if energy_percentile > 0 and filtered:
        before_percentile = len(filtered)
        intensities = [kp[intensity_key] for kp in filtered]
        threshold = np.percentile(intensities, energy_percentile)
        filtered = [kp for kp in filtered if kp[intensity_key] >= threshold]
        print(f"3. 强度百分位过滤后: {len(filtered)} 个 "
              f"(减少 {before_percentile - len(filtered)} 个, percentile={energy_percentile}, threshold={threshold:.3f})")

    # 2. 按最小间隔过滤（在每个间隔内只保留最强的点）
    if min_interval > 0 and filtered:
        before_interval = len(filtered)
        filtered.sort(key=lambda x: x['time'])
        interval_filtered = []
        current_interval_start = filtered[0]['time']
        current_best = filtered[0]

        for kp in filtered[1:]:
            if kp['time'] - current_interval_start < min_interval:
                # 在同一间隔内，保留强度更高的（使用归一化强度比较）
                if kp[intensity_key] > current_best[intensity_key]:
                    current_best = kp
            else:
                # 新间隔，保存之前的最佳点
                interval_filtered.append(current_best)
                current_interval_start = kp['time']
                current_best = kp

        # 添加最后一个
        interval_filtered.append(current_best)
        filtered = interval_filtered
        print(f"3. 最小间隔过滤后: {len(filtered)} 个 (减少 {before_interval - len(filtered)} 个, min_interval={min_interval}s)")

    # 4. 只保留 top_k 个（每种类型分别保留 top_k 个）
    if top_k > 0 and filtered:
        # 按类型分组
        by_type = {}
        for kp in filtered:
            kp_type = kp.get('type', 'Unknown')
            if kp_type not in by_type:
                by_type[kp_type] = []
            by_type[kp_type].append(kp)
        
        before_topk = len(filtered)
        
        # 每种类型分别保留 top_k 个最强的点
        filtered_by_type = []
        type_summary_list = []
        for type_name, points in by_type.items():
            # 按强度降序排序
            points.sort(key=lambda x: x[intensity_key], reverse=True)
            # 每种类型保留 top_k 个（如果该类型点数少于 top_k，则全部保留）
            kept = min(top_k, len(points))
            filtered_by_type.extend(points[:kept])
            type_summary_list.append(f"{type_name}:{kept}")
        
        filtered = filtered_by_type
        filtered.sort(key=lambda x: x['time'])
        
        # 打印每种类型保留的数量
        type_summary = ", ".join(sorted(type_summary_list))
        print(f"5. Top-K 过滤后: {len(filtered)} 个 (减少 {before_topk - len(filtered)} 个, "
              f"每类保留top_k={top_k}, 按类型: {type_summary})")

    print(f"=== 过滤完成，最终保留 {len(filtered)} 个关键点 ===\n")
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


def filter_by_type(
    keypoints: List[dict],
    preferred_types: List[str] = None,
    mode: str = "boost",
    boost_factor: float = 1.5
) -> List[dict]:
    """
    按关键点类型进行过滤或增强

    关键点类型包括:
    - "Downbeat" (重拍): 节奏上的强拍

    Args:
        keypoints: 原始关键点列表
        preferred_types: 优先类型列表，支持部分匹配
                        例如 ["Downbeat"] 会匹配包含这些词的类型
        mode: 过滤模式
              - "only": 只保留指定类型的关键点
              - "boost": 增强指定类型的权重（乘以 boost_factor）
              - "exclude": 排除指定类型
        boost_factor: 当 mode="boost" 时，增强因子（默认 1.5）

    Returns:
        过滤或增强后的关键点列表
    """
    if not keypoints:
        return []

    if not preferred_types:
        return keypoints

    # 将 preferred_types 转为小写以便匹配
    preferred_lower = [t.lower() for t in preferred_types]

    def type_matches(kp_type: str, preferred_list: List[str]) -> bool:
        """检查关键点类型是否匹配优先类型列表"""
        kp_type_lower = kp_type.lower()
        for preferred in preferred_list:
            if preferred in kp_type_lower:
                return True
        return False

    filtered = []

    print(f"\n    🏷️  按类型过滤关键点 (mode={mode}):")
    print(f"       优先类型: {preferred_types}")

    # 统计各类型数量
    type_counts_before = {}
    for kp in keypoints:
        kp_type = kp.get('type', 'Unknown')
        type_counts_before[kp_type] = type_counts_before.get(kp_type, 0) + 1

    if mode == "only":
        # 只保留指定类型
        for kp in keypoints:
            kp_type = kp.get('type', 'Unknown')
            if type_matches(kp_type, preferred_lower):
                filtered.append(kp)

    elif mode == "exclude":
        # 排除指定类型
        for kp in keypoints:
            kp_type = kp.get('type', 'Unknown')
            if not type_matches(kp_type, preferred_lower):
                filtered.append(kp)

    elif mode == "boost":
        # 增强指定类型的权重
        for kp in keypoints:
            kp_copy = dict(kp)
            kp_type = kp_copy.get('type', 'Unknown')
            if type_matches(kp_type, preferred_lower):
                # 增强强度
                if 'normalized_intensity' in kp_copy:
                    kp_copy['normalized_intensity'] = min(1.0, kp_copy['normalized_intensity'] * boost_factor)
                if 'intensity' in kp_copy:
                    kp_copy['intensity'] = kp_copy['intensity'] * boost_factor
                kp_copy['type_boosted'] = True
            filtered.append(kp_copy)

    else:
        print(f"       ⚠️ 未知模式 '{mode}'，返回原始关键点")
        return keypoints

    # 统计过滤后各类型数量
    type_counts_after = {}
    for kp in filtered:
        kp_type = kp.get('type', 'Unknown')
        type_counts_after[kp_type] = type_counts_after.get(kp_type, 0) + 1

    # 打印统计信息
    for kp_type, count_before in type_counts_before.items():
        count_after = type_counts_after.get(kp_type, 0)
        if count_before != count_after:
            print(f"       - {kp_type}: {count_before} -> {count_after}")
        elif mode == "boost" and type_matches(kp_type, preferred_lower):
            print(f"       - {kp_type}: {count_before} (权重增强 x{boost_factor})")

    print(f"       过滤前: {len(keypoints)} 个, 过滤后: {len(filtered)} 个")

    return filtered


def compute_composite_score(
    keypoints: List[dict],
    weight_downbeat: float = 1.0,
    weight_pitch: float = 1.0,
    weight_mel_energy: float = 1.0,
) -> List[dict]:
    """
    为每个关键点计算综合评分（加权组合多个指标）

    composite_score = k1 * downbeat_intensity + k2 * pitch_intensity + k3 * mel_energy_intensity

    注意：每个指标的强度都已在各自类型内归一化到 [0, 1] 范围

    Args:
        keypoints: 关键点列表（必须已经过 normalize_intensity_by_type 处理）
        weight_downbeat: Downbeat 类型的权重 (k1)
        weight_pitch: Pitch 类型的权重 (k2)
        weight_mel_energy: Mel Energy 类型的权重 (k3)

    Returns:
        添加了 composite_score 字段的关键点列表
    """
    if not keypoints:
        return []

    print(f"\n    🎯 计算综合评分 (权重: Downbeat={weight_downbeat:.2f}, Pitch={weight_pitch:.2f}, MelEnergy={weight_mel_energy:.2f})")

    # 为每个点计算综合分数
    for kp in keypoints:
        kp_type = kp.get('type', 'Unknown')
        normalized_intensity = kp.get('normalized_intensity', 0.5)

        # 根据类型应用对应的权重
        if kp_type == 'Downbeat':
            composite_score = weight_downbeat * normalized_intensity
        elif kp_type == 'Pitch':
            composite_score = weight_pitch * normalized_intensity
        elif kp_type == 'MelEnergy':
            composite_score = weight_mel_energy * normalized_intensity
        else:
            # 未知类型，使用平均权重
            avg_weight = (weight_downbeat + weight_pitch + weight_mel_energy) / 3.0
            composite_score = avg_weight * normalized_intensity

        kp['composite_score'] = composite_score

    # 统计每种类型的数量和平均分数
    type_stats = {}
    for kp in keypoints:
        kp_type = kp.get('type', 'Unknown')
        if kp_type not in type_stats:
            type_stats[kp_type] = {'count': 0, 'total_score': 0.0}
        type_stats[kp_type]['count'] += 1
        type_stats[kp_type]['total_score'] += kp['composite_score']

    print(f"    各类型综合评分统计:")
    for kp_type, stats in sorted(type_stats.items()):
        avg_score = stats['total_score'] / stats['count']
        print(f"      - {kp_type}: {stats['count']} 个点, 平均分数={avg_score:.3f}")

    return keypoints


def filter_by_sections(
    keypoints: List[dict],
    sections: List[dict],
    section_min_interval: float = 0.0,
    use_normalized_intensity: bool = True,
    min_segment_duration: float = 3.0,
    max_segment_duration: float = 15.0,
    total_shots: int = 20,
    audio_duration: float = None,
    weight_downbeat: float = 1.0,
    weight_pitch: float = 1.0,
    weight_mel_energy: float = 1.0,
) -> List[dict]:
    """
    基于音乐段落（sections）进行关键点过滤（按比例分配模式）
    根据每个段落的关键点密度，按比例分配总镜头数

    Args:
        keypoints: 原始关键点列表
        sections: 段落列表，每个包含 name, start_time, end_time
        section_min_interval: 全局最小间隔（跨段落应用）
        use_normalized_intensity: 是否使用归一化后的强度进行过滤（推荐True）
        min_segment_duration: 最小片段时长（用于边界检查和合并过短片段）
        max_segment_duration: 用于分割过长片段的最大片段时长（默认15s）
        total_shots: 总镜头数，按比例分配给各section（基于关键点密度）
        audio_duration: 音频总时长（用于边界检查）
        weight_downbeat: Downbeat 类型的权重
        weight_pitch: Pitch 类型的权重
        weight_mel_energy: Mel Energy 类型的权重

    Returns:
        过滤后的关键点列表
    """
    if not keypoints or not sections:
        return keypoints

    # 先按类型归一化强度
    if use_normalized_intensity:
        keypoints = normalize_intensity_by_type(list(keypoints))
        intensity_key = 'normalized_intensity'
    else:
        intensity_key = 'intensity'
        # 确保所有点都有 normalized_intensity 字段
        for kp in keypoints:
            if 'normalized_intensity' not in kp:
                kp['normalized_intensity'] = kp['intensity']

    # 计算综合评分（加权组合多个指标）
    keypoints = compute_composite_score(
        keypoints,
        weight_downbeat=weight_downbeat,
        weight_pitch=weight_pitch,
        weight_mel_energy=weight_mel_energy,
    )
    # 使用综合评分作为排序和比较的依据
    score_key = 'composite_score'

    filtered = []

    print(f"\n    📂 基于 {len(sections)} 个音乐段落进行过滤:")
    print(f"    🎯 按比例分配总镜头数: {total_shots}")

    # 计算每个section应该保留的关键点数量（按比例分配）
    section_top_k_map = {}  # 存储每个section应该保留的关键点数量

    # Step 1: 统计每个section内的keypoint数量
    section_keypoint_counts = {}
    total_keypoints = 0

    for sec in sections:
        start_val = sec.get('start_time', sec.get('Start_Time', 0))
        end_val = sec.get('end_time', sec.get('End_Time', 0))

        try:
            start = parse_time_str(start_val)
            end = parse_time_str(end_val)
        except Exception as e:
            continue

        section_name = sec.get('name', 'Unknown')
        section_points = [kp for kp in keypoints if start <= kp['time'] < end]
        section_keypoint_counts[section_name] = len(section_points)
        total_keypoints += len(section_points)

    # Step 2: 计算比例并分配镜头数
    if total_keypoints > 0:
        print(f"    📊 总关键点数: {total_keypoints}")
        allocated_shots = 0

        for section_name, count in section_keypoint_counts.items():
            ratio = count / total_keypoints
            allocated = max(1, round(total_shots * ratio))  # 至少保留1个
            section_top_k_map[section_name] = allocated
            allocated_shots += allocated
            print(f"       [{section_name}] 关键点: {count} ({ratio*100:.1f}%) -> 分配镜头数: {allocated}")

        # Step 3: 如果分配总数不等于 total_shots，调整最大section的数量
        if allocated_shots != total_shots:
            diff = total_shots - allocated_shots
            # 找到keypoint数量最多的section
            max_section = max(section_keypoint_counts, key=section_keypoint_counts.get)
            section_top_k_map[max_section] += diff
            print(f"    ⚖️  调整 [{max_section}]: {section_top_k_map[max_section] - diff} -> {section_top_k_map[max_section]} (补偿差值: {diff})")

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

        # 使用按比例分配的镜头数
        actual_top_k = section_top_k_map.get(name, 0)

        if actual_top_k == 0:
            # 如果该 section 没有分配到镜头（可能是因为没有关键点），跳过
            continue

        # 获取该段落内的所有关键点
        section_points = [kp for kp in keypoints
                         if start <= kp['time'] < end]

        if not section_points:
            print(f"       [{name}] {start:.1f}s-{end:.1f}s: 无关键点")
            continue

        # 按比例分配模式：直接按综合评分排序取 top_k
        section_points.sort(key=lambda x: x[score_key], reverse=True)
        selected = section_points[:actual_top_k]

        # 为选中的点添加段落信息
        for pt in selected:
            pt['section'] = name

        filtered.extend(selected)

        # 打印保留信息
        print(f"       [{name}] {start:.1f}s-{end:.1f}s ({duration:.1f}s): "
              f"保留 {len(selected)}/{len([kp for kp in keypoints if start <= kp['time'] < end])} 个点"
              f" (分配: {actual_top_k})")

    # 按时间排序
    filtered.sort(key=lambda x: x['time'])

    print(f"    段落过滤后共: {len(filtered)} 个关键点")

    # 4. 全局 min_interval 过滤（跨段落应用）
    # 这一步确保即使来自不同 section 的点也满足最小间隔要求
    if section_min_interval > 0 and len(filtered) > 1:
        before_global = len(filtered)
        global_filtered = []
        current_start = filtered[0]['time']
        current_best = filtered[0]

        for kp in filtered[1:]:
            if kp['time'] - current_start < section_min_interval:
                # 在间隔内，保留综合评分更高的
                if kp[score_key] > current_best[score_key]:
                    current_best = kp
            else:
                # 新间隔，保存之前的最佳点
                global_filtered.append(current_best)
                current_start = kp['time']
                current_best = kp
        
        # 添加最后一个
        global_filtered.append(current_best)
        filtered = global_filtered

        if len(filtered) < before_global:
            print(f"    全局 min_interval 过滤: {before_global} -> {len(filtered)} 个关键点 "
                  f"(移除了 {before_global - len(filtered)} 个跨段落的过近点)")

    # 5. 确保片段长度不超过 max_segment_duration
    # 如果相邻两个关键点之间的距离超过 max_segment_duration，从原始 keypoints 中找最近的点填充
    if max_segment_duration > 0 and len(filtered) > 1:
        before_split = len(filtered)
        filtered.sort(key=lambda x: x['time'])
        
        # 记录已经在 filtered 中的点的时间（用于快速查找）
        filtered_times = set(kp['time'] for kp in filtered)
        
        # 构建原始 keypoints 的时间索引（排除已经在 filtered 中的）
        available_keypoints = [kp for kp in keypoints if kp['time'] not in filtered_times]
        available_keypoints.sort(key=lambda x: x['time'])
        
        split_filtered = []
        
        for i in range(len(filtered)):
            split_filtered.append(filtered[i])
            
            # 检查与下一个点的距离
            if i < len(filtered) - 1:
                current_time = filtered[i]['time']
                next_time = filtered[i + 1]['time']
                gap = next_time - current_time
                
                # 如果间隔超过 max_segment_duration，从原始点中找合适的点填充
                if gap > max_segment_duration:
                    # 计算需要插入多少个点
                    num_splits = int(np.ceil(gap / max_segment_duration)) - 1
                    
                    # 找到位于这个间隔内的所有可用原始点
                    candidates = [kp for kp in available_keypoints 
                                 if current_time < kp['time'] < next_time]
                    
                    if candidates:
                        # 计算理想插入位置
                        ideal_positions = []
                        for j in range(1, num_splits + 1):
                            ideal_time = current_time + (gap * j / (num_splits + 1))
                            ideal_positions.append(ideal_time)
                        
                        # 为每个理想位置找到最接近的候选点
                        selected = []
                        used_indices = set()
                        
                        for ideal_time in ideal_positions:
                            # 找到最接近理想时间的候选点（未被使用的）
                            best_candidate = None
                            best_distance = float('inf')
                            best_idx = -1
                            
                            for idx, candidate in enumerate(candidates):
                                if idx in used_indices:
                                    continue
                                distance = abs(candidate['time'] - ideal_time)
                                if distance < best_distance:
                                    best_distance = distance
                                    best_candidate = candidate
                                    best_idx = idx
                            
                            if best_candidate:
                                selected.append(best_candidate)
                                used_indices.add(best_idx)
                        
                        # 按时间排序并添加选中的点
                        selected.sort(key=lambda x: x['time'])
                        split_filtered.extend(selected)
        
        filtered = split_filtered
        filtered.sort(key=lambda x: x['time'])

        if len(filtered) > before_split:
            print(f"    最大片段限制: 从原始点中补充了 {len(filtered) - before_split} 个关键点 "
                  f"(max_segment={max_segment_duration}s)")

    # 6. 边界检查和短片段处理
    # 确保第一个关键点不会太早，最后一个关键点不会太晚
    if min_segment_duration > 0 and len(filtered) > 0:
        before_boundary = len(filtered)
        boundary_filtered = []

        # 如果提供了audio_duration，进行边界检查
        if audio_duration and audio_duration > 0:
            for i, kp in enumerate(filtered):
                t = kp['time']

                # 检查第一个关键点
                if i == 0:
                    if t < min_segment_duration:
                        # 第一个点太早，跳过（从0到这个点的片段太短）
                        print(f"    ⚠️  跳过第一个关键点 {t:.2f}s (< min_segment={min_segment_duration}s)")
                        continue

                # 检查最后一个关键点
                if i == len(filtered) - 1:
                    remaining = audio_duration - t
                    if remaining < min_segment_duration:
                        # 最后一个点太晚，跳过（从这个点到结尾的片段太短）
                        print(f"    ⚠️  跳过最后一个关键点 {t:.2f}s (剩余 {remaining:.2f}s < min_segment={min_segment_duration}s)")
                        continue

                # 检查相邻关键点之间的间隔
                if len(boundary_filtered) > 0:
                    prev_t = boundary_filtered[-1]['time']
                    gap = t - prev_t

                    if gap < min_segment_duration:
                        # 相邻点间隔太小，保留综合评分更高的
                        if kp.get(score_key, 0) > boundary_filtered[-1].get(score_key, 0):
                            # 当前点评分更高，替换前一个点
                            boundary_filtered[-1] = kp
                            print(f"    ⚠️  合并短片段: [{prev_t:.2f}s - {t:.2f}s] ({gap:.2f}s < {min_segment_duration}s), 保留评分更高的点")
                        # else: 保留前一个点，跳过当前点
                        continue

                boundary_filtered.append(kp)
        else:
            # 没有audio_duration，只做相邻点间隔检查
            for i, kp in enumerate(filtered):
                if len(boundary_filtered) > 0:
                    prev_t = boundary_filtered[-1]['time']
                    gap = kp['time'] - prev_t

                    if gap < min_segment_duration:
                        # 相邻点间隔太小，保留综合评分更高的
                        if kp.get(score_key, 0) > boundary_filtered[-1].get(score_key, 0):
                            boundary_filtered[-1] = kp
                        continue

                boundary_filtered.append(kp)

        filtered = boundary_filtered

        if len(filtered) < before_boundary:
            removed = before_boundary - len(filtered)
            print(f"    边界和短片段处理: {before_boundary} -> {len(filtered)} 个关键点 "
                  f"(移除/合并了 {removed} 个点, min_segment={min_segment_duration}s)")

    return filtered


def main():
    parser = argparse.ArgumentParser(
        description='音频关键点检测 - 支持Downbeat/Pitch/MelEnergy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # Downbeat检测（默认）
  python audio_Madmom.py audio.wav

  # Pitch检测
  python audio_Madmom.py audio.wav --method pitch --pitch-min-distance 0.3 --pitch-threshold 0.8

  # Mel能量检测
  python audio_Madmom.py audio.wav --method mel_energy --mel-min-distance 0.5 --mel-threshold 0.3

  # 检测3/4拍或4/4拍的音乐
  python audio_Madmom.py audio.wav --method downbeat --beats-per-bar 3 4
        """
    )
    parser.add_argument('audio_path', type=str, help='音频文件路径')
    parser.add_argument('--method', type=str, default='downbeat', 
                        choices=['downbeat', 'pitch', 'mel_energy'],
                        help='检测方法 (default: downbeat)')
    
    # === Downbeat/DBN 节拍检测参数 ===
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
    beat_group.add_argument('--dbn-threshold', type=float, default=0.05,
                        help='DBN激活值阈值，默认0.05')
    beat_group.add_argument('--no-correct-beats', action='store_true',
                        help='不对齐节拍到最近的激活峰值')
    beat_group.add_argument('--fps', type=int, default=100,
                        help='帧率(用于节拍检测)，默认100')
    
    # === Pitch检测参数 ===
    pitch_group = parser.add_argument_group('Pitch检测参数')
    pitch_group.add_argument('--pitch-tolerance', type=float, default=0.8,
                        help='Pitch检测容差，默认0.8')
    pitch_group.add_argument('--pitch-threshold', type=float, default=0.8,
                        help='Pitch置信度阈值，默认0.8')
    pitch_group.add_argument('--pitch-min-distance', type=float, default=0.5,
                        help='Pitch检测的最小间隔(秒)，默认0.5')
    pitch_group.add_argument('--pitch-nms', type=str, default='basic',
                        choices=['basic', 'adaptive', 'window'],
                        help='Pitch NMS方法，默认basic')
    pitch_group.add_argument('--pitch-max-points', type=int, default=20,
                        help='Pitch最大保留点数，默认20')
    
    # === Mel能量检测参数 ===
    mel_group = parser.add_argument_group('Mel能量检测参数')
    mel_group.add_argument('--mel-win-size', type=int, default=512,
                        help='FFT窗口大小，默认512')
    mel_group.add_argument('--mel-n-filters', type=int, default=40,
                        help='Mel滤波器数量，默认40')
    mel_group.add_argument('--mel-threshold', type=float, default=0.3,
                        help='能量阈值比例(0-1)，默认0.3')
    mel_group.add_argument('--mel-min-distance', type=float, default=0.5,
                        help='Mel能量检测的最小间隔(秒)，默认0.5')
    mel_group.add_argument('--mel-nms', type=str, default='basic',
                        choices=['basic', 'adaptive', 'window'],
                        help='Mel NMS方法，默认basic')
    mel_group.add_argument('--mel-max-points', type=int, default=20,
                        help='Mel最大保留点数，默认20')
    
    # === 显著性过滤参数 ===
    filter_group = parser.add_argument_group('显著性过滤参数')
    filter_group.add_argument('--min-interval', type=float, default=0.0,
                        help='关键点之间的最小间隔(秒)，间隔内只保留最强的点，默认0.0（不过滤）')
    filter_group.add_argument('--top-k', type=int, default=0,
                        help='只保留强度最高的前K个关键点，默认0（不限制）')
    filter_group.add_argument('--energy-percentile', type=float, default=0.0,
                        help='只保留能量高于该百分位数的点(0-100)，默认0（不过滤）')
    
    # === 基于 Caption 段落过滤参数 ===
    caption_group = parser.add_argument_group('基于Caption段落过滤参数')
    caption_group.add_argument('--caption', type=str, default=None,
                        help='caption.json 文件路径，用于读取音乐段落(sections)划分')
    caption_group.add_argument('--section-top-k', type=int, default=0,
                        help='每个音乐段落内保留的最强点数量，默认0')
    caption_group.add_argument('--section-min-interval', type=float, default=0.0,
                        help='每个音乐段落内的最小间隔(秒)，默认0（不限制）')
    caption_group.add_argument('--section-energy-percentile', type=float, default=0.0,
                        help='每个音乐段落内的强度百分位数阈值(0-100)，只保留高于该阈值的点，默认0（不过滤）')

    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.audio_path):
        print(f"❌ 文件不存在: {args.audio_path}")
        return

    print(f"\n{'='*60}")
    print(f"🎵 音频关键点检测")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        # 根据检测方法创建检测器
        detector = SensoryKeypointDetector(
            detection_method=args.method,
            # Downbeat参数
            beats_per_bar=args.beats_per_bar,
            min_bpm=args.min_bpm,
            max_bpm=args.max_bpm,
            num_tempi=args.num_tempi,
            transition_lambda=args.transition_lambda,
            observation_lambda=args.observation_lambda,
            dbn_threshold=args.dbn_threshold,
            correct_beats=not args.no_correct_beats,
            fps=args.fps,
            # Pitch参数
            pitch_tolerance=args.pitch_tolerance,
            pitch_threshold=args.pitch_threshold,
            pitch_min_distance=args.pitch_min_distance,
            pitch_nms_method=args.pitch_nms,
            pitch_max_points=args.pitch_max_points,
            # Mel参数
            mel_win_s=args.mel_win_size,
            mel_n_filters=args.mel_n_filters,
            mel_threshold_ratio=args.mel_threshold,
            mel_min_distance=args.mel_min_distance,
            mel_nms_method=args.mel_nms,
            mel_max_points=args.mel_max_points,
        )
        result = detector.analyze(args.audio_path)

        elapsed_time = time.time() - start_time

        # 输出结果
        print(f"\n{'='*50}")
        print(f"检测完成! 耗时: {elapsed_time:.2f}秒")
        print(f"{'='*50}")
        
        print(f"\n📊 分析报告:")
        if args.method == "downbeat":
            print(f"  检测到 {len(result['downbeats'])} 个强拍")
        elif args.method == "pitch":
            print(f"  检测到 {len(result['keypoints'])} 个Pitch关键点")
        elif args.method == "mel_energy":
            print(f"  检测到 {len(result['keypoints'])} 个Mel能量关键点")
        
        print(f"  关键点: {len(result['keypoints'])} 个")
        
        # 对于downbeat方法应用显著性过滤
        if args.method == "downbeat":
            original_count = len(result['keypoints'])
            need_filter = (args.min_interval > 0 or args.top_k > 0 or 
                           args.energy_percentile > 0)
            
            if need_filter:
                print(f"\n🔍 应用显著性过滤...")
                filtered_keypoints = filter_significant_keypoints(
                    result['keypoints'],
                    min_interval=args.min_interval,
                    top_k=args.top_k,
                    energy_percentile=args.energy_percentile
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
                        
                        filtered_keypoints = filter_by_sections(
                            result['keypoints'],
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
        print(f"{'时间(秒)':>10} | {'类型':<15} | {'强度':>6}")
        print("-" * 45)
        for pt in result['keypoints'][:15]:
            print(f"{pt['time']:10.3f} | {pt['type']:<15} | {pt['intensity']:6.2f}")
        
        if len(result['keypoints']) > 15:
            print(f"  ... (共 {len(result['keypoints'])} 个关键点)")

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