#!/usr/bin/env python3
"""
快速测试Ruptures是否工作正常
"""

import sys
import time
import numpy as np

print("=" * 60)
print("Ruptures快速测试脚本")
print("=" * 60)

# 测试1: 导入库
print("\n[1/5] 测试导入库...")
try:
    import librosa
    print("  ✓ librosa导入成功")
except ImportError as e:
    print(f"  ✗ librosa导入失败: {e}")
    sys.exit(1)

try:
    import ruptures as rpt
    print("  ✓ ruptures导入成功")
except ImportError as e:
    print(f"  ✗ ruptures导入失败: {e}")
    sys.exit(1)

# 测试2: 生成模拟信号
print("\n[2/5] 生成模拟信号...")
n_samples = 16000 * 60  # 1分钟的音频
signal = np.concatenate([
    np.sin(2 * np.pi * 440 * np.arange(n_samples//3) / 16000),  # 440Hz
    np.sin(2 * np.pi * 880 * np.arange(n_samples//3) / 16000),  # 880Hz
    np.sin(2 * np.pi * 220 * np.arange(n_samples//3) / 16000),  # 220Hz
])
print(f"  ✓ 生成信号: {len(signal)} samples ({len(signal)/16000:.1f}秒)")

# 测试3: 提取特征
print("\n[3/5] 提取RMS特征...")
start = time.time()
features = librosa.feature.rms(y=signal, hop_length=1024)
print(f"  ✓ 特征形状: {features.shape}, 耗时: {time.time()-start:.2f}秒")

# 测试4: 使用Ruptures检测 (l2模型)
print("\n[4/5] 使用Ruptures检测变化点 (model=l2)...")
start = time.time()
algo = rpt.Pelt(model="l2", min_size=10, jump=10)
algo.fit(features.T)
pen = np.log(features.shape[1]) * features.shape[0]
result = algo.predict(pen=pen)
print(f"  ✓ 检测到 {len(result)-1} 个变化点, 耗时: {time.time()-start:.2f}秒")

# 测试5: 如果命令行提供了音频文件，测试真实音频
if len(sys.argv) > 1:
    audio_path = sys.argv[1]
    print(f"\n[5/5] 测试真实音频文件: {audio_path}")

    print("  加载音频...")
    start = time.time()
    audio, sr = librosa.load(audio_path, sr=16000)
    load_time = time.time() - start
    print(f"  ✓ 加载完成: {len(audio)/sr:.2f}秒音频, 耗时: {load_time:.2f}秒")

    print("  提取RMS特征...")
    start = time.time()
    features = librosa.feature.rms(y=audio, hop_length=1024)
    feat_time = time.time() - start
    print(f"  ✓ 特征提取完成: {features.shape}, 耗时: {feat_time:.2f}秒")

    print("  检测变化点 (l2, jump=10)...")
    start = time.time()
    algo = rpt.Pelt(model="l2", min_size=10, jump=10)
    algo.fit(features.T)
    fit_time = time.time() - start
    print(f"  ✓ 拟合完成, 耗时: {fit_time:.2f}秒")

    start = time.time()
    pen = np.log(features.shape[1]) * features.shape[0]
    result = algo.predict(pen=pen)
    pred_time = time.time() - start
    print(f"  ✓ 预测完成: 检测到 {len(result)-1} 个变化点, 耗时: {pred_time:.2f}秒")

    total_time = load_time + feat_time + fit_time + pred_time
    print(f"\n  总耗时: {total_time:.2f}秒")
    print(f"  变化点时间: {librosa.frames_to_time(result[:-1], sr=sr, hop_length=1024).tolist()}")
else:
    print("\n[5/5] 跳过真实音频测试 (未提供音频文件)")

print("\n" + "=" * 60)
print("测试完成！所有基础功能正常")
print("=" * 60)
print("\n使用方法: python test_ruptures_quick.py [音频文件路径]")
