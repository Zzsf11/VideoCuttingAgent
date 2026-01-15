#!/usr/bin/env python3
"""
最简单的 call_vllm_model 测试示例

快速开始：
1. 修改下面的配置
2. 运行: python simple_test.py
"""

from vca import config
from vca.vllm_calling import call_vllm_model

# ========================================
# 配置 - 根据你的环境修改
# ========================================

# vLLM 服务器
VLLM_ENDPOINT = getattr(config, 'VLLM_ENDPOINT', 'http://localhost:8000')
MODEL_NAME = getattr(config, 'VIDEO_ANALYSIS_MODEL', 'Qwen/Qwen2-VL-7B-Instruct')

# 如果要测试视频，修改这里
VIDEO_PATH = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/Dataset/Video/Movie/The_Dark_Knight.mkv"


# ========================================
# 示例 1: 最简单的文本调用
# ========================================

def example_text():
    print("\n=== 示例 1: 纯文本调用 ===\n")

    messages = [
        {"role": "user", "content": "你好，请介绍一下自己。"}
    ]

    response = call_vllm_model(
        messages=messages,
        endpoint=VLLM_ENDPOINT,
        model_name=MODEL_NAME,
        max_tokens=200
    )

    print("响应:", response["content"])


# ========================================
# 示例 2: 视频分析
# ========================================

def example_video():
    print("\n=== 示例 2: 视频分析 ===\n")

    messages = [
        {"role": "user", "content": "请描述这个视频的内容。"}
    ]

    response = call_vllm_model(
        messages=messages,
        endpoint=VLLM_ENDPOINT,
        model_name=MODEL_NAME,
        video_path=VIDEO_PATH,
        video_fps=2.0,         # 采样帧率
        do_sample_frames=True, # 让vLLM采样帧
        max_tokens=500
    )

    print("响应:", response["content"])


# ========================================
# 示例 3: 视频片段分析（使用本地裁剪）
# ========================================

def example_video_clip():
    print("\n=== 示例 3: 视频片段分析（10-15秒） ===\n")

    messages = [
        {"role": "user", "content": "请详细描述这个视频片段中发生了什么。"}
    ]

    response = call_vllm_model(
        messages=messages,
        endpoint=VLLM_ENDPOINT,
        model_name=MODEL_NAME,
        video_path=VIDEO_PATH,
        video_fps=config.VIDEO_FPS,
        video_start_time=10.0,      # 开始时间（秒）
        video_end_time=15.0,        # 结束时间（秒）
        use_local_clipping=True,    # 使用本地ffmpeg裁剪（更快）
        do_sample_frames=False,     # 不再采样
        max_tokens=800
    )

    print("响应:", response["content"])


# ========================================
# 示例 4: JSON 格式输出
# ========================================

def example_json():
    print("\n=== 示例 4: JSON 格式输出 ===\n")

    messages = [
        {
            "role": "system",
            "content": "你是一个数据分析助手，始终以JSON格式返回结果。"
        },
        {
            "role": "user",
            "content": """请分析以下数组并返回JSON格式：
{
  "data": [1, 5, 3, 8, 2],
  "analysis": {
    "max": <最大值>,
    "min": <最小值>,
    "sum": <总和>
  }
}"""
        }
    ]

    response = call_vllm_model(
        messages=messages,
        endpoint=VLLM_ENDPOINT,
        model_name=MODEL_NAME,
        return_json=True,  # 要求JSON输出
        max_tokens=300,
        temperature=0.0
    )

    print("响应:", response["content"])


# ========================================
# 主函数
# ========================================

if __name__ == "__main__":
    import sys

    print("="*60)
    print("call_vllm_model 简单测试")
    print("="*60)
    print(f"\n配置:")
    print(f"  VLLM_ENDPOINT: {VLLM_ENDPOINT}")
    print(f"  MODEL_NAME: {MODEL_NAME}")

    # 选择要运行的示例
    if len(sys.argv) > 1:
        example = sys.argv[1]
    else:
        print("\n可用示例:")
        print("  1. text       - 纯文本")
        print("  2. video      - 完整视频")
        print("  3. video_clip - 视频片段")
        print("  4. json       - JSON输出")
        print("\n用法: python simple_test.py <示例名>")
        print("例如: python simple_test.py text\n")
        example = "text"  # 默认运行文本示例

    try:
        if example == "text":
            example_text()
        elif example == "video":
            example_video()
        elif example == "video_clip":
            example_video_clip()
        elif example == "json":
            example_json()
        else:
            print(f"未知示例: {example}")
            sys.exit(1)

        print("\n✅ 测试完成")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
