#!/usr/bin/env python3
"""
测试 vllm_calling.py 中的 call_vllm_model 函数

使用方法:
    python test_vllm_calling.py --test text             # 测试纯文本调用
    python test_vllm_calling.py --test image            # 测试图片调用
    python test_vllm_calling.py --test video            # 测试完整视频
    python test_vllm_calling.py --test video_clip       # 测试视频片段裁剪
    python test_vllm_calling.py --test function_call    # 测试工具调用
    python test_vllm_calling.py --test json_output      # 测试JSON输出
    python test_vllm_calling.py --test all              # 运行所有测试
"""

import sys
import json
import argparse
from pathlib import Path

# 导入必要的模块
from vca import config
from vca.vllm_calling import call_vllm_model


# ========================================
# 配置部分 - 根据你的环境修改这里
# ========================================

# vLLM 服务器配置
VLLM_ENDPOINT = getattr(config, 'VLLM_ENDPOINT', 'http://localhost:8000')
MODEL_NAME = getattr(config, 'VIDEO_ANALYSIS_MODEL', 'Qwen/Qwen2-VL-7B-Instruct')

# 测试文件路径（请根据实际情况修改）
TEST_IMAGE_PATH = "/path/to/test_image.jpg"  # 修改为实际图片路径
TEST_VIDEO_PATH = "/path/to/test_video.mp4"  # 修改为实际视频路径


# ========================================
# 测试函数
# ========================================

def test_text_only():
    """测试1: 纯文本调用（最简单）"""
    print("\n" + "="*60)
    print("测试 1: 纯文本调用")
    print("="*60)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "请用一句话解释什么是深度学习。"
        }
    ]

    try:
        response = call_vllm_model(
            messages=messages,
            endpoint=VLLM_ENDPOINT,
            model_name=MODEL_NAME,
            max_tokens=200,
            temperature=0.7
        )

        print("\n响应:")
        print(response.get("content", ""))
        print("\n✅ 测试通过")
        return True
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_image_input():
    """测试2: 带图片的调用"""
    print("\n" + "="*60)
    print("测试 2: 图片输入")
    print("="*60)

    if not Path(TEST_IMAGE_PATH).exists():
        print(f"⚠️  跳过测试: 图片文件不存在 {TEST_IMAGE_PATH}")
        print("请修改 TEST_IMAGE_PATH 变量")
        return None

    messages = [
        {
            "role": "user",
            "content": "请描述这张图片中的内容。"
        }
    ]

    try:
        response = call_vllm_model(
            messages=messages,
            endpoint=VLLM_ENDPOINT,
            model_name=MODEL_NAME,
            image_paths=[TEST_IMAGE_PATH],
            max_tokens=500,
            temperature=0.5
        )

        print("\n响应:")
        print(response.get("content", ""))
        print("\n✅ 测试通过")
        return True
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_video_input():
    """测试3: 完整视频输入"""
    print("\n" + "="*60)
    print("测试 3: 完整视频输入")
    print("="*60)

    if not Path(TEST_VIDEO_PATH).exists():
        print(f"⚠️  跳过测试: 视频文件不存在 {TEST_VIDEO_PATH}")
        print("请修改 TEST_VIDEO_PATH 变量")
        return None

    messages = [
        {
            "role": "user",
            "content": "请简要描述这个视频的内容，包括主要场景和动作。"
        }
    ]

    try:
        response = call_vllm_model(
            messages=messages,
            endpoint=VLLM_ENDPOINT,
            model_name=MODEL_NAME,
            video_path=TEST_VIDEO_PATH,
            video_fps=2.0,  # 采样帧率
            do_sample_frames=True,  # 让vLLM采样帧
            max_tokens=800,
            temperature=0.5
        )

        print("\n响应:")
        print(response.get("content", ""))
        print("\n✅ 测试通过")
        return True
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_video_clipping():
    """测试4: 视频片段裁剪"""
    print("\n" + "="*60)
    print("测试 4: 视频片段裁剪 (use_local_clipping=True)")
    print("="*60)

    if not Path(TEST_VIDEO_PATH).exists():
        print(f"⚠️  跳过测试: 视频文件不存在 {TEST_VIDEO_PATH}")
        print("请修改 TEST_VIDEO_PATH 变量")
        return None

    # 裁剪视频的10-20秒片段
    start_time = 10.0  # 秒
    end_time = 20.0    # 秒

    messages = [
        {
            "role": "user",
            "content": f"请详细描述这个{end_time - start_time}秒视频片段中发生了什么。"
        }
    ]

    try:
        response = call_vllm_model(
            messages=messages,
            endpoint=VLLM_ENDPOINT,
            model_name=MODEL_NAME,
            video_path=TEST_VIDEO_PATH,
            video_fps=config.VIDEO_FPS,  # 使用配置中的FPS
            video_start_time=start_time,
            video_end_time=end_time,
            use_local_clipping=True,  # 使用本地ffmpeg裁剪（更快）
            do_sample_frames=False,   # 不再采样
            max_tokens=1000,
            temperature=0.3
        )

        print(f"\n裁剪范围: {start_time}s - {end_time}s")
        print("\n响应:")
        print(response.get("content", ""))
        print("\n✅ 测试通过")
        return True
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_function_calling():
    """测试5: 工具调用（Function Calling）"""
    print("\n" + "="*60)
    print("测试 5: 工具调用 (Function Calling)")
    print("="*60)

    # 定义一个简单的工具
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取指定城市的天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称，例如：北京、上海"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "温度单位"
                        }
                    },
                    "required": ["city"]
                }
            }
        }
    ]

    messages = [
        {
            "role": "user",
            "content": "北京今天天气怎么样？"
        }
    ]

    try:
        response = call_vllm_model(
            messages=messages,
            endpoint=VLLM_ENDPOINT,
            model_name=MODEL_NAME,
            tools=tools,
            tool_choice="auto",
            max_tokens=500,
            temperature=0.0
        )

        print("\n响应:")
        if response.get("tool_calls"):
            print("模型调用了工具:")
            for tool_call in response["tool_calls"]:
                print(f"  - 函数名: {tool_call['function']['name']}")
                print(f"  - 参数: {tool_call['function']['arguments']}")
            print("\n✅ 测试通过（模型正确调用了工具）")
        else:
            print(f"模型返回文本: {response.get('content', '')}")
            print("\n⚠️  警告：模型未调用工具（可能模型不支持function calling）")
        return True
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_json_output():
    """测试6: JSON格式输出"""
    print("\n" + "="*60)
    print("测试 6: JSON 格式输出")
    print("="*60)

    messages = [
        {
            "role": "system",
            "content": "你是一个数据分析助手。请用JSON格式返回结果。"
        },
        {
            "role": "user",
            "content": """请分析以下数据并以JSON格式返回：
{
  "name": "分析报告",
  "items": [1, 5, 3, 8, 2],
  "summary": {
    "max": <最大值>,
    "min": <最小值>,
    "avg": <平均值>
  }
}"""
        }
    ]

    try:
        response = call_vllm_model(
            messages=messages,
            endpoint=VLLM_ENDPOINT,
            model_name=MODEL_NAME,
            return_json=True,  # 要求JSON输出
            max_tokens=500,
            temperature=0.0
        )

        print("\n响应:")
        content = response.get("content", "")
        print(content)

        # 尝试解析JSON
        try:
            parsed_json = json.loads(content)
            print("\n✅ JSON解析成功:")
            print(json.dumps(parsed_json, indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            print("\n⚠️  警告：返回内容不是有效的JSON")

        print("\n✅ 测试通过")
        return True
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_frame_sequence():
    """测试7: 帧序列输入（auto_encode_frames）"""
    print("\n" + "="*60)
    print("测试 7: 帧序列输入 (auto_encode_frames)")
    print("="*60)

    # 这个测试需要一个帧序列目录
    # 假设你有一个目录包含 frame_0000.jpg, frame_0001.jpg, ...
    frame_dir = Path("/path/to/frames")  # 修改为实际路径

    if not frame_dir.exists():
        print(f"⚠️  跳过测试: 帧目录不存在 {frame_dir}")
        print("请修改 frame_dir 变量或创建测试帧")
        return None

    # 获取所有帧
    frame_paths = sorted(frame_dir.glob("*.jpg"))[:30]  # 只取前30帧

    if not frame_paths:
        print(f"⚠️  跳过测试: {frame_dir} 中没有找到图片")
        return None

    messages = [
        {
            "role": "user",
            "content": "请描述这个视频片段的内容。"
        }
    ]

    try:
        response = call_vllm_model(
            messages=messages,
            endpoint=VLLM_ENDPOINT,
            model_name=MODEL_NAME,
            image_paths=[str(p) for p in frame_paths],
            video_fps=2.0,  # 关键：指定帧的采样率
            auto_encode_frames=True,  # 自动编码为视频
            max_tokens=800,
            temperature=0.5
        )

        print(f"\n使用了 {len(frame_paths)} 帧")
        print("\n响应:")
        print(response.get("content", ""))
        print("\n✅ 测试通过")
        return True
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# ========================================
# 主函数
# ========================================

def main():
    parser = argparse.ArgumentParser(
        description="测试 call_vllm_model 函数",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--test',
        choices=['text', 'image', 'video', 'video_clip', 'function_call', 'json_output', 'frame_sequence', 'all'],
        default='text',
        help='选择要运行的测试'
    )

    args = parser.parse_args()

    print("\n" + "🚀 " * 30)
    print("call_vllm_model 测试套件")
    print("🚀 " * 30)

    print(f"\n配置信息:")
    print(f"  - VLLM_ENDPOINT: {VLLM_ENDPOINT}")
    print(f"  - MODEL_NAME: {MODEL_NAME}")

    # 测试映射
    tests = {
        'text': ('纯文本', test_text_only),
        'image': ('图片输入', test_image_input),
        'video': ('完整视频', test_video_input),
        'video_clip': ('视频裁剪', test_video_clipping),
        'function_call': ('工具调用', test_function_calling),
        'json_output': ('JSON输出', test_json_output),
        'frame_sequence': ('帧序列', test_frame_sequence),
    }

    results = {}

    if args.test == 'all':
        # 运行所有测试
        for test_key, (test_name, test_func) in tests.items():
            result = test_func()
            results[test_name] = result
    else:
        # 运行单个测试
        test_name, test_func = tests[args.test]
        result = test_func()
        results[test_name] = result

    # 打印总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)

    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)

    for test_name, result in results.items():
        if result is True:
            status = "✅ 通过"
        elif result is False:
            status = "❌ 失败"
        else:
            status = "⏭️  跳过"
        print(f"  {test_name}: {status}")

    print(f"\n总计: {passed} 通过, {failed} 失败, {skipped} 跳过")

    # 返回退出码
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
