from openai import OpenAI
import cv2
import base64
import io
from PIL import Image

def extract_frames(video_path, start_time=None, end_time=None, fps=1.0, resize=(None, None)):
    """
    从视频中按指定FPS提取帧
    
    Args:
        video_path: 视频文件路径
        start_time: 起始时间（秒），None表示从头开始
        end_time: 结束时间（秒），None表示到视频末尾
        fps: 抽帧频率（帧/秒），可以小于1，例如0.5表示每2秒取1帧
        resize: (height, width) 调整帧的分辨率，None表示保持原始尺寸
    
    Returns:
        frames_base64: base64编码的帧列表
    """
    cap = cv2.VideoCapture(video_path)
    
    # 获取视频信息
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / video_fps
    
    # 确定起止时间
    start_time = start_time if start_time is not None else 0
    end_time = end_time if end_time is not None else video_duration
    
    # 计算起止帧
    start_frame = int(start_time * video_fps)
    end_frame = int(end_time * video_fps)
    
    # 计算需要提取的帧索引
    frame_interval = video_fps / fps  # 每隔多少帧取一帧
    frame_indices = []
    current_frame = start_frame
    while current_frame <= end_frame:
        frame_indices.append(int(current_frame))
        current_frame += frame_interval
    
    frames_base64 = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Resize if specified
            resize_h, resize_w = resize
            if resize_h is not None and resize_w is not None:
                frame = cv2.resize(frame, (resize_w, resize_h))
            
            # 转换BGR到RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转换为PIL Image
            pil_img = Image.fromarray(frame_rgb)
            # 编码为base64
            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG")
            base64_frame = base64.b64encode(buffer.getvalue()).decode('utf-8')
            frames_base64.append(base64_frame)
    
    cap.release()
    return frames_base64

# 提取视频帧
video_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/Dataset/Video/Batman.Begins.2005.1080p.BluRay.x264.YIFY.mp4"
# 示例：从第10秒到第30秒，每秒抽取1帧，resize到720x1280
frames = extract_frames(
    video_path=video_path,
    start_time=1000,      # 起始时间（秒）
    end_time=1600,        # 结束时间（秒）
    fps=1.0,            # 抽帧频率（可以是0.5等小于1的值）
    resize=(144, 144)  # (height, width)，None表示保持原始尺寸
)

# 构建消息内容
content = [{"type": "text", "text": "请描述这个视频的内容"}]
for frame_base64 in frames:
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{frame_base64}"
        }
    })

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1"
)
import time

start_time = time.time()

import subprocess

def get_gpu_utilization():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True
        )
        utils = [int(line) for line in result.stdout.strip().split('\n') if line.strip()]
        if utils:
            # 只使用第一个GPU（可以根据情况调整）
            return utils[0]
        else:
            return None
    except Exception as e:
        print("无法获取GPU利用率:", e)
        return None

gpu_utils = []
import threading

def monitor_gpu():
    while not monitor_stop.is_set():
        util = get_gpu_utilization()
        if util is not None:
            gpu_utils.append(util)
        time.sleep(0.2)  # 每0.2秒采样一次

monitor_stop = threading.Event()
monitor_thread = threading.Thread(target=monitor_gpu)
monitor_thread.start()

response = client.chat.completions.create(
    model="/public_hw/home/cit_shifangzhao/zsf/HF/models/Qwen/Qwen3-VL-30B-A3B-Instruct",
    messages=[
        {
            "role": "user",
            "content": content
        }
    ],
    max_tokens=16384
)

monitor_stop.set()
monitor_thread.join()

if gpu_utils:
    avg_gpu_util = sum(gpu_utils) / len(gpu_utils)
    print(f"请求期间GPU平均利用率：{avg_gpu_util:.2f} %")
else:
    print("没有采集到GPU利用率数据。")
print(time.time() - start_time)
print(response.choices[0].message.content)