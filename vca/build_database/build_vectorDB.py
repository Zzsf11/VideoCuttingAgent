import os
import json
import numpy as np
from vca.build_database.video_caption_ori import convert_seconds_to_hhmmss
from nano_vectordb import NanoVectorDB
from vca import config
import multiprocessing
from tqdm import tqdm
from openai import OpenAI


def init_single_video_db(video_caption_json_path, output_video_db_path, emb_dim):
    # 确保输出目录存在
    output_dir = os.path.dirname(output_video_db_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"创建输出目录: {output_dir}")
    
    vdb = NanoVectorDB(emb_dim, storage_file=output_video_db_path)
    # with open(video_caption_json_path, "r") as f:
    #     captions = json.load(f)
    # subject_registry = captions.pop('subject_registry', captions.pop('character_registry', None))            
    # video_length = max([float(k.split("_")[1]) for k in captions.keys()])
    # if not is_covered(captions.keys(), video_length):
    #     error_msg = (f"Fail to build video database for video {video_caption_json_path.split("/")[-3]}. Get None video clip captions for some clips in the video.")
    #     raise ValueError(error_msg)
    # video_length = convert_seconds_to_hhmmss(video_length)
    if os.path.exists(output_video_db_path):
        print(f"Database {output_video_db_path} already exists.")
    else:
        cap2emb_list = preprocess_captions(video_caption_json_path)
        data = []
        for timestamp, shot_info, emb in cap2emb_list:
            t1 = convert_seconds_to_hhmmss(timestamp[0])
            t2 = convert_seconds_to_hhmmss(timestamp[1])
            prefix = f"[From {t1} to {t2} seconds]\n"
            data.append(
                {
                    "__vector__": np.array(emb),
                    "time_start_secs": timestamp[0],
                    "time_end_secs": timestamp[1],
                    "event_summary": prefix + shot_info['event_summary'],
                    "shot_purpose": shot_info['shot_purpose'],
                    "mood": shot_info['mood'],
                }
            )
        _ = vdb.upsert(data)
        with open(video_caption_json_path, "r") as f:
            captions = json.load(f)
        subject_registry = captions.pop('subject_registry', captions.pop('character_registry', None))
        # 解析时间戳格式：可能是 "0_21" 或 "8393_8396_shot2044_sub18"
        video_length = max([float(k.split("_")[1]) for k in captions.keys() if "_" in k])
        video_length = convert_seconds_to_hhmmss(video_length)
        addtional_data = {
            'subject_registry': subject_registry,
            'video_length': video_length,
            'video_file_root': os.path.dirname(os.path.dirname(video_caption_json_path)),
            'fps': getattr(config, "VIDEO_FPS", 2),
        }
        vdb.store_additional_data(**addtional_data)
        vdb.save()
    return vdb

def preprocess_captions(caption_json_path):
    with open(caption_json_path, "r") as f:
        captions = json.load(f)
    scripts = []
    captions.pop('subject_registry', None)
    captions.pop('character_registry', None)
    for timestamp_key, cap_info in captions.items():
        # 解析时间戳：格式可能是 "0_21" 或 "8393_8396_shot2044_sub18"
        # 只取前两个部分作为时间戳
        parts = timestamp_key.split("_")
        try:
            timestamp = [float(parts[0]), float(parts[1])]
        except (ValueError, IndexError):
            print(f"Invalid timestamp format: {timestamp_key}")
            continue

        # 提取新格式的信息
        event_summary = cap_info.get('action_atoms', {}).get('event_summary', '')
        shot_purpose = cap_info.get('narrative_analysis', {}).get('shot_purpose', '')
        mood = cap_info.get('narrative_analysis', {}).get('mood', '')

        # 兼容旧格式
        if not event_summary:
            event_summary = cap_info.get('caption', '')
            if isinstance(event_summary, list):
                event_summary = event_summary[0] if event_summary else ''
            elif not isinstance(event_summary, str):
                event_summary = str(event_summary) if event_summary else ''

        if not event_summary:
            print(f"Empty event_summary for {timestamp_key} in {caption_json_path}")
            continue

        # 构建结构化的 shot 信息
        shot_info = {
            'event_summary': event_summary,
            'shot_purpose': shot_purpose,
            'mood': mood,
        }

        # 用于 embedding 的文本：组合所有信息
        caption_text = f"Event: {event_summary}"
        if shot_purpose:
            caption_text += f" Purpose: {shot_purpose}"
        if mood:
            caption_text += f" Mood: {mood}"

        scripts.append((timestamp, caption_text, shot_info))

    # batchify
    batch_size = 128
    batched_scripts = []
    print(f"Embedding {len(scripts)} captions...")
    for i in range(0, len(scripts), batch_size):
        batch = scripts[i:i+batch_size]
        batched_scripts.append(batch)
    cap2emb_list = []
    with multiprocessing.Pool(os.cpu_count() // 2) as pool:
        with tqdm(total=len(scripts), desc="Embedding captions...") as pbar:
            for result in pool.imap_unordered(
                single_batch_embedding_task,
                batched_scripts,
            ):
                cap2emb_list.extend(result)
                pbar.update(len(result))
    return cap2emb_list

def single_batch_embedding_task(data):
    """
    使用vLLM本地部署的embedding模型获取文本嵌入
    """
    timestamps, captions, cap_infos = map(list, (zip(*data)))
    
    # 初始化vLLM OpenAI客户端
    # 假设vLLM embedding服务运行在不同端口或者同一个端口
    # 默认使用localhost:8000或者从config读取
    vllm_embedding_endpoint = getattr(config, 'VLLM_EMBEDDING_ENDPOINT', 'http://localhost:8000/v1')
    client = OpenAI(
        api_key="EMPTY",  # vLLM不需要真实的API key
        base_url=vllm_embedding_endpoint,
    )
    
    max_tries = 3
    embs = None
    
    for attempt in range(max_tries):
        try:
            # 调用vLLM的embedding API
            response = client.embeddings.create(
                model=config.EMBEDDING_MODEL,  # 使用本地模型路径或名称
                input=captions,
            )
            embs = [{"embedding": item.embedding} for item in response.data]
            
            if embs is not None and len(embs) == len(captions):
                break
        except Exception as e:
            if attempt < max_tries - 1:
                print(f"Error in embedding (attempt {attempt+1}/{max_tries}): {str(e)}")
                print(f"Retrying for timestamps: {timestamps[:2]}...")  # 只打印前两个timestamp避免输出过长
            else:
                raise ValueError(f"Failed to get embeddings after {max_tries} attempts. Last error: {str(e)}")
    
    if embs is None or len(embs) != len(captions):
        raise ValueError(f"Failed to get embeddings for {len(captions)} captions")
    
    return list(zip(timestamps, cap_infos, [d['embedding'] for d in embs]))

if __name__ == "__main__":
    # benchmark_metadata_path = "/home/xiaoyizhang/event_prediction_model/LVBench/data/video_info.meta.jsonl"
    video_caption_folder = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Database/Batman_Begins_2005_1080p_BluRay_x264_YIFY"
    embedding_dim = config.EMBEDDING_MODEL_DIM  # 使用本地模型的维度

    # with open(benchmark_metadata_path, "r") as f:
    #     lines = f.readlines()
    output_video_db_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Database/Batman_Begins_2005_1080p_BluRay_x264_YIFY/vdb.json"
    video_caption_json_path = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Database/Batman_Begins_2005_1080p_BluRay_x264_YIFY/captions/captions.json"
    video_db = init_single_video_db(video_caption_json_path, output_video_db_path, embedding_dim)

    
