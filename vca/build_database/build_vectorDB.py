import os
import json
import numpy as np
from vca.build_database.video_caption import convert_seconds_to_hhmmss
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
        for idx, (timestamp, cap, emb) in enumerate(cap2emb_list):
            t1 = convert_seconds_to_hhmmss(timestamp[0])
            t2 = convert_seconds_to_hhmmss(timestamp[1])
            prefix = f"[From {t1} to {t2} seconds]\n"
            data.append(
                {
                    "__vector__": np.array(emb),
                    "time_start_secs": timestamp[0],
                    "time_end_secs": timestamp[1],
                    "caption": prefix + cap['caption'],
                }
            )
        _ = vdb.upsert(data)
        with open(video_caption_json_path, "r") as f:
            captions = json.load(f)
        subject_registry = captions.pop('subject_registry', captions.pop('character_registry', None))          
        video_length = max([float(k.split("_")[1]) for k in captions.keys()])
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
    for idx, (timestamp, cap_info) in enumerate(captions.items()):
        if cap_info.get('caption') is None or len(cap_info['caption']) == 0:
            print(f"Empty caption information for {timestamp} in {caption_json_path}")
            continue
        elif isinstance(cap_info['caption'], list):
            cap_info['caption'] = cap_info['caption'][0]
        elif not isinstance(cap_info['caption'], str):
            print(f"Invalid caption type for {cap_info['caption']}")
            cap_info['caption'] = str(cap_info['caption'])
        timestamp = list(map(float, timestamp.split("_")))
        scripts.append((timestamp, cap_info['caption'], cap_info))

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

    
