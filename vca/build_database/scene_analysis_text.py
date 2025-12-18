import os
import json
import re
import concurrent
from openai import OpenAI
from tqdm import tqdm

# ================= 配置 =================
SCENES_DIR = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Database/Batman_Begins_2005_1080p_BluRay_x264_YIFY/captions/scenes"  # 上一步的输出目录
OUTPUT_DIR = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Database/Batman_Begins_2005_1080p_BluRay_x264_YIFY/captions/scene_summaries"
SUBTITLE_FILE = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Database/Batman_Begins_2005_1080p_BluRay_x264_YIFY/subtitles.srt"  # 字幕文件路径
API_URL = "http://localhost:8888/v1"
API_KEY = "EMPTY"
MODEL_NAME = "/public_hw/home/cit_shifangzhao/zsf/HF/models/Qwen/Qwen3-VL-30B-A3B-Instruct" # 推荐使用逻辑强的模型
MAX_WORKERS = 8  # 并发线程数，根据机器性能调整
# =======================================

# ================= PROMPTS =================

# Prompt 1: 人物一致性（结合字幕推断名字）
ENTITY_RESOLUTION_PROMPT = """
[Role]
You are a Film Continuity Editor. Your task is to resolve character identities within a single movie scene, using both visual information and dialogue to infer character names.

[Input Data]
1. **Visual Characters**: Raw character descriptions from sequential shots.
   Format: "Shot ID | Key: [UniqueKey] | Visual: [Visual ID] - [Appearance]"
2. **Dialogue**: Subtitles/transcript from this scene with speaker labels.

[Task]
1. **Infer Names from Dialogue**:
   - Look for names mentioned in dialogue (e.g., "Rachel, let me see" → there's a character named Rachel)
   - Match spoken names to visual characters based on context (who is being addressed, who responds)
   - If someone says "Bruce?" and a boy reacts, that boy is likely Bruce

2. **Cluster Identities**: Group characters that refer to the same person based on:
   - **Visual Similarity**: e.g., "Man in Black Suit" and "Man_Tuxedo" are likely the same
   - **Dialogue Context**: If dialogue mentions a name and visual matches, use the real name
   - **Speaker Matching**: Try to match speaker labels to visual characters

3. **Assign Standard Names**:
   - **Priority 1**: Use real names inferred from dialogue (e.g., "Bruce", "Rachel", "Alfred")
   - **Priority 2**: Use names mentioned in visual_id if they seem like real names
   - **Priority 3**: Use descriptive names as fallback (e.g., "The Detective", "Young Boy")

[Output Schema]
Return a JSON object:
{
  "unique_cast": [
    {
      "standard_name": "Real name if inferred, or descriptive name",
      "visual_summary": "Consolidated appearance description",
      "name_source": "dialogue/visual_id/inferred/descriptive"
    }
  ],
  "mapping": {
    "Shot_ID_Visual_ID": "Standard Name",
    "726_728.json_Boy_in_Tuxedo": "Bruce Wayne"
  },
  "dialogue_names_found": ["List of names explicitly mentioned in dialogue"]
}
"""

# Prompt 2: 解耦摘要
SCENE_ANALYSIS_PROMPT = """
[Role]
You are a Lead Film Critic and Analyst. Analyze the scene based on the provided Shot List and Dialogue.

[Task]
Generate a structured analysis. You must SEPARATE narrative events, emotional shifts, and cinematic techniques.
Use the dialogue to understand character interactions and infer character actions/behaviors.
Do not hallucinate information not present in the shots or dialogue.

[Input Context]
- **Cast**: {CAST_LIST}
- **Shot Sequence**: {SHOT_SEQUENCE}
- **Dialogue**: {DIALOGUE}

[Output Schema]
Return a JSON object:
{
  "narrative_layer": {
    "plot_summary": "A coherent paragraph describing WHAT happens in the scene, integrating both visual actions and spoken dialogue.",
    "conflict_type": "Select one: [Verbal Debate, Physical Combat, Internal Struggle, Investigation, Atmosphere]",
    "outcome": "How the scene ends or what changes by the end."
  },
  "emotional_layer": {
    "mood_flow": "Describe the emotional trajectory (e.g., 'Tense anticipation -> Explosive anger').",
    "character_state": "Briefly describe the emotional state of key characters."
  },
  "cinematic_layer": {
    "visual_style": "Describe lighting, composition, and color palette usage.",
    "pacing": "Fast/Slow/Accelerating/Static",
    "key_technique": "The most notable technique used (e.g., 'Frequent close-ups', 'Long tracking shot')."
  },
  "scene_metadata": {
    "semantic_location": "Inferred specific location name (e.g., 'Batcave')",
    "time_context": "Time of day inferred from context"
  },
  "character_actions": [
    {
      "character": "Character name (use standard name from Cast if available)",
      "actions": ["List of physical actions performed by this character"],
      "dialogue_summary": "Brief summary of what this character says, or null if they don't speak",
      "behavior_intent": "Inferred intent or motivation behind their actions/words"
    }
  ]
}
"""

# ================= 核心工具函数 =================

def parse_srt_timestamp(timestamp_str):
    """将 SRT 时间戳 (HH:MM:SS,mmm) 转换为秒数"""
    timestamp_str = timestamp_str.strip()
    # 支持 HH:MM:SS,mmm 或 HH:MM:SS.mmm 格式
    timestamp_str = timestamp_str.replace(',', '.')
    parts = timestamp_str.split(':')
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    return 0.0


def parse_time_string(time_str):
    """将 HH:MM:SS 格式的时间字符串转换为秒数"""
    time_str = time_str.strip()
    parts = time_str.split(':')
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    return 0.0


def parse_srt_file(srt_path):
    """
    解析 SRT 字幕文件
    返回: 字幕条目列表，每个条目包含 {index, start_sec, end_sec, speaker, text}
    """
    if not os.path.exists(srt_path):
        print(f"Warning: Subtitle file not found: {srt_path}")
        return []

    subtitles = []
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 按空行分割字幕块
    blocks = re.split(r'\n\s*\n', content.strip())

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue

        try:
            # 第一行: 序号
            index = int(lines[0].strip())

            # 第二行: 时间戳
            time_match = re.match(r'(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})', lines[1])
            if not time_match:
                continue

            start_sec = parse_srt_timestamp(time_match.group(1))
            end_sec = parse_srt_timestamp(time_match.group(2))

            # 第三行及之后: 文本内容
            text = ' '.join(lines[2:]).strip()

            # 提取说话人标识 [SPEAKER_XX] 或 [UNKNOWN]
            speaker = None
            speaker_match = re.match(r'\[([^\]]+)\]\s*(.*)', text)
            if speaker_match:
                speaker = speaker_match.group(1)
                text = speaker_match.group(2).strip()

            subtitles.append({
                'index': index,
                'start_sec': start_sec,
                'end_sec': end_sec,
                'speaker': speaker,
                'text': text
            })
        except (ValueError, IndexError):
            continue

    return subtitles


def get_subtitles_in_time_range(subtitles, start_sec, end_sec):
    """
    根据时间范围筛选字幕
    如果字幕的时间与场景时间有任何重叠，则包含该字幕
    """
    filtered = []
    for sub in subtitles:
        # 检查是否有时间重叠
        if sub['end_sec'] >= start_sec and sub['start_sec'] <= end_sec:
            filtered.append(sub)
    return filtered


def format_subtitles_for_prompt(subtitles):
    """将字幕格式化为 prompt 输入文本"""
    if not subtitles:
        return "No dialogue in this scene."

    lines = []
    for sub in subtitles:
        speaker = sub.get('speaker') or 'Unknown'
        text = sub.get('text', '')
        if text:
            lines.append(f"[{speaker}]: \"{text}\"")

    return "\n".join(lines) if lines else "No dialogue in this scene."


def parse_json_safely(text):
    """鲁棒的 JSON 解析"""
    text = text.strip()
    # 去除 markdown 代码块
    if text.startswith("```"):
        text = re.sub(r"^```json\s*|^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 尝试寻找大括号
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
        return None

class SceneAnalyzer:
    def __init__(self, subtitle_file=None):
        # 每个线程内部实例化 client 更安全，或者在此处实例化
        self.client = OpenAI(base_url=API_URL, api_key=API_KEY)
        # 预加载字幕文件
        self.subtitles = []
        if subtitle_file and os.path.exists(subtitle_file):
            self.subtitles = parse_srt_file(subtitle_file)
            print(f"Loaded {len(self.subtitles)} subtitle entries from {subtitle_file}")

    def _call_llm(self, system_prompt, user_content):
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.2, # 较低温度保证结构准确
                max_tokens=2048
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM Call Error: {e}")
            return None

    def resolve_entities(self, shots, dialogue_text="No dialogue in this scene."):
        """步骤 1: 人物实体消解（结合字幕推断名字）"""
        raw_chars_lines = []
        has_chars = False

        for s in shots:
            sid = s.get('source_filename', 'unknown')
            chars = s.get('entities', {}).get('active_characters', [])
            if chars:
                has_chars = True
                for c in chars:
                    vid = c.get('visual_id', 'Unknown')
                    app = c.get('appearance', 'No desc')
                    # 构造唯一键用于映射
                    # Key Format: Filename_VisualID
                    unique_key = f"{sid}_{vid}"
                    raw_chars_lines.append(f"Shot: {sid} | Key: {unique_key} | Visual: {vid} - {app}")

        # 如果整个场景都没人，直接返回空
        if not has_chars:
            return {}, []

        # 构建输入：视觉信息 + 对话
        visual_input = "\n".join(raw_chars_lines)
        prompt_input = (
            f"=== Visual Characters ===\n{visual_input}\n\n"
            f"=== Dialogue ===\n{dialogue_text}"
        )

        llm_output = self._call_llm(ENTITY_RESOLUTION_PROMPT, prompt_input)

        if llm_output:
            data = parse_json_safely(llm_output)
            if data:
                return data.get('mapping', {}), data.get('unique_cast', [])

        return {}, []

    def generate_summary(self, shots, mapping, unique_cast, dialogue_text="No dialogue in this scene."):
        """步骤 2: 生成解耦摘要（包含字幕对话）"""

        # A. 构建 Cast List 文本
        if unique_cast:
            cast_list_text = ", ".join([f"{c['standard_name']} ({c['visual_summary']})" for c in unique_cast])
        else:
            cast_list_text = "No distinct characters detected."

        # B. 构建 Shot Sequence 文本 (使用映射后的名字)
        formatted_shots = []
        for s in shots:
            sid = s.get('source_filename', 'unknown')

            # 处理人物名替换
            chars = s.get('entities', {}).get('active_characters', [])
            resolved_names = []
            for c in chars:
                vid = c.get('visual_id', 'Unknown')
                key = f"{sid}_{vid}"
                # 查表，如果找不到就用原名
                name = mapping.get(key, vid)
                resolved_names.append(name)

            char_str = ", ".join(resolved_names) if resolved_names else "Empty"

            # 获取其他关键信息
            st = s.get('spatio_temporal', {})
            env_tags = ", ".join(st.get('environment_tags', [])[:3])
            action = s.get('action_atoms', {}).get('primary_action', 'None')
            interact = s.get('action_atoms', {}).get('interaction_flow', '')
            func = s.get('narrative_analysis', {}).get('narrative_function', 'Unknown')
            cam = s.get('cinematography', {}).get('shot_scale', '')

            # 整合单行描述
            # 格式: [Shot ID] (Camera) Loc: ... Chars: ... Action: ... Func: ...
            line = (f"[{sid}] ({cam}) Loc: {env_tags}. "
                    f"Chars: {char_str}. "
                    f"Action: {action} ({interact}). "
                    f"Func: {func}.")
            formatted_shots.append(line)

        shot_sequence_text = "\n".join(formatted_shots)

        # C. 替换 Prompt 变量并调用（包含对话）
        final_prompt_content = (
            f"Context:\n"
            f"- Cast: {cast_list_text}\n"
            f"- Sequence:\n{shot_sequence_text}\n"
            f"- Dialogue:\n{dialogue_text}"
        )

        llm_output = self._call_llm(SCENE_ANALYSIS_PROMPT, final_prompt_content)

        if llm_output:
            return parse_json_safely(llm_output)
        return None

    def process_scene_file(self, file_path, output_path):
        """单个文件的完整处理流程"""
        if os.path.exists(output_path):
            return "Skipped"

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                scene_data = json.load(f)

            shots = scene_data.get('shots_data', [])
            if not shots:
                return "Empty Scene"

            # 0. 获取场景时间范围并筛选字幕
            dialogue_text = "No dialogue in this scene."
            time_range = scene_data.get('time_range', {})
            if time_range and self.subtitles:
                start_str = time_range.get('start_seconds', '00:00:00')
                end_str = time_range.get('end_seconds', '00:00:00')
                start_sec = parse_time_string(start_str)
                end_sec = parse_time_string(end_str)

                scene_subtitles = get_subtitles_in_time_range(self.subtitles, start_sec, end_sec)
                dialogue_text = format_subtitles_for_prompt(scene_subtitles)

                # 保存字幕到场景数据
                scene_data['dialogue'] = scene_subtitles

            # 1. 人物对齐（结合字幕推断名字）
            mapping, unique_cast = self.resolve_entities(shots, dialogue_text)
            scene_data['cast_info'] = unique_cast  # 保存一份人物表

            # 2. 深度分析（包含字幕对话）
            analysis_result = self.generate_summary(shots, mapping, unique_cast, dialogue_text)

            if analysis_result:
                scene_data['scene_analysis'] = analysis_result
            else:
                scene_data['scene_analysis'] = {"error": "LLM generation failed"}

            # 3. 保存 (可以选择删除 shots_data 以减小体积，这里保留)
            # del scene_data['shots_data']

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(scene_data, f, indent=2, ensure_ascii=False)

            return "Success"

        except Exception as e:
            return f"Error: {e}"

# ================= 主程序 =================

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. 获取所有场景文件
    scene_files = [f for f in os.listdir(SCENES_DIR) if f.endswith('.json')]
    # 自然排序 scene_0, scene_1...
    scene_files.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
    
    tasks = []
    for f in scene_files:
        in_path = os.path.join(SCENES_DIR, f)
        out_path = os.path.join(OUTPUT_DIR, f)
        tasks.append((in_path, out_path))

    print(f"Starting analysis for {len(tasks)} scenes...")
    print(f"Workers: {MAX_WORKERS} | Model: {MODEL_NAME}")

    # 2. 并发执行
    # 使用 ThreadPoolExecutor，因为 LLM 调用是 I/O 密集型
    analyzer = SceneAnalyzer(subtitle_file=SUBTITLE_FILE)  # 实例化一个即可，内部方法是无状态的(除了client)
    
    # 封装一个 helper function 方便 map 调用
    def worker_func(task_args):
        in_p, out_p = task_args
        # 这里的 analyzer.client 如果不是线程安全的，可以在这里新建 analyzer
        # OpenAI client 是线程安全的，所以复用没问题
        return analyzer.process_scene_file(in_p, out_p)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(
            executor.map(worker_func, tasks), 
            total=len(tasks),
            desc="Analyzing Scenes"
        ))

    # 3. 统计结果
    success_cnt = results.count("Success")
    skip_cnt = results.count("Skipped")
    errors = [r for r in results if r.startswith("Error")]
    
    print("\nProcessing Complete.")
    print(f"Success: {success_cnt}")
    print(f"Skipped: {skip_cnt}")
    if errors:
        print(f"Errors: {len(errors)}")
        print(errors[:5])

if __name__ == "__main__":
    main()