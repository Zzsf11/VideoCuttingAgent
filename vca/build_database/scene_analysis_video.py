"""
场景级视频 Caption 脚本
直接使用已提取的帧进行场景理解，生成以人物为主体的连贯叙事描述
"""
import os
import sys
import json
import re
import base64
import concurrent.futures
from io import BytesIO
from typing import List, Dict, Optional

from PIL import Image
from openai import OpenAI
from tqdm import tqdm

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vca import config

# ================= 配置 =================
# 数据路径
DATABASE_ROOT = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Database/Batman_Begins_2005_1080p_BluRay_x264_YIFY"
FRAMES_DIR = os.path.join(DATABASE_ROOT, "frames")
SCENES_DIR = os.path.join(DATABASE_ROOT, "captions", "scenes")
OUTPUT_DIR = os.path.join(DATABASE_ROOT, "captions", "scene_summaries_video")
SUBTITLE_FILE = os.path.join(DATABASE_ROOT, "subtitles_with_characters.srt")

# 模型配置（从 config 读取）
API_URL = config.VLLM_ENDPOINT.replace("/chat/completions", "")
API_KEY = "EMPTY"
MODEL_NAME = config.VIDEO_ANALYSIS_MODEL

# 帧采样配置
VIDEO_FPS = config.VIDEO_FPS  # 2 fps
MAX_FRAMES_PER_SCENE = 128  # 每个场景最多采样帧数
MIN_FRAMES_PER_SCENE = 6   # 每个场景最少帧数

MAX_WORKERS = 8  # 并发数
# =======================================

# ================= PROMPTS =================

# 场景级视频理解 Prompt（以人物为主体）
SCENE_VIDEO_CAPTION_PROMPT = """
[Role]
You are an expert Film Analyst. Analyze the provided scene frames and generate a CHARACTER-CENTRIC narrative description.

[CRITICAL INSTRUCTION]
You must write a COHERENT STORY, not describe individual frames. Your description should:
1. Use CHARACTER NAMES as sentence subjects (e.g., "Bruce watches in horror" NOT "A boy is shown watching")
2. Tell the COMPLETE EVENT from beginning to end
3. Connect cause and effect (what triggers what, and what are the consequences)
4. Integrate dialogue with visual actions

[Pattern Recognition]
- Gun drawn + Shot fired + Person falls + Child cries = MURDER scene - describe it as "X shoots Y, killing them"
- Formal attire + Child + Parents = FAMILY
- If dialogue mentions names, USE those names for the characters

[Input]
- **Known Characters**: {CHARACTERS}
- **Dialogue**: {DIALOGUE}
- **Frames**: Sequential frames from the scene (in chronological order)

[Output Schema - JSON]
{
  "scene_summary": {
    "narrative": "3-5 sentence coherent story using character names. Example: 'Thomas Wayne, his wife Martha, and their son Bruce leave the opera through a dark alley. A mugger appears and demands their valuables at gunpoint. When Thomas tries to protect his family, the mugger shoots him dead. Martha screams and is also shot. Young Bruce kneels beside his dying parents, traumatized forever by witnessing their murder.'",
    "key_event": "Single most important event (e.g., 'Murder of Thomas and Martha Wayne')",
    "location": "Specific location",
    "time": "Day/Night"
  },
  "characters": [
    {
      "name": "Character name",
      "role": "Protagonist/Antagonist/Victim/Witness",
      "description": "Visual appearance",
      "actions": "Key actions (active voice)",
      "dialogue": "What they say (if any)",
      "fate": "Outcome (killed/escapes/traumatized/etc.)"
    }
  ],
  "narrative_elements": {
    "conflict": "Type of conflict",
    "mood_arc": "Emotional progression",
    "cause_effect": "What triggers the event and its consequences"
  }
}
"""

# 人物识别 Prompt
CHARACTER_ID_PROMPT = """
[Role]
Identify all characters in these scene frames using dialogue and visual cues.

[Input]
- **Previously Known Characters**: {KNOWN_CHARS}
- **Dialogue**: {DIALOGUE}

[Task]
1. Match visual appearances to any known characters
2. Infer names from dialogue (e.g., "Bruce, come here" → the boy is Bruce)
3. Identify relationships (formal dress family = parents and child)

[Output - JSON]
{
  "characters": [
    {
      "name": "Inferred name or descriptive (e.g., 'Bruce Wayne', 'The Mugger')",
      "confidence": "high/medium/low",
      "source": "dialogue/visual/context",
      "appearance": "Visual description",
      "frames_seen": [1, 2, 3]
    }
  ],
  "relationships": [
    {"person1": "Name1", "person2": "Name2", "relation": "parent-child/spouses/etc."}
  ],
  "names_in_dialogue": ["All names mentioned in dialogue"]
}
"""

# ================= 工具函数 =================

def parse_srt_timestamp(ts: str) -> float:
    """SRT 时间戳转秒数"""
    ts = ts.strip().replace(',', '.')
    parts = ts.split(':')
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return 0.0


def parse_time_string(ts: str) -> float:
    """HH:MM:SS 转秒数"""
    parts = ts.strip().split(':')
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return 0.0


def parse_srt_file(srt_path: str) -> List[Dict]:
    """解析 SRT 字幕"""
    if not os.path.exists(srt_path):
        return []

    subtitles = []
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    for block in re.split(r'\n\s*\n', content.strip()):
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        try:
            time_match = re.match(r'(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})', lines[1])
            if not time_match:
                continue

            text = ' '.join(lines[2:]).strip()
            speaker = None
            speaker_match = re.match(r'\[([^\]]+)\]\s*(.*)', text)
            if speaker_match:
                speaker = speaker_match.group(1)
                text = speaker_match.group(2).strip()

            subtitles.append({
                'start_sec': parse_srt_timestamp(time_match.group(1)),
                'end_sec': parse_srt_timestamp(time_match.group(2)),
                'speaker': speaker,
                'text': text
            })
        except:
            continue
    return subtitles


def get_subtitles_in_range(subtitles: List[Dict], start: float, end: float) -> List[Dict]:
    """筛选时间范围内的字幕"""
    return [s for s in subtitles if s['end_sec'] >= start and s['start_sec'] <= end]


def format_subtitles(subtitles: List[Dict]) -> str:
    """格式化字幕"""
    if not subtitles:
        return "No dialogue."
    lines = [f"[{s.get('speaker', 'Unknown')}]: \"{s['text']}\"" for s in subtitles if s.get('text')]
    return "\n".join(lines) if lines else "No dialogue."


def parse_json_safely(text: str) -> Optional[Dict]:
    """安全解析 JSON"""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```json\s*|^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return None


def load_scene_frames(frames_dir: str, frame_range: List[int],
                      max_frames: int = MAX_FRAMES_PER_SCENE,
                      min_frames: int = MIN_FRAMES_PER_SCENE) -> List[Image.Image]:
    """
    从已提取的帧目录加载场景帧
    frame_range: [start_frame, end_frame]
    """
    start_frame, end_frame = frame_range
    total_scene_frames = end_frame - start_frame + 1

    # 计算采样数量
    if total_scene_frames <= min_frames:
        num_samples = total_scene_frames
    elif total_scene_frames <= max_frames:
        num_samples = min(total_scene_frames, max_frames)
    else:
        num_samples = max_frames

    num_samples = max(min_frames, min(num_samples, total_scene_frames))

    # 均匀采样
    if num_samples >= total_scene_frames:
        sample_indices = list(range(start_frame, end_frame + 1))
    else:
        step = total_scene_frames / num_samples
        sample_indices = [int(start_frame + i * step) for i in range(num_samples)]

    # 加载帧
    frames = []
    for idx in sample_indices:
        frame_path = os.path.join(frames_dir, f"frame_{idx:06d}.png")
        if os.path.exists(frame_path):
            try:
                img = Image.open(frame_path).convert('RGB')
                frames.append(img)
            except Exception as e:
                print(f"Warning: Failed to load {frame_path}: {e}")

    return frames


def image_to_base64(img: Image.Image) -> str:
    """PIL Image 转 base64"""
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def extract_known_characters(shots_data: List[Dict]) -> str:
    """从镜头数据提取已知人物"""
    chars = {}
    for shot in shots_data:
        for c in shot.get('entities', {}).get('active_characters', []):
            vid = c.get('visual_id', 'Unknown')
            if vid not in chars:
                chars[vid] = c.get('appearance', 'No description')

    if not chars:
        return "No prior character information."

    return "\n".join([f"- {vid}: {desc}" for vid, desc in chars.items()])


# ================= 核心类 =================

class SceneVideoAnalyzer:
    def __init__(self, frames_dir: str, subtitle_file: Optional[str] = None):
        self.frames_dir = frames_dir
        self.client = OpenAI(base_url=API_URL, api_key=API_KEY)
        self.subtitles = parse_srt_file(subtitle_file) if subtitle_file else []
        if self.subtitles:
            print(f"Loaded {len(self.subtitles)} subtitle entries")

    def _call_vlm(self, system_prompt: str, content: List[Dict], max_tokens: int = 4096) -> Optional[str]:
        """调用 VLM"""
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content}
                ],
                temperature=0.3,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"VLM Error: {e}")
            return None

    def _build_content(self, frames: List[Image.Image], text_parts: List[str]) -> List[Dict]:
        """构建 VLM 输入"""
        content = [{"type": "text", "text": "\n".join(text_parts)}]

        content.append({"type": "text", "text": f"\n=== Scene Frames ({len(frames)} frames) ==="})
        for i, frame in enumerate(frames):
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64(frame)}"}
            })

        return content

    def identify_characters(self, frames: List[Image.Image], dialogue: str, known_chars: str) -> Optional[Dict]:
        """识别人物"""
        prompt = CHARACTER_ID_PROMPT.replace("{KNOWN_CHARS}", known_chars).replace("{DIALOGUE}", dialogue)
        content = self._build_content(frames, [f"Known Characters:\n{known_chars}", f"\nDialogue:\n{dialogue}"])

        result = self._call_vlm(prompt, content, max_tokens=2048)
        return parse_json_safely(result) if result else None

    def generate_caption(self, frames: List[Image.Image], dialogue: str, char_info: Dict) -> Optional[Dict]:
        """生成场景描述"""
        # 格式化人物信息
        if char_info and 'characters' in char_info:
            char_text = "\n".join([
                f"- {c.get('name', 'Unknown')}: {c.get('appearance', 'No desc')}"
                for c in char_info['characters']
            ])
        else:
            char_text = "No characters identified."

        prompt = SCENE_VIDEO_CAPTION_PROMPT.replace("{CHARACTERS}", char_text).replace("{DIALOGUE}", dialogue)
        content = self._build_content(frames, [f"Characters:\n{char_text}", f"\nDialogue:\n{dialogue}"])

        result = self._call_vlm(prompt, content, max_tokens=4096)
        return parse_json_safely(result) if result else None

    def process_scene(self, scene_data: Dict) -> Dict:
        """处理单个场景"""
        # 获取帧范围
        frame_range = scene_data.get('frame_range', [0, 0])

        # 加载帧
        frames = load_scene_frames(self.frames_dir, frame_range)
        if not frames:
            return {"error": "No frames loaded"}

        # 获取时间范围和字幕
        time_range = scene_data.get('time_range', {})
        start_sec = parse_time_string(time_range.get('start_seconds', '00:00:00'))
        end_sec = parse_time_string(time_range.get('end_seconds', '00:00:00'))

        scene_subs = get_subtitles_in_range(self.subtitles, start_sec, end_sec)
        dialogue = format_subtitles(scene_subs)

        # 获取已知人物
        known_chars = extract_known_characters(scene_data.get('shots_data', []))

        # 步骤 1: 人物识别
        char_info = self.identify_characters(frames, dialogue, known_chars)

        # 步骤 2: 场景描述
        caption = self.generate_caption(frames, dialogue, char_info or {})

        return {
            'character_identification': char_info,
            'scene_caption': caption,
            'dialogue': scene_subs,
            'frames_used': len(frames),
            'frame_range': frame_range
        }

    def process_file(self, in_path: str, out_path: str) -> str:
        """处理单个文件"""
        if os.path.exists(out_path):
            return "Skipped"

        try:
            with open(in_path, 'r', encoding='utf-8') as f:
                scene_data = json.load(f)

            result = self.process_scene(scene_data)
            scene_data['video_analysis'] = result

            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(scene_data, f, indent=2, ensure_ascii=False)

            return "Success"
        except Exception as e:
            return f"Error: {e}"


# ================= 主程序 =================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(FRAMES_DIR):
        print(f"Error: Frames directory not found: {FRAMES_DIR}")
        return

    # 获取场景文件
    scene_files = sorted(
        [f for f in os.listdir(SCENES_DIR) if f.endswith('.json')],
        key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)]
    )

    tasks = [(os.path.join(SCENES_DIR, f), os.path.join(OUTPUT_DIR, f)) for f in scene_files]

    print(f"Scene Video Analysis")
    print(f"  Scenes: {len(tasks)}")
    print(f"  Frames dir: {FRAMES_DIR}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Max frames/scene: {MAX_FRAMES_PER_SCENE}")

    analyzer = SceneVideoAnalyzer(FRAMES_DIR, SUBTITLE_FILE)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(
            executor.map(lambda t: analyzer.process_file(t[0], t[1]), tasks),
            total=len(tasks),
            desc="Processing"
        ))

    print(f"\nDone: {results.count('Success')} success, {results.count('Skipped')} skipped")
    errors = [r for r in results if r.startswith("Error")]
    if errors:
        print(f"Errors: {len(errors)}")
        for e in errors[:3]:
            print(f"  {e}")


if __name__ == "__main__":
    main()
