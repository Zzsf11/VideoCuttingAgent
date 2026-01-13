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
DATABASE_ROOT = "/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Database/VLOG_Lisbon"
FRAMES_DIR = os.path.join(DATABASE_ROOT, "frames")
SCENES_DIR = os.path.join(DATABASE_ROOT, "captions", "scenes")
OUTPUT_DIR = os.path.join(DATABASE_ROOT, "captions", "scene_summaries_video_new")
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
You are an expert Film Analyst specializing in video editing material selection. Your job is to:
1. Classify scenes by type and content quality
2. Score scene importance for video editing purposes
3. Generate character-centric narrative descriptions

[CRITICAL INSTRUCTION - SCENE CLASSIFICATION]
FIRST, scan ALL frames and classify the scene. Be especially careful about:

**Scene Types** (check the FIRST few frames carefully):
- **content**: Main story content with characters and meaningful narrative (potentially USABLE)
- **studio_logo**: Production company logos (Warner Bros., DC Comics, Legendary, Syncopy, etc.) - NOT usable
- **title_card**: Movie title cards, chapter titles, or stylized text screens - NOT usable
- **credits**: Opening/ending credits, cast/crew text overlays - NOT usable
- **transition**: Pure black screens, fade transitions, or abstract non-narrative visuals - NOT usable
- **mixed**: Scene STARTS with logo/credits/title but transitions to content - PARTIALLY usable (note the transition point)

**IMPORTANT**: If the FIRST frame shows a logo (e.g., "Warner Bros. Pictures"), the scene_type should be "studio_logo" or "mixed", NOT "content"!

[CRITICAL INSTRUCTION - IMPORTANCE SCORING]
Score each scene's editing value (0-5). Consider TWO dimensions:

**Dimension A - Emotional Intensity** (high emotion = higher score):
- **Intense emotions**: Fighting, kissing, crying, screaming, rage, fear, despair, joy, reunion
- **Physical action**: Combat, chase, explosion, falling, running
- **Intimate moments**: Confession, embrace, death of loved one, betrayal revelation

**Dimension B - Visual Quality** (cinematic shots = higher score):
- **Striking compositions**: Beautiful close-ups, dramatic wide shots, silhouettes
- **Atmospheric shots**: Sunset/sunrise, rain, fog, city skyline at night
- **Dynamic camera**: Sweeping crane shots, intense tracking shots, slow-motion

**Scoring Guide**:
**5 - Essential**:
   - Core plot events with HIGH emotional intensity (murder, climactic fight, passionate kiss, tragic death)
   - Character-defining moments with strong emotion (rage outburst, breakdown crying, triumphant victory)
   - Visually stunning + emotionally powerful combinations
   - Examples: Parent's murder scene, final battle, romantic climax, hero's sacrifice

**4 - Very Important**:
   - Key plot with moderate emotion OR high emotion with less plot significance
   - Beautifully shot emotional moments (tearful goodbye, tense confrontation)
   - Impressive action sequences, dramatic reveals
   - Examples: Intense dialogue confrontation, chase scene, emotional reunion

**3 - Moderately Important**:
   - Supporting scenes with some emotional content or good visual quality
   - Character interactions with tension or warmth
   - Well-composed establishing shots of important locations
   - Examples: Planning scene with conflict, scenic wide shot of Gotham

**2 - Low Importance**:
   - Neutral emotional content, standard cinematography
   - Pure exposition without tension, transitional moments
   - Flashbacks without strong emotion or new revelation
   - Examples: Walking through hallway, casual conversation

**1 - Minimal Value**:
   - Flat emotional content, poor or unremarkable visuals
   - Filler, repetitive, or redundant scenes
   - Examples: Extended static shots, repeated information

**0 - Not Usable**: Non-content (logos, credits, black screens)

**BOOST scores for**: Crying, fighting, kissing, screaming, explosions, beautiful landscapes, dramatic lighting, slow-motion, close-ups showing intense emotion

**Flashback/Memory Scenes**: These are often 2-3 importance unless they reveal critical backstory.
Childhood scenes showing young versions of characters typically score 2-3 unless they depict traumatic/formative events (like parents' death = 5).

[CRITICAL INSTRUCTION - NARRATIVE]
For "content" or "mixed" scenes, write a COHERENT STORY:
1. Use CHARACTER NAMES as sentence subjects (e.g., "Bruce watches in horror" NOT "A boy is shown watching")
2. Tell the COMPLETE EVENT from beginning to end
3. Connect cause and effect (what triggers what, and what are the consequences)
4. Integrate dialogue with visual actions
5. For flashbacks: clearly indicate "In a flashback/memory, young Bruce..."

[Pattern Recognition]
- Gun drawn + Shot fired + Person falls + Child cries = MURDER scene (importance: 5)
- Formal attire + Child + Parents = FAMILY scene
- Young version of main character playing = CHILDHOOD FLASHBACK (importance: 2-3 unless traumatic)
- If dialogue mentions names, USE those names for the characters
- Logo on screen + no characters + abstract background = studio_logo (importance: 0)
- Text overlay listing names/roles = credits (importance: 0)
- Movie title text on screen = title_card (importance: 0)

[Input]
- **Known Characters**: {CHARACTERS}
- **Dialogue**: {DIALOGUE}
- **Frames**: Sequential frames from the scene (in chronological order)

[Output Schema - JSON]
{
  "scene_classification": {
    "scene_type": "content/studio_logo/title_card/credits/transition/mixed",
    "is_usable": true/false,
    "importance_score": 0-5,
    "unusable_reason": "null if fully usable, otherwise explain: 'Studio logo (Warner Bros.)', 'Childhood flashback with low narrative value', 'Opening credits', etc.",
    "contains_non_content": "If mixed scene, describe what non-content elements exist (e.g., 'First 21 seconds contain Warner Bros. logo')"
  },
  "scene_summary": {
    "narrative": "3-5 sentence coherent story using character names. For non-content: brief description. For flashbacks: clearly indicate it's a memory/flashback.",
    "key_event": "Single most important event. For non-content: 'Studio logo display'. For flashbacks: 'Childhood memory of X'",
    "location": "Specific location",
    "time": "Day/Night",
    "scene_function": "plot_progression/character_development/flashback/exposition/action/emotional_beat/establishment/transition"
  },
  "narrative_elements": {
    "conflict": "Type of conflict (or 'None' for non-narrative scenes)",
    "mood_arc": "Emotional progression",
    "cause_effect": "What triggers the event and its consequences",
    "editing_notes": "Specific notes for video editors (e.g., 'Skip first 21 seconds of logo', 'Good establishing shot', 'Contains key dialogue about X')"
  }
}
"""

# 场景级视频理解 Prompt（Vlog旅行风景版）
VLOG_SCENE_CAPTION_PROMPT = """
[Role]
You are an expert Travel Vlog Analyst specializing in video editing material selection. Your job is to:
1. Classify scenes by type and content quality
2. Score scene importance for travel vlog editing purposes
3. Generate journey-centric narrative descriptions focusing on landscapes, experiences, and creator expression

[CRITICAL INSTRUCTION - SCENE CLASSIFICATION]
FIRST, scan ALL frames and classify the scene. Be especially careful about:

**Scene Types** (check the FIRST few frames carefully):
- **scenery**: Beautiful landscapes, natural wonders, cityscapes, architectural landmarks (HIGH VALUE)
- **journey**: Travel process - walking, driving, flying, sailing, exploring new places (HIGH VALUE)
- **creator_moment**: Vlogger speaking to camera, expressing thoughts, sharing experiences (HIGH VALUE)
- **local_culture**: Local food, markets, festivals, people, traditions (MODERATE-HIGH VALUE)
- **b_roll**: Atmospheric shots, detail shots, ambient footage without clear subject (MODERATE VALUE)
- **transition**: Pure black screens, fade transitions, or abstract non-narrative visuals - NOT usable
- **technical_issue**: Blurry footage, accidental recording, equipment malfunction - NOT usable
- **mixed**: Scene contains multiple types above - note the composition

**IMPORTANT**: Prioritize scenes that capture the ESSENCE of travel - the beauty of discovery, the emotion of experiencing new places, and authentic creator expression!

[CRITICAL INSTRUCTION - IMPORTANCE SCORING]
Score each scene's editing value (0-5). Consider THREE dimensions:

**Dimension A - Visual Beauty** (stunning visuals = higher score):
- **Natural landscapes**: Mountains, oceans, sunsets, forests, lakes, waterfalls, starry skies
- **Urban aesthetics**: Skylines, historic architecture, charming streets, night cityscapes
- **Atmospheric moments**: Golden hour light, morning mist, rain, snow, dramatic clouds
- **Unique perspectives**: Drone shots, elevated viewpoints, reflections, silhouettes

**Dimension B - Journey Authenticity** (genuine travel experience = higher score):
- **First encounters**: Arriving at a new place, first glimpse of landmark, initial reactions
- **Immersive moments**: Walking through local markets, tasting local food, interacting with locals
- **Adventure activities**: Hiking, swimming, cycling, exploring hidden spots
- **Transit poetry**: Window views from trains/planes, road trip scenery, boat rides

**Dimension C - Creator Expression** (emotional connection = higher score):
- **Genuine reactions**: Awe, excitement, peace, wonder, gratitude
- **Personal reflections**: Thoughts about the journey, life insights, cultural observations
- **Storytelling moments**: Sharing history, explaining context, narrating experiences
- **Vulnerable moments**: Challenges faced, lessons learned, honest feelings

**Scoring Guide**:
**5 - Essential (Must Include)**:
   - Breathtaking landscape shots with exceptional composition (sunrise over mountains, ocean panorama)
   - Iconic landmark reveals with emotional creator reaction
   - Powerful creator monologue with beautiful backdrop
   - Once-in-a-lifetime moments (aurora, wildlife encounter, perfect sunset)
   - Examples: First view of Eiffel Tower at golden hour, standing atop a mountain summit, emotional reflection at journey's end

**4 - Very Important**:
   - Beautiful scenery with good lighting and composition
   - Meaningful journey moments showing exploration and discovery
   - Engaging creator content with authentic expression
   - Unique local experiences well captured
   - Examples: Walking through charming old town streets, tasting famous local dish, scenic train window views

**3 - Moderately Important**:
   - Pleasant scenery, standard tourist spots well-shot
   - Transitional journey moments that maintain narrative flow
   - Creator content with moderate engagement value
   - Cultural moments that add context
   - Examples: Hotel room tour with nice view, walking to destination, explaining travel plans

**2 - Low Importance**:
   - Average visuals without distinctive beauty
   - Repetitive travel footage (similar walking shots, routine activities)
   - Filler content without strong narrative purpose
   - Examples: Packing luggage, waiting at airport, generic street walking

**1 - Minimal Value**:
   - Poor lighting, shaky footage, unflattering compositions
   - Extended footage without visual interest or narrative purpose
   - Redundant content that doesn't add new information
   - Examples: Long static shots of nothing particular, repeated similar angles

**0 - Not Usable**: Technical issues, accidental recordings, black screens, completely blurry footage

**BOOST scores for**:
- Golden hour/blue hour lighting
- Dramatic weather (clouds, mist, rain adding atmosphere)
- Drone/aerial perspectives
- Genuine emotional reactions from creator
- Unique angles of famous landmarks
- Local life moments (not staged)
- Peaceful/meditative sequences

[CRITICAL INSTRUCTION - NARRATIVE]
For usable scenes, write a JOURNEY-FOCUSED STORY:
1. Describe the VISUAL BEAUTY in evocative language (colors, light, atmosphere)
2. Capture the TRAVEL CONTEXT (where, when, why this matters in the journey)
3. Note CREATOR PRESENCE and expression if visible
4. Convey the MOOD and feeling the scene evokes
5. Identify EDITING POTENTIAL (what makes this shot valuable for the final video)

[Pattern Recognition]
- Wide landscape + golden light + no people = SCENIC ESTABLISHING shot (importance: 4-5)
- Creator facing camera + speaking + nice background = VLOG MOMENT (importance: 3-5 based on content)
- Moving vehicle + window view + passing scenery = JOURNEY TRANSIT (importance: 2-4)
- Food close-up + steam/texture + local setting = CULINARY MOMENT (importance: 3-4)
- Crowd + decorations + music = LOCAL FESTIVAL/EVENT (importance: 3-5)
- Sunrise/sunset + silhouette + landscape = GOLDEN MOMENT (importance: 4-5)
- Creator hiking/walking + scenic path + nature = ADVENTURE SEQUENCE (importance: 3-5)

[Input]
- **Location Context**: {LOCATION}
- **Frames**: Sequential frames from the scene (in chronological order)

[Output Schema - JSON]
{
  "scene_classification": {
    "scene_type": "scenery/journey/creator_moment/local_culture/b_roll/transition/technical_issue/mixed",
    "is_usable": true/false,
    "importance_score": 0-5,
    "unusable_reason": "null if fully usable, otherwise explain: 'Blurry footage', 'Accidental recording', etc.",
    "mixed_composition": "If mixed scene, describe components (e.g., 'Opens with scenery, transitions to creator talking')"
  },
  "visual_analysis": {
    "landscape_type": "mountain/ocean/forest/urban/rural/desert/lake/river/architectural/mixed/indoor/none",
    "lighting_quality": "golden_hour/blue_hour/bright_daylight/overcast/night/artificial/dramatic/flat",
    "composition_notes": "Describe framing, perspective, visual elements",
    "color_palette": "Dominant colors and mood they create",
    "camera_movement": "static/pan/tracking/handheld/drone/gimbal_smooth"
  },
  "journey_context": {
    "narrative": "3-5 sentence evocative description capturing the scene's beauty, travel context, and emotional resonance",
    "key_moment": "Single most impactful visual or emotional beat",
    "location_specificity": "General area and specific spot if identifiable",
    "time_of_day": "Dawn/Morning/Midday/Afternoon/Golden_hour/Dusk/Night",
    "weather_atmosphere": "Clear/Cloudy/Rainy/Misty/Snowy/Dramatic/Calm"
  },
  "creator_presence": {
    "visibility": "on_camera/voice_only/not_present",
    "expression_type": "narrating/reacting/reflecting/explaining/silent/none",
    "emotional_tone": "excited/peaceful/awed/contemplative/joyful/curious/grateful/none",
    "dialogue_summary": "Key points if speaking, null if silent"
  },
  "editing_potential": {
    "suggested_use": "opening/closing/transition/highlight/b_roll/montage/standalone",
    "music_pairing": "upbeat/cinematic/peaceful/emotional/adventurous/none_needed",
    "editing_notes": "Specific notes for video editors (e.g., 'Perfect for slow-motion', 'Great with ambient sound', 'Ideal montage material')"
  }
}
"""

# 根据配置选择使用的 Prompt
def get_scene_caption_prompt():
    """根据 config.SCENE_PROMPT_TYPE 返回对应的 prompt"""
    if config.SCENE_PROMPT_TYPE == "vlog":
        return VLOG_SCENE_CAPTION_PROMPT
    else:  # 默认使用 film prompt
        return SCENE_VIDEO_CAPTION_PROMPT

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
        for frame in frames:
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

    def generate_caption(self, frames: List[Image.Image], dialogue: str, char_info: Dict, max_retries: int = 3) -> Optional[Dict]:
        """生成场景描述，验证必需字段"""
        # 根据配置选择 prompt
        base_prompt = get_scene_caption_prompt()

        # 格式化人物/位置信息
        if config.SCENE_PROMPT_TYPE == "vlog":
            # Vlog 模式：使用 LOCATION 和 DIALOGUE
            location_text = "Unknown location"  # 可从 scene_data 中获取
            prompt = base_prompt.replace("{LOCATION}", location_text).replace("{DIALOGUE}", dialogue)
            content = self._build_content(frames, [f"Location:\n{location_text}", f"\nCreator Speech:\n{dialogue}"])
        else:
            # Film 模式：使用 CHARACTERS 和 DIALOGUE
            if char_info and 'characters' in char_info:
                char_text = "\n".join([
                    f"- {c.get('name', 'Unknown')}: {c.get('appearance', 'No desc')}"
                    for c in char_info['characters']
                ])
            else:
                char_text = "No characters identified."
            prompt = base_prompt.replace("{CHARACTERS}", char_text).replace("{DIALOGUE}", dialogue)
            content = self._build_content(frames, [f"Characters:\n{char_text}", f"\nDialogue:\n{dialogue}"])

        for attempt in range(max_retries):
            result = self._call_vlm(prompt, content, max_tokens=4096)
            parsed = parse_json_safely(result) if result else None

            # 验证必需字段
            if parsed and 'scene_classification' in parsed:
                classification = parsed['scene_classification']
                # 确保有 importance_score
                if 'importance_score' not in classification:
                    # 根据 scene_type 推断默认值
                    scene_type = classification.get('scene_type', 'content')
                    # Film 模式的不可用类型
                    film_unusable_types = ['studio_logo', 'title_card', 'credits', 'transition']
                    # Vlog 模式的不可用类型
                    vlog_unusable_types = ['transition', 'technical_issue']

                    if config.SCENE_PROMPT_TYPE == "vlog":
                        if scene_type in vlog_unusable_types:
                            classification['importance_score'] = 0
                            classification['is_usable'] = False
                        elif scene_type == 'mixed':
                            classification['importance_score'] = 2
                            classification['is_usable'] = True
                        else:
                            classification['importance_score'] = 3  # 默认中等重要性
                            classification['is_usable'] = True
                    else:
                        if scene_type in film_unusable_types:
                            classification['importance_score'] = 0
                            classification['is_usable'] = False
                        elif scene_type == 'mixed':
                            classification['importance_score'] = 2
                            classification['is_usable'] = True
                        else:
                            classification['importance_score'] = 3  # 默认中等重要性
                            classification['is_usable'] = True

                # 确保 is_usable 与 importance_score 一致
                if classification.get('importance_score', 0) == 0:
                    classification['is_usable'] = False

                return parsed

            if attempt < max_retries - 1:
                print(f"Warning: Output missing required fields, retrying ({attempt + 1}/{max_retries})...")

        print(f"Error: Failed to get valid output after {max_retries} attempts")
        return parsed  # 返回最后一次的结果，即使不完整

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
    print(f"  Prompt type: {config.SCENE_PROMPT_TYPE} ({'Travel Vlog' if config.SCENE_PROMPT_TYPE == 'vlog' else 'Film/TV'})")
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
