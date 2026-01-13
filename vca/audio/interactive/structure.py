import soundfile as sf
import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict

# Cache directory for persistent storage
CACHE_DIR = Path.home() / ".cache" / "vca_audio_structure"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

class StructureGenerator:
    """
    Generate audio structure (Level 1 sections) using Omni AI model.
    Separates structure analysis from keypoint-level caption generation.
    Includes caching for structure analysis results.
    """

    def __init__(self):
        self._model = None
        self._processor = None
        # Keep in-memory cache for current session
        self._structure_cache = {}
        self._max_cache_size = 10  # Keep last 10 results in memory
    
    def _get_cache_filename(self, cache_key):
        """Generate cache filename from cache key"""
        try:
            # Create a hash from the cache key for filename
            key_str = str(cache_key)
            key_hash = hashlib.md5(key_str.encode()).hexdigest()
            return CACHE_DIR / f"structure_{key_hash}.json"
        except:
            return None
    
    def _load_from_disk_cache(self, cache_key):
        """Load cached result from disk"""
        try:
            cache_file = self._get_cache_filename(cache_key)
            print(f"  Checking disk cache: {cache_file}")
            if cache_file and cache_file.exists():
                print(f"  ✓ Cache file exists, loading...")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Verify the key matches
                    stored_key = data.get('cache_key')
                    current_key = list(cache_key)
                    print(f"  Stored key: {stored_key}")
                    print(f"  Current key: {current_key}")
                    if stored_key == current_key:
                        print(f"  ✓ Keys match!")
                        return data.get('result')
                    else:
                        print(f"  ✗ Keys don't match")
            else:
                print(f"  ✗ Cache file does not exist")
        except Exception as e:
            print(f"    ⚠️ Failed to load disk cache: {e}")
            import traceback
            traceback.print_exc()
        return None
    
    def _save_to_disk_cache(self, cache_key, result):
        """Save result to disk cache"""
        try:
            cache_file = self._get_cache_filename(cache_key)
            print(f"  Saving to disk cache: {cache_file}")
            if cache_file:
                data = {
                    'cache_key': list(cache_key),
                    'result': result
                }
                print(f"  Cache key being saved: {list(cache_key)}")
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"  ✓ Successfully saved to disk")
                return True
        except Exception as e:
            print(f"    ⚠️ Failed to save disk cache: {e}")
            import traceback
            traceback.print_exc()
        return False
    
    def _get_cache_key(self, audio_path: str, temperature: float, top_p: float, max_tokens: int):
        """Generate cache key based on audio file and parameters"""
        try:
            # Handle different audio_path formats from Gradio
            print(f"  Generating cache key for: {audio_path} (type: {type(audio_path)})")
            if isinstance(audio_path, tuple):
                audio_path = audio_path[0]
                print(f"    → Extracted from tuple: {audio_path}")
            elif hasattr(audio_path, 'name'):
                audio_path = audio_path.name
                print(f"    → Extracted from object: {audio_path}")
            
            # Normalize path to absolute path to ensure consistency
            abs_path = os.path.abspath(audio_path)
            print(f"    → Absolute path: {abs_path}")
            
            # Get file modification time
            mtime = os.path.getmtime(abs_path)
            print(f"    → mtime: {mtime}")
            
            # Round floats to reduce precision issues
            temp_rounded = round(float(temperature), 2)
            top_p_rounded = round(float(top_p), 2)
            max_tokens_int = int(max_tokens)
            
            cache_key = (abs_path, mtime, temp_rounded, top_p_rounded, max_tokens_int)
            print(f"    → Final cache key: {cache_key}")
            return cache_key
        except Exception as e:
            print(f"    ⚠️ Failed to generate cache key: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _load_model(self, model_path: str = None):
        """Load AI model for structure analysis"""
        if self._model is not None and self._processor is not None:
            return self._model, self._processor

        try:
            from vca.audio.audio_caption_madmom import load_model_and_processor
            from vca import config

            if model_path is None:
                model_path = getattr(config, 'AUDIO_ANALYSIS_MODEL', config.VIDEO_ANALYSIS_MODEL)

            self._model, self._processor = load_model_and_processor(
                model_path=model_path,
                use_flash_attn2=True
            )
            return self._model, self._processor
        except Exception as e:
            print(f"Failed to load model: {e}")
            return None, None

    def analyze_structure(
        self,
        audio_path: str,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 2048,
    ) -> Dict:
        """
        Analyze audio structure and return Level 1 sections.
        Results are cached based on audio file path, mtime, and parameters.

        Args:
            audio_path: Path to audio file
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate

        Returns:
            Dict with 'success', 'sections', 'summary', 'duration'
        """
        # Check cache first
        cache_key = self._get_cache_key(audio_path, temperature, top_p, max_tokens)
        
        # Debug info
        print(f"\n{'='*60}")
        print("STRUCTURE ANALYSIS")
        print(f"{'='*60}")
        print(f"Cache key: {cache_key}")
        print(f"Memory cache size: {len(self._structure_cache)}")
        
        # Try memory cache first
        if cache_key and cache_key in self._structure_cache:
            print("✓ Using cached result (from memory)")
            cached_result = self._structure_cache[cache_key]
            return {
                'success': True,
                'sections': cached_result['sections'],
                'summary': cached_result['summary'],
                'duration': cached_result['duration'],
            }
        
        # Try disk cache
        if cache_key:
            disk_result = self._load_from_disk_cache(cache_key)
            if disk_result:
                print("✓ Using cached result (from disk)")
                # Also store in memory cache for faster access
                self._structure_cache[cache_key] = disk_result
                return {
                    'success': True,
                    'sections': disk_result['sections'],
                    'summary': disk_result['summary'],
                    'duration': disk_result['duration'],
                }
        
        print("✗ Cache miss, running new analysis")
        
        try:
            from vca.audio.audio_caption_madmom import (
                generate_overall_analysis,
                extract_json_from_text,
                seconds_to_mmss,
            )
        except ImportError as e:
            return {'success': False, 'error': f'Import error: {e}'}

        # Load model
        model, processor = self._load_model()
        if model is None:
            return {'success': False, 'error': 'Failed to load AI model'}

        # Get audio duration
        try:
            info = sf.info(audio_path)
            audio_duration = info.duration
        except Exception as e:
            return {'success': False, 'error': f'Failed to read audio info: {e}'}

        print(f"\n{'='*60}")
        print("STRUCTURE ANALYSIS (Level 1 Sections)")
        print(f"{'='*60}")
        print(f"Audio: {audio_path}")
        print(f"Duration: {audio_duration:.2f}s")

        # Generate overall analysis
        print("\nGenerating structure analysis with AI...")
        overall_text = generate_overall_analysis(
            audio_path=audio_path,
            model=model,
            processor=processor,
            temperature=temperature,
            top_p=top_p,
            top_k=20,
            max_tokens=max_tokens,
            audio_duration=audio_duration,
        )

        overall_json = extract_json_from_text(overall_text)
        if overall_json:
            sections = overall_json.get("sections", [])
            summary = overall_json.get("summary", "")
            print(f"✓ Found {len(sections)} Level 1 sections")
        else:
            summary = overall_text
            sections = [{
                "name": "Full Audio",
                "description": "",
                "Start_Time": "00:00",
                "End_Time": seconds_to_mmss(audio_duration)
            }]
            print("⚠ Using default section (full audio)")

        # Cache the result (both in memory and disk)
        if cache_key:
            cache_data = {
                'sections': sections,
                'summary': summary,
                'duration': audio_duration,
            }
            
            # Save to memory cache
            self._structure_cache[cache_key] = cache_data
            print(f"✓ Cached result in memory")
            print(f"  Memory cache now has {len(self._structure_cache)} entries")
            
            # Limit memory cache size
            if len(self._structure_cache) > self._max_cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self._structure_cache))
                del self._structure_cache[oldest_key]
                print(f"  (Memory cache limit reached, removed oldest entry)")
            
            # Save to disk cache (persistent)
            if self._save_to_disk_cache(cache_key, cache_data):
                print(f"✓ Cached result to disk: {self._get_cache_filename(cache_key).name}")
        else:
            print("⚠️ Failed to generate cache key, result not cached")

        return {
            'success': True,
            'sections': sections,
            'summary': summary,
            'duration': audio_duration,
        }

    def filter_keypoints_by_sections(
        self,
        keypoints: List[Dict],
        sections: List[Dict],
        audio_duration: float,
        min_interval: float = 3.0,
        min_segment: float = 3.0,
        max_segment: float = 30.0,
        total_shots: int = 20,
        weight_downbeat: float = 1.0,
        weight_pitch: float = 1.0,
        weight_mel_energy: float = 1.0,
    ) -> List[Dict]:
        """
        Filter keypoints based on sections (Level 1 structure).

        Args:
            keypoints: Raw keypoints from Madmom
            sections: Level 1 sections from structure analysis
            audio_duration: Audio duration in seconds
            min_interval: Minimum interval between keypoints (global)
            min_segment: Minimum segment duration (for merge)
            max_segment: Maximum segment duration (for split)
            total_shots: Total shots to allocate proportionally across sections
            weight_downbeat: Weight for Downbeat intensity
            weight_pitch: Weight for Pitch intensity
            weight_mel_energy: Weight for Mel Energy intensity

        Returns:
            Filtered keypoints list
        """
        try:
            from vca.audio.audio_Madmom import filter_by_sections
            from vca.audio.audio_caption_madmom import mmss_to_seconds
        except ImportError:
            # Fallback: return all keypoints
            print("⚠ Could not import filter_by_sections, returning all keypoints")
            return keypoints

        if not keypoints:
            return []

        if not sections:
            return keypoints

        # Convert sections to the format expected by filter_by_sections
        sections_for_filter = []
        for sec in sections:
            try:
                start_time = mmss_to_seconds(sec.get("Start_Time", "00:00"))
                end_time = mmss_to_seconds(sec.get("End_Time", "00:00"))
                if end_time > start_time:
                    sections_for_filter.append({
                        'name': sec.get('name', 'Unknown'),
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': end_time - start_time
                    })
            except Exception as e:
                print(f"⚠ Warning: Failed to parse section: {sec.get('name', 'Unknown')} - {e}")

        if not sections_for_filter:
            return keypoints

        print(f"\nFiltering {len(keypoints)} keypoints by {len(sections_for_filter)} sections...")
        print(f"  Parameters: total_shots={total_shots}, min_interval={min_interval}s, min_segment={min_segment}s")

        # Apply section-based filtering with proportional allocation
        filtered = filter_by_sections(
            keypoints=keypoints,
            sections=sections_for_filter,
            section_min_interval=min_interval,
            use_normalized_intensity=True,
            min_segment_duration=min_segment,
            max_segment_duration=max_segment,
            total_shots=total_shots,
            audio_duration=audio_duration,
            weight_downbeat=weight_downbeat,
            weight_pitch=weight_pitch,
            weight_mel_energy=weight_mel_energy,
        )

        print(f"✓ After section-based filtering: {len(filtered)} keypoints")

        return filtered

# Global structure generator instance
structure_generator = StructureGenerator()
