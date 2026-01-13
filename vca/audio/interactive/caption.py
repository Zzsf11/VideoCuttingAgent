import os
import json
import soundfile as sf
from typing import List, Dict

class CaptionGenerator:
    """
    Generate caption JSON using keypoints and sections from interactive editing.
    Uses the same pipeline as audio_caption_madmom.py but with pre-computed sections.
    """

    def __init__(self):
        self._model = None
        self._processor = None

    def _load_model(self, model_path: str = None):
        """Load AI model for caption generation"""
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

    def generate_caption(
        self,
        audio_path: str,
        keypoints: List[Dict],
        sections: List[Dict],
        output_path: str,
        batch_size: int = 4,
        min_segment_duration: float = 3.0,
        max_segment_duration: float = 30.0,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 20,
        max_tokens: int = 4096,
        overall_summary: str = "",
    ) -> Dict:
        """
        Generate caption JSON from edited keypoints and pre-computed sections.

        Args:
            audio_path: Path to audio file
            keypoints: List of keypoint dicts with 'time', 'type', 'intensity'
            sections: List of section dicts (Level 1 sections from structure analysis)
            output_path: Path to save caption JSON
            batch_size: Batch size for AI inference
            min_segment_duration: Minimum segment duration in seconds
            max_segment_duration: Maximum segment duration in seconds
            overall_summary: Summary from structure analysis

        Returns:
            Caption result dictionary
        """
        try:
                from vca.audio.audio_caption_madmom import (
                generate_audio_captions_batch,
                segment_audio_file,
                extract_json_from_text,
                seconds_to_mmss,
                mmss_to_seconds,
                AUDIO_SEG_KEYPOINT_PROMPT,
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
        print("KEYPOINT ANALYSIS (Level 2 Captions)")
        print(f"{'='*60}")
        print(f"Audio: {audio_path}")
        print(f"Duration: {audio_duration:.2f}s")
        print(f"Keypoints: {len(keypoints)}")
        print(f"Sections: {len(sections)}")

        # Use provided sections or create default
        if not sections:
            sections = [{
                "name": "Full Audio",
                "description": "",
                "Start_Time": "00:00",
                "End_Time": seconds_to_mmss(audio_duration)
            }]

        # Step 1: Create segments from keypoints
        print("\n[Step 1] Creating segments from keypoints...")
        keypoint_times = sorted([kp['time'] for kp in keypoints])
        all_times = [0.0] + keypoint_times + [audio_duration]
        all_times = sorted(set(all_times))

        segments = []
        for i in range(len(all_times) - 1):
            start = all_times[i]
            end = all_times[i + 1]
            duration = end - start

            if duration < 1.0:
                continue

            segments.append({
                'start_time': start,
                'end_time': end,
                'duration': duration
            })

        # Merge short segments
        merged = []
        for seg in segments:
            if seg['duration'] < min_segment_duration and merged:
                merged[-1]['end_time'] = seg['end_time']
                merged[-1]['duration'] = merged[-1]['end_time'] - merged[-1]['start_time']
            else:
                merged.append(seg)
        segments = merged

        print(f"✓ Created {len(segments)} segments")

        # Step 2: Extract and analyze segments in batches
        print("\n[Step 2] Analyzing segments with AI...")
        temp_files = []
        segment_paths = []

        for seg in segments:
            try:
                seg_path = segment_audio_file(audio_path, seg['start_time'], seg['end_time'])
                temp_files.append(seg_path)
                segment_paths.append((seg, seg_path))
            except Exception as e:
                print(f"⚠ Failed to extract segment: {e}")
                segment_paths.append((seg, None))

        # Batch process segments
        valid_paths = [(seg, path) for seg, path in segment_paths if path]
        segment_captions = {}

        for batch_start in range(0, len(valid_paths), batch_size):
            batch = valid_paths[batch_start:batch_start + batch_size]
            batch_paths = [p for _, p in batch]

            print(f"  Processing batch {batch_start // batch_size + 1}...")

            captions = generate_audio_captions_batch(
                audio_paths=batch_paths,
                prompt=AUDIO_SEG_KEYPOINT_PROMPT,
                model=model,
                processor=processor,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
            )

            for (seg, path), caption in zip(batch, captions):
                segment_captions[path] = caption

        # Step 3: Build final structure
        print("\n[Step 3] Building caption structure...")
        final_sections = []

        for stage1_sec in sections:
            sec_start = mmss_to_seconds(stage1_sec.get("Start_Time", "00:00"))
            sec_end = mmss_to_seconds(stage1_sec.get("End_Time", seconds_to_mmss(audio_duration)))

            # Find segments within this section
            section_segs = []
            for seg, path in segment_paths:
                if sec_start <= seg['start_time'] < sec_end:
                    sub_section = {
                        "name": f"Segment {len(section_segs) + 1}",
                        "description": "",
                        "Start_Time": seconds_to_mmss(seg['start_time'] - sec_start),
                        "End_Time": seconds_to_mmss(seg['end_time'] - sec_start)
                    }

                    if path and path in segment_captions:
                        caption_json = extract_json_from_text(segment_captions[path])
                        if caption_json:
                            sub_section["description"] = caption_json.get("summary", "")
                            if "emotion" in caption_json:
                                sub_section["Emotional_Tone"] = caption_json["emotion"]
                            if "energy" in caption_json:
                                sub_section["energy"] = caption_json["energy"]
                            if "rhythm" in caption_json:
                                sub_section["rhythm"] = caption_json["rhythm"]

                    section_segs.append(sub_section)

            final_sections.append({
                "name": stage1_sec.get("name", "Section"),
                "description": stage1_sec.get("description", ""),
                "Start_Time": stage1_sec.get("Start_Time"),
                "End_Time": stage1_sec.get("End_Time"),
                "detailed_analysis": {
                    "summary": "",
                    "sections": section_segs
                }
            })

        # Cleanup temp files
        for f in temp_files:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except:
                pass

        # Build result
        # Extract keypoint details for metadata
        keypoints_detail = []
        for kp in keypoints:
            kp_info = {
                'time': float(kp.get('time', 0.0)),
                'type': kp.get('type', 'Unknown'),
                'intensity': float(kp.get('intensity', 0.0)),
            }
            # Add normalized intensity if available
            if 'normalized_intensity' in kp:
                kp_info['normalized_intensity'] = float(kp.get('normalized_intensity', 0.0))
            # Add composite score if available
            if 'composite_score' in kp:
                kp_info['composite_score'] = float(kp.get('composite_score', 0.0))
            # Add section info if available
            if 'section' in kp:
                kp_info['section'] = kp.get('section')
            keypoints_detail.append(kp_info)

        result = {
            "audio_path": audio_path,
            "overall_analysis": {
                "summary": overall_summary
            },
            "sections": final_sections,
            "_source": "interactive_editing",
            "_keypoints_used": len(keypoints),
            "_keypoints_detail": keypoints_detail
        }

        # Save to file
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print(f"✓ Caption saved to: {output_path}")
        print(f"  - Level 1 sections: {len(final_sections)}")
        total_subs = sum(len(s.get('detailed_analysis', {}).get('sections', [])) for s in final_sections)
        print(f"  - Level 2 sub-segments: {total_subs}")
        print(f"  - Keypoints saved: {len(keypoints_detail)}")

        # Print keypoint type statistics
        type_counts = {}
        for kp in keypoints_detail:
            kp_type = kp.get('type', 'Unknown')
            type_counts[kp_type] = type_counts.get(kp_type, 0) + 1
        print(f"  - Keypoint types: {dict(sorted(type_counts.items()))}")
        print(f"{'='*60}")

        return {'success': True, 'output_path': output_path, 'result': result}


# Global caption generator instance
caption_generator = CaptionGenerator()
