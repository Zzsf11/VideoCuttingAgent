import gradio as gr
import json
import random
import numpy as np
from .detectors import MadmomDetector, OmniDetector
from .visualization import create_full_width_player, format_table, _unique_keypoint_types, _filter_keypoints_by_types
from .utils import get_audio_data

madmom_detector = MadmomDetector()
omni_detector = OmniDetector()

def run_analysis(model_name, audio_file, current_kp, current_sec, *args):
    """Unified handler for running analysis"""
    if audio_file is None:
        return (
            gr.update(value="<div style='color:red'>Please upload audio first</div>"),
            "Please upload audio",
            current_kp,
            current_sec,
            gr.update(),
            ""
        )

    # Handle filepath
    if isinstance(audio_file, str): audio_path = audio_file
    elif isinstance(audio_file, tuple): audio_path = audio_file[0]
    else: audio_path = audio_file.name

    new_keypoints = []
    new_sections = []
    info_text = ""
    extra_features = {}

    if model_name == "Madmom":
        # Minimal core params (accept both old and new UIs; extra args are ignored)
        onset_th = args[0] if len(args) > 0 else 0.6
        bpb = args[1] if len(args) > 1 else 4
        dbn_th = args[2] if len(args) > 2 else 0.05
        merge_close = args[3] if len(args) > 3 else 0.1
        min_interval = args[4] if len(args) > 4 else 0.0
        top_k = args[5] if len(args) > 5 else 0

        # Backward/forward-compatible parsing:
        # - older UI: args[6] = silence_db (negative dB)
        # - newer UI: args[6] = energy_percentile (0..100), args[7] = silence_db
        energy_percentile = 0.0
        silence_db = -45.0
        if len(args) > 6:
            try:
                v6 = float(args[6])
            except Exception:
                v6 = 0.0
            if v6 < 0:
                silence_db = v6
            else:
                energy_percentile = v6
                if len(args) > 7:
                    silence_db = args[7]

        params = {
            'onset_threshold': onset_th,
            'beats_per_bar': int(bpb),
            'dbn_threshold': dbn_th,
            'merge_close': merge_close,
            'min_interval': min_interval,
            'top_k': int(top_k),
            'energy_percentile': float(energy_percentile),
            'silence_filter': True,
            'silence_threshold_db': silence_db,
        }
        res = madmom_detector.detect(audio_path, **params)
        if res['success']:
            new_keypoints = res['keypoints']
            info_text = f"**Madmom Results**\nFound {len(new_keypoints)} keypoints.\n\n" + format_table(new_keypoints)
            extra_features = {}

    elif model_name == "Omni":
        # args: (temp, top_p, max_tok, type)
        params = {
            'temperature': args[0], 'top_p': args[1],
            'max_tokens': args[2], 'analysis_type': args[3]
        }
        res = omni_detector.detect(audio_path, **params)
        if res['success']:
            new_keypoints = res['keypoints']
            new_sections = res.get('sections', [])
            # Build info text with sections
            info_text = f"**Omni Results**\n{res.get('summary','')}\n\n"
            if new_sections:
                info_text += "**Sections:**\n\n"
                info_text += "| # | Name | Start | End | Description |\n|---|---|---|---|---|\n"
                for i, sec in enumerate(new_sections):
                    start = sec.get('Start_Time', '00:00')
                    end = sec.get('End_Time', '00:00')
                    name = sec.get('name', f'Section {i+1}')
                    desc = sec.get('description', '')[:50]
                    info_text += f"| {i+1} | {name} | {start} | {end} | {desc} |\n"
                info_text += "\n"
            info_text += "**Keypoints:**\n\n" + format_table(new_keypoints)

    # Default filter = show all types found in result
    type_choices = _unique_keypoint_types(new_keypoints)
    selected_types = type_choices

    # Player renders ALL keypoints and sections; filtering is applied client-side to avoid
    # re-rendering the audio element (which would reset progress).
    player_html = create_full_width_player(
        audio_path, new_keypoints,
        title=f"{model_name} Analysis Result",
        sections=new_sections,
        extra_features=extra_features
    )

    # Generate script to sync client-side filter state
    reset_script = f"""
    <div style="display:none;" id="_vca_reset_trigger_{random.randint(0, 100000)}">
    <script>
    (function() {{
        window._vcaSelectedTypes = {json.dumps(selected_types)};
    }})();
    </script>
    </div>
    """

    return (
        player_html,
        info_text,
        new_keypoints,
        new_sections,
        gr.update(choices=type_choices, value=selected_types),
        reset_script
    )


def edit_keypoint(action, audio_file, time_val, del_idx, current_keypoints, current_sections, selected_types):
    """Unified handler for Adding/Deleting keypoints"""
    if not audio_file:
        return gr.update(), "No audio loaded", current_keypoints, current_sections, gr.update(), ""

    if isinstance(audio_file, str): audio_path = audio_file
    elif isinstance(audio_file, tuple): audio_path = audio_file[0]
    else: audio_path = audio_file.name

    kp_list = list(current_keypoints) if current_keypoints else []
    sections = list(current_sections) if current_sections else []

    # Normalize filter selection from UI (used for delete mapping).
    type_choices_before = _unique_keypoint_types(kp_list)
    if selected_types is None:
        selected_types_before = type_choices_before
    else:
        selected_types_before = [t for t in list(selected_types) if t in type_choices_before]

    if action == "add":
        if time_val < 0:
            return gr.update(), "Invalid time", kp_list, sections, gr.update(choices=type_choices_before, value=selected_types_before), ""
        kp_list.append({'time': float(time_val), 'type': 'Manual', 'intensity': 1.0})
        kp_list.sort(key=lambda x: x['time'])

    elif action == "delete":
        try:
            idx = int(del_idx) - 1
        except Exception:
            return gr.update(), f"Invalid index {del_idx}", kp_list, sections, gr.update(choices=type_choices_before, value=selected_types_before), ""

        shown_before = _filter_keypoints_by_types(kp_list, selected_types_before)
        if not shown_before:
            return gr.update(), "No keypoints in current filter (nothing to delete)", kp_list, sections, gr.update(choices=type_choices_before, value=selected_types_before), ""

        if 0 <= idx < len(shown_before):
            target = shown_before[idx]
            remove_index = None
            for j, kp in enumerate(kp_list):
                if kp is target:
                    remove_index = j
                    break
            if remove_index is None:
                # Fallback: match by fields
                for j, kp in enumerate(kp_list):
                    if (kp.get('time') == target.get('time') and kp.get('type') == target.get('type') and kp.get('intensity') == target.get('intensity')):
                        remove_index = j
                        break
            if remove_index is None:
                return gr.update(), "Failed to locate target keypoint to delete", kp_list, sections, gr.update(choices=type_choices_before, value=selected_types_before), ""

            kp_list.pop(remove_index)
        else:
            return gr.update(), f"Invalid index {del_idx} for current filter", kp_list, sections, gr.update(choices=type_choices_before, value=selected_types_before), ""

    # Update filter choices; keep user's selection when possible.
    type_choices = _unique_keypoint_types(kp_list)
    # Note: CheckboxGroup passes [] when user deselects all; preserve that.
    if selected_types is None:
        selected_types = type_choices
    else:
        selected_types = [t for t in list(selected_types) if t in type_choices]

    shown_keypoints = _filter_keypoints_by_types(kp_list, selected_types)

    # Re-render player with ALL keypoints and sections; only the table is filtered.
    player_html = create_full_width_player(audio_path, kp_list, title="Edited Analysis Result", sections=sections)
    info_text = f"**Keypoints Shown ({len(shown_keypoints)}/{len(kp_list)})**\n\n{format_table(shown_keypoints)}"

    # Sync script
    sync_script = f"""
    <div style="display:none;" id="_vca_sync_trigger_{random.randint(0, 100000)}">
    <script>
    (function() {{
        window._vcaSelectedTypes = {json.dumps(selected_types)};
    }})();
    </script>
    </div>
    """

    return player_html, info_text, kp_list, sections, gr.update(choices=type_choices, value=selected_types), sync_script


def apply_type_filter(audio_file, current_keypoints, selected_types):
    """Apply type filter - update info table and return script to filter markers."""
    kp_list = list(current_keypoints) if current_keypoints else []

    # Handle None or empty selected_types
    if selected_types is None:
        selected_types = []
    else:
        selected_types = list(selected_types)

    shown_keypoints = _filter_keypoints_by_types(kp_list, selected_types)
    info_text = f"**Keypoints Shown ({len(shown_keypoints)}/{len(kp_list)})**\n\n{format_table(shown_keypoints)}"

    # Generate a script that applies the filter to SVG markers
    # This runs when Gradio updates the hidden HTML component
    filter_script = f"""
    <div style="display:none;" id="_vca_filter_trigger">
    <script>
    (function() {{
        window._vcaSelectedTypes = {json.dumps(selected_types)};
        document.querySelectorAll('svg[id$="_markers_svg"]').forEach(function(svg) {{
            var selectedSet = new Set(window._vcaSelectedTypes);
            var hasSelection = selectedSet.size > 0;
            svg.querySelectorAll('g[data-kptype]').forEach(function(g) {{
                var t = g.getAttribute('data-kptype') || 'Unknown';
                g.style.display = hasSelection && selectedSet.has(t) ? '' : 'none';
            }});
        }});
    }})();
    </script>
    </div>
    """

    return info_text, filter_script


def preview_segments(audio_file, keypoints, min_segment, max_segment):
    """
    Preview how segments will be merged/split based on current keypoints and parameters.

    Returns a markdown table showing the resulting segments.
    """
    if not audio_file or not keypoints:
        return "❌ No audio or keypoints. Run analysis first."

    # Get audio path
    if isinstance(audio_file, str):
        audio_path = audio_file
    elif isinstance(audio_file, tuple):
        audio_path = audio_file[0]
    else:
        audio_path = audio_file.name

    # Get audio duration
    try:
        _, duration = get_audio_data(audio_path)
    except Exception as e:
        return f"❌ Error reading audio: {e}"

    # Sort keypoints by time
    keypoint_times = sorted([kp['time'] for kp in keypoints])

    # Create initial segments from keypoints
    all_times = [0.0] + keypoint_times + [duration]
    all_times = sorted(set(all_times))

    segments = []
    for i in range(len(all_times) - 1):
        start = all_times[i]
        end = all_times[i + 1]
        seg_duration = end - start
        if seg_duration >= 1.0:  # Skip very short segments
            segments.append({
                'start': start,
                'end': end,
                'duration': seg_duration,
                'status': 'original'
            })

    # Step 1: Merge short segments
    merged_segments = []
    for seg in segments:
        if seg['duration'] < min_segment and merged_segments:
            # Merge with previous
            prev = merged_segments[-1]
            merged_segments[-1] = {
                'start': prev['start'],
                'end': seg['end'],
                'duration': seg['end'] - prev['start'],
                'status': 'merged'
            }
        elif seg['duration'] < min_segment and len(segments) > len(merged_segments) + 1:
            # Will be merged with next (mark for now)
            segments[len(merged_segments) + 1]['start'] = seg['start']
            segments[len(merged_segments) + 1]['duration'] = segments[len(merged_segments) + 1]['end'] - seg['start']
            segments[len(merged_segments) + 1]['status'] = 'merged'
        else:
            merged_segments.append(seg)

    # Step 2: Split long segments
    final_segments = []
    for seg in merged_segments:
        if seg['duration'] > max_segment:
            # Need to split
            num_parts = int(np.ceil(seg['duration'] / max_segment))
            part_duration = seg['duration'] / num_parts
            for i in range(num_parts):
                part_start = seg['start'] + i * part_duration
                part_end = seg['start'] + (i + 1) * part_duration
                if i == num_parts - 1:
                    part_end = seg['end']  # Ensure last part ends exactly
                final_segments.append({
                    'start': part_start,
                    'end': part_end,
                    'duration': part_end - part_start,
                    'status': 'split'
                })
        else:
            final_segments.append(seg)

    # Format output
    def fmt_time(t):
        m = int(t // 60)
        s = t % 60
        return f"{m:02d}:{s:05.2f}"

    # Build markdown table
    lines = [
        f"### 📊 Segment Preview",
        f"",
        f"**Parameters:** min_segment={min_segment}s, max_segment={max_segment}s",
        f"",
        f"**Original keypoints:** {len(keypoints)} | **Resulting segments:** {len(final_segments)}",
        f"",
        f"| # | Start | End | Duration | Status |",
        f"|---|---|---|---|---|"
    ]

    status_emoji = {
        'original': '✅ OK',
        'merged': '🔗 Merged',
        'split': '✂️ Split'
    }

    total_duration = 0
    for i, seg in enumerate(final_segments):
        status = status_emoji.get(seg['status'], seg['status'])
        lines.append(f"| {i+1} | {fmt_time(seg['start'])} | {fmt_time(seg['end'])} | {seg['duration']:.1f}s | {status} |")
        total_duration += seg['duration']

    lines.append(f"")
    lines.append(f"**Total duration:** {fmt_time(total_duration)} ({total_duration:.1f}s)")

    # Summary stats
    short_count = sum(1 for seg in final_segments if seg['duration'] < min_segment)
    long_count = sum(1 for seg in final_segments if seg['duration'] > max_segment)

    if short_count > 0 or long_count > 0:
        lines.append(f"")
        lines.append(f"⚠️ **Warnings:**")
        if short_count > 0:
            lines.append(f"  - {short_count} segments still shorter than {min_segment}s")
        if long_count > 0:
            lines.append(f"  - {long_count} segments still longer than {max_segment}s")

    return "\n".join(lines)
