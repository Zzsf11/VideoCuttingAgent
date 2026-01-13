import os
import gradio as gr
import argparse
from .config import load_saved_params, save_params_to_file, reset_params_to_default
from .detectors import MadmomDetector
from .structure import structure_generator
from .caption import caption_generator
from .utils import get_audio_data, _patch_gradio_checkboxgroup_none_payload
from .visualization import (
    create_full_width_player, format_table, _unique_keypoint_types, _filter_keypoints_by_types
)
from .logic import preview_segments

# Apply patch
_patch_gradio_checkboxgroup_none_payload()

madmom_detector = MadmomDetector()

def create_gradio_interface(default_audio_path: str = None, default_detection_method: str = "downbeat"):
    # Load saved parameters
    saved_params = load_saved_params()
    # Detection methods: migrated from single selection ("detection_method") to multi-select.
    # Keep backward compatibility with older saved configs.
    saved_params.setdefault("detection_method", default_detection_method)
    initial_methods = saved_params.get("detection_methods")
    if not initial_methods:
        initial_methods = saved_params.get("detection_method", default_detection_method)
    if isinstance(initial_methods, str):
        initial_methods = [initial_methods]
    if not isinstance(initial_methods, (list, tuple)):
        initial_methods = [default_detection_method]

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"), title="Audio Segmenter Pro") as demo:

        gr.Markdown(
            """
            <div style="text-align:center; margin-bottom:10px;">
                <h1>Audio Segmentation Interactive Tool</h1>
                <p>Workflow: Madmom Detection → Structure Analysis → Filter & Edit → Keypoint Analysis → Save</p>
            </div>
            """
        )

        # 1. Global Input
        with gr.Row():
            audio_input = gr.Audio(label="Source Audio", type="filepath", value=default_audio_path)

        # ==================== State Variables ====================
        # Raw keypoints from Madmom (before structure-based filtering)
        raw_keypoints = gr.State([])
        # Filtered keypoints (after structure-based filtering, can be edited)
        filtered_keypoints = gr.State([])
        # Level 1 sections from structure analysis
        current_sections = gr.State([])
        # Feature arrays (rms, flux, etc.)
        current_features = gr.State({})
        # Structure summary
        structure_summary = gr.State("")
        # Audio duration
        audio_duration = gr.State(0.0)

        # Hidden HTML component to run filter scripts (updates marker visibility)
        filter_script_runner = gr.HTML(visible=False, sanitize_html=False)

        # ==================== STEP 1: Madmom Detection ====================
        gr.Markdown("---")
        gr.Markdown("## Step 1: Madmom Detection (Signal Processing)")
        gr.Markdown("Detect raw keypoints using Madmom signal processing algorithms.")

        with gr.Row():
            with gr.Column(scale=2, variant="panel"):
                with gr.Accordion("Madmom Parameters", open=True):
                    gr.Markdown("选择检测方法和相应的参数。")

                    # Detection method selection (multi)
                    m_method = gr.CheckboxGroup(
                        choices=["downbeat", "pitch", "mel_energy"],
                        value=list(initial_methods),
                        label="Detection Methods",
                        info="You can select multiple algorithms and merge their keypoints."
                    )

                    # Downbeat-specific parameters (shown when method="downbeat")
                    with gr.Group(visible=True) as downbeat_group:
                        gr.Markdown("**Downbeat Detection Parameters**")
                        m_bpb = gr.Slider(
                            1,
                            8,
                            value=saved_params.get("beats_per_bar", 4),
                            step=1,
                            label="Beats/Bar",
                            info="Usually 4. Use 3 for waltz (3/4).",
                        )
                        m_dbn_th = gr.Slider(
                            0.01,
                            0.5,
                            value=saved_params.get("dbn_threshold", 0.05),
                            label="DBN Threshold",
                            info="Higher = stricter beat/downbeat tracking.",
                        )
                        m_correct_beats = gr.Checkbox(
                            value=saved_params.get("correct_beats", True),
                            label="Correct Beats",
                            info="Align beats to nearest activation peak (recommended).",
                        )
                        m_min_bpm = gr.Slider(
                            20, 200, value=saved_params.get("min_bpm", 55.0),
                            label="Min BPM",
                            info="Minimum tempo to detect"
                        )
                        m_max_bpm = gr.Slider(
                            100, 300, value=saved_params.get("max_bpm", 215.0),
                            label="Max BPM",
                            info="Maximum tempo to detect"
                        )
                        m_num_tempi = gr.Slider(
                            10, 120,
                            value=saved_params.get("num_tempi", 60),
                            step=1,
                            label="Num Tempi",
                            info="Number of tempo states for beat tracking.",
                        )
                        m_transition_lambda = gr.Slider(
                            1, 300,
                            value=saved_params.get("transition_lambda", 100.0),
                            step=1,
                            label="Transition Lambda",
                            info="Higher = prefer constant tempo.",
                        )
                        m_observation_lambda = gr.Slider(
                            4, 32,
                            value=saved_params.get("observation_lambda", 16),
                            step=1,
                            label="Observation Lambda",
                            info="Subdivisions per beat cycle.",
                        )
                        m_fps = gr.Slider(
                            50, 200,
                            value=saved_params.get("fps", 100),
                            step=1,
                            label="FPS",
                            info="Frame rate used for beat tracking.",
                        )

                    # Pitch-specific parameters (shown when method="pitch")
                    with gr.Group(visible=False) as pitch_group:
                        gr.Markdown("**Pitch Detection Parameters**")
                        m_pitch_tolerance = gr.Slider(
                            0.1, 1.0,
                            value=saved_params.get("pitch_tolerance", 0.8),
                            label="Pitch Tolerance",
                            info="Detection sensitivity (higher = more tolerant)"
                        )
                        m_pitch_threshold = gr.Slider(
                            0.1, 1.0,
                            value=saved_params.get("pitch_threshold", 0.8),
                            label="Pitch Confidence Threshold",
                            info="Minimum confidence to keep pitch points"
                        )
                        m_pitch_min_distance = gr.Slider(
                            0.1, 2.0,
                            value=saved_params.get("pitch_min_distance", 0.5),
                            label="Min Distance (s)",
                            info="Minimum time between detected pitch points"
                        )
                        m_pitch_nms = gr.Dropdown(
                            choices=["basic", "adaptive", "window"],
                            value=saved_params.get("pitch_nms_method", "basic"),
                            label="Pitch NMS",
                            info="Non-maximum suppression strategy for pitch points.",
                        )
                        m_pitch_max_points = gr.Slider(
                            1, 200,
                            value=saved_params.get("pitch_max_points", 20),
                            step=1,
                            label="Pitch Max Points",
                            info="Maximum number of pitch points to keep.",
                        )

                    # Mel energy-specific parameters (shown when method="mel_energy")
                    with gr.Group(visible=False) as mel_group:
                        gr.Markdown("**Mel Energy Detection Parameters**")
                        m_mel_win_size = gr.Slider(
                            128, 4096,
                            value=saved_params.get("mel_win_s", 512),
                            step=64,
                            label="FFT Window Size",
                            info="FFT window size used for mel-energy.",
                        )
                        m_mel_n_filters = gr.Slider(
                            10, 128,
                            value=saved_params.get("mel_n_filters", 40),
                            step=1,
                            label="Mel Filters",
                            info="Number of mel filters.",
                        )
                        m_mel_threshold = gr.Slider(
                            0.1, 1.0,
                            value=saved_params.get("mel_threshold_ratio", 0.3),
                            label="Energy Threshold Ratio",
                            info="Threshold as fraction of max energy (lower = more points)"
                        )
                        m_mel_min_distance = gr.Slider(
                            0.1, 2.0,
                            value=saved_params.get("mel_min_distance", 0.5),
                            label="Min Distance (s)",
                            info="Minimum time between detected energy peaks"
                        )
                        m_mel_nms = gr.Dropdown(
                            choices=["basic", "adaptive", "window"],
                            value=saved_params.get("mel_nms_method", "basic"),
                            label="Mel NMS",
                            info="Non-maximum suppression strategy for mel-energy peaks.",
                        )
                        m_mel_max_points = gr.Slider(
                            1, 200,
                            value=saved_params.get("mel_max_points", 20),
                            step=1,
                            label="Mel Max Points",
                            info="Maximum number of mel-energy points to keep.",
                        )

                    def _update_method_groups(selected_methods):
                        methods = selected_methods or []
                        if isinstance(methods, str):
                            methods = [methods]
                        return (
                            gr.update(visible="downbeat" in methods),
                            gr.update(visible="pitch" in methods),
                            gr.update(visible="mel_energy" in methods),
                        )

                    m_method.change(
                        fn=_update_method_groups,
                        inputs=[m_method],
                        outputs=[downbeat_group, pitch_group, mel_group],
                    )

                    gr.Markdown("**Output Stability**")
                    m_merge = gr.Slider(
                        0,
                        0.5,
                        value=saved_params["merge_close"],
                        label="Merge Close (s)",
                        info="Merge keypoints within this window.",
                    )
                    m_min_int = gr.Slider(
                        0,
                        2.0,
                        value=saved_params["min_interval"],
                        label="Min Interval (s)",
                        info="Minimum time between keypoints.",
                    )
                    m_topk = gr.Slider(
                        0,
                        100,
                        value=saved_params["top_k"],
                        step=1,
                        label="Top K",
                        info="Keep strongest keypoints per type. Each type (Downbeat/Onset/etc.) gets a proportional share. 0 = keep all.",
                    )
                    m_energy_pct = gr.Slider(
                        0,
                        100,
                        value=saved_params.get("energy_percentile", 0),
                        step=5,
                        label="Energy Percentile",
                        info="Keep only keypoints above this energy percentile (0=keep all, 50=keep top 50%).",
                    )

                    gr.Markdown("**Silence Filter (Recommended)**")
                    m_silence_th = gr.Slider(
                        -80.0,
                        -10.0,
                        value=saved_params.get("silence_threshold_db", -45.0),
                        label="Silence Threshold (dB)",
                        info="More negative = less aggressive. If silent areas still have points, increase to e.g. -35.",
                    )

                    gr.Markdown("---")
                    with gr.Row():
                        save_params_btn = gr.Button("💾 Save Parameters", size="sm")
                        reset_params_btn = gr.Button("🔄 Reset to Default", size="sm", variant="secondary")
                    params_status = gr.Markdown("")

            with gr.Column(scale=1):
                madmom_btn = gr.Button("🚀 Run Madmom Detection", variant="primary", size="lg")
                madmom_status = gr.Markdown("*Click to detect raw keypoints*")

        # Raw keypoints visualization (player and filter always visible, only table collapsible)
        raw_kp_player = gr.HTML(label="Raw Keypoints Visualization", sanitize_html=False)
        raw_type_filter = gr.CheckboxGroup(
            choices=[],
            value=[],
            label="Filter Keypoint Types",
            elem_id="raw_type_filter",
        )
        with gr.Accordion("📊 Raw Keypoints Table", open=False):
            raw_kp_info = gr.Markdown("*Run Madmom detection to see raw keypoints*")

        # ==================== STEP 2: Structure Analysis ====================
        gr.Markdown("---")
        gr.Markdown("## Step 2: Structure Analysis (AI)")
        gr.Markdown("Use Omni AI to analyze overall audio structure and identify Level 1 sections (Intro, Verse, Chorus, etc.). Results are cached for each audio file.")

        with gr.Row():
            with gr.Column(scale=2, variant="panel"):
                with gr.Accordion("Structure Analysis Settings", open=False):
                    s_temp = gr.Slider(0.1, 1.5, value=saved_params["structure_temperature"], label="Temperature",
                        info="Higher = more creative section detection.")
                    s_top_p = gr.Slider(0.1, 1.0, value=saved_params["structure_top_p"], label="Top P",
                        info="Nucleus sampling threshold.")
                    s_max_tokens = gr.Slider(512, 4096, value=saved_params["structure_max_tokens"], label="Max Tokens",
                        info="Maximum tokens for structure generation.")

            with gr.Column(scale=1):
                with gr.Row():
                    structure_btn = gr.Button("✨ Analyze Structure", variant="secondary", size="lg")
                    clear_cache_btn = gr.Button("🗑️ Clear Cache", variant="stop", size="sm", scale=0)
                structure_status = gr.Markdown("*First run Madmom detection, then analyze structure*")

        # Structure result (collapsible)
        with gr.Accordion("📋 Structure Analysis Result", open=True):
            structure_info = gr.Markdown("*Run structure analysis to see Level 1 sections*")

        # ==================== STEP 3: Filter by Structure & Edit ====================
        gr.Markdown("---")
        gr.Markdown("## Step 3: Filter & Edit Keypoints")
        gr.Markdown("Filter raw keypoints based on structure sections, then preview and edit the filtered keypoints.")

        with gr.Row():
            with gr.Column(scale=2, variant="panel"):
                gr.Markdown("**Filter Parameters**")
                f_total_shots = gr.Slider(5, 200, value=saved_params.get("filter_total_shots", 20), step=1, label="Total Shots",
                    info="Total shots to allocate across sections by proportion (based on keypoint density).")
                f_min_interval = gr.Slider(0.5, 10.0, value=saved_params["filter_min_interval"], step=0.5, label="Min Interval (s)",
                    info="Minimum interval between filtered keypoints (global, applied across all sections).")

                gr.Markdown("**Composite Score Weights** (k1*Downbeat + k2*Pitch + k3*MelEnergy)")
                f_weight_downbeat = gr.Slider(0.0, 2.0, value=saved_params.get("weight_downbeat", 1.0), step=0.1, label="k1: Downbeat Weight",
                    info="Weight for Downbeat intensity in composite score.")
                f_weight_pitch = gr.Slider(0.0, 2.0, value=saved_params.get("weight_pitch", 1.0), step=0.1, label="k2: Pitch Weight",
                    info="Weight for Pitch intensity in composite score.")
                f_weight_mel_energy = gr.Slider(0.0, 2.0, value=saved_params.get("weight_mel_energy", 1.0), step=0.1, label="k3: Mel Energy Weight",
                    info="Weight for Mel Energy intensity in composite score.")

                gr.Markdown("**Segment Parameters (for merge/split)**")
                f_min_seg = gr.Slider(1.0, 15.0, value=saved_params["filter_min_segment"], step=0.5, label="Min Segment Duration (s)",
                    info="Segments shorter than this will be merged.")
                f_max_seg = gr.Slider(0.0, 30.0, value=min(saved_params["filter_max_segment"], 30.0), step=1.0, label="Max Segment Duration (s)",
                    info="Segments longer than this will be split. Set to 0 for no limit.")

            with gr.Column(scale=1):
                filter_btn = gr.Button("🔍 Filter Keypoints", variant="secondary", size="lg")
                preview_seg_btn = gr.Button("👁️ Preview Segments", size="sm")
                filter_status = gr.Markdown("*Run structure analysis first, then filter keypoints*")

        # Filtered keypoints visualization and editing
        gr.Markdown("### Filtered Keypoints (Editable)")
        filtered_kp_player = gr.HTML(label="Filtered Keypoints Visualization", sanitize_html=False)

        # Editing controls
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                gr.Markdown("**Edit Filtered Keypoints**")
            with gr.Column(scale=2):
                with gr.Row():
                    add_time = gr.Number(label="Time (s)", precision=3, show_label=False, container=False,
                        placeholder="Add: time in seconds (e.g. 12.345)")
                    add_btn = gr.Button("➕ Add Point", size="sm")
            with gr.Column(scale=2):
                with gr.Row():
                    del_idx = gr.Number(label="Index #", precision=0, show_label=False, container=False,
                        placeholder="Delete: index in table (starts at 1)")
                    del_btn = gr.Button("➖ Delete Point", size="sm", variant="stop")

        # Filtered keypoints table (collapsible)
        with gr.Accordion("📋 Filtered Keypoints Table", open=True):
            filtered_type_filter = gr.CheckboxGroup(
                choices=[],
                value=[],
                label="Filter by Type",
                elem_id="filtered_type_filter",
            )
            filtered_kp_info = gr.Markdown("*Filter keypoints to see the list*")
            segment_preview = gr.Markdown("*Click 'Preview Segments' to see how keypoints will be merged/split*")

        # ==================== STEP 4: Keypoint Analysis & Save ====================
        gr.Markdown("---")
        gr.Markdown("## Step 4: Keypoint Analysis & Save")
        gr.Markdown("Use Omni AI to analyze each segment and generate detailed captions, then save to JSON.")

        with gr.Row():
            with gr.Column(scale=2, variant="panel"):
                with gr.Accordion("Keypoint Analysis Settings", open=False):
                    k_batch = gr.Slider(1, 8, value=saved_params["keypoint_batch_size"], step=1, label="Batch Size",
                        info="Number of segments to process in parallel.")
                    k_temp = gr.Slider(0.1, 1.5, value=saved_params["keypoint_temperature"], step=0.1, label="Temperature",
                        info="Higher = more creative captions.")
                    k_top_p = gr.Slider(0.1, 1.0, value=saved_params["keypoint_top_p"], step=0.05, label="Top P",
                        info="Nucleus sampling threshold.")
                    k_max_tokens = gr.Slider(512, 8192, value=saved_params["keypoint_max_tokens"], step=512, label="Max Tokens",
                        info="Maximum tokens for caption generation.")

                caption_output_path = gr.Textbox(
                    label="Output Path",
                    placeholder="e.g., ./output/audio_caption.json",
                    value="./audio_caption_interactive.json",
                    info="Path to save the generated caption JSON file"
                )

            with gr.Column(scale=1):
                generate_btn = gr.Button("🚀 Generate & Save Caption", variant="primary", size="lg")
                caption_status = gr.Markdown("*Edit filtered keypoints first, then generate captions*")

        # ==================== Callbacks ====================

        # Helper to simplify inputs for Madmom
        madmom_inputs = [
            m_method,
            # downbeat
            m_bpb, m_dbn_th, m_correct_beats, m_min_bpm, m_max_bpm, m_num_tempi, m_transition_lambda, m_observation_lambda, m_fps,
            # pitch
            m_pitch_tolerance, m_pitch_threshold, m_pitch_min_distance, m_pitch_nms, m_pitch_max_points,
            # mel
            m_mel_win_size, m_mel_n_filters, m_mel_threshold, m_mel_min_distance, m_mel_nms, m_mel_max_points,
            m_merge, m_min_int, m_topk, m_energy_pct, m_silence_th
        ]

        # ==================== STEP 1: Madmom Detection Callback ====================
        def run_madmom_detection(audio_file, *args):
            """Run Madmom detection and return raw keypoints"""
            if audio_file is None:
                return (
                    gr.update(),
                    "❌ Please upload audio first",
                    [],
                    0.0,
                    gr.update(),
                    gr.update(),
                    {},
                    gr.update()  # caption_output_path
                )

            # Handle filepath
            if isinstance(audio_file, str):
                audio_path = audio_file
            elif isinstance(audio_file, tuple):
                audio_path = audio_file[0]
            else:
                audio_path = audio_file.name
            
            # Generate output path based on audio filename
            audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
            output_dir = f"/public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Audio/{audio_basename}"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "audio_caption_interactive.json")

            # Get audio duration
            try:
                _, duration = get_audio_data(audio_path)
            except Exception as e:
                return (
                    gr.update(),
                    f"❌ Error reading audio: {e}",
                    [],
                    0.0,
                    gr.update(),
                    gr.update(),
                    {},
                    gr.update()  # caption_output_path
                )

            # Parse args based on new madmom_inputs order
            idx = 0
            detection_methods = args[idx]; idx += 1
            # downbeat
            bpb = args[idx]; idx += 1
            dbn_th = args[idx]; idx += 1
            correct_beats = args[idx]; idx += 1
            min_bpm = args[idx]; idx += 1
            max_bpm = args[idx]; idx += 1
            num_tempi = args[idx]; idx += 1
            transition_lambda = args[idx]; idx += 1
            observation_lambda = args[idx]; idx += 1
            fps = args[idx]; idx += 1
            # pitch
            pitch_tolerance = args[idx]; idx += 1
            pitch_threshold = args[idx]; idx += 1
            pitch_min_distance = args[idx]; idx += 1
            pitch_nms_method = args[idx]; idx += 1
            pitch_max_points = args[idx]; idx += 1
            # mel
            mel_win_s = args[idx]; idx += 1
            mel_n_filters = args[idx]; idx += 1
            mel_threshold = args[idx]; idx += 1
            mel_min_distance = args[idx]; idx += 1
            mel_nms_method = args[idx]; idx += 1
            mel_max_points = args[idx]; idx += 1
            merge_close = args[idx]; idx += 1
            min_interval = args[idx]; idx += 1
            top_k = args[idx]; idx += 1
            energy_percentile = args[idx]; idx += 1
            silence_threshold = args[idx]; idx += 1

            methods = detection_methods or []
            if isinstance(methods, str):
                methods = [methods]
            methods = [m for m in methods if m in {"downbeat", "pitch", "mel_energy"}]
            if not methods:
                methods = ["downbeat"]

            # Run selected methods and merge keypoints.
            # We do post-filtering (min_interval/top_k/percentile) after merging so it applies globally.
            merged_keypoints = []
            for method in methods:
                params = {
                    'detection_method': method,
                    # Downbeat parameters
                    'beats_per_bar': int(bpb),
                    'dbn_threshold': dbn_th,
                    'correct_beats': bool(correct_beats),
                    'min_bpm': float(min_bpm),
                    'max_bpm': float(max_bpm),
                    'num_tempi': int(num_tempi),
                    'transition_lambda': float(transition_lambda),
                    'observation_lambda': int(observation_lambda),
                    'fps': int(fps),
                    # Pitch parameters
                    'pitch_tolerance': float(pitch_tolerance),
                    'pitch_threshold': float(pitch_threshold),
                    'pitch_min_distance': float(pitch_min_distance),
                    'pitch_nms_method': str(pitch_nms_method),
                    'pitch_max_points': int(pitch_max_points),
                    # Mel parameters
                    'mel_win_s': int(mel_win_s),
                    'mel_n_filters': int(mel_n_filters),
                    'mel_threshold_ratio': float(mel_threshold),
                    'mel_min_distance': float(mel_min_distance),
                    'mel_nms_method': str(mel_nms_method),
                    'mel_max_points': int(mel_max_points),
                    # Silence gating
                    'silence_filter': True,
                    'silence_threshold_db': float(silence_threshold),
                    # Disable per-method post filtering; apply after merge.
                    'min_interval': 0.0,
                    'top_k': 0,
                    'energy_percentile': 0.0,
                }

                res = madmom_detector.detect(audio_path, **params)
                if not res.get('success'):
                    return (
                        gr.update(),
                        f"❌ Detection failed ({method}): {res.get('error', 'Unknown error')}",
                        [],
                        0.0,
                        gr.update(),
                        gr.update(),
                        {},
                        gr.update()  # caption_output_path
                    )
                merged_keypoints.extend(res.get('keypoints', []))

            keypoints = merged_keypoints
            # Apply post-filtering globally (after merge)
            if (min_interval and float(min_interval) > 0) or (top_k and int(top_k) > 0) or (energy_percentile and float(energy_percentile) > 0):
                try:
                    from vca.audio.audio_Madmom import filter_significant_keypoints  # noqa: WPS433 (runtime import)
                    keypoints = filter_significant_keypoints(
                        keypoints,
                        min_interval=float(min_interval),
                        top_k=int(top_k),
                        energy_percentile=float(energy_percentile),
                        use_normalized_intensity=True,
                    )
                except Exception:
                    # If filtering fails, still show unfiltered merged output.
                    pass

            if True:
                
                extra_features = {}

                # Generate player HTML
                player_html = create_full_width_player(
                    audio_path, keypoints,
                    title="Raw Keypoints (Madmom)",
                    extra_features=extra_features
                )

                # Get unique types for filter
                type_choices = _unique_keypoint_types(keypoints)

                # Format info
                info_text = f"**Found {len(keypoints)} raw keypoints**\n\n{format_table(keypoints)}"

                key_text = f" (methods: {', '.join(methods)})" if methods else ""

                return (
                    gr.update(value=player_html),
                    f"✅ Detected {len(keypoints)} raw keypoints (duration: {duration:.1f}s){key_text}",
                    keypoints,
                    duration,
                    gr.update(choices=type_choices, value=type_choices),
                    info_text,
                    extra_features,
                    output_path  # caption_output_path
                )

        madmom_btn.click(
            fn=run_madmom_detection,
            inputs=[audio_input] + madmom_inputs,
            outputs=[raw_kp_player, madmom_status, raw_keypoints, audio_duration, raw_type_filter, raw_kp_info, current_features, caption_output_path]
        )

        # ==================== STEP 2: Structure Analysis Callback ====================
        def run_structure_analysis(audio_file, raw_kps, temp, top_p, max_tokens):
            """Run Omni structure analysis"""
            if audio_file is None:
                return (
                    "❌ Please upload audio first",
                    [],
                    "",
                    "❌ No audio"
                )

            if not raw_kps:
                return (
                    "⚠️ Run Madmom detection first to get raw keypoints",
                    [],
                    "",
                    "⚠️ No raw keypoints"
                )

            # Handle filepath
            if isinstance(audio_file, str):
                audio_path = audio_file
            elif isinstance(audio_file, tuple):
                audio_path = audio_file[0]
            else:
                audio_path = audio_file.name

            # Run structure analysis
            result = structure_generator.analyze_structure(
                audio_path=audio_path,
                temperature=temp,
                top_p=top_p,
                max_tokens=int(max_tokens)
            )

            if result['success']:
                sections = result['sections']
                summary = result['summary']

                # Format sections info
                info_lines = [f"**Summary:** {summary}", "", "**Sections:**", ""]
                info_lines.append("| # | Name | Start | End | Description |")
                info_lines.append("|---|---|---|---|---|")
                for i, sec in enumerate(sections):
                    start = sec.get('Start_Time', '00:00')
                    end = sec.get('End_Time', '00:00')
                    name = sec.get('name', f'Section {i+1}')
                    desc = sec.get('description', '')[:50]
                    info_lines.append(f"| {i+1} | {name} | {start} | {end} | {desc} |")

                return (
                    "\n".join(info_lines),
                    sections,
                    summary,
                    f"✅ Found {len(sections)} Level 1 sections"
                )
            else:
                return (
                    f"❌ Structure analysis failed: {result.get('error', 'Unknown error')}",
                    [],
                    "",
                    f"❌ Error"
                )

        def clear_structure_cache():
            """Clear structure analysis cache (memory and disk)"""
            # Clear memory cache
            structure_generator._structure_cache.clear()
            
            # Clear disk cache
            import shutil
            from pathlib import Path
            cache_dir = Path.home() / ".cache" / "vca_audio_structure"
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
                cache_dir.mkdir(parents=True, exist_ok=True)
            
            return "✅ Structure cache cleared (memory and disk)"

        structure_btn.click(
            fn=run_structure_analysis,
            inputs=[audio_input, raw_keypoints, s_temp, s_top_p, s_max_tokens],
            outputs=[structure_info, current_sections, structure_summary, structure_status]
        )

        clear_cache_btn.click(
            fn=clear_structure_cache,
            inputs=[],
            outputs=[structure_status]
        )

        # ==================== STEP 3: Filter Keypoints Callback ====================
        def run_filter_keypoints(audio_file, raw_kps, sections, duration,
                                 total_shots, min_interval, min_seg, max_seg,
                                 weight_downbeat, weight_pitch, weight_mel_energy, features):
            """Filter keypoints by sections"""
            if audio_file is None:
                return (
                    gr.update(),
                    [],
                    "❌ No audio",
                    gr.update(),
                    gr.update()
                )

            if not raw_kps:
                return (
                    gr.update(),
                    [],
                    "⚠️ Run Madmom detection first",
                    gr.update(),
                    gr.update()
                )

            if not sections:
                return (
                    gr.update(),
                    [],
                    "⚠️ Run structure analysis first",
                    gr.update(),
                    gr.update()
                )

            # Handle filepath
            if isinstance(audio_file, str):
                audio_path = audio_file
            elif isinstance(audio_file, tuple):
                audio_path = audio_file[0]
            else:
                audio_path = audio_file.name

            # Filter keypoints using proportional allocation
            filtered = structure_generator.filter_keypoints_by_sections(
                keypoints=raw_kps,
                sections=sections,
                audio_duration=duration,
                min_interval=min_interval,
                min_segment=min_seg,
                max_segment=max_seg,
                total_shots=int(total_shots),
                weight_downbeat=float(weight_downbeat),
                weight_pitch=float(weight_pitch),
                weight_mel_energy=float(weight_mel_energy),
            )

            # Generate player HTML with sections overlay
            player_html = create_full_width_player(
                audio_path, filtered,
                title="Filtered Keypoints",
                sections=sections,
                extra_features=features
            )

            # Get unique types for filter
            type_choices = _unique_keypoint_types(filtered)

            # Format info
            info_text = f"**Filtered to {len(filtered)} keypoints** (from {len(raw_kps)} raw)\n\n{format_table(filtered)}"

            return (
                gr.update(value=player_html),
                filtered,
                f"✅ Filtered to {len(filtered)} keypoints",
                gr.update(choices=type_choices, value=type_choices),
                info_text
            )

        filter_btn.click(
            fn=run_filter_keypoints,
            inputs=[audio_input, raw_keypoints, current_sections, audio_duration,
                    f_total_shots, f_min_interval, f_min_seg, f_max_seg,
                    f_weight_downbeat, f_weight_pitch, f_weight_mel_energy, current_features],
            outputs=[filtered_kp_player, filtered_keypoints, filter_status, filtered_type_filter, filtered_kp_info]
        )

        # ==================== Edit Filtered Keypoints Callbacks ====================
        def add_filtered_keypoint(audio_file, time_val, filtered_kps, sections, features):
            """Add a keypoint to filtered list"""
            if audio_file is None or time_val is None or time_val < 0:
                return gr.update(), filtered_kps, gr.update(), gr.update()

            # Handle filepath
            if isinstance(audio_file, str):
                audio_path = audio_file
            elif isinstance(audio_file, tuple):
                audio_path = audio_file[0]
            else:
                audio_path = audio_file.name

            kp_list = list(filtered_kps) if filtered_kps else []
            kp_list.append({'time': float(time_val), 'type': 'Manual', 'intensity': 1.0})
            kp_list.sort(key=lambda x: x['time'])

            # Generate player HTML
            player_html = create_full_width_player(
                audio_path, kp_list,
                title="Filtered Keypoints (Edited)",
                sections=sections,
                extra_features=features
            )

            # Get unique types
            type_choices = _unique_keypoint_types(kp_list)
            info_text = f"**{len(kp_list)} keypoints**\n\n{format_table(kp_list)}"

            return gr.update(value=player_html), kp_list, gr.update(choices=type_choices, value=type_choices), info_text

        add_btn.click(
            fn=add_filtered_keypoint,
            inputs=[audio_input, add_time, filtered_keypoints, current_sections, current_features],
            outputs=[filtered_kp_player, filtered_keypoints, filtered_type_filter, filtered_kp_info]
        )

        def delete_filtered_keypoint(audio_file, del_index, filtered_kps, sections, selected_types, features):
            """Delete a keypoint from filtered list"""
            if audio_file is None or not filtered_kps:
                return gr.update(), filtered_kps, gr.update(), gr.update()

            # Handle filepath
            if isinstance(audio_file, str):
                audio_path = audio_file
            elif isinstance(audio_file, tuple):
                audio_path = audio_file[0]
            else:
                audio_path = audio_file.name

            kp_list = list(filtered_kps)

            # Get shown keypoints based on filter
            if selected_types:
                shown = _filter_keypoints_by_types(kp_list, selected_types)
            else:
                shown = kp_list

            try:
                idx = int(del_index) - 1
                if 0 <= idx < len(shown):
                    target = shown[idx]
                    # Find and remove from full list
                    for i, kp in enumerate(kp_list):
                        if kp.get('time') == target.get('time') and kp.get('type') == target.get('type'):
                            kp_list.pop(i)
                            break
            except:
                pass

            # Generate player HTML
            player_html = create_full_width_player(
                audio_path, kp_list,
                title="Filtered Keypoints (Edited)",
                sections=sections,
                extra_features=features
            )

            # Get unique types
            type_choices = _unique_keypoint_types(kp_list)
            new_selected = [t for t in (selected_types or []) if t in type_choices]
            info_text = f"**{len(kp_list)} keypoints**\n\n{format_table(kp_list)}"

            return gr.update(value=player_html), kp_list, gr.update(choices=type_choices, value=new_selected), info_text

        del_btn.click(
            fn=delete_filtered_keypoint,
            inputs=[audio_input, del_idx, filtered_keypoints, current_sections, filtered_type_filter, current_features],
            outputs=[filtered_kp_player, filtered_keypoints, filtered_type_filter, filtered_kp_info]
        )

        # Preview segments
        preview_seg_btn.click(
            fn=preview_segments,
            inputs=[audio_input, filtered_keypoints, f_min_seg, f_max_seg],
            outputs=[segment_preview]
        )

        # ==================== Type Filter Callbacks ====================
        def apply_raw_type_filter(raw_kps, selected_types):
            """Apply type filter to raw keypoints table"""
            kp_list = list(raw_kps) if raw_kps else []
            if selected_types is None:
                selected_types = []
            else:
                selected_types = list(selected_types)

            shown_keypoints = _filter_keypoints_by_types(kp_list, selected_types)
            if not selected_types:
                # Show all if nothing selected
                shown_keypoints = kp_list
            info_text = f"**Keypoints Shown ({len(shown_keypoints)}/{len(kp_list)})**\n\n{format_table(shown_keypoints)}"
            return info_text

        raw_type_filter.change(
            fn=apply_raw_type_filter,
            inputs=[raw_keypoints, raw_type_filter],
            outputs=[raw_kp_info]
        )

        def apply_filtered_type_filter(filtered_kps, selected_types):
            """Apply type filter to filtered keypoints table"""
            kp_list = list(filtered_kps) if filtered_kps else []
            if selected_types is None:
                selected_types = []
            else:
                selected_types = list(selected_types)

            shown_keypoints = _filter_keypoints_by_types(kp_list, selected_types)
            if not selected_types:
                # Show all if nothing selected
                shown_keypoints = kp_list
            info_text = f"**Keypoints Shown ({len(shown_keypoints)}/{len(kp_list)})**\n\n{format_table(shown_keypoints)}"
            return info_text

        filtered_type_filter.change(
            fn=apply_filtered_type_filter,
            inputs=[filtered_keypoints, filtered_type_filter],
            outputs=[filtered_kp_info]
        )

        # ==================== STEP 4: Generate & Save Caption Callback ====================
        def generate_caption_callback(audio_file, filtered_kps, sections, summary, output_path,
                                      batch_size, temperature, top_p, max_tokens, min_seg, max_seg):
            """Generate captions and save to JSON"""
            if not audio_file:
                return "❌ **Error**: Please upload an audio file first"

            if not filtered_kps or len(filtered_kps) == 0:
                return "❌ **Error**: No filtered keypoints. Complete Steps 1-3 first."

            if not sections:
                return "❌ **Error**: No structure sections. Run Step 2 first."

            if not output_path or not output_path.strip():
                return "❌ **Error**: Please specify an output path"

            # Get audio path
            if isinstance(audio_file, str):
                audio_path = audio_file
            elif isinstance(audio_file, tuple):
                audio_path = audio_file[0]
            else:
                audio_path = audio_file.name

            output_path = output_path.strip()

            print(f"\n⏳ Generating caption...")
            print(f"  Audio: {os.path.basename(audio_path)}")
            print(f"  Keypoints: {len(filtered_kps)}")
            print(f"  Sections: {len(sections)}")
            print(f"  Output: {output_path}")

            try:
                result = caption_generator.generate_caption(
                    audio_path=audio_path,
                    keypoints=filtered_kps,
                    sections=sections,
                    output_path=output_path,
                    batch_size=int(batch_size),
                    min_segment_duration=float(min_seg),
                    max_segment_duration=float(max_seg),
                    temperature=float(temperature),
                    top_p=float(top_p),
                    max_tokens=int(max_tokens),
                    overall_summary=summary or "",
                )

                if result.get('success'):
                    final_result = result.get('result', {})
                    n_sections = len(final_result.get('sections', []))
                    n_subsegments = sum(
                        len(s.get('detailed_analysis', {}).get('sections', []))
                        for s in final_result.get('sections', [])
                    )
                    return (
                        f"✅ **Caption Generated Successfully!**\n\n"
                        f"- **Output file**: `{output_path}`\n"
                        f"- **Level 1 sections**: {n_sections}\n"
                        f"- **Level 2 sub-segments**: {n_subsegments}\n"
                        f"- **Keypoints used**: {len(filtered_kps)}\n\n"
                        f"You can now use this caption file for video editing."
                    )
                else:
                    error_msg = result.get('error', 'Unknown error')
                    return f"❌ **Error**: {error_msg}"

            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                return f"❌ **Error**: {str(e)}\n\n```\n{tb}\n```"

        generate_btn.click(
            fn=generate_caption_callback,
            inputs=[audio_input, filtered_keypoints, current_sections, structure_summary, caption_output_path,
                    k_batch, k_temp, k_top_p, k_max_tokens, f_min_seg, f_max_seg],
            outputs=[caption_status]
        )

        # ==================== Parameter Save/Reset Callbacks ====================

        def save_current_params(
            methods,
            bpb, dbn_th, correct_beats, min_bpm, max_bpm, num_tempi, transition_lambda, observation_lambda, fps,
            pitch_tolerance, pitch_threshold, pitch_min_distance, pitch_nms_method, pitch_max_points,
            mel_win_s, mel_n_filters, mel_threshold_ratio, mel_min_distance, mel_nms_method, mel_max_points,
            merge, min_int, topk, energy_pct, silence_th,
            s_temp, s_top_p, s_max_tok,
            f_total_shots, f_min_int, f_min_seg, f_max_seg,
            f_weight_db, f_weight_p, f_weight_mel,
            k_batch, k_temp, k_top_p, k_max_tok
        ):
            m = methods or []
            if isinstance(m, str):
                m = [m]
            params = {
                "detection_methods": list(m),
                # Keep old single-selection key for backward compatibility.
                "detection_method": (list(m)[0] if m else "downbeat"),
                "beats_per_bar": bpb,
                "dbn_threshold": dbn_th,
                "correct_beats": bool(correct_beats),
                "min_bpm": float(min_bpm),
                "max_bpm": float(max_bpm),
                "num_tempi": int(num_tempi),
                "transition_lambda": float(transition_lambda),
                "observation_lambda": int(observation_lambda),
                "fps": int(fps),
                # Pitch
                "pitch_tolerance": float(pitch_tolerance),
                "pitch_threshold": float(pitch_threshold),
                "pitch_min_distance": float(pitch_min_distance),
                "pitch_nms_method": str(pitch_nms_method),
                "pitch_max_points": int(pitch_max_points),
                # Mel
                "mel_win_s": int(mel_win_s),
                "mel_n_filters": int(mel_n_filters),
                "mel_threshold_ratio": float(mel_threshold_ratio),
                "mel_min_distance": float(mel_min_distance),
                "mel_nms_method": str(mel_nms_method),
                "mel_max_points": int(mel_max_points),
                "merge_close": merge,
                "min_interval": min_int,
                "top_k": topk,
                "energy_percentile": energy_pct,
                "silence_threshold_db": silence_th,
                # Other params
                "structure_temperature": s_temp,
                "structure_top_p": s_top_p,
                "structure_max_tokens": s_max_tok,
                "filter_total_shots": int(f_total_shots),
                "filter_min_interval": f_min_int,
                "filter_min_segment": f_min_seg,
                "filter_max_segment": f_max_seg,
                "weight_downbeat": float(f_weight_db),
                "weight_pitch": float(f_weight_p),
                "weight_mel_energy": float(f_weight_mel),
                "keypoint_batch_size": k_batch,
                "keypoint_temperature": k_temp,
                "keypoint_top_p": k_top_p,
                "keypoint_max_tokens": k_max_tok,
            }
            msg = save_params_to_file(params)
            return msg

        save_params_btn.click(
            fn=save_current_params,
            inputs=madmom_inputs + [s_temp, s_top_p, s_max_tokens,
                                    f_total_shots, f_min_interval, f_min_seg, f_max_seg,
                                    f_weight_downbeat, f_weight_pitch, f_weight_mel_energy,
                                    k_batch, k_temp, k_top_p, k_max_tokens],
            outputs=[params_status]
        )

        def reset_all_params():
            defaults = reset_params_to_default()
            return (
                [default_detection_method],
                defaults["beats_per_bar"],
                defaults["dbn_threshold"],
                defaults.get("correct_beats", True),
                defaults.get("min_bpm", 55.0),
                defaults.get("max_bpm", 215.0),
                defaults.get("num_tempi", 60),
                defaults.get("transition_lambda", 100.0),
                defaults.get("observation_lambda", 16),
                defaults.get("fps", 100),
                defaults.get("pitch_tolerance", 0.8),
                defaults.get("pitch_threshold", 0.8),
                defaults.get("pitch_min_distance", 0.5),
                defaults.get("pitch_nms_method", "basic"),
                defaults.get("pitch_max_points", 20),
                defaults.get("mel_win_s", 512),
                defaults.get("mel_n_filters", 40),
                defaults.get("mel_threshold_ratio", 0.3),
                defaults.get("mel_min_distance", 0.5),
                defaults.get("mel_nms_method", "basic"),
                defaults.get("mel_max_points", 20),
                defaults["merge_close"],
                defaults["min_interval"],
                defaults["top_k"],
                defaults.get("energy_percentile", 0),
                defaults.get("silence_threshold_db", -45.0),
                # Other params
                defaults["structure_temperature"],
                defaults["structure_top_p"],
                defaults["structure_max_tokens"],
                defaults.get("filter_total_shots", 20),
                defaults["filter_min_interval"],
                defaults["filter_min_segment"],
                defaults["filter_max_segment"],
                defaults.get("weight_downbeat", 1.0),
                defaults.get("weight_pitch", 1.0),
                defaults.get("weight_mel_energy", 1.0),
                defaults["keypoint_batch_size"],
                defaults["keypoint_temperature"],
                defaults["keypoint_top_p"],
                defaults["keypoint_max_tokens"],
                "🔄 Parameters reset to default values"
            )

        reset_params_btn.click(
            fn=reset_all_params,
            inputs=[],
            outputs=madmom_inputs + [s_temp, s_top_p, s_max_tokens,
                                     f_total_shots, f_min_interval, f_min_seg, f_max_seg,
                                     f_weight_downbeat, f_weight_pitch, f_weight_mel_energy,
                                     k_batch, k_temp, k_top_p, k_max_tokens,
                                     params_status]
        )

    return demo
