import random
import json
import html
import numpy as np
from typing import List, Dict
from .utils import get_audio_data, get_audio_base64

TYPE_COLORS = {
    'Downbeat': '#FF4444', 'Attack': '#44AA44', 'Onset': '#44AA44',
    'Spectral': '#FF8800', 'Energy': '#AA44AA', 'Timbre': '#8B4513',
    'Tonality': '#00FF00', # Bright Green for Tonality/Whistle
    'ZCR': '#00FFFF', # Cyan
    'Rolloff': '#FFFF00', # Yellow
    'Flatness': '#FFC0CB', # Pink
    'Centroid': '#ADD8E6', # Light Blue
    # Volume Features
    'DynamicRange': '#FF6B9D', # Pink-Red for Dynamic Range
    'LoudnessEnvelope': '#9D4EDD', # Purple for Loudness Envelope
    'VolumeGradient': '#FF9F1C', # Orange for Volume Gradient
    # Spectral Features
    'SpectralBandwidth': '#06FFA5', # Mint Green for Spectral Bandwidth
    'SpectralContrast': '#4ECDC4', # Teal for Spectral Contrast
    'MFCCChange': '#FF006E', # Hot Pink for MFCC Change
    # Auto-generated
    'AutoSplit': '#999999', # Gray for auto-split points
    'Section Start': '#4444FF', 'Section End': '#00CCCC',
    'Manual': '#FF00FF', 'Unknown': '#888888',
}

def get_keypoint_color(kp_type: str) -> str:
    """Get color for keypoint type."""
    for type_key, color in TYPE_COLORS.items():
        if type_key.lower() in kp_type.lower():
            return color
    return TYPE_COLORS['Unknown']

def compute_waveform_data(audio_path: str, num_bars: int = 400) -> List[float]: # Increased bars for full width
    """Compute waveform data"""
    audio_data, duration = get_audio_data(audio_path)
    if num_bars <= 0:
        return []
    # Guard against very short audio where integer division would be 0.
    samples_per_bar = max(1, len(audio_data) // num_bars)
    waveform = []
    for i in range(num_bars):
        start_idx = i * samples_per_bar
        end_idx = min(start_idx + samples_per_bar, len(audio_data))
        if start_idx < len(audio_data):
            segment = audio_data[start_idx:end_idx]
            rms = np.sqrt(np.mean(segment ** 2)) if len(segment) > 0 else 0
            waveform.append(float(rms))
        else:
            waveform.append(0.0)
    max_val = max(waveform) if waveform else 1.0
    return [v / max_val for v in waveform] if max_val > 0 else waveform

def generate_waveform_svg(waveform_data: List[float], width_percent: int = 100) -> str:
    """Generate SVG string"""
    num_bars = len(waveform_data)
    if num_bars == 0: return ""
    
    # We use percentage for width to fill container
    bar_width = 100.0 / num_bars
    bars_svg = []
    
    for i, amp in enumerate(waveform_data):
        height = max(5, amp * 90)
        y = (100 - height) / 2
        x = i * bar_width
        # Use simple rects
        bars_svg.append(
            f'<rect x="{x:.2f}%" y="{y:.1f}%" width="{bar_width*0.8:.2f}%" height="{height:.1f}%" fill="url(#grad1)" />'
        )

    return f'''
    <svg width="100%" height="100%" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="grad1" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" style="stop-color:#00d9ff;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#0055ff;stop-opacity:1" />
            </linearGradient>
        </defs>
        {''.join(bars_svg)}
    </svg>
    '''


# Section colors for Omni structure visualization
SECTION_COLORS = [
    'rgba(255, 99, 132, 0.3)',   # Red
    'rgba(54, 162, 235, 0.3)',   # Blue
    'rgba(255, 206, 86, 0.3)',   # Yellow
    'rgba(75, 192, 192, 0.3)',   # Teal
    'rgba(153, 102, 255, 0.3)',  # Purple
    'rgba(255, 159, 64, 0.3)',   # Orange
    'rgba(199, 199, 199, 0.3)',  # Gray
    'rgba(83, 102, 255, 0.3)',   # Indigo
]


def _mmss_to_seconds(mmss) -> float:
    """Convert MM:SS or numeric value to seconds."""
    if isinstance(mmss, (int, float)):
        return float(mmss)
    try:
        parts = str(mmss).split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(mmss)
    except:
        return 0.0


def generate_sections_svg(uid: str, sections: List[Dict], duration: float) -> str:
    """Render section regions as colored rectangles on the waveform."""
    if not sections or not duration or duration <= 0:
        return ""

    rect_elems: List[str] = []
    label_elems: List[str] = []

    for i, sec in enumerate(sections):
        start = _mmss_to_seconds(sec.get('Start_Time', 0))
        end = _mmss_to_seconds(sec.get('End_Time', duration))
        name = sec.get('name', f'Section {i+1}')

        # Calculate position as percentage
        x = (start / duration) * 100.0
        width = ((end - start) / duration) * 100.0
        x = max(0.0, min(100.0, x))
        width = max(0.0, min(100.0 - x, width))

        color = SECTION_COLORS[i % len(SECTION_COLORS)]

        # Create rectangle for section (in stretched SVG)
        rect_elems.append(
            f'<rect x="{x:.2f}%" y="0" width="{width:.2f}%" height="100%" '
            f'fill="{color}" style="pointer-events:none;" />'
        )

        # Create HTML label (not stretched)
        label_x = x + 0.3
        label_elems.append(
            f'<div style="position:absolute; left:{label_x:.2f}%; top:4px; '
            f'font-size:11px; font-weight:bold; color:white; '
            f'text-shadow: 1px 1px 2px black, -1px -1px 2px black; '
            f'pointer-events:none; white-space:nowrap; z-index:3;">'
            f'{html.escape(name)}</div>'
        )

    # SVG for rectangles only (stretched)
    svg_part = (
        f'<svg id="{uid}_sections_svg" width="100%" height="100%" viewBox="0 0 100 100" preserveAspectRatio="none" '
        'xmlns="http://www.w3.org/2000/svg" style="position:absolute; inset:0; z-index:2;">'
        + ''.join(rect_elems)
        + '</svg>'
    )

    # HTML div for labels (not stretched)
    labels_part = ''.join(label_elems)

    return svg_part + labels_part


def generate_markers_svg(uid: str, keypoints: List[Dict], duration: float) -> str:
    """Render keypoint markers as a single SVG overlay (fewer DOM nodes than div markers)."""
    if not keypoints or not duration or duration <= 0:
        return ""

    elems: List[str] = []
    for i, kp in enumerate(keypoints):
        t = float(kp.get('time', 0.0) or 0.0)
        # Strip whitespace to ensure consistency with filter logic
        kp_type = str(kp.get('type', 'Unknown') or 'Unknown').strip()
        color = get_keypoint_color(kp_type)
        x = (t / duration) * 100.0
        x = max(0.0, min(100.0, x))
        title = f"#{i+1} {kp_type} ({t:.2f}s)"
        onclick = f"var a=document.getElementById('{uid}_audio'); if(a){{ a.currentTime={t:.6f}; if(a._vcaUpdate) a._vcaUpdate(); }}"

        left = max(0.0, x - 0.6)
        right = min(100.0, x + 0.6)

        # Use html.escape to safely embed the type in the data attribute
        safe_type = html.escape(kp_type, quote=True)

        # Group per marker so we can toggle visibility by type.
        # data-kptype is read by client-side filter logic.
        elems.append(
            f'<g data-kptype="{safe_type}">'
            f'<title>{title}</title>'
            f'<line x1="{x:.4f}" y1="0" x2="{x:.4f}" y2="100" '
            f'stroke="{color}" stroke-width="0.25" opacity="0.65" style="pointer-events:none;" />'
            f'<polygon points="{left:.4f},0 {right:.4f},0 {x:.4f},4" '
            f'fill="{color}" opacity="0.95" onclick="{onclick}" '
            f'style="cursor:pointer; pointer-events:all;" />'
            '</g>'
        )

    return (
        f'<svg id="{uid}_markers_svg" width="100%" height="100%" viewBox="0 0 100 100" preserveAspectRatio="none" '
        'xmlns="http://www.w3.org/2000/svg" style="position:absolute; inset:0; z-index:6;">'
        + ''.join(elems)
        + '</svg>'
    )

def create_full_width_player(audio_path: str, keypoints: List[Dict], title: str = "Analysis Result", sections: List[Dict] = None, extra_features: Dict = None) -> str:
    """
    Generates the HTML for the full-width player.
    CRITICAL FIX: Uses requestAnimationFrame for smooth playback visualization.

    Args:
        audio_path: Path to audio file
        keypoints: List of keypoint dicts with time, type, intensity
        title: Title to display
        sections: Optional list of section dicts with name, Start_Time, End_Time (for Omni structure)
        extra_features: Optional dict containing feature arrays (rms, spectral_flux, etc.)
    """
    if not audio_path:
        return '<div style="padding:20px;text-align:center;color:#666;">No audio loaded. Run analysis to visualize.</div>'

    _, duration = get_audio_data(audio_path)
    b64_audio = get_audio_base64(audio_path)
    waveform = compute_waveform_data(audio_path, num_bars=300)
    svg_waveform = generate_waveform_svg(waveform)

    # Generate unique ID with timestamp to force refresh on each render
    import time
    uid = f"player_{int(time.time() * 1000)}_{random.randint(10000, 99999)}"

    # Render markers as a single SVG overlay (less DOM, smoother when many points)
    markers_svg = generate_markers_svg(uid, keypoints, duration)

    # Render sections as colored regions (for Omni structure visualization)
    sections_svg = generate_sections_svg(uid, sections or [], duration)

    dur_min = int(duration // 60)
    dur_sec = duration % 60

    # Header info text
    header_info = f"{len(keypoints)} Markers"
    if sections:
        header_info += f" | {len(sections)} Sections"

    # We keep all runtime JS in inline event handlers because Gradio's HTML is often
    # injected via innerHTML, where <script> tags may not execute.
    duration_js = f"{float(duration):.8f}"

    # Prepare features JSON
    features_json = "{}"
    if extra_features:
        features_data = {
            'rms': extra_features.get('rms', []),
            'flux': extra_features.get('spectral_flux', []),
            'change': extra_features.get('energy_change', []),
            'centroid': extra_features.get('centroid_change', []),
            'zcr': extra_features.get('zcr', []),
            'rolloff': extra_features.get('rolloff', []),
            'flatness': extra_features.get('flatness', [])
        }
        features_json = json.dumps(features_data)

    html = f"""
    <div id="{uid}_container" style="background:#1a1a2e; color:white; border-radius:8px; overflow:hidden; border:1px solid #333; margin-top:10px;">
        <div style="padding:10px 15px; background:#16213e; border-bottom:1px solid #333; display:flex; justify-content:space-between; align-items:center;">
            <span style="font-weight:bold; color:#00d9ff;">{title}</span>
            <span style="font-size:12px; color:#aaa;">{header_info}</span>
        </div>

        <div id="{uid}_data" style="display:none;">{features_json}</div>

        <div id="{uid}_wave" style="position:relative; height:150px; background:#0f172a; cursor:pointer;"
             onclick="var r=this.getBoundingClientRect(); var p=(event.clientX-r.left)/r.width; var t = Math.max(0, Math.min({duration_js}, p*{duration_js})); var a=document.getElementById('{uid}_audio'); if(a){{ a.currentTime = t; if(a._vcaUpdate) a._vcaUpdate(); }} var m=document.getElementById('{uid}_metrics'); var d=document.getElementById('{uid}_data'); if(m && d){{ if(!d._parsed && d.textContent){{ try{{ d._parsed=JSON.parse(d.textContent); }}catch(e){{}} }} var f=d._parsed; if(f){{ var h=''; var g=function(arr,tm){{ if(!arr||arr.length==0)return 'N/A'; var i=Math.floor(tm/{duration_js}*arr.length); i=Math.max(0,Math.min(arr.length-1,i)); return arr[i].toFixed(4); }}; if(f.rms&&f.rms.length) h+='<span style=\\'margin-right:15px\\'>Energy: '+g(f.rms,t)+'</span>'; if(f.flux&&f.flux.length) h+='<span style=\\'margin-right:15px\\'>Flux: '+g(f.flux,t)+'</span>'; if(f.change&&f.change.length) h+='<span style=\\'margin-right:15px\\'>Change: '+g(f.change,t)+'</span>'; if(f.centroid&&f.centroid.length) h+='<span style=\\'margin-right:15px\\'>Centroid: '+g(f.centroid,t)+'</span>'; if(f.zcr&&f.zcr.length) h+='<span style=\\'margin-right:15px\\'>ZCR: '+g(f.zcr,t)+'</span>'; if(f.rolloff&&f.rolloff.length) h+='<span style=\\'margin-right:15px\\'>Rolloff: '+g(f.rolloff,t)+'</span>'; if(f.flatness&&f.flatness.length) h+='<span>Flatness: '+g(f.flatness,t)+'</span>'; m.innerHTML=h; }} }} ">

            {sections_svg}

            <div style="position:absolute; width:100%; height:100%; opacity:0.8; z-index:3;">
                {svg_waveform}
            </div>

            {markers_svg}
            
            <div id="{uid}_prog" style="position:absolute; top:0; left:0; height:100%; width:100%; background:rgba(255,255,255,0.14); pointer-events:none; border-right:1px solid rgba(255,255,255,0.45); transform-origin: 0% 50%; transform: scaleX(0); will-change: transform; z-index:4;"></div>
            
        </div>
        
        <div style="padding:10px 15px; display:flex; align-items:center; gap:15px; background:#16213e;">
            <button id="{uid}_btn" 
                onclick="var a = document.getElementById('{uid}_audio'); if(a) {{ if(a.paused) a.play(); else a.pause(); }}"
                style="background:#00d9ff; border:none; border-radius:4px; padding:5px 15px; color:#000; font-weight:bold; cursor:pointer; width:80px;">
                ▶ Play
            </button>
            
            <div style="font-family:monospace; font-size:14px; background:#000; padding:4px 8px; border-radius:4px; border:1px solid #333;">
                <span id="{uid}_time" style="color:#00d9ff;">00:00.0</span> 
                <span style="color:#666;">/ {dur_min:02d}:{dur_sec:05.2f}</span>
            </div>
            
            <div style="flex-grow:1; text-align:right; font-size:12px; color:#888;">
                <span style="color:#FF4444">■ Downbeat</span> 
                <span style="color:#44AA44; margin-left:8px;">■ Onset</span>
                <span style="color:#4444FF; margin-left:8px;">■ Section</span>
                <span style="color:#FF00FF; margin-left:8px;">■ Manual</span>
            </div>
        </div>

        <div id="{uid}_metrics" style="padding:5px 15px; background:#16213e; border-top:1px solid #333; font-family:monospace; font-size:12px; color:#00d9ff; min-height:28px; display:flex; align-items:center;">
            <span style="color:#666;">Click on waveform to see metrics...</span>
        </div>

        <audio id="{uid}_audio" src="{b64_audio}" preload="auto" playsinline
            onloadedmetadata="(function(a){{
                // Clean up old player instances to prevent stale state
                if(window._vcaApplyTypeFilters) {{
                    window._vcaApplyTypeFilters = window._vcaApplyTypeFilters.filter(function(fn) {{
                        // Remove filters that reference removed DOM elements
                        try {{
                            var testCall = fn.toString();
                            return true; // Keep all for now, cleanup happens automatically
                        }} catch(e) {{ return false; }}
                    }});
                }}

                if(a._vcaUpdate) a._vcaUpdate();
                if(!a._vcaApplyFilter){{
                    a._vcaApplyFilter=function(){{
                        var svg=document.getElementById('{uid}_markers_svg');
                        if(!svg) return;

                        // Use global selected types if available (set by Gradio callback or delegated listener)
                        var selectedTypes = window._vcaSelectedTypes;
                        if(!selectedTypes || !Array.isArray(selectedTypes)) {{
                            // Show all by default
                            svg.querySelectorAll('g[data-kptype]').forEach(function(g){{ g.style.display=''; }});
                            return;
                        }}

                        var selectedSet = new Set(selectedTypes);
                        var hasSelection = selectedSet.size > 0;
                        svg.querySelectorAll('g[data-kptype]').forEach(function(g){{
                            var t = g.getAttribute('data-kptype') || 'Unknown';
                            g.style.display = hasSelection && selectedSet.has(t) ? '' : 'none';
                        }});
                    }};
                }}

                // Global delegated listener for checkbox changes
                if(!window._vcaTypeFilterDelegated){{
                    window._vcaTypeFilterDelegated = true;
                    window._vcaApplyTypeFilters = [];
                    document.addEventListener('change', function(ev){{
                        var tgt = ev && ev.target;
                        if(tgt && tgt.type === 'checkbox'){{
                            var tf = tgt.closest('#raw_type_filter') || tgt.closest('#filtered_type_filter') || tgt.closest('#type_filter');
                            if(tf){{
                                // Extract selected types from DOM and store globally
                                window._vcaSelectedTypes = [];
                                tf.querySelectorAll('input[type=checkbox]:checked').forEach(function(inp){{
                                    var label = inp.closest('label');
                                    if(!label) return;

                                    // Method 1: Try input value (Gradio often sets this to the choice value)
                                    var v = (inp.value || '').trim();
                                    if(v && v !== 'on' && v !== 'true') {{
                                        window._vcaSelectedTypes.push(v);
                                        return;
                                    }}

                                    // Method 2: Find label text - look for spans from last to first
                                    // (first span is often the checkbox indicator, last is often the label)
                                    var spans = label.querySelectorAll('span');
                                    for(var i = spans.length - 1; i >= 0; i--) {{
                                        var txt = (spans[i].textContent || '').trim();
                                        // Skip checkbox indicators (single char, checkmarks, empty)
                                        if(txt && txt.length > 1 && !/^[✓✔☐☑✅❎⬜⬛▢▣□■]+$/.test(txt)) {{
                                            window._vcaSelectedTypes.push(txt);
                                            return;
                                        }}
                                    }}

                                    // Method 3: Full label text as fallback
                                    var fullText = (label.textContent || '').replace(/[✓✔☐☑✅❎⬜⬛▢▣□■]/g, '').trim();
                                    if(fullText) window._vcaSelectedTypes.push(fullText);
                                }});

                                // Call all filter functions
                                (window._vcaApplyTypeFilters || []).forEach(function(fn){{
                                    try {{ fn(); }} catch(e) {{}}
                                }});
                            }}
                        }}
                    }}, true);
                }}
                if(window._vcaApplyTypeFilters.indexOf(a._vcaApplyFilter) < 0){{
                    window._vcaApplyTypeFilters.push(a._vcaApplyFilter);
                }}
            }})(this);"
            ontimeupdate="if(this._vcaUpdate) this._vcaUpdate();"
            onseeking="if(this._vcaUpdate) this._vcaUpdate();"
            onplay="(function(a){{
                const duration={duration_js};
                if(!a._vcaRefs){{
                    a._vcaRefs={{
                        prog: document.getElementById('{uid}_prog'),
                        timeEl: document.getElementById('{uid}_time'),
                        btn: document.getElementById('{uid}_btn')
                    }};
                }}
                const r=a._vcaRefs;
                if(!r || !r.prog || !r.timeEl || !r.btn) return;

                if(!a._vcaFmt){{
                    a._vcaFmt=function(t){{
                        if(!isFinite(t) || t < 0) t = 0;
                        const m = Math.floor(t / 60);
                        const s = t - m * 60;
                        const mm = String(m).padStart(2, '0');
                        const ss = s.toFixed(1).padStart(4, '0');
                        return mm + ':' + ss;
                    }};
                }}

                if(!a._vcaUpdate){{
                    a._vcaUpdate=function(){{
                        const t = a.currentTime || 0;
                        const p = duration > 0 ? Math.max(0, Math.min(1, t / duration)) : 0;
                        r.prog.style.transform = 'scaleX(' + p + ')';
                        r.timeEl.textContent = a._vcaFmt(t);
                    }};
                }}

                r.btn.textContent='⏸ Pause';
                if(a._vcaRaf){{ cancelAnimationFrame(a._vcaRaf); a._vcaRaf=null; }}
                a._vcaLastTs = 0;
                if(!a._vcaTick){{
                    a._vcaTick=function(ts){{
                        if(a.paused || a.ended){{ a._vcaRaf=null; a._vcaUpdate(); return; }}
                        if(!a._vcaLastTs || (ts - a._vcaLastTs) >= 33){{ a._vcaUpdate(); a._vcaLastTs = ts; }}
                        a._vcaRaf = requestAnimationFrame(a._vcaTick);
                    }};
                }}

                a._vcaUpdate();
                if(a._vcaApplyFilter) a._vcaApplyFilter();
                a._vcaRaf = requestAnimationFrame(a._vcaTick);
            }})(this);"
            onpause="if(this._vcaRaf){{ cancelAnimationFrame(this._vcaRaf); this._vcaRaf=null; }} var b=document.getElementById('{uid}_btn'); if(b) b.textContent='▶ Play'; if(this._vcaUpdate) this._vcaUpdate();"
            onended="if(this._vcaRaf){{ cancelAnimationFrame(this._vcaRaf); this._vcaRaf=null; }} var b=document.getElementById('{uid}_btn'); if(b) b.textContent='▶ Play'; if(this._vcaUpdate) this._vcaUpdate();">
        </audio>
    </div>
    """
    return html

def format_table(data_list: List[Dict], type_key='type') -> str:
    if not data_list: return "No data."
    headers = list(data_list[0].keys())
    # Simplify for keypoints - show raw features
    if 'time' in headers:
        lines = ["| # | Time (s) | Type | Description | Intensity |", "|---|---|---|---|---|"]
        for i, d in enumerate(data_list[:100]):
            kp_type = d.get(type_key, '')
            intensity = d.get('intensity', 0.0)
            
            # Determine description based on type
            desc = "-"
            if 'VolumeGradient' in kp_type:
                desc = "Detect volume change rate"
            elif 'DynamicRange' in kp_type:
                desc = "Detect volume dynamic range changes"
            elif 'LoudnessEnvelope' in kp_type:
                desc = "Detect loudness envelope peaks"
            elif 'SpectralBandwidth' in kp_type:
                desc = "Detect spectral bandwidth changes"
            elif 'SpectralContrast' in kp_type:
                desc = "Detect spectral contrast changes"
            elif 'MFCCChange' in kp_type:
                desc = "Detect MFCC feature changes"
            elif 'Downbeat' in kp_type:
                desc = "Rhythmic strong beat"
            elif 'Onset' in kp_type:
                desc = "Note onset / Attack"
            elif 'Flux' in kp_type:
                desc = "Spectral flux change"
            elif 'Change' in kp_type:
                desc = "Energy change"
            elif 'Centroid' in kp_type:
                desc = "Timbre/Brightness change"
            elif 'Tonality' in kp_type:
                desc = "Tonality change"
            elif 'ZCR' in kp_type:
                desc = "Zero Crossing Rate peak"
            elif 'Rolloff' in kp_type:
                desc = "Spectral Rolloff peak"
            elif 'Flatness' in kp_type:
                desc = "Spectral Flatness peak"
            
            lines.append(f"| {i+1} | {d.get('time',0):.3f} | {kp_type} | {desc} | {intensity:.2f} |")
    else:
        lines = [f"| {' | '.join(headers)} |", f"| {' | '.join(['---']*len(headers))} |"]
        for d in data_list:
            lines.append(f"| {' | '.join([str(d.get(h,'')) for h in headers])} |")
    return "\n".join(lines)


def _unique_keypoint_types(keypoints: List[Dict]) -> List[str]:
    types: List[str] = []
    seen = set()
    for kp in keypoints or []:
        t = str(kp.get('type', 'Unknown') or 'Unknown').strip()
        if t not in seen:
            seen.add(t)
            types.append(t)
    return sorted(types)


def _filter_keypoints_by_types(keypoints: List[Dict], selected_types: List[str]) -> List[Dict]:
    if not keypoints:
        return []
    # If nothing selected, show nothing.
    if not selected_types:
        return []
    selected = set(str(t).strip() for t in selected_types)
    return [kp for kp in keypoints if str(kp.get('type', 'Unknown') or 'Unknown').strip() in selected]
