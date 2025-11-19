#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional


def parse_time_to_seconds(hhmmss: str) -> float:
    parts = hhmmss.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid time format: {hhmmss}")
    h, m, s = map(int, parts)
    return h * 3600 + m * 60 + s


def seconds_to_time_label(seconds_value: float) -> str:
    total_seconds = int(seconds_value)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    frac = seconds_value - int(seconds_value)
    if abs(frac - 0.5) < 1e-6:
        return f"{h:02d}:{m:02d}:{s:02d}.5"
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_time_hhmmss_or_mmss(text: str) -> float:
    text = text.strip()
    if re.match(r"^\d{2}:\d{2}:\d{2}$", text):
        return parse_time_to_seconds(text)
    if re.match(r"^\d{2}:\d{2}$", text):
        m, s = text.split(":")
        return int(m) * 60 + int(s)
    raise ValueError(f"Invalid time format: {text}")


def load_segments_time_window(report_html_path: Path) -> Tuple[float, float]:
    seg_path = report_html_path.with_name("segments.json")
    if not seg_path.exists():
        html_text = report_html_path.read_text(encoding="utf-8")
        times = re.findall(r"时间段:\s*(\d{2}:\d{2}:\d{2})\s*to\s*(\d{2}:\d{2}:\d{2})", html_text)
        if not times:
            raise FileNotFoundError("segments.json not found and no time ranges detected in HTML.")
        starts = [parse_time_to_seconds(t0) for t0, _ in times]
        ends = [parse_time_to_seconds(t1) for _, t1 in times]
        return min(starts), max(ends)

    data = json.loads(seg_path.read_text(encoding="utf-8"))
    starts: List[float] = []
    ends: List[float] = []
    for item in data:
        tr = item.get("time_range", "").strip()
        m = re.match(r"^(\d{2}:\d{2}:\d{2})\s*to\s*(\d{2}:\d{2}:\d{2})$", tr)
        if m:
            starts.append(parse_time_to_seconds(m.group(1)))
            ends.append(parse_time_to_seconds(m.group(2)))
    if not starts or not ends:
        raise ValueError("No valid time_range entries found in segments.json")
    return min(starts), max(ends)


def list_frame_files(frames_dir: Path) -> List[Path]:
    candidates = [p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    def key_fn(p: Path):
        m = re.search(r"(\d+)", p.stem)
        return int(m.group(1)) if m else 0
    return sorted(candidates, key=key_fn)


def sample_indices(num_items: int, max_items: int) -> List[int]:
    if num_items <= max_items:
        return list(range(num_items))
    step = (num_items - 1) / (max_items - 1)
    return [round(i * step) for i in range(max_items)]


def build_gallery_html(relative_dir: str, frame_names: List[str], frame_ids: List[int], fps: float) -> str:
    items_html = []
    for name, fid in zip(frame_names, frame_ids):
        t_seconds = fid / fps
        label = seconds_to_time_label(t_seconds)
        items_html.append(
            "            <div style='display:flex;flex-direction:column'>\n"
            f"              <img src='{relative_dir}/{name}' alt='{name}' style='width:100%;border-radius:4px;border:1px solid #333'/>\n"
            f"              <div style='font-size:12px;color:#aaa;margin-top:4px;text-align:center'>{label}</div>\n"
            "            </div>"
        )

    gallery = (
        "\n            <div style='margin:24px 0;padding:12px 0;border-top:1px solid #2a2a2a'>\n"
        "              <h3 style='margin:0 0 12px 0;color:#ddd'>帧预览（fps=2）</h3>\n"
        "              <div style='display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:8px'>\n"
        + "\n".join(items_html)
        + "\n              </div>\n"
        + "            </div>\n"
    )
    return gallery


def inject_gallery_into_html(html_text: str, gallery_block: str) -> str:
    marker = "\n          </div>\n        </body>"
    idx = html_text.rfind(marker)
    if idx == -1:
        body_idx = html_text.rfind("\n        </body>")
        if body_idx == -1:
            return html_text + gallery_block
        return html_text[:body_idx] + gallery_block + html_text[body_idx:]
    insert_pos = idx
    return html_text[:insert_pos] + gallery_block + html_text[insert_pos:]


def detect_frame_padding(example_name: str) -> int:
    m = re.search(r"(\d+)$", Path(example_name).stem)
    if not m:
        return 6
    return len(m.group(1))


def extract_frame_id(path: Path) -> Optional[int]:
    m = re.search(r"(\d+)", path.stem)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def load_segments_list(report_html_path: Path) -> List[Tuple[float, float]]:
    seg_path = report_html_path.with_name("segments.json")
    segments: List[Tuple[float, float]] = []
    if seg_path.exists():
        data = json.loads(seg_path.read_text(encoding="utf-8"))
        for item in data:
            tr = item.get("time_range", "")
            m = re.match(r"^(\d{2}:\d{2}:\d{2})\s*to\s*(\d{2}:\d{2}:\d{2})$", tr)
            if not m:
                continue
            s0 = parse_time_to_seconds(m.group(1))
            s1 = parse_time_to_seconds(m.group(2))
            segments.append((s0, s1))
        return segments

    html_text = report_html_path.read_text(encoding="utf-8")
    times = re.findall(r"时间段:\s*(\d{2}:\d{2}:\d{2})\s*to\s*(\d{2}:\d{2}:\d{2})", html_text)
    for t0, t1 in times:
        segments.append((parse_time_to_seconds(t0), parse_time_to_seconds(t1)))
    return segments


def find_existing_frame_near(frames_dir: Path, target_id: int, pad: int, search_radius: int = 30) -> Optional[Path]:
    candidate = frames_dir / f"frame_{target_id:0{pad}d}.png"
    if candidate.exists():
        return candidate
    for ext in (".jpg", ".jpeg"):
        jpg = candidate.with_suffix(ext)
        if jpg.exists():
            return jpg
    for delta in range(1, search_radius + 1):
        for sign in (-1, 1):
            idx = target_id + sign * delta
            p = frames_dir / f"frame_{idx:0{pad}d}.png"
            if p.exists():
                return p
            for ext in (".jpg", ".jpeg"):
                pj = p.with_suffix(ext)
                if pj.exists():
                    return pj
    return None


def fix_segment_images(report_path: Path, frames_src: Path, fps: float) -> None:
    report_dir = report_path.parent
    html_text = report_path.read_text(encoding="utf-8")

    html_text = re.sub(r"src='[^']*/(segment_\d{2}\.png)'", r"src='\1'", html_text)

    segments = load_segments_list(report_path)
    if not segments:
        report_path.write_text(html_text, encoding="utf-8")
        return

    all_frames = list_frame_files(frames_src)
    if not all_frames:
        report_path.write_text(html_text, encoding="utf-8")
        return

    pad = detect_frame_padding(all_frames[0].name)

    for i, (s0, s1) in enumerate(segments):
        mid_s = (s0 + s1) / 2.0
        target_id = int(round(mid_s * fps))
        src = find_existing_frame_near(frames_src, target_id, pad)
        if not src:
            continue
        dst = report_dir / f"segment_{i:02d}.png"
        try:
            shutil.copy2(src, dst)
        except Exception:
            continue

    report_path.write_text(html_text, encoding="utf-8")


def run_ffmpeg_clip(src_mp3: Path, start_s: float, end_s: float, dest_path: Path) -> None:
    duration = max(0.0, end_s - start_s)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{start_s:.3f}",
        "-t", f"{duration:.3f}",
        "-i", str(src_mp3),
        "-c", "copy",
        str(dest_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        raise FileNotFoundError("ffmpeg not found")
    except subprocess.CalledProcessError:
        cmd2 = [
            "ffmpeg",
            "-y",
            "-ss", f"{start_s:.3f}",
            "-t", f"{duration:.3f}",
            "-i", str(src_mp3),
            "-acodec", "mp3",
            str(dest_path),
        ]
        try:
            subprocess.run(cmd2, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError:
            raise FileNotFoundError("ffmpeg not found")


def ensure_segment_js(html_text: str) -> str:
    if "function __segAudioInit()" in html_text:
        return html_text
    script = (
        "\n        <script>\n"
        "        function __segAudioInit(){\n"
        "          const players = document.querySelectorAll('audio.seg-audio');\n"
        "          players.forEach(a=>{\n"
        "            const start=parseFloat(a.getAttribute('data-start')||'0');\n"
        "            const end=parseFloat(a.getAttribute('data-end')||'0');\n"
        "            let armed=false;\n"
        "            a.addEventListener('play',()=>{ if(!isNaN(start)) { a.currentTime=start; armed=true; } });\n"
        "            a.addEventListener('timeupdate',()=>{ if(armed && !isNaN(end) && a.currentTime>=end){ a.pause(); armed=false; } });\n"
        "          });\n"
        "        }\n"
        "        if(document.readyState==='loading'){ document.addEventListener('DOMContentLoaded', __segAudioInit);} else { __segAudioInit(); }\n"
        "        </script>\n"
    )
    ib = html_text.rfind("\n        </body>")
    if ib == -1:
        return html_text + script
    return html_text[:ib] + script + html_text[ib:]


def parse_proposal_sections_times(proposal_html: Path) -> List[Tuple[int, float, float]]:
    text = proposal_html.read_text(encoding="utf-8")
    sections: List[Tuple[int, float, float]] = []
    sec_iter = list(re.finditer(r"Section\s+(\d+)", text))
    for i, m in enumerate(sec_iter):
        sec_id = int(m.group(1))
        start_idx = m.end()
        end_idx = sec_iter[i + 1].start() if i + 1 < len(sec_iter) else len(text)
        block = text[start_idx:end_idx]
        tmatch = re.search(r"时间\s*</b>:\s*(\d{2}:\d{2})\s*[→\-\u2192]+\s*(\d{2}:\d{2})", block)
        if not tmatch:
            continue
        t0 = parse_time_hhmmss_or_mmss(tmatch.group(1))
        t1 = parse_time_hhmmss_or_mmss(tmatch.group(2))
        sections.append((sec_id, t0, t1))
    return sections


def inject_audio_into_proposal(proposal_html: Path, audio_src: Path, dest_subdir: str = "audio") -> None:
    text = proposal_html.read_text(encoding="utf-8")
    sec_times = parse_proposal_sections_times(proposal_html)
    if not sec_times:
        return
    html_dir = proposal_html.parent
    audio_dir = html_dir / dest_subdir
    audio_dir.mkdir(parents=True, exist_ok=True)

    inserts: List[Tuple[int, str]] = []
    ffmpeg_ok = True
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        ffmpeg_ok = False

    full_rel_name = None
    if not ffmpeg_ok:
        full_rel_name = audio_src.name
        full_copy = audio_dir / full_rel_name
        if not full_copy.exists():
            shutil.copy2(audio_src, full_copy)

    for sec_id, s0, s1 in sec_times:
        if ffmpeg_ok:
            out_name = f"proposal_section_{sec_id}.mp3"
            out_path = audio_dir / out_name
            try:
                run_ffmpeg_clip(audio_src, s0, s1, out_path)
                audio_rel_src = f"{dest_subdir}/{out_name}"
                extra_attrs = ""
            except FileNotFoundError:
                ffmpeg_ok = False
                full_rel_name = audio_src.name
                full_copy = audio_dir / full_rel_name
                if not full_copy.exists():
                    shutil.copy2(audio_src, full_copy)
                audio_rel_src = f"{dest_subdir}/{full_rel_name}"
                extra_attrs = f" class='seg-audio' data-start='{s0:.3f}' data-end='{s1:.3f}'"
        else:
            audio_rel_src = f"{dest_subdir}/{full_rel_name}"
            extra_attrs = f" class='seg-audio' data-start='{s0:.3f}' data-end='{s1:.3f}'"

        sec_pat = re.compile(rf"(Section\s+{sec_id}[\s\S]*?时间\s*</b>:\s*\d{{2}}:\d{{2}}\s*[→\-\u2192]+\s*\d{{2}}:\d{{2}})([\s\S]*?)</div>")
        m = sec_pat.search(text)
        if not m:
            continue
        insert_pos = m.end()
        audio_tag = (
            f"\n              <div style='margin-top:6px'>"
            f"<audio controls preload='none' src='{audio_rel_src}'{extra_attrs}></audio>"
            f"</div>"
        )
        inserts.append((insert_pos, audio_tag))

    for pos, snippet in sorted(inserts, key=lambda x: x[0], reverse=True):
        text = text[:pos] + snippet + text[pos:]
    if not ffmpeg_ok:
        text = ensure_segment_js(text)
    proposal_html.write_text(text, encoding="utf-8")


def inject_audio_into_shot_plan(shot_plan_html: Path, proposal_html: Path, audio_src: Path, dest_subdir: str = "audio") -> None:
    text = shot_plan_html.read_text(encoding="utf-8")
    mh = re.search(r"Section\s+(\d+)\s*·\s*Shot\s*Plan", text)
    if not mh:
        return
    sec_id = int(mh.group(1))
    sec_times = {sid: (s0, s1) for sid, s0, s1 in parse_proposal_sections_times(proposal_html)}
    if sec_id not in sec_times:
        return
    s0, s1 = sec_times[sec_id]

    html_dir = shot_plan_html.parent
    audio_dir = html_dir / dest_subdir
    audio_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_ok = True
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        ffmpeg_ok = False

    if ffmpeg_ok:
        out_name = f"shot_plan_section_{sec_id}.mp3"
        out_path = audio_dir / out_name
        try:
            run_ffmpeg_clip(audio_src, s0, s1, out_path)
            audio_rel_src = f"{dest_subdir}/{out_name}"
            extra_attrs = ""
        except FileNotFoundError:
            ffmpeg_ok = False
    if not ffmpeg_ok:
        full_rel_name = audio_src.name
        full_copy = audio_dir / full_rel_name
        if not full_copy.exists():
            shutil.copy2(audio_src, full_copy)
        audio_rel_src = f"{dest_subdir}/{full_rel_name}"
        extra_attrs = f" class='seg-audio' data-start='{s0:.3f}' data-end='{s1:.3f}'"

    mh2 = re.search(r"<div class='header'>[\s\S]*?</div>", text)
    if not mh2:
        return
    insert_pos = mh2.end()
    audio_tag = (
        f"\n            <div style='margin:8px 0'>"
        f"<audio controls preload='none' src='{audio_rel_src}'{extra_attrs}></audio>"
        f"</div>"
    )
    new_text = text[:insert_pos] + audio_tag + text[insert_pos:]
    if not ffmpeg_ok:
        new_text = ensure_segment_js(new_text)
    shot_plan_html.write_text(new_text, encoding="utf-8")


def _process_single_report(report_path: Path, frames_src: Path, fps: float, max_frames: int, dest_name: str, copy_mode: str, do_fix_segments: bool) -> None:
    if do_fix_segments:
        fix_segment_images(report_path, frames_src, fps=fps)

    start_s, end_s = load_segments_time_window(report_path)

    all_frames = list_frame_files(frames_src)
    if not all_frames:
        raise RuntimeError("No frame images found in frames directory")

    frame_ids = [fid for fid in (extract_frame_id(p) for p in all_frames) if fid is not None]
    if not frame_ids:
        raise RuntimeError("Could not parse frame ids from frames directory")
    id_to_path = {extract_frame_id(p): p for p in all_frames if extract_frame_id(p) is not None}

    start_id = int(max(0, round(start_s * fps)))
    end_id = int(round(end_s * fps))
    window_ids = [fid for fid in frame_ids if start_id <= fid <= end_id]
    if not window_ids:
        window_ids = frame_ids

    idxs = sample_indices(len(window_ids), max_items=max_frames)
    sampled_ids = [window_ids[i] for i in idxs]
    sampled_paths = [id_to_path[fid] for fid in sampled_ids]

    report_dir = report_path.parent
    dest_dir = report_dir / dest_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    rel_names: List[str] = []
    for src in sampled_paths:
        dst = dest_dir / src.name
        if copy_mode == "copy":
            shutil.copy2(src, dst)
        else:
            if dst.exists():
                dst.unlink()
            os.symlink(src, dst)
        rel_names.append(src.name)

    html_text = report_path.read_text(encoding="utf-8")
    # Basic guard against duplicate gallery injection
    if "帧预览（fps=2）" in html_text:
        # Skip reinject to avoid duplicates; could be enhanced to replace
        return
    gallery_block = build_gallery_html(dest_name, rel_names, sampled_ids, fps=fps)
    new_html = inject_gallery_into_html(html_text, gallery_block)
    report_path.write_text(new_html, encoding="utf-8")


def _find_targets(reports_root: Path) -> Tuple[List[Path], List[Path], Optional[Path]]:
    report_pages: List[Path] = []
    shot_plan_pages: List[Path] = []
    proposal_page: Optional[Path] = None
    # Collect report.html files
    for p in reports_root.rglob("report.html"):
        report_pages.append(p)
    # Collect shot_plan.html files
    for p in reports_root.rglob("shot_plan.html"):
        shot_plan_pages.append(p)
    # Proposal at root (or any depth, prefer root)
    candidate = reports_root / "proposal.html"
    if candidate.exists():
        proposal_page = candidate
    else:
        # fallback: first match anywhere under reports
        for p in reports_root.rglob("proposal.html"):
            proposal_page = p
            break
    return report_pages, shot_plan_pages, proposal_page


def main():
    parser = argparse.ArgumentParser(description="Inject frames gallery and optional audio players into HTML reports")
    parser.add_argument("--report", help="Path to a single report.html to modify")
    parser.add_argument("--reports-root", help="Root directory containing many HTML reports to process (batch mode)")
    parser.add_argument("--frames-dir", required=True, help="Directory containing extracted frames (fps=2)")
    parser.add_argument("--fps", type=float, default=2.0, help="Frames per second used for extraction (default: 2.0)")
    parser.add_argument("--max-frames", type=int, default=60, help="Maximum frames to include (evenly sampled)")
    parser.add_argument("--dest-name", default="frames", help="Relative directory (next to report.html) to store copied frames")
    parser.add_argument("--copy-mode", choices=["copy", "link"], default="copy", help="Copy files or create symlinks into dest directory")
    parser.add_argument("--fix-segment-images", action="store_true", help="Fix <img> src for segments and update images from frames")
    parser.add_argument("--proposal", help="Path to proposal.html for audio injection")
    parser.add_argument("--shot-plan", help="Path to shot_plan.html for audio injection")
    parser.add_argument("--audio-src", help="Path to source MP3 file to slice and inject")
    parser.add_argument("--audio-dest", default="audio", help="Destination subdirectory for generated audio files")
    args = parser.parse_args()

    frames_src = Path(args.frames_dir).resolve()
    if not frames_src.exists() or not frames_src.is_dir():
        raise FileNotFoundError(f"Frames directory not found: {frames_src}")
    # Batch mode takes precedence if provided
    if args.reports_root:
        root = Path(args.reports_root).resolve()
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Reports root not found: {root}")
        report_pages, shot_plan_pages, proposal_page = _find_targets(root)
        # Process all report.html
        processed = 0
        for rp in report_pages:
            try:
                _process_single_report(
                    report_path=rp,
                    frames_src=frames_src,
                    fps=args.fps,
                    max_frames=args.max_frames,
                    dest_name=args.dest_name,
                    copy_mode=args.copy_mode,
                    do_fix_segments=args.fix_segment_images,
                )
                processed += 1
            except Exception as e:
                print(f"[warn] skipping {rp}: {e}")

        # Audio injection once for proposal, then each shot plan
        if args.audio_src and (proposal_page or shot_plan_pages):
            audio_src = Path(args.audio_src).resolve()
            if proposal_page:
                try:
                    inject_audio_into_proposal(proposal_page, audio_src, dest_subdir=args.audio_dest)
                except Exception as e:
                    print(f"[warn] proposal audio injection failed: {e}")
            # Pass proposal_page if available for time ranges
            if proposal_page:
                for sp in shot_plan_pages:
                    try:
                        inject_audio_into_shot_plan(sp, proposal_page, audio_src, dest_subdir=args.audio_dest)
                    except Exception as e:
                        print(f"[warn] shot_plan audio injection failed for {sp}: {e}")
        print(f"Batch processed reports: {processed}; audio injected where requested.")
        return

    # Single-file mode
    if not args.report:
        raise SystemExit("Either --report or --reports-root must be provided")
    report_path = Path(args.report).resolve()
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")
    _process_single_report(
        report_path=report_path,
        frames_src=frames_src,
        fps=args.fps,
        max_frames=args.max_frames,
        dest_name=args.dest_name,
        copy_mode=args.copy_mode,
        do_fix_segments=args.fix_segment_images,
    )
    if args.audio_src and args.proposal:
        inject_audio_into_proposal(Path(args.proposal).resolve(), Path(args.audio_src).resolve(), dest_subdir=args.audio_dest)
    if args.audio_src and args.shot_plan and args.proposal:
        inject_audio_into_shot_plan(Path(args.shot_plan).resolve(), Path(args.proposal).resolve(), Path(args.audio_src).resolve(), dest_subdir=args.audio_dest)
    print(f"Processed single report: {report_path}; audio injected where requested.")


if __name__ == "__main__":
    main()

# python3 /public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/inject_frames_into_html.py \
#   --reports-root /public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/reports \
#   --frames-dir /public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/video_database/Database/Batman_Begins_2005_1080p_BluRay_x264_YIFY/frames \
#   --fps 2 --max-frames 60 --fix-segment-images \
#   --audio-src /public_hw/home/cit_shifangzhao/zsf/VideoCuttingAgent/Dataset/Audio/Call_of_Slience/CallofSilence.mp3
