"""
Helmet Detector — Streamlit App (dashboard UI)
Run with: streamlit run app.py
Requires Streamlit >= 1.39 (uses container keys for CSS targeting).
"""

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(page_title="Helmet Detector", page_icon="🪖", layout="wide",
                   initial_sidebar_state="expanded")

# ── Theme ────────────────────────────────────────────────────────────────────
BG = "#0a1224"
SIDEBAR = "#0d1830"
CARD = "#111c34"
BORDER = "#1e2d4d"
BLUE = "#3b82f6"
GREEN = "#22c55e"
RED = "#ef4444"
PURPLE = "#8b5cf6"
MUTED = "#8fa0bf"

st.markdown(
    f"""
<style>
    header[data-testid="stHeader"] {{ background: transparent; }}
    .stApp {{ background: radial-gradient(circle at 20% 0%, #10204a 0%, {BG} 45%); color:#e5e7eb; }}
    .block-container {{ padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1500px; }}
    section[data-testid="stSidebar"] {{ background:{SIDEBAR}; border-right:1px solid {BORDER}; }}

    /* Bordered containers = panels */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background:{CARD}; border:1px solid {BORDER} !important; border-radius:14px; padding:.4rem .5rem;
    }}
    .panel-head {{ display:flex; align-items:center; gap:.8rem; margin-bottom:.6rem; }}
    .icon-sq {{ width:46px; height:46px; border-radius:12px; display:flex; align-items:center;
               justify-content:center; font-size:1.4rem; background:#172645; }}
    .panel-head .t {{ font-weight:700; font-size:1.15rem; }}
    .panel-head .s {{ color:{MUTED}; font-size:.85rem; }}

    /* Sidebar navigation radio -> buttons */
    section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap:.35rem; }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label {{
        padding:.7rem 1rem; border-radius:10px; width:100%; cursor:pointer;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {{ display:none; }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
        background:linear-gradient(90deg,#2554d6,#2f66ee); color:white; font-weight:600;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{ background:#16244a; }}

    /* Input source radio -> 4 cards */
    .st-key-input_source div[role="radiogroup"] {{
        display:grid; grid-template-columns:repeat(4,1fr); gap:.8rem; width:100%;
    }}
    .st-key-input_source div[role="radiogroup"] label {{
        background:#0e1a33; border:1px solid {BORDER}; border-radius:12px; padding:1rem .8rem;
        justify-content:center; margin:0; cursor:pointer; min-height:74px;
    }}
    .st-key-input_source div[role="radiogroup"] label:has(input:checked) {{
        border-color:{BLUE}; background:#132550;
    }}

    /* Stat cards */
    .stat {{ background:{CARD}; border:1px solid {BORDER}; border-radius:14px; padding:1rem 1.2rem;
             display:flex; gap:1rem; align-items:center; }}
    .stat .ico {{ width:52px; height:52px; border-radius:12px; display:flex; align-items:center;
                 justify-content:center; font-size:1.5rem; }}
    .stat .l {{ color:{MUTED}; font-size:.85rem; }}
    .stat .v {{ font-size:1.8rem; font-weight:700; line-height:1.2; }}
    .stat .n {{ font-size:.78rem; color:{MUTED}; }}

    /* Primary button (Run Detection) */
    div.stButton > button[kind="primary"], div.stDownloadButton > button {{
        background:linear-gradient(90deg,#1d6cf2,#2f8cff); border:none; color:white;
        border-radius:12px; padding:.85rem 1rem; font-weight:700; font-size:1.05rem; width:100%;
    }}
    /* Uploader dropzone */
    section[data-testid="stFileUploaderDropzone"] {{
        background:#0d1830; border:2px dashed {BLUE}; border-radius:14px; padding:2rem 1rem;
    }}
    .welcome h2 {{ margin:0; font-size:1.5rem; }}
    .welcome p {{ margin:0 0 1rem; color:{MUTED}; }}
    .chip {{ background:rgba(34,197,94,.15); color:{GREEN}; border:1px solid rgba(34,197,94,.4);
            padding:.25rem .75rem; border-radius:999px; font-size:.82rem; font-weight:600; }}
</style>
""",
    unsafe_allow_html=True,
)

# ── Session state ────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ── Model ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model(weights_path: str):
    from ultralytics import YOLO
    return YOLO(weights_path)


DEFAULT_WEIGHTS = Path(r"C:\Users\Nithin T\Desktop\CV_Projects\safety helmet detection\best.pt")


def get_model(weights_file):
    if weights_file is not None:
        p = Path(tempfile.gettempdir()) / weights_file.name
        p.write_bytes(weights_file.getbuffer())
        return load_model(str(p)), weights_file.name
    if DEFAULT_WEIGHTS.exists():
        return load_model(str(DEFAULT_WEIGHTS)), DEFAULT_WEIGHTS.name
    return None, None


# ── Detection helpers (unchanged logic) ──────────────────────────────────────
DATASET_CLASS_NAMES = ["helmet", "head", "person"]
CLASS_TO_DISPLAY = {"helmet": "With Helmet", "head": "Without Helmet",
                    "no-helmet": "Without Helmet", "person": "Person"}


def normalize_class_name(name):
    return str(name).strip().lower().replace("_", "-").replace(" ", "-")


def classify_detection(name):
    return CLASS_TO_DISPLAY.get(normalize_class_name(name), str(name))


def run_inference(model, image, conf, iou, return_raw_names=False):
    r = model.predict(image, conf=conf, iou=iou, agnostic_nms=True, verbose=False)[0]
    names, rows = r.names, []
    for box in r.boxes:
        cid = int(box.cls[0])
        cname = names.get(cid, str(cid)) if isinstance(names, dict) else names[cid]
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        row = {"Class": classify_detection(cname), "Confidence": round(float(box.conf[0]), 3),
               "X1": round(x1, 1), "Y1": round(y1, 1), "X2": round(x2, 1), "Y2": round(y2, 1)}
        if return_raw_names:
            row["RawClassName"], row["RawClassID"] = cname, cid
        rows.append(row)
    cols = ["Class", "Confidence", "X1", "Y1", "X2", "Y2"] + (["RawClassName", "RawClassID"] if return_raw_names else [])
    return pd.DataFrame(rows, columns=cols)


def match_person_to_ppe(df, head_fraction=0.35, overlap_thresh=0.25):
    people = df[df["Class"] == "Person"].reset_index(drop=True)
    ppe = df[df["Class"].isin(["With Helmet", "Without Helmet"])].reset_index(drop=True)
    out = []
    for _, p in people.iterrows():
        head_y2 = p["Y1"] + (p["Y2"] - p["Y1"]) * head_fraction
        best, best_ov = None, 0.0
        for _, h in ppe.iterrows():
            iw = max(0, min(p["X2"], h["X2"]) - max(p["X1"], h["X1"]))
            ih = max(0, min(head_y2, h["Y2"]) - max(p["Y1"], h["Y1"]))
            ov = (iw * ih) / max(1e-6, (h["X2"] - h["X1"]) * (h["Y2"] - h["Y1"]))
            if ov > best_ov:
                best_ov, best = ov, h
        if best is not None and best_ov >= overlap_thresh:
            status, conf = best["Class"], best["Confidence"]
        else:
            status, conf = "Unknown", p["Confidence"]
        out.append({"Class": status, "Confidence": conf,
                    "X1": p["X1"], "Y1": p["Y1"], "X2": p["X2"], "Y2": p["Y2"]})
    return pd.DataFrame(out, columns=["Class", "Confidence", "X1", "Y1", "X2", "Y2"])


def draw_detections(image, df, pad=0.0):
    img = image.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    W, H = img.size
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=max(14, image.width // 60))
    except Exception:
        font = ImageFont.load_default()
    colors = {"With Helmet": GREEN, "Without Helmet": RED, "Person": BLUE, "Unknown": "#eab308"}
    labels = {"With Helmet": "Helmet", "Without Helmet": "No Helmet", "Person": "Person", "Unknown": "Unknown"}
    for _, r in df.iterrows():
        c = colors.get(r["Class"], "#eab308")
        x1, y1, x2, y2 = r["X1"], r["Y1"], r["X2"], r["Y2"]
        if pad > 0:
            px, py = (x2 - x1) * pad, (y2 - y1) * pad
            x1, y1, x2, y2 = max(0, x1 - px), max(0, y1 - py), min(W, x2 + px), min(H, y2 + py)
        label = f'{labels.get(r["Class"], r["Class"])} {r["Confidence"]:.2f}'
        d.rectangle([x1, y1, x2, y2], outline=c, width=3)
        tb = d.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        d.rectangle([x1, y1 - th - 8, x1 + tw + 10, y1], fill=c)
        d.text((x1 + 5, y1 - th - 6), label, fill="black", font=font)
    return img


def run_detection_pipeline(model, image, cfg):
    raw = run_inference(model, image, cfg["conf"], cfg["iou"])
    if cfg["compliance"]:
        det = match_person_to_ppe(raw)
        draw_df = pd.concat([det, raw[raw["Class"] == "Person"]], ignore_index=True) if cfg["persons"] else det
    else:
        draw_df = raw if cfg["persons"] else raw[raw["Class"] != "Person"]
        det = draw_df
    return summarize(det, draw_detections(image, draw_df, cfg["pad"]))


def summarize(det, annotated):
    h = det[det["Class"].isin(["With Helmet", "Without Helmet"])]
    w = int((h["Class"] == "With Helmet").sum())
    return {"annotated_img": annotated, "detections_df": det,
            "avg_conf": h["Confidence"].mean() if len(h) else 0.0,
            "total_detections": len(h), "with_helmet": w, "without_helmet": len(h) - w,
            "unknown_count": int((det["Class"] == "Unknown").sum())}


def log_run(result, source):
    st.session_state.history.append({
        "Time": datetime.now().strftime("%d %b %Y, %I:%M %p"), "Source": source,
        "Total": result["total_detections"], "With Helmet": result["with_helmet"],
        "Without Helmet": result["without_helmet"], "Avg Confidence": round(result["avg_conf"], 2)})


# ── Rendering helpers ────────────────────────────────────────────────────────
def head(icon, title, sub=""):
    st.markdown(f'<div class="panel-head"><div class="icon-sq">{icon}</div><div>'
                f'<div class="t">{title}</div><div class="s">{sub}</div></div></div>', unsafe_allow_html=True)


def stat_cards(container):
    hist = st.session_state.history
    total = sum(h["Total"] for h in hist)
    wh = sum(h["With Helmet"] for h in hist)
    wo = sum(h["Without Helmet"] for h in hist)
    last = hist[-1]["Time"].split(", ")[-1] if hist else "—"
    cards = [
        ("🛡️", "#1c3d8f", "Total Detections", total, "This session"),
        ("🪖", "#0f6a4b", "Helmet Detected", wh, f"{wh / total * 100:.0f}% of total" if total else "—"),
        ("⚠️", "#a3243a", "No Helmet Detected", wo, f"{wo / total * 100:.0f}% of total" if total else "—"),
        ("🕐", "#4b3499", "Last Scan", last, "● Real-time monitoring ready"),
    ]
    with container.container():
        for col, (ic, bg, lbl, val, note) in zip(st.columns(4), cards):
            col.markdown(f'<div class="stat"><div class="ico" style="background:{bg}">{ic}</div><div>'
                         f'<div class="l">{lbl}</div><div class="v">{val}</div><div class="n">{note}</div>'
                         f'</div></div>', unsafe_allow_html=True)


def render_results(result, key):
    if result["unknown_count"]:
        st.warning(f"{result['unknown_count']} person(s) detected with no confidently-matched helmet status "
                   "(shown in yellow). This does not mean they are compliant.")
    m = st.columns(3)
    m[0].metric("With Helmet", result["with_helmet"])
    m[1].metric("Without Helmet", result["without_helmet"])
    m[2].metric("Average Confidence", f'{result["avg_conf"]:.2f}')
    c1, c2 = st.columns([1, 1.4])
    with c1, st.container(border=True):
        head("📊", "Detection Overview")
        st.bar_chart(pd.DataFrame({"Category": ["Total", "With Helmet", "Without Helmet"],
                                   "Count": [result["total_detections"], result["with_helmet"],
                                             result["without_helmet"]]}).set_index("Category"),
                     color=GREEN)
    with c2, st.container(border=True):
        head("🪪", "Detection Details")
        df = result["detections_df"]
        if len(df):
            d = df.copy()
            d.insert(0, "#", range(1, len(d) + 1))
            st.dataframe(d, hide_index=True, use_container_width=True)
            st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode(), "detections.csv",
                               "text/csv", key=f"csv_{key}")
        else:
            st.caption("No detections found.")


def render_image_result(orig, result, key):
    l, r = st.columns(2)
    with l, st.container(border=True):
        head("🖼️", "Original Image")
        st.image(orig, use_container_width=True)
    with r, st.container(border=True):
        head("🎯", "Detection Result")
        st.markdown(f'<span class="chip">✅ Helmet Detection: {result["avg_conf"]:.2f}</span>',
                    unsafe_allow_html=True)
        st.image(result["annotated_img"], use_container_width=True)
    render_results(result, key)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:.7rem;margin-bottom:1.4rem;">
      <div style="background:{BLUE};border-radius:12px;width:50px;height:50px;display:flex;
                  align-items:center;justify-content:center;font-size:1.6rem;">🪖</div>
      <div><div style="font-weight:700;font-size:1.25rem;">Helmet Detector</div>
      <div style="color:{MUTED};font-size:.8rem;">AI-Powered Safety Monitoring</div></div>
    </div>""", unsafe_allow_html=True)
    page = st.radio("Navigation", ["🏠  Home", "☁️  Upload", "🕐  History", "⚙️  Settings"],
                    label_visibility="collapsed")
    model_card = st.container()
    st.markdown(f'<div style="margin-top:3rem;color:{MUTED};font-style:italic;font-size:.85rem;">'
                'Safer Workplaces<br>Build a Brighter Future</div>', unsafe_allow_html=True)

# ── Main header + stat cards ─────────────────────────────────────────────────
st.markdown('<div class="welcome"><h2>Welcome back, Nithin 👋</h2>'
            '<p>Detect and monitor safety helmets using advanced AI models</p></div>', unsafe_allow_html=True)
stats_ph = st.empty()
stat_cards(stats_ph)
st.markdown("<br>", unsafe_allow_html=True)

# ── History / Settings pages ─────────────────────────────────────────────────
if page.strip().endswith("History"):
    with st.container(border=True):
        head("🕐", "Scan History", "Runs from this session")
        if st.session_state.history:
            st.dataframe(pd.DataFrame(st.session_state.history[::-1]), hide_index=True, use_container_width=True)
            if st.button("Clear history"):
                st.session_state.history = []
                st.rerun()
        else:
            st.caption("No scans yet — run a detection from Home.")
    st.stop()

if page.strip().endswith("Settings"):
    with st.container(border=True):
        head("⚙️", "Settings", "Model and display options live in the Model Settings panel on Home")
        st.caption("Expected classes: helmet, head, person. Default weights path: " + str(DEFAULT_WEIGHTS))
    st.stop()

# ── Workspace: left (input) / right (model settings) ─────────────────────────
left, right = st.columns([2.1, 1])

with right:
    with st.container(border=True):
        head("⚙️", "Model Settings")
        weights_file = st.file_uploader("Model Weights (.pt)", type=["pt"])
        st.caption("Max 200MB per file • YOLO .pt weights")
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01,
                                   help="Higher value = fewer false positives")
        iou_threshold = st.slider("IoU Threshold (NMS)", 0.0, 1.0, 0.70, 0.01,
                                  help="Controls overlapping box suppression")
        with st.expander("Advanced"):
            show_person_boxes = st.checkbox("Show 'Person' boxes", value=False)
            compliance_mode = st.checkbox(
                "Person-level compliance matching", value=False,
                help="Matches each Person to the Helmet/Head box near their head. Leave off if your "
                     "model's person-class mAP is poor.")
            box_padding = st.slider("Box padding (visual only)", 0, 100, 20) / 100.0
            debug_mode = st.checkbox("🔍 Debug mode", value=False)
        run_clicked = st.button("▶  Run Detection", type="primary", use_container_width=True)

cfg = {"conf": conf_threshold, "iou": iou_threshold, "compliance": compliance_mode,
       "persons": show_person_boxes, "pad": box_padding}

model, weights_name = get_model(weights_file)
with model_card:
    ready = model is not None
    st.markdown(f"""
    <div style="margin-top:1.4rem;background:{CARD};border:1px solid {BORDER};border-radius:12px;padding:.8rem 1rem;">
      <div style="font-weight:600;">{'🟢' if ready else '🔴'} {'Model Ready' if ready else 'No Model'}</div>
      <div style="color:{MUTED};font-size:.78rem;">{('YOLO • ' + weights_name) if ready else 'Upload .pt weights'}</div>
    </div>""", unsafe_allow_html=True)

with left:
    upload_panel = st.container(border=True)
    source_panel = st.container(border=True)

result_slot = st.container()  # full-width results below both columns

with source_panel:
    head("📥", "Input Source")
    with st.container(key="input_source"):
        input_source = st.radio(
            "Input source",
            ["📷 Upload Image", "🎥 Upload Video", "📸 Live Snapshot", "🔴 Continuous Live Feed"],
            horizontal=True, label_visibility="collapsed")

if model is None:
    with upload_panel:
        st.info("👉 Upload your trained `.pt` weights in Model Settings (or place `best.pt` at the configured path).")
    st.stop()

names = getattr(model, "names", {})
loaded = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
if loaded and {normalize_class_name(n) for n in loaded} != {normalize_class_name(n) for n in DATASET_CLASS_NAMES}:
    st.warning(f"Loaded model classes {sorted(map(normalize_class_name, loaded))} differ from expected "
               f"{DATASET_CLASS_NAMES}. Check CLASS_TO_DISPLAY.")

# ── Modes ────────────────────────────────────────────────────────────────────
with upload_panel:
    if input_source == "📷 Upload Image":
        head("☁️", "Upload Image", "Drag and drop your image here or click to browse")
        img_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], key="img",
                                    label_visibility="collapsed")
    elif input_source == "🎥 Upload Video":
        head("🎥", "Upload Video", "Drag and drop your video here or click to browse")
        vid_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"], key="vid",
                                    label_visibility="collapsed")
        c1, c2 = st.columns(2)
        frame_skip = c1.slider("Process every Nth frame", 1, 30, 5)
        max_frames = c2.slider("Max frames to process", 10, 300, 60)
    elif input_source == "📸 Live Snapshot":
        head("📸", "Live Snapshot", "Take a single photo with your camera")
        snap = st.camera_input("Take a snapshot", label_visibility="collapsed")
    else:
        head("🔴", "Continuous Live Feed", "Real-time detection on a webcam stream (needs streamlit-webrtc)")

with result_slot:
    if input_source == "📷 Upload Image":
        if img_file is None:
            st.stop()
        orig = Image.open(img_file).convert("RGB")
        if debug_mode:
            with st.expander("🔍 Debug: raw model output", expanded=True):
                st.caption(f"model.names: `{names}`")
                st.dataframe(run_inference(model, orig, conf_threshold, iou_threshold, True), hide_index=True)
                st.markdown("**conf=0.01 pass:**")
                st.dataframe(run_inference(model, orig, 0.01, iou_threshold, True), hide_index=True)
        if run_clicked:
            with st.spinner("Running detection…"):
                res = run_detection_pipeline(model, orig, cfg)
            log_run(res, "Image")
            stat_cards(stats_ph)
            render_image_result(orig, res, "image")
        else:
            st.info("Image ready — click **Run Detection**.")

    elif input_source == "🎥 Upload Video":
        import cv2
        if vid_file is None:
            st.stop()
        if run_clicked:
            tmp_in = Path(tempfile.gettempdir()) / f"input_{vid_file.name}"
            tmp_in.write_bytes(vid_file.getbuffer())
            cap = cv2.VideoCapture(str(tmp_in))
            fps = cap.get(cv2.CAP_PROP_FPS) or 20
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            tmp_out = Path(tempfile.gettempdir()) / "annotated_output.mp4"
            writer = cv2.VideoWriter(str(tmp_out), cv2.VideoWriter_fourcc(*"mp4v"), max(fps / frame_skip, 1), (w, h))
            dets, idx, done = [], 0, 0
            bar = st.progress(0, text="Processing frames…")
            while cap.isOpened() and done < max_frames:
                ok, fr = cap.read()
                if not ok:
                    break
                if idx % frame_skip == 0:
                    res = run_detection_pipeline(model, Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)), cfg)
                    dets.append(res["detections_df"])
                    writer.write(cv2.cvtColor(np.array(res["annotated_img"]), cv2.COLOR_RGB2BGR))
                    done += 1
                    bar.progress(min(done / max_frames, 1.0), text=f"Processed {done}/{max_frames} frames…")
                idx += 1
            cap.release()
            writer.release()
            bar.empty()
            st.success(f"Done — processed {done} frames.")
            st.video(str(tmp_out))
            if dets:
                agg = summarize(pd.concat(dets, ignore_index=True), None)
                log_run(agg, "Video")
                stat_cards(stats_ph)
                st.caption("Aggregate across frames (the same person in several frames counts several times).")
                render_results(agg, "video")
        else:
            st.info("Set the options, then click **Run Detection**.")

    elif input_source == "📸 Live Snapshot":
        if snap is None:
            st.stop()
        orig = Image.open(snap).convert("RGB")
        with st.spinner("Running detection…"):
            res = run_detection_pipeline(model, orig, cfg)
        log_run(res, "Snapshot")
        stat_cards(stats_ph)
        render_image_result(orig, res, "snapshot")

    else:
        try:
            import av
            import cv2
            from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

            class HelmetProcessor(VideoProcessorBase):
                def __init__(self):
                    self.n, self.last = 0, None

                def recv(self, frame):
                    bgr = frame.to_ndarray(format="bgr24")
                    self.n += 1
                    if self.n % 3 == 0 or self.last is None:
                        res = run_detection_pipeline(model, Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), cfg)
                        self.last = cv2.cvtColor(np.array(res["annotated_img"]), cv2.COLOR_RGB2BGR)
                    return av.VideoFrame.from_ndarray(self.last, format="bgr24")

            st.caption("Only the annotated video is shown live; use Live Snapshot for stats.")
            webrtc_streamer(key="live", video_processor_factory=HelmetProcessor,
                            media_stream_constraints={"video": True, "audio": False})
        except ImportError:
            st.error("Install the live-feed dependencies: `pip install streamlit-webrtc av`, then restart.")
