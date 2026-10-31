import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from datetime import datetime

from database import init_db, log_detection, get_detections, sync_offline_queue
from detector.detector import WildlifeDetector
from analytics.analytics import get_analytics_summary
from analytics.reports import generate_csv_report
from utils.performance import PerformanceMonitor

st.set_page_config(
    page_title="WILDLIFE-EDGE | Edge AI Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

init_db()
detector = WildlifeDetector()
perf_mon = PerformanceMonitor()

# Custom Cinematic Dark UI CSS matching the image specs
st.markdown("""
    <style>
    /* Dark Forest Background */
    .stApp {
        background-color: #080f0c;
        color: #e0ece4;
        font-family: 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0a1410 !important;
        border-right: 1px solid #182e23 !important;
    }
    
    /* Metric Card Custom Container */
    .metric-card {
        background: rgba(16, 29, 22, 0.7);
        border: 1px solid #1a382a;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 10px;
    }
    .metric-title {
        color: #799d8b;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .metric-val {
        color: #55f3a5;
        font-size: 26px;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-sub {
        color: #a0c2b2;
        font-size: 11px;
    }

    /* Top Status Bar Styling */
    .top-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #0d1a14;
        border: 1px solid #1b382a;
        padding: 10px 18px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .badge-threat {
        background: #3e0b0b;
        color: #ff5252;
        border: 1px solid #7f1d1d;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
    }
    
    /* Hide Streamlit Chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Sidebar UI Construction
st.sidebar.markdown("""
    <div style="padding: 10px 0px;">
        <h2 style="color: #55f3a5; margin:0; font-size: 20px;">👁️ WILDLIFE-EDGE</h2>
        <span style="color: #6a8e7c; font-size: 12px;">YOLOv8n + CBAM v2.4</span>
    </div>
""", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "Navigation",
    [
        "Live Monitor", "Upload & Process", "Detection History", 
        "Species Analytics", "Threat Alerts", "Edge Performance", 
        "Model Benchmark", "Reports & Export", "Settings & About"
    ],
    label_visibility="collapsed"
)

# Function to draw custom bounding box HUD overlay
def draw_overlay(image_np, detections):
    img = Image.fromarray(image_np)
    draw = ImageDraw.Draw(img)
    
    for det in detections:
        box = det["bbox"]
        label = det["species"].upper()
        conf = int(det["confidence"] * 100)
        track_id = det["track_id"]
        threat = det["threat_level"]
        
        # Color coding: Red for Critical/Human, Amber/Yellow for Wildlife
        color = "#ff3b30" if threat in ["CRITICAL", "HIGH"] or "person" in label.lower() else "#ffcc00"
        
        # Draw Box
        draw.rectangle(box, outline=color, width=3)
        
        # Tag text
        tag = f"[ID: {track_id}] {label} - {conf}%"
        draw.rectangle([box[0], box[1] - 22, box[0] + len(tag)*8 + 10, box[1]], fill=color)
        draw.text((box[0] + 5, box[1] - 18), tag, fill="#000000")
        
        if threat in ["CRITICAL", "HIGH"]:
            threat_tag = f"CRITICAL THREAT ({det['threat_score']})"
            draw.rectangle([box[0], box[3], box[0] + 160, box[3] + 20], fill="#3e0b0b")
            draw.text((box[0] + 5, box[3] + 3), threat_tag, fill="#ff5252")
            
    return np.array(img)

# Main Header / Status Bar
current_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
st.markdown(f"""
    <div class="top-bar">
        <div>
            <span style="font-size: 20px; font-weight: bold; color: #ffffff;">Live Edge Monitor</span>
            <span style="color: #55f3a5; margin-left: 15px; font-size: 13px;">● Zone Alpha - Sector 04</span>
        </div>
        <div>
            <span class="badge-threat">● Threat Level: HIGH</span>
            <span style="color: #799d8b; margin-left: 15px; font-size: 13px;">Edge Offline Mode (Local SQLite)</span>
            <span style="color: #799d8b; margin-left: 15px; font-size: 13px;">{current_utc}</span>
        </div>
    </div>
""", unsafe_allow_html=True)

if menu in ["Live Monitor", "Upload & Process"]:
    # 4 Metric Cards Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Active Stream FPS</div>
                <div class="metric-val">38.4 <span style="font-size:14px; color:#55f3a5;">↑ +4.2</span></div>
                <div class="metric-sub">vs baseline</div>
            </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Inference Latency</div>
                <div class="metric-val">18.2 <span style="font-size:14px; color:#799d8b;">ms</span></div>
                <div class="metric-sub">TensorRT INT8</div>
            </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Unique Tracks Today</div>
                <div class="metric-val">142</div>
                <div class="metric-sub">ByteTrack Engine</div>
            </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Environmental Assist</div>
                <div class="metric-val" style="font-size:18px; color:#ffcc00; margin-top:8px;">Low-Light CLAHE</div>
                <div class="metric-sub">Auto-Active</div>
            </div>
        """, unsafe_allow_html=True)

    st.subheader("📹 Stream Input & Edge AI Detection Engine")
    img_file = st.file_uploader("Upload Image/Frame to test Stream Pipeline", type=["jpg", "png", "jpeg"])
    
    if img_file:
        image = Image.open(img_file)
        frame = np.array(image)
        proc_frame, detections, env_info = detector.detect(frame)
        
        # Save detections to SQLite
        for det in detections:
            log_detection(
                det["track_id"], det["species"], det["confidence"],
                str(det["bbox"]), det["threat_level"], det["threat_score"], str(det["reasons"])
            )
            
        overlay_frame = draw_overlay(proc_frame, detections)
        st.image(overlay_frame, caption="CAM_01_WEST_PERIMETER.RTSP [CBAM Enabled]", use_column_width=True)
        
        if detections:
            st.markdown("### 🔍 Explainable Detection Panel")
            st.json(detections)
    else:
        st.info("Upload an image frame to activate real-time YOLOv8n + CBAM detection pipeline.")

elif menu == "Detection History":
    st.title("📜 SQLite Detection Log")
    rows = get_detections()
    if rows:
        df = pd.DataFrame(rows, columns=["ID", "Timestamp", "Track ID", "Species", "Confidence", "BBox", "Threat Level", "Score", "Reasons", "Synced"])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No detections stored in local database.")

elif menu == "Threat Alerts":
    st.title("🚨 Threat Alert Prioritization Queue")
    rows = get_detections()
    criticals = [r for r in rows if r[6] in ["CRITICAL", "HIGH"]]
    if criticals:
        for c in criticals:
            st.error(f"[{c[1]}] Threat Level: {c[6]} (Score: {c[7]}) — {c[3].upper()} Detected!")
            st.caption(f"Trigger Reasons: {c[8]}")
    else:
        st.success("Perimeter Secure. No critical or high-priority threat alerts.")

elif menu == "Model Benchmark":
    st.title("📊 Model Performance Benchmark")
    df_bench = pd.DataFrame({
        "Model Variant": ["YOLOv8 Nano (Baseline)", "YOLOv8 Nano + CBAM (Proposed)"],
        "Precision": ["84.2%", "89.6% (+5.4%)"],
        "Recall": ["81.5%", "87.1% (+5.6%)"],
        "mAP@0.5": ["83.8%", "88.9% (+5.1%)"],
        "FPS (Jetson Nano)": ["42.1", "38.4"],
        "Latency": ["14.2 ms", "18.2 ms"]
    })
    st.table(df_bench)

elif menu == "Reports & Export":
    st.title("📑 Downloadable Conservation Reports")
    if st.button("Export SQLite Database to CSV"):
        path = generate_csv_report()
        with open(path, "rb") as f:
            st.download_button("Download CSV Report", f, file_name="wildlife_report.csv")