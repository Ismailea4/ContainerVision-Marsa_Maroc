import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
from datetime import datetime
import io
import warnings
from src.pipeline import container_detection

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

st.set_page_config(page_title="Container Vision", page_icon="📦", layout="wide")

CAMERAS = [
    {"id": "zone_a", "name": "Zone A - Quai 1"},
    {"id": "zone_b", "name": "Zone B - Quai 2"},
    {"id": "zone_c", "name": "Zone C - Quai 3"},
    {"id": "zone_d", "name": "Zone D - Quai 4"},
    {"id": "zone_e", "name": "Zone E - Parking"},
    {"id": "zone_f", "name": "Zone F - Sortie"}
]

if "history" not in st.session_state:
    st.session_state["history"] = []
if "detections_by_zone" not in st.session_state:
    st.session_state["detections_by_zone"] = {z["id"]: [] for z in CAMERAS}

st.title("📦 Container Vision System")
tabs = st.tabs(["📺 Camera Zones", "📜 History", "📊 Statistics"])

# 1. CAMERA ZONES TAB (6 rectangles)
with tabs[0]:
    st.header("Camera Zones (Live Upload & Detection)")
    cols = st.columns(3)
    for idx, cam in enumerate(CAMERAS):
        with cols[idx % 3]:
            st.subheader(cam["name"])
            uploaded = st.file_uploader(f"Upload image to {cam['name']}", type=["jpg", "jpeg", "png"], key=cam["id"])
            if uploaded:
                img = Image.open(uploaded)
                st.image(img, use_container_width=True, caption="Uploaded image")
                if st.button(f"Run Detection ({cam['name']})", key=f"run_{cam['id']}"):
                    temp_path = f"temp_{cam['id']}.jpg"
                    img.save(temp_path)
                    result = container_detection(temp_path, object_type=['code', 'character', 'seal'], conf=0.25, iou=0.45, display=False)
                    os.remove(temp_path)
                    detections = result.get("detections", {})
                    processed_img = result.get("predictions", None)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    # Convert processed image to PIL for display
                    processed_pil = None
                    if isinstance(processed_img, np.ndarray):
                        processed_pil = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
                    # Store result (store both original and processed image)
                    history_entry = {
                        "datetime": timestamp,
                        "zone_id": cam["id"],
                        "zone": cam["name"],
                        "original_image": img,
                        "processed_image": processed_pil,
                        "detections": detections,
                        "container": detections.get("CN", {}).get("value", ""),
                        "confidence": round(float(detections.get("CN", {}).get("confidence", 0)), 2) if "CN" in detections else 0
                    }
                    st.session_state["history"].append(history_entry)
                    st.session_state["detections_by_zone"][cam["id"]].insert(0, history_entry)
                    st.success("Detection complete!")
                    # Only results shown below
                    with st.expander("Detection Results", expanded=True):
                        if processed_pil:
                            st.image(processed_pil, use_container_width=True, caption="Processed image (with rectangles)")
                        if detections:
                            for k, v in detections.items():
                                st.write(f"{k}: {v['value']} (Confidence: {v['confidence']:.2f})")
                        else:
                            st.info("No valid codes detected (after validation).")
                        # Fullscreen show original image on click
                        if st.button("Show Original (Fullscreen)", key=f"fullscreen_{cam['id']}"):
                            st.image(img, use_container_width=True, caption="Original Image (Fullscreen)")
            # Show last 3 detections for this zone
            last = st.session_state["detections_by_zone"][cam["id"]][:3]
            if last:
                st.caption("Last 3 detections:")
                for det in last:
                    conf = det["confidence"]
                    color = "🟢" if conf >= 90 else "🟡" if conf >= 75 else "🔴"
                    st.write(f"{det['datetime']} | {det['container']} | {color} {conf}%")

# 2. HISTORY TAB
with tabs[1]:
    st.header("Prediction History (All Zones)")
    hist = st.session_state["history"]
    if hist:
        # Quick summary table
        df = pd.DataFrame([{
            "Date/Time": h["datetime"],
            "Zone": h["zone"],
            "Container": h["container"],
            "Confidence": h["confidence"]
        } for h in hist])
        st.dataframe(df, use_container_width=True)
        # Download CSV
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button("⬇ Download Full History as CSV", csv_buf.getvalue(), file_name="container_history.csv", mime="text/csv")
        # Detailed cards (only processed image in expander, original in fullscreen)
        for i, h in enumerate(reversed(hist)):
            with st.expander(f"{h['datetime']} | {h['zone']} | {h['container'] or 'No CN'} ({h['confidence']}%)", expanded=False):
                if h["processed_image"]:
                    st.image(h["processed_image"], use_container_width=True, caption="Processed Image (with rectangles)")
                if h["detections"]:
                    st.write("*Detection Results:*")
                    for k, v in h["detections"].items():
                        st.write(f"- {k}: {v['value']} (Confidence: {v['confidence']:.2f})")
                # Button to view original image fullscreen
                if st.button(f"Show Original Image (Fullscreen) {i}", key=f"fullscreen_hist_{i}"):
                    st.image(h["original_image"], use_container_width=True, caption="Original Image (Fullscreen)")
    else:
        st.info("No predictions have been made yet.")

# 3. STATISTICS TAB - ONLY BAR AND LINE CHARTS
with tabs[2]:
    st.header("System Statistics & Exports")
    hist = st.session_state["history"]
    if hist:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Predictions", len(hist))
        with col2:
            today = datetime.today().strftime("%Y-%m-%d")
            count_today = len([h for h in hist if h["datetime"].startswith(today)])
            st.metric("Today's Predictions", count_today)
        with col3:
            all_conf = [h["confidence"] for h in hist]
            avg_conf = np.mean(all_conf) if all_conf else 0
            st.metric("Average Confidence", f"{avg_conf:.2f}")
        # Per-zone stats
        st.subheader("Zone Statistics")
        zone_stats = []
        for c in CAMERAS:
            last = st.session_state["detections_by_zone"][c["id"]][:1]
            last_activity = last[0]["datetime"] if last else "Never"
            zone_stats.append({
                "Zone": c["name"],
                "Detections": len(st.session_state["detections_by_zone"][c["id"]]),
                "Last Activity": last_activity
            })
        df_zone = pd.DataFrame(zone_stats)
        st.dataframe(df_zone, use_container_width=True)
        zone_csv_buf = io.StringIO()
        df_zone.to_csv(zone_csv_buf, index=False)
        st.download_button("⬇ Download Detections by Zone (CSV)", zone_csv_buf.getvalue(), file_name="detections_by_zone.csv", mime="text/csv")
        # Confidence distribution (bar)
        st.subheader("Confidence Distribution")
        conf_levels = {
            "High (≥90%)": sum(1 for h in hist if h["confidence"] >= 90),
            "Medium (75-89%)": sum(1 for h in hist if 75 <= h["confidence"] < 90),
            "Low (<75%)": sum(1 for h in hist if h["confidence"] < 75)
        }
        conf_df = pd.DataFrame(list(conf_levels.items()), columns=["Confidence", "Count"])
        st.bar_chart(conf_df.set_index("Confidence"))
        stats_csv_buf = io.StringIO()
        conf_df.to_csv(stats_csv_buf, index=False)
        st.download_button("⬇ Download Statistics (CSV)", stats_csv_buf.getvalue(), file_name="stats.csv", mime="text/csv")
        # Detection type bar
        st.subheader("Detection Type Distribution")
        type_counts = {}
        for h in hist:
            for k in h["detections"].keys():
                type_counts[k] = type_counts.get(k, 0) + 1
        if type_counts:
            st.bar_chart(pd.DataFrame.from_dict(type_counts, orient="index", columns=["count"]))
        # Timeline plot: Confidence over time
        st.subheader("Confidence Over Time")
        time_df = pd.DataFrame([{"datetime": h["datetime"], "confidence": h["confidence"]} for h in hist])
        if not time_df.empty:
            time_df["datetime"] = pd.to_datetime(time_df["datetime"])
            time_df = time_df.sort_values("datetime")
            st.line_chart(time_df.set_index("datetime"))
        # Per-zone prediction count plot
        st.subheader("Predictions Per Zone")
        zone_counts = {z["name"]: len(st.session_state["detections_by_zone"][z["id"]]) for z in CAMERAS}
        st.bar_chart(pd.DataFrame.from_dict(zone_counts, orient="index", columns=["Count"]))
    else:
        st.info("No statistics available yet.")