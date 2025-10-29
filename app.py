# streamlit run "C:\Users\rselv\OneDrive\Documents\YOLOv8\object detection ppe project\app.py"
# https://192.0.0.4:8080

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import pandas as pd
import threading, queue, time
import matplotlib.pyplot as plt
from alert_system import trigger_alert  # assumes this exists
from violation import append_detection    # assumes this exists
from sklearn.cluster import KMeans
import plotly.graph_objects as go

# ---------------- Config ----------------
st.set_page_config(page_title="PPE Detection Dashboard", layout="wide")
LOG_CSV = "violations.csv"  # you said file is violations.csv

# ---------------- CSS ----------------
st.markdown("""
<style>
.stApp { background: linear-gradient(to right, #0f2027, #203a43, #2c5364); color: white; }
h1,h2,h3,h4 { color: #00E6E6; text-shadow:1px 1px 3px #000; }
section[data-testid="stSidebar"] { background-color: #111827; color: #FFF; border-right:2px solid #00E6E6; }
div[data-testid="stMetricValue"] { color:#00FFC6; font-size:28px; font-weight:700; }
[data-testid="stDataFrame"] { border:2px solid #00E6E6; border-radius:12px; box-shadow:0px 0px 10px rgba(0,255,200,0.4);}
</style>
""", unsafe_allow_html=True)

# ---------------- Title ----------------
st.title("🚧 AI-Powered PPE Detection Dashboard")
st.markdown("""
<div style="background: linear-gradient(90deg, #1F2937, #111827); padding:14px; border-radius:10px;">
  <h3 style="color:#00E6E6;">Smart PPE Monitoring — Real-time detection + Historical analysis</h3>
</div>
""", unsafe_allow_html=True)

# ---------------- Sidebar ----------------
st.sidebar.header("Input & Settings")
input_mode = st.sidebar.radio("Input Source:", ["Image", "Video", "Webcam", "IP Camera"])
CONF_THRESHOLD = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
start_btn = st.sidebar.button("▶ Start")
stop_btn = st.sidebar.button("⏹ Stop")

if "stop" not in st.session_state:
    st.session_state.stop = False
if stop_btn:
    st.session_state.stop = True
if start_btn:
    st.session_state.stop = False

# ---------------- Load YOLO model ----------------
cv2.setNumThreads(1)
model = YOLO("runs/detect/train9/weights/best.pt")  # keep path to your trained weights

# ---------------- helpers ----------------
PPE_ITEMS = ["Helmet", "Vest", "Gloves", "Goggles", "Shoes"]

def count_ppe_from_results(results):
    mapping = {"helmet":"Helmet","vest":"Vest","gloves":"Gloves","goggles":"Goggles","shoes":"Shoes"}
    counts = {p:0 for p in mapping.values()}
    # results[0].boxes exists for ultralytics
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = results[0].names.get(cls_id, "").lower()
        if label in mapping:
            counts[mapping[label]] += 1
    return counts

def classify_risk_from_counts(ppe_counts):
    # simple rule-based classifier (you can replace with ML models)
    missing = sum(1 for v in ppe_counts.values() if v == 0)
    if missing >= 3:
        return "High"
    elif missing >= 1:
        return "Medium"
    else:
        return "Low"

# ---------------- Real-time video processing ----------------
def process_video(cap):
    stframe = st.empty()
    metrics = st.container()
    chart_slot = st.empty()
    frame_q = queue.Queue(maxsize=2)
    stop_flag = False

    def reader():
        while cap.isOpened() and not stop_flag:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            frame = cv2.resize(frame, (416, 416))
            if not frame_q.full():
                frame_q.put(frame)
            else:
                try:
                    frame_q.get_nowait()
                    frame_q.put(frame)
                except:
                    pass

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    while cap.isOpened() and not st.session_state.stop:
        if frame_q.empty():
            time.sleep(0.01)
            continue
        frame = frame_q.get()
        results = model.predict(frame, conf=CONF_THRESHOLD, device="cpu", verbose=False)
        annotated = results[0].plot()
        ppe_counts = count_ppe_from_results(results)
        risk = classify_risk_from_counts(ppe_counts)

        # Log and alert
        append_detection(str(ppe_counts), camera_id="cam_real", source="video", predicted_risk=risk)
        trigger_alert(ppe_counts)

        # UI
        stframe.image(annotated, channels="BGR", use_column_width=True)
        with metrics:
            st.markdown("<h4 style='color:#00FFC6;'>Live PPE</h4>", unsafe_allow_html=True)
            cols = st.columns(len(ppe_counts))
            for i,(k,v) in enumerate(ppe_counts.items()):
                cols[i].metric(k, v)
            st.markdown(f"<b>Risk:</b> <span style='color:#FF4D4D'>{risk}</span>", unsafe_allow_html=True)

        chart_slot.bar_chart(pd.Series(ppe_counts))
        time.sleep(0.7)

    stop_flag = True
    cap.release()

# ---------------- Input handlers ----------------
if input_mode == "Image":
    up = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
    if up and start_btn:
        image = Image.open(up).convert("RGB")
        arr = np.array(image)
        results = model.predict(arr, conf=CONF_THRESHOLD, device="cpu", verbose=False)
        annotated = results[0].plot()
        ppe_counts = count_ppe_from_results(results)
        risk = classify_risk_from_counts(ppe_counts)

        # Logging
        append_detection(str(ppe_counts), camera_id="cam_img", source="image", predicted_risk=risk)
        trigger_alert(ppe_counts)

        # Show annotated image + metrics
        st.image(annotated, use_column_width=True)
        st.markdown(f"## ⚠️ Risk Level: **{risk}**")
        st.subheader("Detected PPE counts")
        st.table(pd.DataFrame([ppe_counts]).T.rename(columns={0:"Count"}))

        # Per-input chart: bar + pie of missing vs present
        st.subheader("Per-input visualization")
        st.bar_chart(pd.Series(ppe_counts))

        present = sum(1 for v in ppe_counts.values() if v > 0)
        missing = len(PPE_ITEMS) - present
        fig1, ax1 = plt.subplots(figsize=(3,3))
        ax1.pie([present, missing], labels=["Present","Missing"], autopct="%1.1f%%")
        ax1.set_title("PPE Present vs Missing")
        st.pyplot(fig1)

elif input_mode == "Video":
    up = st.file_uploader("Upload video", type=["mp4","avi","mov","mkv"])
    if up and start_btn:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(up.read())
        cap = cv2.VideoCapture(tfile.name)
        process_video(cap)

elif input_mode == "Webcam":
    if start_btn:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        process_video(cap)

elif input_mode == "IP Camera":
    ip = st.text_input("IP camera URL (e.g. http://192.0.0.4:8080/video )")
    if ip and start_btn:
        cap = cv2.VideoCapture(ip)
        process_video(cap)

# ---------------- Historical / Overall Dashboard ----------------
st.markdown("---")
st.header("📊 Overall Analysis (from violations.csv)")

def safe_read_csv(path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        return None

df = safe_read_csv(LOG_CSV)
if df is None:
    st.info(f"Log file `{LOG_CSV}` not found or unreadable. Start detection to create logs.")
else:
    # Normalize column names
    df.columns = df.columns.str.strip()
    # attempt to find a 'Detected PPE' style column (case-insensitive)
    detected_col = None
    for c in df.columns:
        if c.lower() in ("detected ppe", "detected_ppe", "detectedppe", "detected"):
            detected_col = c
            break

    # If separate PPE columns exist already, prefer them (case-insensitive match)
    found_ppe_cols = {}
    for p in PPE_ITEMS:
        for c in df.columns:
            if c.lower() == p.lower() or c.lower() == p.lower() + "_count" or c.lower().startswith(p.lower()):
                found_ppe_cols[p] = c
                break

    # If detected_col exists but separate columns not present, parse strings like "Helmet:1, Vest:0,..."
    if detected_col and len(found_ppe_cols) < len(PPE_ITEMS):
        df_detect_parsed = df[detected_col].astype(str)
        for p in PPE_ITEMS:
            def extract_val(s, item=p):
                try:
                    if item in s:
                        idx = s.find(f"{item}:")
                        if idx == -1:
                            return 0
                        tail = s[idx + len(f"{item}:"):]
                        val_str = tail.split(",")[0].strip()
                        return int(''.join(ch for ch in val_str if ch.isdigit())) if any(ch.isdigit() for ch in val_str) else 0
                    else:
                        return 0
                except:
                    return 0
            df[p] = df_detect_parsed.apply(extract_val)
    else:
        for p in PPE_ITEMS:
            if p in found_ppe_cols:
                df[p] = df[found_ppe_cols[p]]
            elif p.lower() in (c.lower() for c in df.columns):
                real = [c for c in df.columns if c.lower()==p.lower()][0]
                df[p] = df[real]
            else:
                df[p] = 0

    for p in PPE_ITEMS:
        df[p] = pd.to_numeric(df[p], errors='coerce').fillna(0).astype(int)

    # Ensure time_of_day column exists
    time_col = None
    for c in df.columns:
        if c.lower() in ("timestamp", "time", "datetime", "ts"):
            time_col = c
            break
    if time_col:
        try:
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            def get_slot(h):
                if pd.isna(h): return "unknown"
                h = int(h)
                if 6 <= h < 12: return "morning"
                if 12 <= h < 17: return "afternoon"
                if 17 <= h < 21: return "evening"
                return "night"
            df["time_of_day"] = df[time_col].dt.hour.apply(get_slot)
        except:
            df["time_of_day"] = "unknown"
    else:
        if "time_of_day" in df.columns:
            df["time_of_day"] = df["time_of_day"].astype(str).str.lower()
        else:
            df["time_of_day"] = "unknown"

    # Risk column normalization
    risk_col = None
    for c in df.columns:
        if c.lower() in ("risk", "risk_level", "risklabel", "predicted_risk"):
            risk_col = c
            break
    if risk_col:
        df["risk_level"] = df[risk_col].astype(str).str.capitalize()
    else:
        df["risk_level"] = df[PPE_ITEMS].apply(lambda row: classify_risk_from_counts(row.to_dict()), axis=1)

    # === FILTERS ===
    st.markdown("### Filters")
    c1, c2 = st.columns(2)
    with c1:
        time_sel = st.multiselect("Time of day", options=sorted(df["time_of_day"].unique()), default=sorted(df["time_of_day"].unique()))
    with c2:
        ppe_sel = st.multiselect("PPE Type (filter rows containing)", options=PPE_ITEMS, default=[])

    filtered = df.copy()
    if time_sel:
        filtered = filtered[filtered["time_of_day"].isin(time_sel)]
    if ppe_sel:
        if detected_col:
            filtered = filtered[detected_col].astype(str).apply(lambda s: any(p in s for p in ppe_sel))
        else:
            filtered = filtered[filtered[ppe_sel].sum(axis=1) > 0]

    st.markdown("### Log snapshot")
    st.dataframe(filtered.head(200), use_container_width=True)

    # === Overall charts ===
    if len(filtered) > 0:
        st.subheader("PPE Count vs Time of Day")
        session_ppe = filtered.groupby("time_of_day")[PPE_ITEMS].sum().reindex(["morning","afternoon","evening","night"], fill_value=0)
        st.line_chart(session_ppe)

        st.subheader("Risk distribution by Time of Day")
        risk_table = filtered.groupby(["time_of_day","risk_level"]).size().unstack(fill_value=0).reindex(index=["morning","afternoon","evening","night"], fill_value=0)
        st.bar_chart(risk_table)

        st.subheader("Most-missing PPE per Time of Day")
        counts_per_session = filtered.groupby("time_of_day")[PPE_ITEMS].sum().reindex(index=["morning","afternoon","evening","night"], fill_value=0)
        rows_per_session = filtered.groupby("time_of_day").size().reindex(index=["morning","afternoon","evening","night"], fill_value=0)
        missing_per_session = pd.DataFrame(index=counts_per_session.index, columns=PPE_ITEMS)
        for sess in counts_per_session.index:
            total = rows_per_session.loc[sess] if sess in rows_per_session.index else 0
            if total > 0:
                missing_per_session.loc[sess] = total - counts_per_session.loc[sess]
            else:
                missing_per_session.loc[sess] = 0
        st.dataframe(missing_per_session.fillna(0).astype(int))

        total_missing = missing_per_session.sum(axis=1).astype(int)
        if not total_missing.empty:
            most_missing_session = total_missing.idxmax()
            st.markdown(f"### ⚠️ Session with most missing PPE: **{most_missing_session.capitalize()}**")
            series = missing_per_session.loc[most_missing_session].sort_values(ascending=False)
            top_missing = [p for p,v in series.items() if v>0]
            if top_missing:
                st.markdown(f"Most commonly missing PPE in that session: **{', '.join(top_missing)}**")
            else:
                st.markdown("No missing PPE recorded for that session (based on logs).")

        try:
            X = filtered[PPE_ITEMS].values
            if len(filtered) >= 3 and X.shape[0] >= 3:
                kmeans = KMeans(n_clusters=3, random_state=42)
                clusters = kmeans.fit_predict(X)
                filtered["cluster"] = clusters
                cluster_summary = filtered.groupby("cluster")[PPE_ITEMS].sum()
                st.subheader("K-Means cluster summary (PPE sums per cluster)")
                st.dataframe(cluster_summary)
        except Exception:
            pass

        # --- Risk score gauge chart ---
        def compute_risk_score(row):
            total = len(PPE_ITEMS)
            missing = sum(1 for p in PPE_ITEMS if row[p] == 0)
            score = (missing / total) * 100
            return score

        filtered['risk_score'] = filtered.apply(compute_risk_score, axis=1)
        avg_risk_score = filtered['risk_score'].mean()

        st.subheader("⚠️ Average Risk Score")
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=avg_risk_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk Score (0-100)", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red"},
                'steps': [
                    {'range': [0, 33], 'color': "green"},
                    {'range': [33, 66], 'color': "yellow"},
                    {'range': [66, 100], 'color': "red"}],
                'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': avg_risk_score}}))
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No log rows after applying filters — cannot draw overall charts.")

# ---------------- End ----------------
