
import streamlit as st
from ultralytics import YOLO
import cv2, numpy as np, pandas as pd, re, time, tempfile
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from alert_system import trigger_alert
from violation import append_detection

# ---------------- Config ----------------
st.set_page_config(page_title="PPE Detection Dashboard", layout="wide")
LOG_CSV = "violations.csv"
cv2.setNumThreads(1)

# ---------------- Power BI-Style CSS ----------------
st.markdown("""
<style>
.stApp {
    background-color: #f4f6f8;
    color: #111;
    font-family: 'Segoe UI', sans-serif;
}
h1,h2,h3,h4 {
    color: #0078D7;
    font-weight: 600;
}
section[data-testid="stSidebar"] {
    background-color: #111827;
    color: white;
    border-right: 2px solid #00A8A8;
}
div.block-container {
    padding-top: 1rem;
}
.metric-card {
    background: white;
    padding: 18px;
    border-radius: 12px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    transition: all 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 4px 10px rgba(0,0,0,0.25);
}
[data-testid="stMetricValue"] {
    color: #0078D7;
    font-weight: 700;
    font-size: 24px;
}
[data-testid="stDataFrame"] {
    border-radius: 10px;
    border: 1px solid #ddd;
    background-color: white;
}
hr { border: 1px solid #ccc; }
</style>
""", unsafe_allow_html=True)

# ---------------- Title ----------------
st.markdown("""
<div style="background:#0078D7;padding:12px 18px;border-radius:8px;">
<h2 style="color:white;">⚙️ AI-Powered PPE Detection Dashboard</h2>
</div>
""", unsafe_allow_html=True)
st.caption("Interactive Analytics • Power BI-style Layout • Real-time + Historical Insights")

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Input Settings")
input_mode = st.sidebar.radio("Input Source:", ["Image", "Video", "Webcam", "IP Camera"])
CONF_THRESHOLD = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
start_btn = st.sidebar.button("▶ Start Detection")
stop_btn = st.sidebar.button("⏹ Stop Detection")

if "stop" not in st.session_state:
    st.session_state.stop = False
if stop_btn:
    st.session_state.stop = True
if start_btn:
    st.session_state.stop = False

# ---------------- Load YOLO ----------------
@st.cache_resource
def load_model():
    return YOLO("runs/detect/train9/weights/best.pt")

model = load_model()
PPE_ITEMS = ["Helmet", "Vest", "Gloves", "Goggles", "Shoes"]

# ---------------- Helper Functions ----------------
def count_ppe_from_results(results):
    mapping = {"helmet": "Helmet", "vest": "Vest", "gloves": "Gloves", "goggles": "Goggles", "shoes": "Shoes"}
    counts = {p: 0 for p in mapping.values()}
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = results[0].names.get(cls_id, "").lower()
        if label in mapping:
            counts[mapping[label]] += 1
    return counts

def classify_risk_from_counts(ppe_counts):
    missing = sum(1 for v in ppe_counts.values() if v == 0)
    return "High" if missing >= 3 else "Medium" if missing >= 1 else "Low"

# ---------------- Optimized Stream / Video Processor ----------------
def process_video(source, label="video"):
    stframe = st.empty()
    metrics = st.empty()
    fps_display = st.sidebar.empty()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        st.error("⚠️ Failed to open video or camera.")
        return

    frame_count = 0
    start_time = time.time()

    while cap.isOpened() and not st.session_state.stop:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Convert properly
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame, (640, 640))

        # YOLO detection
        results = model.predict(resized, conf=CONF_THRESHOLD, device="cpu", verbose=False)
        annotated = results[0].plot()

        # PPE and risk logic
        ppe_counts = count_ppe_from_results(results)
        risk = classify_risk_from_counts(ppe_counts)

        append_detection(str(ppe_counts), "video_cam", label, risk)
        trigger_alert(ppe_counts)

        # FPS calculation
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        fps_display.write(f"📹 **Current FPS:** {fps:.1f}")

        # Draw border color based on risk
        color = (0, 255, 0) if risk == "Low" else (255, 255, 0) if risk == "Medium" else (255, 0, 0)
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        cv2.rectangle(annotated, (0, 0), (annotated.shape[1], annotated.shape[0]), color, 5)

        # Display frame
        stframe.image(annotated, channels="RGB", use_column_width=True)

        # Live PPE metrics
        with metrics:
            cols = st.columns(len(ppe_counts))
            for i, (k, v) in enumerate(ppe_counts.items()):
                cols[i].metric(k, v)
            st.markdown(f"### ⚠️ Risk: **{risk}**")

        # Small delay to stabilize FPS
        time.sleep(0.03)

    cap.release()
    st.success("✅ Stream Ended")

# ---------------- Input Handlers ----------------
if input_mode == "Image":
    up = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if up and start_btn:
        img = Image.open(up).convert("RGB")
        arr = np.array(img)
        results = model.predict(arr, conf=CONF_THRESHOLD, device="cpu", verbose=False)
        annotated = results[0].plot()
        ppe_counts = count_ppe_from_results(results)
        risk = classify_risk_from_counts(ppe_counts)
        append_detection(str(ppe_counts), "img", "image", risk)
        trigger_alert(ppe_counts)
        st.image(annotated, use_column_width=True)
        st.markdown(f"### ⚠️ Risk Level: **{risk}**")

elif input_mode == "Video":
    up = st.file_uploader("🎞️ Upload Video", type=["mp4", "avi", "mov", "mkv"])
    if up and start_btn:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(up.read())
        process_video(tfile.name, "video")

elif input_mode == "Webcam":
    if start_btn:
        process_video(0, "webcam")

elif input_mode == "IP Camera":
    ip = st.text_input("Enter IP Camera URL (e.g. http://192.0.0.4:8080/video)")
    if ip and start_btn:
        process_video(ip, "ipcam")

# ---------------- Historical Dashboard ----------------
st.markdown("---")
st.markdown("## 📈 Historical Analytics")

def safe_read_csv(path):
    try: return pd.read_csv(path)
    except: return None

df = safe_read_csv(LOG_CSV)
if df is not None:
    df.columns = df.columns.str.strip()
    detected_col = next((c for c in df.columns if "ppe" in c.lower()), None)

    def extract_ppe_values(s):
        counts = {p: 0 for p in PPE_ITEMS}
        for p in PPE_ITEMS:
            m = re.search(rf"{p}\s*[:=]\s*(\d+)", str(s), re.IGNORECASE)
            if m: counts[p] = int(m.group(1))
        return counts

    if detected_col:
        parsed = df[detected_col].apply(extract_ppe_values)
        for p in PPE_ITEMS:
            df[p] = parsed.apply(lambda d: d[p])

    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        def get_session(ts):
            if pd.isna(ts): return "unknown"
            h = ts.hour
            return ("morning" if 6 <= h < 12 else
                    "afternoon" if 12 <= h < 17 else
                    "evening" if 17 <= h < 21 else
                    "night")
        df["time_of_day"] = df["Timestamp"].apply(get_session)

    df["risk_level"] = df[PPE_ITEMS].apply(lambda r: classify_risk_from_counts(r.to_dict()), axis=1)

    st.markdown("### 🔍 Data Filter")
    time_filter = st.selectbox("Filter by Time of Day:", ["All", "morning", "afternoon", "evening", "night"])
    df_filtered = df if time_filter == "All" else df[df["time_of_day"] == time_filter]

    # KPI Cards
    total_logs = len(df_filtered)
    avg_risk = (df_filtered[PPE_ITEMS] == 0).mean().mean() * 100 if total_logs > 0 else 0
    high = df_filtered[df_filtered["risk_level"] == "High"].shape[0]
    med = df_filtered[df_filtered["risk_level"] == "Medium"].shape[0]
    low = df_filtered[df_filtered["risk_level"] == "Low"].shape[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f"<div class='metric-card'><h4>Total Logs</h4><h2>{total_logs}</h2></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='metric-card'><h4>Average Risk %</h4><h2>{avg_risk:.1f}%</h2></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='metric-card'><h4>High Risk</h4><h2 style='color:red'>{high}</h2></div>", unsafe_allow_html=True)
    with c4: st.markdown(f"<div class='metric-card'><h4>Low Risk</h4><h2 style='color:green'>{low}</h2></div>", unsafe_allow_html=True)

    st.markdown("### 🧾 Log Snapshot")
    st.dataframe(df_filtered[["Timestamp"] + PPE_ITEMS + ["risk_level", "time_of_day"]].tail(200), use_container_width=True)

    # Charts
    st.markdown("### 📊 Analytics Overview")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### PPE Count by Time of Day")
        session_ppe = df_filtered.groupby("time_of_day")[PPE_ITEMS].sum()
        st.bar_chart(session_ppe)
    with colB:
        st.markdown("#### Risk Distribution")
        risk_by = df_filtered.groupby(["time_of_day", "risk_level"]).size().unstack(fill_value=0)
        st.area_chart(risk_by)

    st.markdown("#### ⚠️ Average Risk Gauge")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_risk,
        title={'text': "Average PPE Risk %"},
        gauge={'axis': {'range': [0, 100]},
               'steps': [{'range': [0, 33], 'color': "green"},
                         {'range': [33, 66], 'color': "yellow"},
                         {'range': [66, 100], 'color': "red"}],
               'bar': {'color': "#0078D7"}}))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No log data found. Run detection to generate logs.")
