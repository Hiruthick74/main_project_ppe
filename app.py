# streamlit run "C:\Users\rselv\OneDrive\Documents\YOLOv8\object detection ppe project\app.py"

# https://192.0.0.4:8080

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import pandas as pd
from collections import Counter
import time

# Load YOLOv8 Model
model = YOLO("runs/detect/train9/weights/best.pt")

st.set_page_config(page_title="PPE Detection Dashboard", layout="wide")
st.title("🚧 AI-Powered PPE Detection Dashboard")

# Sidebar options
st.sidebar.header("Input Options")
option = st.sidebar.radio("Choose Input Source:", ["Image", "Video", "Webcam", "IP Camera"])

CONF_THRESHOLD = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

start_button = st.sidebar.button("▶ Start Detection")
stop_button = st.sidebar.button("⏹ Stop Detection")

if "stop" not in st.session_state:
    st.session_state.stop = False
if stop_button:
    st.session_state.stop = True

# ---------------- Function: Count PPE dynamically ----------------
def count_ppe(results):
    counts = Counter()
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names.get(cls_id, "Unknown")
        counts[label] += 1
    return counts

# ---------------- Function: Run detection loop ----------------
def process_video(cap):
    stframe = st.empty()
    chart_placeholder = st.empty()
    metric_placeholder = st.container()

    while cap.isOpened() and not st.session_state.stop:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for smoother FPS
        frame = cv2.resize(frame, (640, 480))

        # YOLO Inference on CPU
        results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False, device="cpu")
        annotated = results[0].plot()
        ppe_count = count_ppe(results)

        # Show annotated frame
        stframe.image(annotated, channels="BGR")

        # Update dashboard metrics
        with metric_placeholder:
            st.subheader("📊 Live Metrics")
            cols = st.columns(len(ppe_count) if ppe_count else 1)
            for i, (item, value) in enumerate(ppe_count.items()):
                cols[i].metric(item, value)

        # Update bar chart
        if ppe_count:
            chart_placeholder.bar_chart(pd.Series(ppe_count))

        # Small delay for smoother FPS
        time.sleep(0.02)

    cap.release()

# ---------------- IMAGE INPUT ----------------
if option == "Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file and start_button:
        image = Image.open(uploaded_file)
        if image.mode != "RGB":
            image = image.convert("RGB")

        results = model.predict(np.array(image), conf=CONF_THRESHOLD, device="cpu")
        annotated = results[0].plot()
        ppe_count = count_ppe(results)

        st.image(annotated, caption="Detection Result", use_column_width=True)

        st.subheader("📊 Detected PPE Counts")
        for item, value in ppe_count.items():
            st.metric(item, value)
        st.bar_chart(pd.Series(ppe_count))

# ---------------- VIDEO INPUT ----------------
elif option == "Video":
    uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file and start_button:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        process_video(cap)

# ---------------- WEBCAM INPUT ----------------
elif option == "Webcam":
    if start_button:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        process_video(cap)

# ---------------- IP CAMERA INPUT ----------------
elif option == "IP Camera":
    ip_url = st.text_input("Enter IP Camera URL (e.g., http://192.0.0.4:8080/video):")
    if ip_url and start_button:
        cap = cv2.VideoCapture(ip_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # reduce lag
        process_video(cap)
