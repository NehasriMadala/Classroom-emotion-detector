# app.py
import streamlit as st
import cv2
import tempfile
import os
from deepface import DeepFace
from collections import Counter
import numpy as np
from tqdm import tqdm

st.set_page_config(layout="wide")
st.title("🎓 Classroom Emotion Detection (Video Upload)")
st.markdown("Upload a classroom video. The app will sample frames, detect faces, run emotion detection on faces, and produce an output video with labels plus an emotion summary.")

uploaded = st.file_uploader("Upload classroom video (mp4 / avi / mov)", type=["mp4", "avi", "mov"])

# Performance knobs (user can change)
st.sidebar.header("Performance settings")
SAMPLE_EVERY_N_FRAMES = st.sidebar.number_input("Process every Nth frame (higher = faster)", min_value=1, max_value=30, value=5, step=1)
MAX_FRAMES = st.sidebar.number_input("Max frames to analyse (0 = all)", min_value=0, max_value=2000, value=0, step=50)

if uploaded is not None:
    # Save uploaded file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded.read())
    tfile.flush()
    input_path = tfile.name

    st.video(input_path)  # show original video

    analyze_button = st.button("Start Analysis")

    if analyze_button:
        with st.spinner("Preparing video and detector..."):
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                st.error("Could not open the video file.")
                raise SystemExit

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            # prepare output video writer
            out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            out_path = out_tmp.name
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            # face detector (fast) - Haar cascade shipped with OpenCV
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

            emotion_counter = Counter()
            frame_display = st.empty()
            stats_display = st.empty()
            progress_bar = st.progress(0)

            # limit frames to process (for speed)
            frames_to_process = total_frames if MAX_FRAMES == 0 else min(total_frames, MAX_FRAMES)
            if frames_to_process == 0:
                frames_to_process = total_frames

            processed_frames = 0
            # iterate frames
            frame_idx = 0
            pbar_total = max(1, frames_to_process // SAMPLE_EVERY_N_FRAMES)
            pbar = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # always write original frame to output (we'll annotate and write)
                annotate_frame = frame.copy()

                if frame_idx % SAMPLE_EVERY_N_FRAMES == 0:
                    # detect faces (grayscale)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))

                    # For each detected face, crop and analyze
                    for (x, y, w, h) in faces:
                        # expand ROI slightly to capture full face
                        pad = int(0.15 * w)
                        x1 = max(0, x - pad)
                        y1 = max(0, y - pad)
                        x2 = min(frame.shape[1], x + w + pad)
                        y2 = min(frame.shape[0], y + h + pad)
                        face_img = frame[y1:y2, x1:x2]

                        # DeepFace expects RGB
                        try:
                            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                            # run emotion detection (fast mode: enforce_detection=False to avoid raising)
                            result = DeepFace.analyze(face_rgb, actions=["emotion"], enforce_detection=False)
                            # DeepFace may return dict or list (handle both)
                            if isinstance(result, list) and len(result) > 0:
                                dom = result[0].get("dominant_emotion", None)
                            elif isinstance(result, dict):
                                dom = result.get("dominant_emotion", None)
                            else:
                                dom = None

                        except Exception as e:
                            dom = None

                        label = dom if dom is not None else "unknown"
                        emotion_counter[label] += 1

                        # Draw bounding box + label
                        cv2.rectangle(annotate_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotate_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    # update progress
                    pbar += 1
                    progress = min(1.0, pbar / pbar_total)
                    progress_bar.progress(progress)

                # write annotated frame to output video
                out_writer.write(annotate_frame)
                frame_display.image(cv2.cvtColor(annotate_frame, cv2.COLOR_BGR2RGB), channels="RGB")
                processed_frames += 1
                frame_idx += 1

                # break early if processed enough frames
                if MAX_FRAMES != 0 and frame_idx >= MAX_FRAMES:
                    break

            cap.release()
            out_writer.release()

        st.success("✅ Processing complete.")
        # show summary
        stats_display.subheader("Emotion Summary (counts from analysed faces)")
        if len(emotion_counter) == 0:
            st.warning("No faces / emotions were detected. Try lowering 'Process every Nth frame' to 1 and increasing 'Max frames' or use a clearer video.")
        else:
            st.table(sorted(emotion_counter.items(), key=lambda x: -x[1]))

        st.subheader("Processed Video with emotion labels")
        st.video(out_path)

        # provide download link
        with open(out_path, "rb") as f:
            st.download_button("Download processed video", data=f, file_name="processed_video.mp4", mime="video/mp4")

        # clean up temporary input file
        try:
            os.remove(input_path)
        except Exception:
            pass
