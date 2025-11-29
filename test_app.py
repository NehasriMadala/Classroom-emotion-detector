import streamlit as st

st.title("Emotion Detection Test App")

st.write("✅ Streamlit is working correctly!")

video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if video is not None:
    st.success("Video uploaded successfully!")
    st.video(video)
