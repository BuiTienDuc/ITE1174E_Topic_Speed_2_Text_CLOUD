import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import whisper
import soundfile as sf

# ==========================
# 🎯 TRANG ANALYSIS
# ==========================
def show():

    st.markdown(
        "<h3 style='color:#2b6f3e;'>Analysis – Phân tích Audio & Speech to Text</h3>",
        unsafe_allow_html=True
    )

    st.write(
        """
        Trang này cho phép người dùng tải lên file audio tiếng Việt,
        phân tích tín hiệu âm thanh và thực hiện chuyển giọng nói thành văn bản
        bằng mô hình Speech-to-Text mã nguồn mở.
        """
    )

    st.write("---")

    # ==========================
    # 🎵 UPLOAD AUDIO
    # ==========================
    audio_file = st.file_uploader(
        "📤 Upload Vietnamese audio file (WAV / MP3 / FLAC)",
        type=["wav", "mp3", "flac"]
    )

    if audio_file is None:
        st.info("Vui lòng upload file audio để bắt đầu phân tích.")
        return

    # Lưu file tạm
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        audio_path = tmp.name

    # ==========================
    # 📊 THÔNG TIN AUDIO
    # ==========================
    y, sr = librosa.load(audio_path, sr=None)

    duration = librosa.get_duration(y=y, sr=sr)

    st.subheader("🔍 Thông tin Audio")
    col1, col2, col3 = st.columns(3)
    col1.metric("Sample Rate (Hz)", sr)
    col2.metric("Duration (seconds)", f"{duration:.2f}")
    col3.metric("Channels", "Mono")

    st.write("---")

    # ==========================
    # 📈 WAVEFORM
    # ==========================
    st.subheader("📈 Waveform")

    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    # ==========================
    # 📊 SPECTROGRAM
    # ==========================
    st.subheader("📊 Spectrogram")

    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        D,
        sr=sr,
        x_axis="time",
        y_axis="hz",
        ax=ax
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    st.pyplot(fig)

    st.write("---")

    # ==========================
    # 🧠 SPEECH TO TEXT
    # ==========================
    st.subheader("🧠 Vietnamese Speech to Text")

    st.write(
        """
        Hệ thống sử dụng mô hình **Whisper (open-source)** để chuyển đổi
        giọng nói tiếng Việt thành văn bản.
        """
    )

    if st.button("▶️ Thực hiện Speech-to-Text"):
        with st.spinner("Đang nhận dạng giọng nói..."):
            model = whisper.load_model("base")
            result = model.transcribe(audio_path, language="vi")

        transcript = result["text"]

        st.success("Hoàn thành nhận dạng!")

        # ==========================
        # 📝 KẾT QUẢ TRANSCRIPT
        # ==========================
        st.subheader("📝 Transcript (có thể chỉnh sửa)")

        edited_text = st.text_area(
            "Nội dung chuyển giọng nói → văn bản:",
            transcript,
            height=300
        )

        # ==========================
        # 📤 EXPORT
        # ==========================
        st.write("---")
        st.subheader("📤 Xuất Transcript")

        st.download_button(
            label="⬇️ Tải file TXT",
            data=edited_text,
            file_name="meeting_transcript.txt",
            mime="text/plain"
        )

        # Thống kê đơn giản
        word_count = len(edited_text.split())
        st.info(f"📊 Số từ trong transcript: **{word_count}**")
