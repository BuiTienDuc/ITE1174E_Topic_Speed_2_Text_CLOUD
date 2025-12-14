import streamlit as st
import numpy as np
import tempfile
import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf

import whisper


# ==========================
# ✅ Helper UI: yellow box
# ==========================
def yellow_box(html: str):
    st.markdown(
        f"""
        <div style="
            background-color:#fff7cc;
            padding:18px;
            border-radius:10px;
            border:1px solid #e6d784;
            font-size:16px;
            line-height:1.6;">
            {html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================
# ✅ Load Whisper model (Cloud-safe)
# ==========================
@st.cache_resource
def load_whisper(model_size: str = "base"):
    """
    Load Whisper model trực tiếp từ thư viện.
    Phù hợp Streamlit Cloud – không lưu file .pkl
    """
    return whisper.load_model(model_size)


# ==========================
# ✅ Audio utils
# ==========================
def normalize_audio_to_wav(audio_path: str, target_sr: int = 16000):
    """Load audio -> mono 16kHz WAV PCM16"""
    y, sr = librosa.load(audio_path, sr=target_sr, mono=True)

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0:
        y = y / peak

    out_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    out_wav.close()
    sf.write(out_wav.name, y, target_sr, subtype="PCM_16")

    return out_wav.name, target_sr, y


def chunk_signal(y: np.ndarray, sr: int, chunk_seconds: int):
    total_samples = len(y)
    chunk_len = int(chunk_seconds * sr)

    if chunk_len <= 0 or total_samples == 0:
        return [(0, total_samples)]

    ranges = []
    for start in range(0, total_samples, chunk_len):
        end = min(start + chunk_len, total_samples)
        ranges.append((start, end))

    return ranges


def format_timestamp(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


# ==========================
# 🎯 PAGE
# ==========================
def show():
    st.markdown(
        "<h3 style='color:#2b6f3e;'>Analysis – Phân tích Audio & Speech-to-Text</h3>",
        unsafe_allow_html=True,
    )

    yellow_box(
        """
        <h4 style="color:#b30000; margin:0;">Mục tiêu</h4>
        <ul style="margin:10px 0 0 18px;">
            <li>Upload audio (WAV / MP3 / FLAC)</li>
            <li>Chuẩn hoá về <b>mono – 16kHz – WAV</b></li>
            <li>Speech-to-Text bằng <b>Whisper Base</b></li>
            <li>Xử lý audio dài bằng <b>chunking</b></li>
        </ul>
        """
    )

    st.write("---")

    # ==========================
    # Upload audio
    # ==========================
    audio_file = st.file_uploader(
        "📤 Upload audio",
        type=["wav", "mp3", "flac"],
    )

    if audio_file is None:
        st.info("Vui lòng upload file audio để bắt đầu.")
        return

    suffix = "." + audio_file.name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_file.read())
        raw_path = tmp.name

    # ==========================
    # Options
    # ==========================
    st.subheader("⚙️ Tuỳ chọn")

    col1, col2 = st.columns(2)

    with col1:
        lang_mode = st.selectbox(
            "Ngôn ngữ",
            ["Auto-detect", "Vietnamese (vi)", "English (en)"],
            index=0,
        )

    with col2:
        chunk_seconds = st.selectbox(
            "Độ dài mỗi đoạn (giây)",
            [15, 30, 45, 60],
            index=1,
        )

    # ==========================
    # Normalize
    # ==========================
    st.subheader("🧼 Chuẩn hoá audio")
    with st.spinner("Đang chuẩn hoá audio..."):
        norm_path, sr, y = normalize_audio_to_wav(raw_path)

    duration = librosa.get_duration(y=y, sr=sr)
    st.success(f"Chuẩn hoá xong | Duration: {duration:.2f}s")

    st.audio(open(norm_path, "rb").read())

    # ==========================
    # Visualization
    # ==========================
    st.subheader("📈 Waveform")
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    st.pyplot(fig)

    # ==========================
    # Speech-to-Text
    # ==========================
    st.write("---")
    st.subheader("🧠 Speech-to-Text")

    lang_param = None
    if lang_mode == "Vietnamese (vi)":
        lang_param = "vi"
    elif lang_mode == "English (en)":
        lang_param = "en"

    ranges = chunk_signal(y, sr, int(chunk_seconds))
    st.write(f"🔹 Số đoạn: **{len(ranges)}**")

    if st.button("▶️ Thực hiện Speech-to-Text"):
        model = load_whisper("base")

        progress = st.progress(0.0)
        transcripts = []
        detected_lang = None

        for i, (s0, s1) in enumerate(ranges, start=1):
            chunk_y = y[s0:s1]

            tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp_wav.close()
            sf.write(tmp_wav.name, chunk_y, sr, subtype="PCM_16")

            kwargs = {"fp16": False}
            if lang_param:
                kwargs["language"] = lang_param

            with st.spinner(f"Đang xử lý đoạn {i}/{len(ranges)}..."):
                result = model.transcribe(tmp_wav.name, **kwargs)

            if detected_lang is None:
                detected_lang = result.get("language")

            text = (result.get("text") or "").strip()
            header = f"[{format_timestamp(s0/sr)} - {format_timestamp(s1/sr)}]"
            transcripts.append(f"{header} {text}")

            progress.progress(i / len(ranges))

        if lang_mode == "Auto-detect" and detected_lang:
            st.info(f"🌍 Ngôn ngữ phát hiện: **{detected_lang}**")

        full_text = "\n".join(transcripts)

        st.success("✅ Hoàn thành Speech-to-Text")

        edited_text = st.text_area(
            "📝 Transcript",
            value=full_text,
            height=300,
        )

        st.download_button(
            "⬇️ Tải transcript (.txt)",
            data=edited_text,
            file_name="transcript.txt",
            mime="text/plain",
        )
