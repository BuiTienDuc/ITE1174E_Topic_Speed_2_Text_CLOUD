import streamlit as st
import os
import pickle
import torch

# ==========================
# 📦 Model persistence
# ==========================
MODEL_DIR = "models"


# ==========================
# 🎨 HỘP HIỂN THỊ (giống Topic 3)
# ==========================
def info_box(html: str):
    st.markdown(
        f"""
        <div style="
            background-color:#fff7cc;
            padding:18px;
            border-radius:10px;
            border:1px solid #e6d784;
            font-size:16px;
            line-height:1.6;
            margin-bottom:15px;">
            {html}
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================
# 🔍 ĐỌC THÔNG TIN MODEL OBJECT
# ==========================
def load_model_info():
    models_info = []

    if not os.path.exists(MODEL_DIR):
        return models_info

    for fname in os.listdir(MODEL_DIR):
        if not fname.endswith(".pkl"):
            continue

        fpath = os.path.join(MODEL_DIR, fname)

        try:
            with open(fpath, "rb") as f:
                model = pickle.load(f)

            param_count = sum(p.numel() for p in model.parameters())
            trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

            models_info.append({
                "name": fname.replace(".pkl", ""),
                "file": fname,
                "size_mb": round(os.path.getsize(fpath) / (1024 * 1024), 2),
                "total_params": param_count,
                "trainable_params": trainable_count,
                "device": "CPU"
            })

        except Exception as e:
            models_info.append({
                "name": fname.replace(".pkl", ""),
                "file": fname,
                "error": str(e)
            })

    return models_info


# ==========================
# 🎯 TRANG TRAINING INFO
# ==========================
def show():

    st.markdown(
        "<h3 style='color:#2b6f3e;'>Training Info – Thông tin mô hình Speech-to-Text</h3>",
        unsafe_allow_html=True
    )

    st.write(
        """
        Trang này trình bày **quy trình xử lý – mô hình – kết quả – so sánh**
        của hệ thống Vietnamese Speech-to-Text, tương tự bố cục Topic 3.
        """
    )

    st.write("---")

    # ==========================================================
    # 1️⃣ HIỆN DỮ LIỆU THÔ
    # ==========================================================
    info_box("""
    <h4 style="color:#b30000;">1. Dữ liệu thô (Raw Audio Data)</h4>
    <ul>
        <li>Dữ liệu đầu vào là các file audio cuộc họp, thảo luận, phỏng vấn.</li>
        <li>Định dạng phổ biến: <b>MP3, WAV, FLAC</b>.</li>
        <li>Audio có thể có nhiễu nền, nhiều người nói.</li>
    </ul>
    """)

    # ==========================================================
    # 2️⃣ TIỀN XỬ LÝ
    # ==========================================================
    info_box("""
    <h4 style="color:#b30000;">2. Tiền xử lý dữ liệu</h4>
    <ul>
        <li>Chuẩn hóa audio về <b>WAV – PCM16 – mono – 16kHz</b>.</li>
        <li>Chia audio dài thành các đoạn nhỏ (chunking).</li>
    </ul>
    """)

    # ==========================================================
    # 3️⃣ KIẾN TRÚC MÔ HÌNH
    # ==========================================================
    info_box("""
    <h4 style="color:#b30000;">3. Kiến trúc mô hình</h4>
    <ul>
        <li>Sử dụng mô hình <b>Whisper</b> (OpenAI).</li>
        <li>Transformer Encoder–Decoder.</li>
        <li>Huấn luyện đa ngôn ngữ.</li>
    </ul>
    """)

    # ==========================================================
    # 4️⃣ THÔNG TIN MODEL ĐÃ LƯU (OBJECT)
    # ==========================================================
    st.write("---")
    st.subheader("📦 Thông tin Model Object đã lưu")

    models_info = load_model_info()

    if not models_info:
        st.warning("⚠️ Chưa tìm thấy model .pkl trong thư mục models/")
    else:
        for m in models_info:
            if "error" in m:
                st.error(f"❌ {m['file']}: {m['error']}")
                continue

            info_box(f"""
            <h4 style="color:#b30000;">{m['name']}</h4>
            <ul>
                <li><b>File:</b> {m['file']}</li>
                <li><b>Dung lượng:</b> {m['size_mb']} MB</li>
                <li><b>Tổng số tham số:</b> {m['total_params']:,}</li>
                <li><b>Tham số trainable:</b> {m['trainable_params']:,}</li>
                <li><b>Thiết bị inference:</b> {m['device']}</li>
                <li><b>Định dạng lưu:</b> Pickle (.pkl)</li>
            </ul>
            """)

    # ==========================================================
    # 5️⃣ ĐÁNH GIÁ
    # ==========================================================
    info_box("""
    <h4 style="color:#b30000;">5. Đánh giá & độ tin cậy</h4>
    <ul>
        <li>Whisper base cho độ chính xác tốt với tiếng Việt phổ thông.</li>
        <li>Model được cache và load từ object giúp tăng tốc độ hệ thống.</li>
        <li>Phù hợp triển khai trên CPU (Streamlit).</li>
    </ul>
    """)

    # ==========================================================
    # 6️⃣ SO SÁNH
    # ==========================================================
    info_box("""
    <h4 style="color:#b30000;">6. So sánh các mô hình</h4>
    <table style="width:100%; border-collapse:collapse;" border="1">
        <tr style="background:#f2f2f2;">
            <th>Mô hình</th>
            <th>Tham số</th>
            <th>Tốc độ</th>
            <th>Phù hợp</th>
        </tr>
        <tr>
            <td>Whisper tiny</td>
            <td>~39M</td>
            <td>Rất nhanh</td>
            <td>Demo</td>
        </tr>
        <tr>
            <td>Whisper base</td>
            <td>~74M</td>
            <td>Nhanh</td>
            <td>Khuyến nghị</td>
        </tr>
        <tr>
            <td>Whisper small</td>
            <td>~244M</td>
            <td>Chậm hơn</td>
            <td>Audio ngắn</td>
        </tr>
    </table>
    """)
