import streamlit as st

# ==========================
# 🎨 HỘP HIỂN THỊ NỘI DUNG
# ==========================
def intro_box(text: str):
    st.markdown(
        f"""
        <div style="
            background-color:#fff7cc;
            padding:20px;
            border-radius:10px;
            border:1px solid #e6d784;
            font-size:18px;
            line-height:1.7;
        ">
        {text}
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================
# 🎯 TRANG HOME (BẮT BUỘC CÓ show)
# ==========================
def show():

    st.markdown(
        "<h3 style='color:#2b6f3e;'>Giới thiệu Đề tài</h3>",
        unsafe_allow_html=True
    )

    intro_box("""
    <h3 style="color:#b30000;">1. Bối cảnh và Lý do chọn đề tài</h3>
    Trong các cuộc họp, thảo luận, phỏng vấn hoặc thuyết trình,
    việc ghi biên bản thủ công thường tốn thời gian và dễ sai sót.
    <br><br>
    Đề tài này xây dựng hệ thống <b>Vietnamese Speech-to-Text</b>
    giúp tự động chuyển giọng nói tiếng Việt thành văn bản,
    phục vụ cho hành chính, giáo dục và doanh nghiệp.
    """)

    intro_box("""
    <h3 style="color:#b30000;">2. Mục tiêu Đề tài</h3>
    <ul>
        <li>Xây dựng hệ thống Speech-to-Text tiếng Việt bằng mô hình mã nguồn mở.</li>
        <li>Cho phép tải lên audio cuộc họp (WAV/MP3).</li>
        <li>Hiển thị waveform và spectrogram.</li>
        <li>Tạo transcript tự động và cho phép chỉnh sửa.</li>
    </ul>
    """)

    intro_box("""
    <h3 style="color:#b30000;">3. Phạm vi thực hiện</h3>
    <ul>
        <li>Tiền xử lý audio.</li>
        <li>Nhận dạng tiếng nói bằng Whisper.</li>
        <li>Triển khai web app bằng Streamlit.</li>
        <li>Deploy trên Streamlit Cloud.</li>
    </ul>
    """)

    intro_box("""
    <h3 style="color:#b30000;">4. Ý nghĩa khoa học và thực tiễn</h3>
    <ul>
        <li>Ứng dụng AI vào xử lý tiếng nói tiếng Việt.</li>
        <li>Hỗ trợ tự động hóa ghi biên bản cuộc họp.</li>
        <li>Có khả năng mở rộng sang tóm tắt và phân tích cuộc họp.</li>
    </ul>
    """)

    st.write("---")

    

   