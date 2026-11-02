import os
import joblib
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from preprocess import preprocess

st.set_page_config(page_title="Phân loại tin tức tiếng Việt", page_icon="📰", layout="wide")

TEN_CHUYEN_MUC = {
    "chinh_tri_xa_hoi": "Chính trị - Xã hội",
    "doi_song": "Đời sống",
    "khoa_hoc": "Khoa học",
    "kinh_doanh": "Kinh doanh",
    "phap_luat": "Pháp luật",
    "suc_khoe": "Sức khỏe",
    "the_gioi": "Thế giới",
    "the_thao": "Thể thao",
    "van_hoa": "Văn hóa",
    "vi_tinh": "Vi tính",
}

MAU = {
    "(Tự nhập nội dung)": "",
    "Mẫu: thể thao": "Đội tuyển Việt Nam thắng đậm đối thủ trong trận đấu tối qua nhờ cú đúp của tiền đạo. "
                     "Huấn luyện viên cho biết các cầu thủ đã thi đấu rất quyết tâm ở hiệp hai và giành ba điểm quan trọng tại vòng loại.",
    "Mẫu: sức khỏe": "Các bác sĩ khuyến cáo bệnh nhân tiểu đường cần kiểm tra đường huyết định kỳ và uống thuốc đúng liều. "
                     "Chế độ ăn ít đường và vận động đều đặn giúp giảm nguy cơ biến chứng tim mạch.",
    "Mẫu: vi tính": "Microsoft vừa phát hành bản cập nhật Windows mới nhằm vá lỗi bảo mật nghiêm trọng. "
                    "Người dùng máy tính nên cài đặt phần mềm diệt virus và cập nhật hệ điều hành để tránh bị tấn công qua mạng internet.",
    "Mẫu: kinh doanh": "Ngân hàng điều chỉnh lãi suất huy động, thị trường chứng khoán biến động mạnh khi nhà đầu tư lo ngại về lạm phát. "
                       "Nhiều doanh nghiệp xuất khẩu báo lỗ trong quý này do chi phí tăng.",
}


def ten_hien_thi(ma):
    return TEN_CHUYEN_MUC.get(ma, ma)


@st.cache_resource
def tai_mo_hinh():
    b = joblib.load(os.path.join("model", "demo_model.pkl"))
    b["vocab"] = np.array(b["tfidf"].get_feature_names_out())
    return b


bundle = tai_mo_hinh()
tfidf, nb = bundle["tfidf"], bundle["nb"]
X_index = bundle["X_index"]
index_labels = bundle["index_labels"]
index_snippets = bundle["index_snippets"]
vocab = bundle["vocab"]

with st.sidebar:
    st.header("Thông tin mô hình")
    st.write("**Thuật toán:** Naive Bayes đa thức (alpha = 0,1)")
    st.write("**Đặc trưng:** TF-IDF, 50.000 từ")
    st.write("**Dữ liệu huấn luyện:** bộ VNTC, 10 chuyên mục")
    st.write(f"**Số bài trong chỉ mục gợi ý:** {X_index.shape[0]:,}")
    st.caption("Bộ dữ liệu VNTC khá cũ và cách chia chuyên mục có thể khác với các báo hiện nay, "
               "nên kết quả với tin tức mới có thể không trùng với chuyên mục của báo.")

st.title("📰 Phân loại tin tức tiếng Việt và gợi ý bài tương tự")

chon = st.selectbox("Chèn bài mẫu:", list(MAU))
text = st.text_area("Dán nội dung bài báo vào đây:", value=MAU[chon], height=250)

if st.button("Phân loại", type="primary"):
    if not text.strip():
        st.warning("Bạn chưa nhập nội dung.")
        st.stop()

    clean = preprocess(text)
    v = tfidf.transform([clean])
    if v.nnz == 0:
        st.warning("Không tìm thấy từ nào quen thuộc với mô hình. Hãy nhập đoạn văn tiếng Việt dài hơn.")
        st.stop()

    proba = nb.predict_proba(v)[0]
    thu_tu = np.argsort(-proba)
    top1 = nb.classes_[thu_tu[0]]

    cot_trai, cot_phai = st.columns(2)

    with cot_trai:
        st.subheader("Kết quả phân loại")
        st.success(f"Chuyên mục dự đoán: **{ten_hien_thi(top1)}** ({proba[thu_tu[0]]:.1%})")
        if proba[thu_tu[0]] < 0.6:
            st.info("Mô hình chưa thật chắc chắn. Hãy xem các chuyên mục có xác suất cao tiếp theo.")

        df_p = pd.DataFrame({
            "Chuyên mục": [ten_hien_thi(nb.classes_[i]) for i in thu_tu],
            "Xác suất": [float(proba[i]) for i in thu_tu],
        })
        bieu_do = alt.Chart(df_p).mark_bar().encode(
            y=alt.Y("Chuyên mục:N", sort=None, axis=alt.Axis(title=None, labelLimit=200)),
            x=alt.X("Xác suất:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            tooltip=["Chuyên mục", alt.Tooltip("Xác suất:Q", format=".1%")],
        ).properties(height=320)
        st.altair_chart(bieu_do, width="stretch")

        top3 = " · ".join(
            f"{ten_hien_thi(nb.classes_[i])} {proba[i]:.0%}" for i in thu_tu[:3])
        st.markdown(f"**3 chuyên mục cao nhất:** {top3}")
        st.caption("Xác suất của Naive Bayes thường 'quá tự tin', nên hãy dùng để so sánh thứ hạng giữa các chuyên mục.")

        st.markdown("**Từ khóa nổi bật trong bài:**")
        v_coo = v.tocoo()
        top_idx = np.argsort(-v_coo.data)[:8]
        st.write(", ".join(vocab[v_coo.col[top_idx]]))

    with cot_phai:
        st.subheader("5 bài tương tự nhất trong chỉ mục")
        sim = (v @ X_index.T).toarray().ravel()
        top5 = np.argsort(-sim)[:5]
        for hang, i in enumerate(top5, 1):
            st.markdown(f"**{hang}. [{ten_hien_thi(index_labels[i])}]** — độ giống {sim[i]:.0%}")
            st.write(index_snippets[i] + "...")

    with st.expander("Xem văn bản sau khi tiền xử lý"):
        st.write(clean)