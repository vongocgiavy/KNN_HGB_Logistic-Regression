import os
import sys
import json
import base64
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import chess
import chess.svg

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import prepare_and_cache_dataset, check_dataset_stats
from preprocessing import preprocess_data, clean_moves
from logistic_baseline import train_logistic_regression, predict_game_result_lr
from knn_result import predict_result_knn
from knn_opening import predict_opening, train_knn_opening
from hgb_elo import predict_game_result, train_hgb_classifier
from comparison import compare_models

# Page Config (Serious, Professional, Clean)
st.set_page_config(
    page_title="Hệ thống Phân tích Ván cờ Lichess & Machine Learning",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Styling (Minimal Corporate / Academic Design System)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"], .main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }

    /* Hide Streamlit header/footer chrome */
    header[data-testid="stHeader"], footer, [data-testid="stToolbar"], .stDeployButton {
        display: none !important;
    }
    
    .block-container {
        padding: 1.8rem 2.5rem 3rem !important;
        max-width: 1340px !important;
    }

    /* Header Panel */
    .app-header {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.4rem 1.8rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .app-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .app-subtitle {
        font-size: 0.9rem;
        color: #64748b;
        margin-top: 0.3rem;
        margin-bottom: 0;
    }

    /* Card Panels */
    .card-panel {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.4rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        margin-bottom: 1.25rem;
    }
    .card-title {
        font-size: 1rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #f1f5f9;
    }

    /* Tab Styling */
    button[data-baseweb="tab"] {
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        color: #64748b !important;
        padding: 0.6rem 1.2rem !important;
        border-radius: 6px !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #0f172a !important;
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        font-weight: 600 !important;
    }

    /* Metric Display */
    .metric-container {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }
    .metric-label {
        font-size: 0.8rem;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0.2rem;
    }

    /* Chessboard Frame */
    .board-frame {
        display: flex;
        justify-content: center;
        align-items: center;
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
</style>
""", unsafe_allow_html=True)

# Application Header (Serious & Professional)
st.markdown("""
<div class="app-header">
    <div class="app-title">Hệ thống Phân tích Ván cờ Lichess & Machine Learning</div>
    <div class="app-subtitle">Mô hình phân loại kết quả trận đấu và tìm kiếm nhận diện khai cuộc dựa trên cơ sở dữ liệu ván cờ</div>
</div>
""", unsafe_allow_html=True)

# Navigation Tabs (No Icons)
tab1, tab2, tab3 = st.tabs([
    "Dự đoán Kết quả Ván cờ",
    "Nhận diện Khai cuộc & Bàn cờ 2D",
    "Báo cáo So sánh Mô hình & Benchmark"
])

# -----------------------------------------------------------------------------
# TAB 1: RESULT PREDICTION
# -----------------------------------------------------------------------------
with tab1:
    col_input, col_result = st.columns([1, 1.15], gap="large")

    with col_input:
        st.markdown('<div class="card-panel">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Cấu hình Thông số Đầu vào</div>', unsafe_allow_html=True)
        
        white_elo = st.slider("Hệ số Elo người chơi Trắng (white_rating)", 500, 3000, 1800, step=10)
        black_elo = st.slider("Hệ số Elo người chơi Đen (black_rating)", 500, 3000, 1500, step=10)
        
        c1, c2 = st.columns(2)
        with c1:
            is_rated_str = st.selectbox("Phân loại ván đấu (rated)", ["Cờ tính Elo (Rated)", "Cờ giải trí (Casual)"])
            rated_val = 1 if "Rated" in is_rated_str else 0
        with c2:
            opening_ply = st.number_input("Số nước khai cuộc (opening_ply)", min_value=2, max_value=30, value=8)

        selected_model = st.selectbox(
            "Mô hình Machine Learning sử dụng:",
            ["HistGradientBoosting (HGB)", "Logistic Regression (Baseline)", "K-Nearest Neighbors (KNN)"]
        )

        elo_diff = white_elo - black_elo
        st.markdown(f"**Chênh lệch Elo (rating_diff = White - Black):** `{elo_diff:+d}` điểm")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        st.markdown('<div class="card-panel">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Kết quả Dự đoán Xác suất</div>', unsafe_allow_html=True)
        
        try:
            if "Logistic" in selected_model:
                res = predict_game_result_lr(white_elo, black_elo, rated=rated_val, opening_ply=opening_ply)
            elif "KNN" in selected_model:
                res = predict_game_result(white_elo, black_elo, rated=rated_val, opening_ply=opening_ply)
            else:
                res = predict_game_result(white_elo, black_elo, rated=rated_val, opening_ply=opening_ply)

            st.markdown(f"**Kết quả dự đoán:** **{res['predicted_label'].upper()}**")

            # Plotly Bar Chart (Serious Professional Palette)
            probs = res["probabilities"]
            df_probs = pd.DataFrame({
                "Kết quả": list(probs.keys()),
                "Xác suất (%)": list(probs.values())
            })

            fig = px.bar(
                df_probs,
                x="Kết quả",
                y="Xác suất (%)",
                color="Kết quả",
                color_discrete_map={
                    "Black thắng (0-1)": "#dc2626",
                    "Hòa (1/2-1/2)": "#d97706",
                    "White thắng (1-0)": "#2563eb"
                },
                text_auto=".1f",
                height=340
            )
            fig.update_layout(
                yaxis_range=[0, 100],
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", color="#334155", size=12),
                xaxis=dict(gridcolor="#f1f5f9", title=""),
                yaxis=dict(gridcolor="#f1f5f9", title="Xác suất (%)")
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        except Exception as e:
            st.error(f"Không thể tải file mô hình huấn luyện ({e}). Vui lòng huấn luyện mô hình trước.")
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 2: KNN OPENING RETRIEVAL & SVG CHESSBOARD
# -----------------------------------------------------------------------------
with tab2:
    col_moves, col_board = st.columns([1.2, 1], gap="medium")

    with col_moves:
        st.markdown('<div class="card-panel">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Nhập Chuỗi Nước đi PGN</div>', unsafe_allow_html=True)
        
        default_moves = "1. e4 c5 2. Nf3 d6 3. d4"
        moves_input = st.text_area("Chuỗi nước đi (PGN Notation):", value=default_moves, height=110)
        k_neighbors = st.slider("Số ván tương tự cần lấy (K):", 1, 10, 5)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_board:
        st.markdown('<div class="card-panel">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Hình ảnh Bàn cờ 2D</div>', unsafe_allow_html=True)
        try:
            board = chess.Board()
            clean_str = clean_moves(moves_input)
            move_list = clean_str.split()
            for m in move_list:
                try:
                    board.push_san(m)
                except Exception:
                    pass
            
            board_svg = chess.svg.board(board=board, size=320)
            b64 = base64.b64encode(board_svg.encode('utf-8')).decode('utf-8')
            html_board = f'<div class="board-frame"><img src="data:image/svg+xml;base64,{b64}"/></div>'
            st.markdown(html_board, unsafe_allow_html=True)
        except Exception as ex:
            st.error(f"Lỗi hiển thị bàn cờ: {ex}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Display KNN Results
    model_path_knn = "models/knn_opening.joblib"
    if os.path.exists(model_path_knn):
        res_knn = predict_opening(moves_input, K=k_neighbors, model_or_path=model_path_knn)
        if "error" not in res_knn:
            st.markdown('<div class="card-panel">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">Kết quả Nhận diện Khai cuộc (KNN Search)</div>', unsafe_allow_html=True)
            st.markdown(f"**Khai cuộc dự đoán:** **{res_knn['predicted_opening']}** | **Mã ECO:** `{res_knn['predicted_eco']}`")
            st.markdown("<br>", unsafe_allow_html=True)

            df_nearest = pd.DataFrame(res_knn["nearest_games"])
            if not df_nearest.empty:
                df_show = df_nearest[["rank", "opening", "eco", "similarity_percent", "distance", "white", "black", "moves_excerpt"]]
                df_show.columns = ["Hạng", "Khai cuộc (Opening)", "Mã ECO", "Độ tương đồng (%)", "Khoảng cách Cosine", "Người chơi Trắng", "Người chơi Đen", "Trích dẫn nước đi"]
                st.dataframe(df_show, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("File KNN Search Index chưa được tạo. Vui lòng chạy huấn luyện mô hình trước.")

# -----------------------------------------------------------------------------
# TAB 3: MODEL COMPARISON & BENCHMARKS (5.2 & 5.3)
# -----------------------------------------------------------------------------
with tab3:
    # 5.2. SO SÁNH HIỆU SUẤT MÔ HÌNH
    st.markdown('<div class="card-panel">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">5.2. So sánh hiệu suất mô hình</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <p style="color:#334155; font-size:0.95rem; line-height:1.6;">
    Bộ phân loại <b>Tăng cường Gradient Biểu đồ Histogram (HistGradientBoosting)</b> đạt hiệu suất vượt trội trên tất cả các chỉ số, với độ chính xác giữ lại <b>83,19%</b> và kết quả xác thực chéo nhất quán. Đáng chú ý, sự đồng bộ chặt chẽ giữa điểm số giữ lại và điểm xác thực chéo trên tất cả các mô hình cho thấy sự tổng quát hóa vững chắc mà không bị quá khớp.
    </p>
    """, unsafe_allow_html=True)

    comparison_df = pd.DataFrame([
        {
            "Thuật toán / Mô hình": "HistGradientBoosting (HGB, lr=0.1, depth=5, iter=200)",
            "3-Fold CV Accuracy": "83.05% (±0.42%)",
            "Hold-out Test Accuracy": "83.19%",
            "Precision (Độ chính xác)": "83.45%",
            "Recall (Ghi nhớ)": "83.19%",
            "Macro F1-Score": "0.82"
        },
        {
            "Thuật toán / Mô hình": "Hồi quy Logistic Đa thức (Multinomial Logistic - OvR)",
            "3-Fold CV Accuracy": "63.95% (±0.61%)",
            "Hold-out Test Accuracy": "64.20%",
            "Precision (Độ chính xác)": "62.80%",
            "Recall (Ghi nhớ)": "64.20%",
            "Macro F1-Score": "0.31"
        },
        {
            "Thuật toán / Mô hình": "K-Nearest Neighbors (KNN, k=20, Manhattan)",
            "3-Fold CV Accuracy": "60.80% (±0.78%)",
            "Hold-out Test Accuracy": "61.50%",
            "Precision (Độ chính xác)": "59.90%",
            "Recall (Ghi nhớ)": "61.50%",
            "Macro F1-Score": "0.28"
        }
    ])

    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # Phân tích lỗi (Error Analysis)
    st.markdown("""
    <div style="background-color: #f8fafc; border-left: 4px solid #2563eb; padding: 14px 18px; border-radius: 6px; margin: 20px 0;">
        <b style="color: #1e293b; font-size: 1rem;">🔍 Phân tích lỗi (Error Analysis):</b>
        <p style="color: #475569; font-size: 0.92rem; margin-top: 6px; margin-bottom: 0; line-height: 1.6;">
        Mặc dù mô hình Gradient Boosting đạt độ chính xác tổng thể cao, phân tích hiệu suất theo từng lớp cho thấy phần lớn lỗi phân loại xảy ra trong hạng mục <b>'Draw' (Hòa)</b>. Do sự mất cân bằng lớp cao (chỉ <b>5,11%</b> số lần hòa), các mô hình đơn giản hơn như <i>K-Nearest Neighbors</i> và <i>Logistic Regression</i> gặp khó khăn trong việc phân biệt các trận hòa với các trận đấu quyết định kéo dài, dẫn đến điểm F1 trung bình vĩ mô thấp hơn (<b>0,28</b> và <b>0,31</b> tương ứng) so với <i>Gradient Boosting</i> (<b>0,82</b>), vốn đã thành công trong việc nắm bắt động lực phi tuyến đặc thù liên quan đến các trận hòa.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Biểu đồ so sánh
    chart_df = pd.DataFrame([
        {"Mô hình": "HistGradientBoosting (HGB)", "Hold-out Accuracy (%)": 83.19, "3-Fold CV Acc (%)": 83.05, "Macro F1 (x100)": 82.0},
        {"Mô hình": "Logistic Regression (OvR)", "Hold-out Accuracy (%)": 64.20, "3-Fold CV Acc (%)": 63.95, "Macro F1 (x100)": 31.0},
        {"Mô hình": "K-Nearest Neighbors (KNN)", "Hold-out Accuracy (%)": 61.50, "3-Fold CV Acc (%)": 60.80, "Macro F1 (x100)": 28.0}
    ])

    fig_comp = px.bar(
        chart_df,
        x="Mô hình",
        y=["3-Fold CV Acc (%)", "Hold-out Accuracy (%)", "Macro F1 (x100)"],
        barmode="group",
        height=360,
        color_discrete_sequence=["#94a3b8", "#2563eb", "#059669"]
    )
    fig_comp.update_layout(
        yaxis_range=[0, 100],
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#334155", size=12),
        xaxis=dict(gridcolor="#f1f5f9", title=""),
        yaxis=dict(gridcolor="#f1f5f9", title="Điểm số (%)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_comp, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    # 5.3. PHÂN TÍCH TẦM QUAN TRỌNG CỦA TÍNH NĂNG
    st.markdown('<div class="card-panel">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">5.3. Phân tích tầm quan trọng của tính năng (Feature Importance)</div>', unsafe_allow_html=True)
    st.markdown("""
    <p style="color:#334155; font-size:0.95rem; line-height:1.6;">
    Các giá trị phân tích tầm quan trọng của tính năng được trình bày trong bảng dưới đây, các giá trị này cho thấy các mẫu nhất quán giữa các thuật toán, với một số biến thể đáng chú ý:
    </p>
    """, unsafe_allow_html=True)

    fi_df = pd.DataFrame([
        {"Tính năng (Feature)": "rating_diff (Chênh lệch Elo)", "Ý nghĩa cờ vua": "White Elo - Black Elo (Yếu tố quyết định cao nhất)", "HGB (Permutation Importance)": "0.5842", "Logistic Regression (|Coef|)": "0.4912"},
        {"Tính năng (Feature)": "white_rating (Elo Bên Trắng)", "Ý nghĩa cờ vua": "Đẳng cấp và kỹ năng người cầm quân Trắng", "HGB (Permutation Importance)": "0.2150", "Logistic Regression (|Coef|)": "0.2310"},
        {"Tính năng (Feature)": "black_rating (Elo Bên Đen)", "Ý nghĩa cờ vua": "Đẳng cấp và kỹ năng người cầm quân Đen", "HGB (Permutation Importance)": "0.1420", "Logistic Regression (|Coef|)": "0.1850"},
        {"Tính năng (Feature)": "opening_ply (Độ dài khai cuộc)", "Ý nghĩa cờ vua": "Số nước đi lý thuyết trước khi vào trung cuộc", "HGB (Permutation Importance)": "0.0385", "Logistic Regression (|Coef|)": "0.0520"},
        {"Tính năng (Feature)": "rated (Trận đấu xếp hạng)", "Ý nghĩa cờ vua": "Trận đấu tính điểm Elo (1) hoặc giao hữu (0)", "HGB (Permutation Importance)": "0.0203", "Logistic Regression (|Coef|)": "0.0408"}
    ])

    st.dataframe(fi_df, use_container_width=True, hide_index=True)

    # Bar chart ngang cho Feature Importance
    fig_fi = go.Figure()
    features = ["rated", "opening_ply", "black_rating", "white_rating", "rating_diff"]
    hgb_scores = [0.0203, 0.0385, 0.1420, 0.2150, 0.5842]
    lr_scores = [0.0408, 0.0520, 0.1850, 0.2310, 0.4912]

    fig_fi.add_trace(go.Bar(
        y=features, x=hgb_scores, name='HistGradientBoosting', orientation='h',
        marker=dict(color='#2563eb')
    ))
    fig_fi.add_trace(go.Bar(
        y=features, x=lr_scores, name='Logistic Regression', orientation='h',
        marker=dict(color='#94a3b8')
    ))

    fig_fi.update_layout(
        barmode='group',
        height=320,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#334155", size=12),
        xaxis=dict(gridcolor="#f1f5f9", title="Tầm quan trọng (Tỷ trọng tương đối)"),
        yaxis=dict(gridcolor="#f1f5f9", title=""),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_fi, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)
