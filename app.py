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
from logistic_baseline import train_logistic_regression
from knn_result import predict_result_knn
from knn_opening import predict_opening, train_knn_opening
from hgb_elo import predict_game_result
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
            ["HistGradientBoosting (HGB)", "Logistic Regression (Baseline)"]
        )

        elo_diff = white_elo - black_elo
        st.markdown(f"**Chênh lệch Elo (rating_diff = White - Black):** `{elo_diff:+d}` điểm")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        st.markdown('<div class="card-panel">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Kết quả Dự đoán Xác suất</div>', unsafe_allow_html=True)
        
        try:
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
# TAB 3: MODEL COMPARISON & BENCHMARKS
# -----------------------------------------------------------------------------
with tab3:
    metrics_lr_path = "outputs/logistic_metrics.json"
    metrics_hgb_path = "outputs/hgb_metrics.json"

    if os.path.exists(metrics_lr_path) and os.path.exists(metrics_hgb_path):
        with open(metrics_lr_path, "r", encoding="utf-8") as f: lr_m = json.load(f)
        with open(metrics_hgb_path, "r", encoding="utf-8") as f: hgb_m = json.load(f)

        st.markdown('<div class="card-panel">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Chỉ số Hiệu năng Kiểm thử & Cross-Validation</div>', unsafe_allow_html=True)
        
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Logistic Regression Baseline</div>
                <div class="metric-value">{lr_m['f1_score']:.4f}</div>
                <div style="font-size:0.8rem; color:#64748b; margin-top:4px;">5-Fold CV: {lr_m.get('cv_f1_mean', 0):.4f} ± {lr_m.get('cv_f1_std', 0):.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">HistGradientBoosting (HGB)</div>
                <div class="metric-value">{hgb_m['f1_score']:.4f}</div>
                <div style="font-size:0.8rem; color:#64748b; margin-top:4px;">5-Fold CV: {hgb_m.get('cv_f1_mean', 0):.4f} ± {hgb_m.get('cv_f1_std', 0):.4f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Comparison Chart
        comp_df = pd.DataFrame([
            {"Model": "Logistic Baseline", "Accuracy": lr_m['accuracy'], "F1-Score": lr_m['f1_score']},
            {"Model": "HistGradientBoosting", "Accuracy": hgb_m['accuracy'], "F1-Score": hgb_m['f1_score']}
        ])

        fig_comp = px.bar(
            comp_df, x="Model", y=["Accuracy", "F1-Score"], barmode="group",
            title="",
            height=360,
            color_discrete_sequence=["#2563eb", "#64748b"]
        )
        fig_comp.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#334155", size=12),
            xaxis=dict(gridcolor="#f1f5f9", title="Mô hình"),
            yaxis=dict(gridcolor="#f1f5f9", title="Điểm số (0.0 - 1.0)", range=[0, 1])
        )
        st.plotly_chart(fig_comp, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

        # Feature Importance Table
        st.markdown('<div class="card-panel">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Độ quan trọng của Đặc trưng (Feature Importance)</div>', unsafe_allow_html=True)
        feats = lr_m.get("features_used", [])
        fi_rows = []
        for ft in feats:
            fi_rows.append({
                "Đặc trưng (Feature)": ft,
                "Logistic Regression (Coef Mag)": f"{lr_m.get('feature_importance', {}).get(ft, 0.0):.4f}",
                "HistGradientBoosting (Permutation)": f"{hgb_m.get('feature_importance', {}).get(ft, 0.0):.4f}"
            })
        st.dataframe(pd.DataFrame(fi_rows), use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Báo cáo so sánh mô hình chưa có sẵn. Vui lòng thực hiện huấn luyện các mô hình trước.")
