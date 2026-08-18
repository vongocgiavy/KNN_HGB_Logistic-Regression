import os
import sys
import argparse
import pandas as pd

# Ensure UTF-8 output encoding for Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from data_loader import prepare_and_cache_dataset, check_dataset_stats
from preprocessing import preprocess_data, clean_moves
from logistic_baseline import train_logistic_regression
from knn_result import train_knn_classifier, predict_result_knn
from knn_opening import train_knn_opening, predict_opening
from hgb_elo import train_hgb_classifier, predict_game_result
from comparison import compare_models


def load_or_process_all_data(max_games=100000):
    """
    Ensures dataset is loaded, cleaned, and features engineered.
    """
    df = prepare_and_cache_dataset(max_games=max_games)
    df_clean, X, y, df_knn = preprocess_data(df)
    return df_clean, X, y, df_knn


def prompt_and_predict_result(model_name="HGB", predict_fn=predict_game_result, model_path="models/hgb_elo.joblib"):
    """
    Interactively prompts for White and Black rating and displays predicted Result probabilities.
    """
    try:
        w_str = input("\nNhập Elo người chơi Trắng (white_rating) [Mặc định 1800]: ").strip()
        w_rating = int(w_str) if w_str.isdigit() else 1800

        b_str = input("Nhập Elo người chơi Đen (black_rating) [Mặc định 1500]: ").strip()
        b_rating = int(b_str) if b_str.isdigit() else 1500
    except Exception as e:
        print(f"[!] Giá trị không hợp lệ. Dùng mặc định White=1800, Black=1500 ({e})")
        w_rating, b_rating = 1800, 1500

    res = predict_fn(w_rating, b_rating, model_path=model_path)
    print("\n" + "=" * 60)
    print(f"        KẾT QUẢ DỰ ĐOÁN VÁN CỜ BẰNG {model_name.upper()}")
    print("=" * 60)
    print(f"white_rating : {res['white_rating']}")
    print(f"black_rating : {res['black_rating']}")
    print(f"rating_diff  : {res['rating_diff']} ( = White - Black)")
    print(f"\n=> ĐỰ ĐOÁN KẾT QUẢ: {res['predicted_label'].upper()}")
    print("\nXác suất dự đoán từng kết quả:")
    for label, prob in res["probabilities"].items():
        bar = "█" * int(prob / 5)
        print(f"  {label:<20}: {prob:>6.2f}% {bar}")
    print("=" * 60 + "\n")


def run_option_1(X=None, y=None, interactive=True):
    """
    1. Logistic Regression - BASELINE (5 Features -> Result)
    """
    print("\n>>> [CHỨC NĂNG 1] LOGISTIC REGRESSION (BASELINE DỰ ĐOÁN RESULT DỰA TRÊN 5 FEATURES)")
    model_path = "models/logistic_baseline.joblib"
    if not os.path.exists(model_path) or X is not None:
        if X is None or y is None:
            _, X, y, _ = load_or_process_all_data()
        pipeline, metrics, _ = train_logistic_regression(X, y)
    else:
        metrics = {}

    if interactive:
        prompt_and_predict_result("Logistic Regression Baseline", predict_game_result, model_path=model_path)
    return None, metrics


def run_option_2(X=None, y=None, interactive=True):
    """
    2. HistGradientBoosting (HGB) - DỰ ĐOÁN RESULT (5 Features -> Result)
    """
    print("\n>>> [CHỨC NĂNG 2] HIST GRADIENT BOOSTING (HGB DỰ ĐOÁN RESULT DỰA TRÊN 5 FEATURES)")
    model_path = "models/hgb_elo.joblib"
    if not os.path.exists(model_path) or X is not None:
        if X is None or y is None:
            _, X, y, _ = load_or_process_all_data()
        hgb_model, metrics, _, _ = train_hgb_classifier(X, y)
    else:
        metrics = {}

    if interactive:
        prompt_and_predict_result("HistGradientBoosting", predict_game_result, model_path=model_path)
    return None, metrics


def run_option_3(df_knn=None, k_value=5):
    """
    3. KNN - TÌM OPENING TƯƠNG TỰ (Moves -> Opening)
    """
    print("\n>>> [CHỨC NĂNG 3] KNN TÌM OPENING TƯƠNG TỰ DỰA TRÊN NƯỚC ĐỊ (MOVES)")

    model_path = "models/knn_opening.joblib"
    if not os.path.exists(model_path):
        print("[!] KNN model index not found. Building index now...")
        if df_knn is None:
            _, _, _, df_knn = load_or_process_all_data()
        train_knn_opening(df_knn)

    moves_input = input("\nNhập chuỗi nước đi (Ví dụ: '1. e4 c5 2. Nf3 d6 3. d4' hoặc 'e4 c5 Nf3 d6 d4'): ").strip()
    if not moves_input:
        moves_input = "e4 c5 Nf3 d6 d4"
        print(f"[i] Không nhập nước đi. Sử dụng nước đi mẫu: '{moves_input}'")

    try:
        k_in = input(f"Nhập số K ván gần nhất muốn lấy [Mặc định = {k_value}]: ").strip()
        if k_in.isdigit():
            k_value = int(k_in)
    except Exception:
        pass

    result = predict_opening(moves_input, K=k_value, model_or_path=model_path)

    if "error" in result:
        print(f"[!] Lỗi: {result['error']}")
        return

    print("\n" + "=" * 65)
    print("                KẾT QUẢ TÌM OPENING TƯƠNG TỰ (KNN)")
    print("=" * 65)
    print(f"Chuỗi nước đi gốc    : {result['input_moves_raw']}")
    print(f"Chuỗi nước đi đã xử lý: {result['input_moves_cleaned']}")
    print(f"Opening dự đoán       : {result['predicted_opening']}")
    print(f"Mã ECO dự đoán        : {result['predicted_eco']}")
    print(f"Số ván gần nhất (K)   : {result['K']}")
    print("-" * 65)
    print(f"{'Hạng':<5} | {'Kc Distance':<11} | {'Độ tương đồng':<13} | {'ECO':<5} | {'Opening'}")
    print("-" * 65)

    for g in result["nearest_games"]:
        print(f"{g['rank']:<5} | {g['distance']:<11.4f} | {g['similarity_percent']:<12.1f}% | {g['eco']:<5} | {g['opening']}")

    print("-" * 65 + "\n")


def run_option_4(max_games=100000):
    """
    4. Chạy toàn bộ pipeline: Parse dataset -> Preprocess -> Train LR -> Train HGB -> Train KNN Search -> Compare
    """
    print("\n" + "#" * 70)
    print("        CHẠY TOÀN BỘ PROJECT MACHINE LEARNING LICHESS CHESS")
    print("#" * 70)

    # Step 1 & 2: Load & Preprocess
    df_clean, X, y, df_knn = load_or_process_all_data(max_games=max_games)

    # Step 3: Logistic Regression Baseline
    _, lr_metrics = run_option_1(X, y, interactive=False)

    # Step 4: HistGradientBoosting Classifier
    _, hgb_metrics = run_option_2(X, y, interactive=False)

    # Step 5: KNN Opening Finder
    artifacts_knn = train_knn_opening(df_knn, K_list=[3, 5, 7, 9])
    demo_moves = "1. e4 c5 2. Nf3 d6 3. d4"
    knn_res = predict_opening(demo_moves, K=5, model_or_path=artifacts_knn)
    print(f"[+] Predicted Opening for '{demo_moves}': {knn_res['predicted_opening']} (ECO: {knn_res['predicted_eco']})")

    # Step 6: Compare Baseline vs HGB
    comp_df, report = compare_models(lr_metrics, hgb_metrics)

    print("\n[+] HOÀN THÀNH CHẠY TOÀN BỘ DỰ ÁN PHÂN TÍCH VÀ SO SÁNH MÔ HÌNH!")
    print("#" * 70 + "\n")


def main_menu():
    """
    Interactive Console Interface
    """
    while True:
        print("\n" + "=" * 65)
        print("     HỆ THỐNG MACHINE LEARNING PHÂN TÍCH VÁN CỜ LICHESS")
        print("=" * 65)
        print("1. Logistic Regression (BASELINE dự đoán Result từ 5 features)")
        print("2. HistGradientBoosting (HGB dự đoán Result từ 5 features)")
        print("3. KNN tìm Opening tương tự (Dựa trên chuỗi nước đi Moves)")
        print("4. Chạy toàn bộ (Full Pipeline & So sánh mô hình)")
        print("0. Thoát")
        print("=" * 65)

        choice = input("Vui lòng chọn chức năng (0-4): ").strip()

        if choice == "1":
            run_option_1()
        elif choice == "2":
            run_option_2()
        elif choice == "3":
            run_option_3()
        elif choice == "4":
            run_option_4()
        elif choice == "0":
            print("\nCảm ơn bạn đã sử dụng hệ thống Machine Learning! Tạm biệt.\n")
            break
        else:
            print("[!] Lựa chọn không hợp lệ. Vui lòng nhập từ 0 đến 4.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lichess Chess ML System")
    parser.add_argument("--mode", type=int, choices=[1, 2, 3, 4], help="Run mode directly (1-4)")
    parser.add_argument("--moves", type=str, help="Input moves string for KNN mode 3")
    parser.add_argument("--white-elo", type=int, default=1800, help="White Elo")
    parser.add_argument("--black-elo", type=int, default=1500, help="Black Elo")
    parser.add_argument("--max-games", type=int, default=100000, help="Max games to parse")

    args = parser.parse_args()

    if args.mode == 1:
        run_option_1()
    elif args.mode == 2:
        run_option_2()
    elif args.mode == 3:
        model_path = "models/knn_opening.joblib"
        if not os.path.exists(model_path):
            _, _, _, df_knn = load_or_process_all_data(max_games=args.max_games)
            train_knn_opening(df_knn)
        moves = args.moves if args.moves else "e4 c5 Nf3 d6 d4"
        res = predict_opening(moves, K=5, model_or_path=model_path)
        print(f"Predicted Opening for '{moves}': {res['predicted_opening']} (ECO: {res['predicted_eco']})")
    elif args.mode == 4:
        run_option_4(max_games=args.max_games)
    else:
        main_menu()
