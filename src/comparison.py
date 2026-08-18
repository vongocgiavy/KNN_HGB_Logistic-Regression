import os
import sys
import json
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def compare_models(lr_metrics=None, hgb_metrics=None,
                   lr_json_path="outputs/logistic_metrics.json",
                   hgb_json_path="outputs/hgb_metrics.json",
                   output_path="outputs/model_comparison.txt"):
    """
    Compares Result Classification models:
    1. Logistic Regression (BASELINE)
    2. HistGradientBoosting (HGB)

    Note: KNN is dedicated exclusively to Opening Retrieval (Moves -> Opening).
    """
    if lr_metrics is None:
        if os.path.exists(lr_json_path):
            with open(lr_json_path, "r", encoding="utf-8") as f:
                lr_metrics = json.load(f)
        else:
            raise FileNotFoundError(f"Baseline metrics file not found at '{lr_json_path}'. Train Logistic Regression first.")

    if hgb_metrics is None:
        if os.path.exists(hgb_json_path):
            with open(hgb_json_path, "r", encoding="utf-8") as f:
                hgb_metrics = json.load(f)
        else:
            raise FileNotFoundError(f"HGB metrics file not found at '{hgb_json_path}'. Train HGB first.")

    comparison_data = [
        {
            "Model": "Logistic Regression (BASELINE)",
            "5-Fold CV F1": f"{lr_metrics.get('cv_f1_mean', 0.0):.4f} ± {lr_metrics.get('cv_f1_std', 0.0):.4f}",
            "Accuracy": f"{lr_metrics['accuracy']:.4f}",
            "Precision": f"{lr_metrics['precision']:.4f}",
            "Recall": f"{lr_metrics['recall']:.4f}",
            "F1-Score": f"{lr_metrics['f1_score']:.4f}"
        },
        {
            "Model": "HistGradientBoosting (HGB)",
            "5-Fold CV F1": f"{hgb_metrics.get('cv_f1_mean', 0.0):.4f} ± {hgb_metrics.get('cv_f1_std', 0.0):.4f}",
            "Accuracy": f"{hgb_metrics['accuracy']:.4f}",
            "Precision": f"{hgb_metrics['precision']:.4f}",
            "Recall": f"{hgb_metrics['recall']:.4f}",
            "F1-Score": f"{hgb_metrics['f1_score']:.4f}"
        }
    ]

    comp_df = pd.DataFrame(comparison_data)

    header = "=" * 80 + "\n"
    header += "       SO SÁNH MÔ HÌNH DỰ ĐOÁN KẾT QUẢ VÁN CỜ: LOGISTIC BASELINE VÀ HGB\n"
    header += "=" * 80 + "\n\n"

    table_str = comp_df.to_string(index=False)

    diff_acc = hgb_metrics['accuracy'] - lr_metrics['accuracy']
    diff_f1 = hgb_metrics['f1_score'] - lr_metrics['f1_score']

    analysis = "\n\n--- ĐÁNH GIÁ VÀ NHẬN XÉT CHI TIẾT ---\n"
    analysis += f"1. Baseline Logistic Regression: Accuracy = {lr_metrics['accuracy']:.4f}, F1-Score = {lr_metrics['f1_score']:.4f}\n"
    analysis += f"2. HistGradientBoosting (HGB): Accuracy = {hgb_metrics['accuracy']:.4f}, F1-Score = {hgb_metrics['f1_score']:.4f}\n"

    if diff_acc > 0:
        analysis += f"3. HistGradientBoosting vượt trội hơn Baseline Logistic Regression {abs(diff_acc)*100:.2f}% về Accuracy.\n"
    else:
        analysis += f"3. Logistic Regression duy trì hiệu năng tương đương hoặc vượt nhẹ ({abs(diff_acc)*100:.2f}% điểm).\n"

    # Feature Importance table
    analysis += "\n--- ĐỘ QUAN TRỌNG CỦA ĐẶC TRƯNG (FEATURE IMPORTANCE) ---\n"
    features = lr_metrics.get("features_used", [])
    fi_data = []
    for f in features:
        fi_data.append({
            "Feature": f,
            "Logistic (Coef Mag)": f"{lr_metrics.get('feature_importance', {}).get(f, 0.0):.4f}",
            "HGB (Permutation)": f"{hgb_metrics.get('feature_importance', {}).get(f, 0.0):.4f}"
        })
    fi_df = pd.DataFrame(fi_data)
    fi_str = fi_df.to_string(index=False)

    full_report = header + table_str + analysis + "\n" + fi_str + "\n\n" + "=" * 80 + "\n"

    print(full_report)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_report)
    print(f"[+] Comparison report saved to '{output_path}'.")

    return comp_df, full_report


if __name__ == "__main__":
    compare_models()
