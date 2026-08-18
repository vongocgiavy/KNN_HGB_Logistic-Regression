import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Ensure UTF-8 output encoding for Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from data_loader import prepare_and_cache_dataset


def analyze_result_distribution(csv_path="data/processed_games.csv"):
    if not os.path.exists(csv_path):
        df = prepare_and_cache_dataset(max_games=100000)
    else:
        df = pd.read_csv(csv_path, dtype=str)

    print("\n" + "=" * 65)
    print("      THỐNG KÊ TỈ LỆ THẮNG - THUA - HÒA TOÀN BỘ TỆP DỮ LIỆU")
    print("=" * 65)

    # Clean results
    valid_results = ["1-0", "0-1", "1/2-1/2"]
    df_clean = df[df["Result"].isin(valid_results)].copy()
    total_games = len(df_clean)

    # Result mapping
    result_labels = {
        "1-0": "Trắng thắng (White Win)",
        "0-1": "Đen thắng (Black Win)",
        "1/2-1/2": "Hòa (Draw)"
    }
    df_clean["Result_Label"] = df_clean["Result"].map(result_labels)

    # Counts & Percentages via Pandas
    counts = df_clean["Result_Label"].value_counts()
    percentages = (counts / total_games) * 100

    stats_df = pd.DataFrame({
        "Mã Result": ["1-0", "0-1", "1/2-1/2"],
        "Kết quả": ["Trắng thắng (White Win)", "Đen thắng (Black Win)", "Hòa (Draw)"],
        "Số lượng (Ván)": [counts.get("Trắng thắng (White Win)", 0), counts.get("Đen thắng (Black Win)", 0), counts.get("Hòa (Draw)", 0)],
        "Tỉ lệ (%)": [percentages.get("Trắng thắng (White Win)", 0.0), percentages.get("Đen thắng (Black Win)", 0.0), percentages.get("Hòa (Draw)", 0.0)]
    })

    # Format percentage display
    stats_df["Tỉ lệ (%)"] = stats_df["Tỉ lệ (%)"].map("{:.2f}%".format)
    stats_df["Số lượng (Ván)"] = stats_df["Số lượng (Ván)"].map("{:,}".format)

    print(f"\nTổng số ván cờ hợp lệ phân tích: {total_games:,} ván\n")
    print(stats_df.to_string(index=False))
    print("=" * 65)

    # Create Visualization Image using Pandas + Matplotlib
    os.makedirs("outputs", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Thống kê Phân bố Kết quả Ván cờ Lichess Dataset", fontsize=14, fontweight="bold", y=0.98)

    colors = ["#2563eb", "#dc2626", "#f59e0b"]
    labels = ["Trắng thắng (1-0)", "Đen thắng (0-1)", "Hòa (1/2-1/2)"]
    raw_counts = [counts.get("Trắng thắng (White Win)", 0), counts.get("Đen thắng (Black Win)", 0), counts.get("Hòa (Draw)", 0)]

    # Bar chart
    bars = ax1.bar(labels, raw_counts, color=colors, width=0.55, edgecolor="#1e293b", linewidth=1)
    ax1.set_title("Số lượng ván cờ theo kết quả", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Số lượng ván cờ", fontsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f"{height:,}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Pie chart
    wedges, texts, autotexts = ax2.pie(
        raw_counts,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2)
    )
    ax2.set_title("Tỉ lệ phần trăm kết quả (%)", fontsize=11, fontweight="bold")
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_weight("bold")

    plt.tight_layout()
    output_img = "outputs/result_distribution.png"
    plt.savefig(output_img, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n[+] Đã xuất file hình ảnh trực quan hóa: '{output_img}'\n")


if __name__ == "__main__":
    analyze_result_distribution()
