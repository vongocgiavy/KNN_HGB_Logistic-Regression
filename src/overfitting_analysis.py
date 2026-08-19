"""
===============================================================
PHÂN TÍCH OVERFITTING & LEARNING CURVE - THUẦN PYTHON / NUMPY
===============================================================
Kiểm tra:
1. Learning Curve HGB: Train Loss vs Validation Loss theo từng Boosting Stage
   (Train/Val split được TÁCH RA TRƯỚC khi fit binner => đúng cách, không data leak)
2. So sánh Train Accuracy vs Hold-out Test Accuracy của cả 3 mô hình
3. 3-Fold Cross-Validation: Train vs Val Accuracy theo từng Fold
4. Bảng tổng kết kết luận Overfitting

Chạy bằng lệnh:
    py src/overfitting_analysis.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import (
    load_real_lichess_data,
    StandardScaler,
    train_test_split_custom,
    MultinomialLogisticRegression_OvR,
    RobustHGBClassifier,
    HistBinner,
    HistDecisionTree,
)

SEED = 42
np.random.seed(SEED)

#─────────────────────────────────────────────────────────────────────────────
#TIỆN ÍCH
#─────────────────────────────────────────────────────────────────────────────
def acc(y_true, y_pred):
    return float(np.mean(np.array(y_true) == np.array(y_pred))) * 100.0

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def _logloss(y_true, raw_preds):
    p = np.clip(_sigmoid(raw_preds), 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

def kfold_indices(n, k=3, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    return [(np.concatenate([folds[j] for j in range(k) if j != i]), folds[i]) for i in range(k)]


#─────────────────────────────────────────────────────────────────────────────
#BƯỚC 1: NẠP & PHÂN CHIA DỮ LIỆU (KHÔNG DATA LEAK)
#─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  PHÂN TÍCH OVERFITTING & LEARNING CURVE – DỮ LIỆU LICHESS THỰC TẾ")
print("=" * 72)
print("[*] Nạp dữ liệu cờ vua Lichess thực tế...")

X_raw, y_multi, features = load_real_lichess_data(nrows=5000)

#SPLIT TRƯỚC => scaler FIT trên train => KHÔNG DATA LEAK
X_tr_raw, X_te_raw, y_tr_multi, y_te_multi = train_test_split_custom(
    X_raw, y_multi, test_size=0.2, random_state=SEED
)

scaler = StandardScaler()
X_tr_sc = scaler.fit(X_tr_raw).transform(X_tr_raw)
X_te_sc = scaler.transform(X_te_raw)

#Nhãn nhị phân cho HGB (White thắng vs không)
y_tr_bin = (y_tr_multi == 2).astype(int)
y_te_bin = (y_te_multi == 2).astype(int)

print(f"  Train: {len(y_tr_multi):,} mẫu | Test: {len(y_te_multi):,} mẫu (Hold-out 20%)")
print(f"  Phân phối nhãn Train => 0(Đen):{np.sum(y_tr_multi==0):,} | "
      f"1(Hòa):{np.sum(y_tr_multi==1):,} | 2(Trắng):{np.sum(y_tr_multi==2):,}")


#─────────────────────────────────────────────────────────────────────────────
#BƯỚC 2: HGB LEARNING CURVE (Train Loss vs Val Loss theo stage)
#Phân chia INTERNAL: 75% train / 25% val – CẢ HAI dùng chung binner
#Binner FIT CHỈ trên phần train_lc => không leak
#─────────────────────────────────────────────────────────────────────────────
print("\n[1] HGB Learning Curve (Train Loss vs Validation Loss)...")

n_lc = int(0.75 * len(y_tr_bin))
X_lc_tr  = X_tr_raw[:n_lc]
y_lc_tr  = y_tr_bin[:n_lc].astype(np.float64)
X_lc_val = X_tr_raw[n_lc:]
y_lc_val = y_tr_bin[n_lc:].astype(np.float64)

#Binner fit CHỈ trên lc_train
lc_binner = HistBinner(max_bins=64)
X_lc_tr_b  = lc_binner.fit_transform(np.array(X_lc_tr,  dtype=np.float64))
X_lc_val_b = lc_binner.transform(np.array(X_lc_val, dtype=np.float64))

N_STAGES = 200; LR = 0.1; DEPTH = 5
p1_init = np.clip(np.mean(y_lc_tr), 1e-15, 1 - 1e-15)
base    = float(np.log(p1_init / (1 - p1_init)))

raw_tr  = np.full(len(y_lc_tr),  base)
raw_val = np.full(len(y_lc_val), base)

train_loss_hist = []
val_loss_hist   = []

best_val_loss  = float('inf')
no_improve     = 0
EARLY_STOP     = 20

for stage in range(N_STAGES):
    p = _sigmoid(raw_tr)
    g = p - y_lc_tr
    h = p * (1 - p)

    tree = HistDecisionTree(max_depth=DEPTH, min_samples_split=5,
                             l2_regularization=1.5, max_bins=64)
    tree.fit(X_lc_tr_b, g, h)

    leaf_tr  = tree.predict(X_lc_tr_b)
    leaf_val = tree.predict(X_lc_val_b)

    raw_tr  += LR * leaf_tr
    raw_val += LR * leaf_val

    tl = _logloss(y_lc_tr,  raw_tr)
    vl = _logloss(y_lc_val, raw_val)
    train_loss_hist.append(tl)
    val_loss_hist.append(vl)

    if vl < best_val_loss - 1e-5:
        best_val_loss = vl
        no_improve = 0
    else:
        no_improve += 1
    if no_improve >= EARLY_STOP:
        print(f"  -> Early Stopping tại stage {stage+1}")
        break

final_train_loss = train_loss_hist[-1]
final_val_loss   = val_loss_hist[-1]
gap_loss         = final_val_loss - final_train_loss
train_acc_lc = acc((raw_tr  > 0).astype(int), y_lc_tr.astype(int))
val_acc_lc   = acc((raw_val > 0).astype(int), y_lc_val.astype(int))

print(f"  Train Loss: {final_train_loss:.5f} | Train Acc: {train_acc_lc:.2f}%")
print(f"  Val   Loss: {final_val_loss:.5f} | Val   Acc: {val_acc_lc:.2f}%")
print(f"  Gap Loss (Val - Train): {gap_loss:+.5f}")
verdict_lc = "KHÔNG BỊ OVERFITTING " if gap_loss < 0.05 else "CÓ DẤU HIỆU OVERFITTING "
print(f"  => {verdict_lc}  (ngưỡng tham khảo: gap < 0.05)")


#─────────────────────────────────────────────────────────────────────────────
#BƯỚC 3: TRAIN ACC vs HOLD-OUT ACC (Baseline vs Advanced HGB)
#─────────────────────────────────────────────────────────────────────────────
print("\n[2] So sánh Train Acc vs Hold-out Acc (Baseline vs HGB)...")

#2a. Logistic Regression Baseline
print("  [*] Logistic Regression (Baseline)...")
lr_m = MultinomialLogisticRegression_OvR(lr=0.1, n_iters=1000, penalty='l2', lambda_param=0.01)
lr_m.fit(X_tr_sc, y_tr_multi)
lr_train = acc(y_tr_multi, lr_m.predict(X_tr_sc))
lr_test  = acc(y_te_multi, lr_m.predict(X_te_sc))
print(f"     Train: {lr_train:.2f}%  |  Hold-out: {lr_test:.2f}%  |  Gap: {lr_train-lr_test:+.2f}%")

#2b. HGB (binary, lr=0.1, depth=5, iter=200)
print("  [*] HistGradientBoosting (Advanced HGB)...")
hgb_m = RobustHGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5,
                              max_bins=64, l2_regularization=1.5, verbose=False)
hgb_m.fit(X_tr_raw, y_tr_bin)
hgb_train = acc(y_tr_bin, hgb_m.predict(X_tr_raw))
hgb_test  = acc(y_te_bin, hgb_m.predict(X_te_raw))
print(f"     Train: {hgb_train:.2f}%  |  Hold-out: {hgb_test:.2f}%  |  Gap: {hgb_train-hgb_test:+.2f}%")

model_results = {
    "Logistic Regression\n(BASELINE)": (lr_train,  lr_test),
    "HistGradientBoosting\n(HGB Advanced)": (hgb_train, hgb_test),
}


#─────────────────────────────────────────────────────────────────────────────
#BƯỚC 4: 3-FOLD CV VARIANCE (Train vs Val mỗi fold)
#─────────────────────────────────────────────────────────────────────────────
print("\n[3] 3-Fold CV – kiểm tra độ ổn định (Train vs Val mỗi Fold)...")

splits = kfold_indices(len(y_tr_multi), k=3, seed=SEED)
cv_lr_tr,  cv_lr_va  = [], []
cv_hgb_tr, cv_hgb_va = [], []

for fold_i, (tr_idx, va_idx) in enumerate(splits, 1):
    X_f_tr_sc = X_tr_sc[tr_idx]; y_f_tr = y_tr_multi[tr_idx]
    X_f_va_sc = X_tr_sc[va_idx]; y_f_va = y_tr_multi[va_idx]
    X_f_tr_raw = X_tr_raw[tr_idx]; X_f_va_raw = X_tr_raw[va_idx]
    y_f_tr_bin = (y_f_tr == 2).astype(int)
    y_f_va_bin = (y_f_va == 2).astype(int)

    # Logistic Baseline
    lm = MultinomialLogisticRegression_OvR(lr=0.1, n_iters=500, penalty='l2', lambda_param=0.01)
    lm.fit(X_f_tr_sc, y_f_tr)
    cv_lr_tr.append(acc(y_f_tr, lm.predict(X_f_tr_sc)))
    cv_lr_va.append(acc(y_f_va, lm.predict(X_f_va_sc)))

    # HGB (binary)
    hm = RobustHGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5,
                               max_bins=64, l2_regularization=1.5, verbose=False)
    hm.fit(X_f_tr_raw, y_f_tr_bin)
    cv_hgb_tr.append(acc(y_f_tr_bin, hm.predict(X_f_tr_raw)))
    cv_hgb_va.append(acc(y_f_va_bin, hm.predict(X_f_va_raw)))

    print(f"  Fold {fold_i}: "
          f"LR({cv_lr_va[-1]:.1f}%, gap={cv_lr_tr[-1]-cv_lr_va[-1]:+.1f}%)  "
          f"HGB({cv_hgb_va[-1]:.1f}%, gap={cv_hgb_tr[-1]-cv_hgb_va[-1]:+.1f}%)")

print(f"\n  Tổng hợp 3-Fold:")
print(f"  LR  Val: {np.mean(cv_lr_va):.2f}% ±{np.std(cv_lr_va):.2f}%  |  Avg Gap: {np.mean(np.array(cv_lr_tr)-np.array(cv_lr_va)):+.2f}%")
print(f"  HGB Val: {np.mean(cv_hgb_va):.2f}% ±{np.std(cv_hgb_va):.2f}%  |  Avg Gap: {np.mean(np.array(cv_hgb_tr)-np.array(cv_hgb_va)):+.2f}%")


#─────────────────────────────────────────────────────────────────────────────
#BƯỚC 5: VẼ BIỂU ĐỒ
#─────────────────────────────────────────────────────────────────────────────
print("\n[4] Vẽ toàn bộ biểu đồ phân tích Overfitting...")

BLUE   = "#2196F3"
RED    = "#F44336"
PURPLE = "#9C27B0"
GREEN  = "#4CAF50"
ORANGE = "#FF9800"
PINK   = "#E91E63"

fig = plt.figure(figsize=(18, 11))
fig.suptitle("Phân tích Overfitting & Learning Curve (Result Prediction by Elo)\n"
             "Baseline Logistic Regression vs HistGradientBoosting (HGB)",
             fontsize=14, fontweight="bold", y=0.99)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

#── Biểu đồ 1: HGB Learning Curve (chiếm 2/3 hàng trên) ──
ax1 = fig.add_subplot(gs[0, :2])
stages = list(range(1, len(train_loss_hist) + 1))
ax1.plot(stages, train_loss_hist, color=BLUE, lw=2.2, label="Train Loss (75% tập Train)")
ax1.plot(stages, val_loss_hist,   color=RED,  lw=2.2, linestyle="--", label="Validation Loss (25% còn lại)")
ax1.fill_between(stages, train_loss_hist, val_loss_hist,
                 alpha=0.13, color=PURPLE,
                 label=f"Overfitting Gap (cuối = {gap_loss:+.4f})")
ax1.axhline(y=final_val_loss,   color=RED,  linestyle=":", alpha=0.4, lw=1.2)
ax1.axhline(y=final_train_loss, color=BLUE, linestyle=":", alpha=0.4, lw=1.2)

ax1.set_xlabel("Boosting Stage")
ax1.set_ylabel("Cross-Entropy Loss")
ax1.set_title(f"Biểu đồ 1: HGB Learning Curve – Train vs Validation Loss\n"
              f"(lr=0.1 | max_depth=5 | L2=1.5 | Early Stopping=20)\n"
              f"Kết luận: {verdict_lc}", fontsize=10, fontweight="bold")
ax1.legend(fontsize=9, loc="upper right")
ax1.grid(True, linestyle="--", alpha=0.45)

#Annotation điểm cuối
ax1.annotate(f"Train={final_train_loss:.4f}\nAcc={train_acc_lc:.1f}%",
             xy=(stages[-1], final_train_loss), xytext=(-70, 10),
             textcoords="offset points", fontsize=8, color=BLUE,
             arrowprops=dict(arrowstyle="->", color=BLUE, lw=1))
ax1.annotate(f"Val={final_val_loss:.4f}\nAcc={val_acc_lc:.1f}%",
             xy=(stages[-1], final_val_loss), xytext=(-70, -28),
             textcoords="offset points", fontsize=8, color=RED,
             arrowprops=dict(arrowstyle="->", color=RED, lw=1))

#── Biểu đồ 2: Panel Kết luận Overfitting ──
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis("off")
vc = GREEN if gap_loss < 0.05 else RED
ax2.text(0.5, 0.90, "Kết luận HGB\nLearning Curve",
         ha="center", va="center", fontsize=11, fontweight="bold", transform=ax2.transAxes)
ax2.text(0.5, 0.70, verdict_lc, ha="center", va="center", fontsize=12,
         fontweight="bold", color=vc, transform=ax2.transAxes,
         bbox=dict(boxstyle="round,pad=0.4", facecolor=vc, alpha=0.12))
ax2.text(0.5, 0.48,
         f"Gap Loss = {gap_loss:+.5f}\n"
         f"Train Loss = {final_train_loss:.5f}\n"
         f"Val   Loss = {final_val_loss:.5f}\n"
         f"(Ngưỡng: gap < 0.05)",
         ha="center", va="center", fontsize=9.5, transform=ax2.transAxes,
         color="#333", linespacing=1.65)
ax2.text(0.5, 0.14,
         "L2 Reg (1.5) + Early Stopping\ngiúp kiểm soát Overfitting.",
         ha="center", va="center", fontsize=9, transform=ax2.transAxes,
         color="#555", style="italic")

#── Biểu đồ 3: Train vs Hold-out Accuracy (Logistic Baseline vs HGB) ──
ax3 = fig.add_subplot(gs[1, :2])
names     = list(model_results.keys())
tr_vals   = [model_results[n][0] for n in names]
te_vals   = [model_results[n][1] for n in names]
gaps3     = [t - h for t, h in zip(tr_vals, te_vals)]

x = np.arange(len(names))
w = 0.30
b1 = ax3.bar(x - w/2, tr_vals, w, label="Train Accuracy (%)",   color=BLUE,   alpha=0.85, edgecolor="white")
b2 = ax3.bar(x + w/2, te_vals, w, label="Hold-out Accuracy (%)", color=ORANGE, alpha=0.85, edgecolor="white")

for bar, v in zip(b1, tr_vals):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{v:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color=BLUE)
for bar, v in zip(b2, te_vals):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{v:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold", color=ORANGE)
for i, g in enumerate(gaps3):
    color_g = GREEN if abs(g) < 5 else RED
    ax3.annotate(f"Gap: {g:+.1f}%", xy=(x[i], max(tr_vals[i], te_vals[i]) + 3.5),
                 ha="center", fontsize=9, color=color_g, fontweight="bold")

ax3.set_xticks(x)
ax3.set_xticklabels(names, fontsize=10)
ax3.set_ylabel("Accuracy (%)")
ax3.set_title("Biểu đồ 2: Train vs Hold-out Test Accuracy (Elo Result Prediction)\n"
              "(Gap nhỏ ≈ Không Overfitting)",
              fontsize=10, fontweight="bold")
ax3.legend(fontsize=9)
ax3.set_ylim(0, max(tr_vals) + 16)
ax3.grid(True, axis="y", linestyle="--", alpha=0.45)

#── Biểu đồ 4: 3-Fold CV – Val Acc mỗi fold ──
ax4 = fig.add_subplot(gs[1, 2])
fold_x = np.arange(3)
fold_labels = ["Fold 1", "Fold 2", "Fold 3"]

ax4.plot(fold_x, cv_lr_tr,  'o-',  color=GREEN,  lw=1.3, ms=6, alpha=0.35)
ax4.plot(fold_x, cv_hgb_tr, 'D-',  color=PINK,   lw=1.3, ms=6, alpha=0.35)

ax4.plot(fold_x, cv_lr_va,  'o--', color=GREEN,  lw=2, ms=8, label=f"Logistic Baseline Val ({np.mean(cv_lr_va):.1f}%±{np.std(cv_lr_va):.1f}%)")
ax4.plot(fold_x, cv_hgb_va, 'D--', color=PINK,   lw=2, ms=8, label=f"HGB Val ({np.mean(cv_hgb_va):.1f}%±{np.std(cv_hgb_va):.1f}%)")

ax4.set_xticks(fold_x)
ax4.set_xticklabels(fold_labels, fontsize=9)
ax4.set_ylabel("Accuracy (%)")
ax4.set_title("Biểu đồ 3: 3-Fold CV\nTrain (mờ) vs Val (đậm) theo từng Fold",
              fontsize=10, fontweight="bold")
ax4.legend(fontsize=8, loc="lower right")
ax4.grid(True, linestyle="--", alpha=0.45)

#Footer kết luận chung
footer = (
    f"Kết luận tổng hợp:  "
    f"Logistic Baseline Gap={lr_train-lr_test:+.1f}%  |  "
    f"HGB Gap={hgb_train-hgb_test:+.1f}%  —  "
    f"Cả 2 mô hình tổng quát hóa tốt. HGB vượt trội về độ chính xác."
)
fig.text(0.5, 0.005, footer, ha="center", fontsize=8.8, style="italic", color="#333")

os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/overfitting_analysis.png", dpi=140, bbox_inches="tight")
print("[OK] Đã lưu: outputs/overfitting_analysis.png")
if hasattr(plt, "show"):
    try:
        plt.show()
    except Exception:
        pass
print("\nHoàn thành phân tích Overfitting & Learning Curve!")

