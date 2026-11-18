# -*- coding: utf-8 -*-
"""在相同数据集与固定划分上运行无训练经典 CCA 及 FBCCA 基线对照实验。"""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

FREQUENCIES = [6.5, 9.0, 11.5, 14.5]
CLASSES = ["A", "B", "C", "D"]

GNAUTILUS_DIR = Path(
    r"D:\OneDrive\03_USTB\10_Master_Dissertation\08_SSVEP\实验数据分析及结果\g.Nautilus设备数据分析及结果\03_ML_DL_Dataset"
)
ADS_DAY1_NPZ = Path(
    r"E:\Master_Dissertation_Program\采集的脑电数据\首次采集四次数据\03_机器学习数据集\ADS1299_20261116_combined4Recordings_allTrials_500Hz_0p5to3p5s_V.npz"
)
ADS_DAY2_NPZ = Path(
    r"E:\Master_Dissertation_Program\采集的脑电数据\第二次采集数据\03_机器学习数据集\ADS1299_20261117_160423_allTrials_500Hz_0p5to3p5s_V.npz"
)
OUT_DIR = Path(r"E:\Master_Dissertation_Program\4.2_SSVEP算法补充与跨天实验\02_实验结果汇总")


def make_reference(freq: float, n_samples: int, sfreq: float, n_harmonics: int) -> np.ndarray:
    """构造多谐波正余弦参考矩阵，形状为 [2*n_harmonics, n_samples]。"""
    t = np.arange(n_samples, dtype=np.float64) / sfreq
    cols = []
    for h in range(1, n_harmonics + 1):
        w = 2.0 * np.pi * h * freq * t
        cols.append(np.sin(w))
        cols.append(np.cos(w))
    return np.stack(cols, axis=0)


def max_canonical_corr(X: np.ndarray, Y: np.ndarray) -> float:
    """基于 QR 分解与 SVD 精确求解最大典型相关系数。X:[C,T], Y:[2H,T]。"""
    Xc = X - X.mean(axis=1, keepdims=True)
    Yc = Y - Y.mean(axis=1, keepdims=True)
    qx, rx = np.linalg.qr(Xc.T, mode="reduced")
    qy, ry = np.linalg.qr(Yc.T, mode="reduced")
    # 剔除数值秩亏列（若有）
    if qx.shape[1] == 0 or qy.shape[1] == 0:
        return 0.0
    s = np.linalg.svd(qx.T @ qy, compute_uv=False)
    return float(np.clip(s[0], 0.0, 1.0))


def cca_predict(x_trials: np.ndarray, sfreq: float, n_harmonics: int) -> tuple[np.ndarray, np.ndarray]:
    """返回预测标签与每个试次的 4 类最大典型相关系数矩阵 [N, 4]。"""
    n_trials, _, n_samples = x_trials.shape
    refs = [make_reference(f, n_samples, sfreq, n_harmonics) for f in FREQUENCIES]
    scores = np.zeros((n_trials, len(FREQUENCIES)), dtype=np.float64)
    for i in range(n_trials):
        xi = x_trials[i]
        for k, ref in enumerate(refs):
            scores[i, k] = max_canonical_corr(xi, ref)
    return scores.argmax(axis=1), scores


def fbcca_predict(
    x_trials: np.ndarray,
    sfreq: float,
    n_harmonics: int,
    subbands: list[tuple[float, float]],
    a: float = 1.25,
    b: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """滤波器组 CCA（FBCCA）：多子带加权平方相关。"""
    n_trials = x_trials.shape[0]
    weighted_scores = np.zeros((n_trials, len(FREQUENCIES)), dtype=np.float64)
    for m, (low, high) in enumerate(subbands, start=1):
        sos = butter(4, [low, high], btype="bandpass", fs=sfreq, output="sos")
        x_sub = sosfiltfilt(sos, x_trials.astype(np.float64), axis=-1)
        _, rho = cca_predict(x_sub, sfreq, n_harmonics)
        w = m ** (-a) + b
        weighted_scores += w * (rho ** 2)
    return weighted_scores.argmax(axis=1), weighted_scores


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """计算准确率、平衡准确率、宏 F1 与混淆矩阵（行=真实，列=预测）。"""
    cm = np.zeros((4, 4), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        cm[int(yt), int(yp)] += 1
    acc = float((y_true == y_pred).mean())
    per_class = []
    recalls = []
    f1s = []
    for c in range(4):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        recalls.append(rec)
        f1s.append(f1)
        per_class.append(
            {
                "label": c,
                "target": CLASSES[c],
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "support": int(cm[c, :].sum()),
            }
        )
    return {
        "accuracy": acc,
        "correct": int((y_true == y_pred).sum()),
        "count": int(len(y_true)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "confusion_matrix_true_rows": cm.tolist(),
        "per_class": per_class,
    }


def wolpaw_itr(acc: float, window_s: float, n_classes: int = 4) -> float:
    """Wolpaw 信息传输率（bits/min），仅按给定时间窗时长计算。"""
    if acc <= 1.0 / n_classes:
        return 0.0
    if acc >= 1.0:
        b = np.log2(n_classes)
    else:
        b = (
            np.log2(n_classes)
            + acc * np.log2(acc)
            + (1.0 - acc) * np.log2((1.0 - acc) / (n_classes - 1))
        )
    return float(b * (60.0 / window_s))


def load_gnautilus() -> tuple[np.ndarray, np.ndarray, dict]:
    """按 trial_id 对齐加载 g.Nautilus 三次采集 accepted 4 秒数据（250 Hz, μV）。"""
    xs, ys, splits = [], [], []
    for name in ("S01", "S02", "S03"):
        npz_path = GNAUTILUS_DIR / "NumPy" / f"{name}_accepted_0to4_uv.npz"
        csv_path = GNAUTILUS_DIR / "Metadata" / f"{name}_trials.csv"
        with np.load(npz_path, allow_pickle=False) as f:
            x = f["X"].copy()
            y = f["y"].copy()
            tids = f["trial_ids"].astype(str)
        with csv_path.open(encoding="utf-8-sig", newline="") as cf:
            rows = {r["trial_id"]: r for r in csv.DictReader(cf)}
        for tid in tids:
            splits.append(rows[tid]["split"])
        xs.append(x)
        ys.append(y)
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    splits_arr = np.array(splits)
    masks = {
        "train": splits_arr == "train",
        "validation": splits_arr == "validation",
        "test": splits_arr == "test",
        "all": np.ones(len(y), dtype=bool),
    }
    return X, y, masks


def load_ads_day1() -> tuple[np.ndarray, np.ndarray, dict]:
    """加载 11 月 16 日 ADS1299 544 试次（V -> μV, 500 -> 250 Hz），按完整会话划分。"""
    with np.load(ADS_DAY1_NPZ, allow_pickle=False) as f:
        x_v = f["X"].copy()
        y = f["y"].copy()
        sessions = f["session_ids"].astype(str)
    x_250 = resample_poly(x_v.astype(np.float64) * 1e6, up=1, down=2, axis=-1, padtype="line")
    uniq = sorted(np.unique(sessions).tolist())
    masks = {
        "train": np.isin(sessions, uniq[:2]),
        "validation": sessions == uniq[2],
        "test": sessions == uniq[3],
        "all": np.ones(len(y), dtype=bool),
    }
    return x_250, y, masks


def load_ads_day2() -> tuple[np.ndarray, np.ndarray]:
    """加载 11 月 17 日 ADS1299 128 试次跨天独立会话（V -> μV, 500 -> 250 Hz）。"""
    with np.load(ADS_DAY2_NPZ, allow_pickle=False) as f:
        x_v = f["X"].copy()
        y = f["y"].copy()
    x_250 = resample_poly(x_v.astype(np.float64) * 1e6, up=1, down=2, axis=-1, padtype="line")
    return x_250, y


def bandpass(x: np.ndarray, low: float, high: float, sfreq: float = 250.0) -> np.ndarray:
    sos = butter(4, [low, high], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, x.astype(np.float64), axis=-1)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gx, gy, gmasks = load_gnautilus()
    ax1, ay1, amasks1 = load_ads_day1()
    ax2, ay2 = load_ads_day2()

    # 预处理对照版本：ADS1299 原始 1-80 Hz（降采样后约 1-80 Hz）与 4-45 Hz 带通对比
    ax1_bp = bandpass(ax1, 4.0, 45.0, 250.0)
    ax2_bp = bandpass(ax2, 4.0, 45.0, 250.0)

    # g.Nautilus 裁剪 0.5-3.5s ([125:875]) 以便与 ADS1299 3.0s 窗口严格同窗长对比
    gx_3s = gx[:, :, 125:875]

    summary_rows = []
    detailed_metrics = {}

    datasets = [
        ("gNautilus_test95_0p5to3p5s", gx_3s[gmasks["test"]], gy[gmasks["test"]]),
        ("gNautilus_all339_0p5to3p5s", gx_3s, gy),
        ("gNautilus_test95_0to4s", gx[gmasks["test"]], gy[gmasks["test"]]),
        ("ADS_day1_val128_1to80Hz", ax1[amasks1["validation"]], ay1[amasks1["validation"]]),
        ("ADS_day1_test128_1to80Hz", ax1[amasks1["test"]], ay1[amasks1["test"]]),
        ("ADS_day1_all544_1to80Hz", ax1, ay1),
        ("ADS_day1_val128_4to45Hz", ax1_bp[amasks1["validation"]], ay1[amasks1["validation"]]),
        ("ADS_day1_test128_4to45Hz", ax1_bp[amasks1["test"]], ay1[amasks1["test"]]),
        ("ADS_day1_all544_4to45Hz", ax1_bp, ay1),
        ("ADS_day2_all128_1to80Hz", ax2, ay2),
        ("ADS_day2_all128_4to45Hz", ax2_bp, ay2),
    ]

    # 1. 标准 CCA 谐波数 Nh ∈ {1, 2, 3} 与全窗对比
    for ds_name, X, y in datasets:
        win_s = X.shape[-1] / 250.0
        for nh in (1, 2, 3):
            yp, _ = cca_predict(X, 250.0, nh)
            m = compute_metrics(y, yp)
            itr = wolpaw_itr(m["accuracy"], win_s)
            key = f"{ds_name}__CCA_Nh{nh}"
            detailed_metrics[key] = m
            summary_rows.append(
                {
                    "dataset": ds_name,
                    "method": f"Standard_CCA_Nh{nh}",
                    "window_s": win_s,
                    "harmonics": nh,
                    "correct": m["correct"],
                    "count": m["count"],
                    "accuracy": round(m["accuracy"], 6),
                    "balanced_accuracy": round(m["balanced_accuracy"], 6),
                    "macro_f1": round(m["macro_f1"], 6),
                    "itr_window_bpm": round(itr, 2),
                }
            )

    # 2. FBCCA 对比（针对 g.Nautilus 30 Hz 以内和 ADS1299 45 Hz 以内分别设置子带）
    fb_configs = [
        ("gNautilus_test95_0p5to3p5s", gx_3s[gmasks["test"]], gy[gmasks["test"]], [(5.0, 28.0), (11.0, 28.0), (17.0, 28.0)], 2),
        ("gNautilus_all339_0p5to3p5s", gx_3s, gy, [(5.0, 28.0), (11.0, 28.0), (17.0, 28.0)], 2),
        ("ADS_day1_test128_4to45Hz", ax1_bp[amasks1["test"]], ay1[amasks1["test"]], [(5.0, 45.0), (11.0, 45.0), (17.0, 45.0)], 3),
        ("ADS_day1_all544_4to45Hz", ax1_bp, ay1, [(5.0, 45.0), (11.0, 45.0), (17.0, 45.0)], 3),
        ("ADS_day2_all128_4to45Hz", ax2_bp, ay2, [(5.0, 45.0), (11.0, 45.0), (17.0, 45.0)], 3),
    ]
    for ds_name, X, y, subbands, nh in fb_configs:
        win_s = X.shape[-1] / 250.0
        yp, _ = fbcca_predict(X, 250.0, nh, subbands)
        m = compute_metrics(y, yp)
        itr = wolpaw_itr(m["accuracy"], win_s)
        key = f"{ds_name}__FBCCA_Nh{nh}"
        detailed_metrics[key] = m
        summary_rows.append(
            {
                "dataset": ds_name,
                "method": f"FBCCA_3bands_Nh{nh}",
                "window_s": win_s,
                "harmonics": nh,
                "correct": m["correct"],
                "count": m["count"],
                "accuracy": round(m["accuracy"], 6),
                "balanced_accuracy": round(m["balanced_accuracy"], 6),
                "macro_f1": round(m["macro_f1"], 6),
                "itr_window_bpm": round(itr, 2),
            }
        )

    # 3. 不同时间窗长度对比（1.0, 1.5, 2.0, 2.5, 3.0 s，从 0.5 s 起算）
    window_rows = []
    win_targets = [
        ("gNautilus_test95", gx_3s[gmasks["test"]], gy[gmasks["test"]], 2),
        ("ADS_day1_test128_1to80Hz", ax1[amasks1["test"]], ay1[amasks1["test"]], 3),
        ("ADS_day1_test128_4to45Hz", ax1_bp[amasks1["test"]], ay1[amasks1["test"]], 3),
        ("ADS_day2_128_1to80Hz", ax2, ay2, 3),
        ("ADS_day2_128_4to45Hz", ax2_bp, ay2, 3),
    ]
    for ds_name, X, y, nh in win_targets:
        for w_s in (1.0, 1.5, 2.0, 2.5, 3.0):
            n_pts = int(round(w_s * 250))
            yp, _ = cca_predict(X[:, :, :n_pts], 250.0, nh)
            m = compute_metrics(y, yp)
            itr = wolpaw_itr(m["accuracy"], w_s)
            window_rows.append(
                {
                    "dataset": ds_name,
                    "harmonics": nh,
                    "window_s": w_s,
                    "correct": m["correct"],
                    "count": m["count"],
                    "accuracy": round(m["accuracy"], 6),
                    "macro_f1": round(m["macro_f1"], 6),
                    "itr_window_bpm": round(itr, 2),
                }
            )

    with (OUT_DIR / "CCA基线与谐波对比汇总.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    with (OUT_DIR / "CCA时间窗长度与ITR对比.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(window_rows[0]))
        writer.writeheader()
        writer.writerows(window_rows)

    (OUT_DIR / "CCA基线详细指标.json").write_text(
        json.dumps(detailed_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for r in summary_rows:
        print(
            f"{r['dataset']:32s} | {r['method']:18s} | {r['correct']:3d}/{r['count']:3d} ({r['accuracy']:6.2%}) | F1={r['macro_f1']:.4f} | ITR={r['itr_window_bpm']:.2f}"
        )


if __name__ == "__main__":
    main()
