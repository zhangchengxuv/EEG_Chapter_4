# -*- coding: utf-8 -*-
"""基于真实实验结果 JSON/CSV 统一绘制第 4 章论文插图（白底、低饱和蓝青灰、300 dpi PNG + SVG）。"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

REMOTE_SYNC = Path(r"E:\Master_Dissertation_Program\远程SSVEP工程同步\SSVEP")
RES_DIR = Path(r"E:\Master_Dissertation_Program\4.2_SSVEP算法补充与跨天实验\02_实验结果汇总")
FIG_DIR = Path(r"E:\Master_Dissertation_Program\4.2_SSVEP算法补充与跨天实验\03_论文插图重绘")

BLUE = "#4D7E98"
DEEP_BLUE = "#2E5266"
TEAL = "#63918B"
GRAY = "#8C9397"
LIGHT_BLUE = "#86A8BC"
WARM_ACCENT = "#B67B5B"

plt.rcParams.update(
    {
        "font.family": ["Times New Roman", "SimSun"],
        "font.size": 11.5,
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.75,
        "lines.linewidth": 1.35,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "svg.fonttype": "none",
    }
)


def save_fig(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def draw_cm(ax: plt.Axes, cm: np.ndarray, title: str, vmax: int | None = None, show_ylabel: bool = True) -> matplotlib.image.AxesImage:
    if vmax is None:
        vmax = int(cm.max())
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
    for r in range(4):
        for c in range(4):
            val = int(cm[r, c])
            ax.text(
                c,
                r,
                str(val),
                ha="center",
                va="center",
                fontsize=10.5,
                color="white" if val > vmax * 0.55 else "#1E293B",
            )
    labels = ["A\n(6.5 Hz)", "B\n(9.0 Hz)", "C\n(11.5 Hz)", "D\n(14.5 Hz)"]
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_yticklabels(["A", "B", "C", "D"] if show_ylabel else ["", "", "", ""], fontsize=10)
    ax.set_xlabel("预测类别", fontsize=10.5)
    if show_ylabel:
        ax.set_ylabel("真实类别", fontsize=10.5)
    ax.set_title(title, fontsize=11, pad=8)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    return im


def plot_fig_4_30() -> None:
    """图 4-30 g.Nautilus 测试集（95 试次）方法性能对比与对数能量 EEGNet 混淆矩阵。"""
    gn_dir = REMOTE_SYNC / "02_gNautilus/current/results/eegnet_merged_search"
    test_sum = {r["model"]: r for r in read_csv_rows(gn_dir / "test_summary.csv")}
    sel_metrics = read_json(gn_dir / "selected_model/metrics.json")
    cca_rows = read_csv_rows(RES_DIR / "CCA基线与谐波对比汇总.csv")

    def find_cca(ds: str, method: str) -> dict:
        return next(r for r in cca_rows if r["dataset"] == ds and r["method"] == method)

    items = [
        ("原版 EEGNet\n(0~4 s)", float(test_sum["merged_original"]["accuracy"]), float(test_sum["merged_original"]["macro_f1"])),
        ("CCA (Nh=1)\n(0.5~3.5 s)", float(find_cca("gNautilus_test95_0p5to3p5s", "Standard_CCA_Nh1")["accuracy"]), float(find_cca("gNautilus_test95_0p5to3p5s", "Standard_CCA_Nh1")["macro_f1"])),
        ("CCA (Nh=2)\n(0.5~3.5 s)", float(find_cca("gNautilus_test95_0p5to3p5s", "Standard_CCA_Nh2")["accuracy"]), float(find_cca("gNautilus_test95_0p5to3p5s", "Standard_CCA_Nh2")["macro_f1"])),
        ("FBCCA (Nh=2)\n(0.5~3.5 s)", float(find_cca("gNautilus_test95_0p5to3p5s", "FBCCA_3bands_Nh2")["accuracy"]), float(find_cca("gNautilus_test95_0p5to3p5s", "FBCCA_3bands_Nh2")["macro_f1"])),
        ("CCA (Nh=2)\n(0~4 s)", float(find_cca("gNautilus_test95_0to4s", "Standard_CCA_Nh2")["accuracy"]), float(find_cca("gNautilus_test95_0to4s", "Standard_CCA_Nh2")["macro_f1"])),
        ("对数能量 EEGNet\n(0~4 s)", float(test_sum["selected_model"]["accuracy"]), float(test_sum["selected_model"]["macro_f1"])),
    ]

    fig = plt.figure(figsize=(11.8, 4.6), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    x = np.arange(len(items))
    w = 0.36
    accs = [it[1] for it in items]
    f1s = [it[2] for it in items]
    b1 = ax0.bar(x - w / 2, accs, width=w, color=BLUE, label="分类准确率")
    b2 = ax0.bar(x + w / 2, f1s, width=w, color=TEAL, label="宏平均 F1")
    for bar in b1:
        v = bar.get_height()
        ax0.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v*100:.1f}%", ha="center", va="bottom", fontsize=8.5)
    for bar in b2:
        v = bar.get_height()
        ax0.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=8.2, color="#35524F")
    ax0.axhline(0.25, color=GRAY, ls="--", lw=1.0, label="四分类随机水平 (25%)")
    ax0.set_xticks(x)
    ax0.set_xticklabels([it[0] for it in items], fontsize=9.2)
    ax0.set_ylim(0, 1.12)
    ax0.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax0.set_ylabel("测试性能指标", fontsize=10.5)
    ax0.set_title("(a) g.Nautilus 固定测试集（95 试次）不同方法性能对比", fontsize=11)
    ax0.grid(axis="y", alpha=0.22)
    ax0.legend(loc="upper left", frameon=False, fontsize=9.2)

    cm = np.array(sel_metrics["confusion_matrix_true_rows_predicted_columns"])
    im = draw_cm(ax1, cm, f"(b) 对数能量 EEGNet 混淆矩阵\n(准确率 {sel_metrics['accuracy']:.2%}, 86/95)", vmax=24)
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="试次数")

    save_fig(fig, "Fig_4_30_gNautilus_Paradigm_Validation")


def plot_fig_4_31() -> None:
    """图 4-31 ADS1299 首日模型训练过程与同日留出测试对比。"""
    hist_path = REMOTE_SYNC / "01_ADS1299/current/results/run_110epochs/ads1299/candidates/eegnet_long_seed42/history.csv"
    history = read_csv_rows(hist_path)
    cca_rows = read_csv_rows(RES_DIR / "CCA基线与谐波对比汇总.csv")
    ads_metrics = read_json(REMOTE_SYNC / "01_ADS1299/current/results/run_110epochs/ads1299/metrics.json")

    epochs = [int(r["epoch"]) for r in history]
    tr_loss = [float(r["train_loss"]) for r in history]
    va_loss = [float(r["validation_loss"]) for r in history]
    va_acc = [float(r["validation_accuracy"]) for r in history]

    fig = plt.figure(figsize=(11.8, 4.4), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    ax0.plot(epochs, tr_loss, color=GRAY, ls="--", label="训练损失 (Dropout 启用)")
    ax0.plot(epochs, va_loss, color=BLUE, label="验证损失 (评估模式)")
    ax0.axvline(109, color=WARM_ACCENT, ls=":", lw=1.2, label="选中轮次 (第 109 轮)")
    ax0.set_xlabel("训练轮次", fontsize=10.5)
    ax0.set_ylabel("交叉熵损失", fontsize=10.5)
    ax0.set_title("(a) ADS1299 EEGNet-16,2 训练与验证损失过程", fontsize=11)
    ax0.grid(alpha=0.22)
    ax0_r = ax0.twinx()
    ax0_r.spines["top"].set_visible(False)
    ax0_r.plot(epochs, va_acc, color=TEAL, lw=1.4, label="验证准确率")
    ax0_r.set_ylabel("验证准确率", fontsize=10.5, color=TEAL)
    ax0_r.set_ylim(0, 1.08)
    ax0_r.yaxis.set_major_formatter(PercentFormatter(1.0))
    lines1, labels1 = ax0.get_legend_handles_labels()
    lines2, labels2 = ax0_r.get_legend_handles_labels()
    ax0.legend(lines1 + lines2, labels1 + labels2, loc="lower left", frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=8.5)

    def get_row(ds: str, m: str) -> dict:
        return next(r for r in cca_rows if r["dataset"] == ds and r["method"] == m)

    comp = [
        ("标准 CCA\n(1~80 Hz)", float(get_row("ADS_day1_test128_1to80Hz", "Standard_CCA_Nh3")["accuracy"]), float(get_row("ADS_day1_test128_1to80Hz", "Standard_CCA_Nh3")["macro_f1"])),
        ("标准 CCA\n(4~45 Hz)", float(get_row("ADS_day1_test128_4to45Hz", "Standard_CCA_Nh3")["accuracy"]), float(get_row("ADS_day1_test128_4to45Hz", "Standard_CCA_Nh3")["macro_f1"])),
        ("FBCCA\n(4~45 Hz)", float(get_row("ADS_day1_test128_4to45Hz", "FBCCA_3bands_Nh3")["accuracy"]), float(get_row("ADS_day1_test128_4to45Hz", "FBCCA_3bands_Nh3")["macro_f1"])),
        ("EEGNet-16,2\n(run_110epochs)", float(ads_metrics["test"]["accuracy"]), float(ads_metrics["test"]["macro_f1"])),
    ]
    x = np.arange(len(comp))
    w = 0.35
    b1 = ax1.bar(x - w / 2, [c[1] for c in comp], width=w, color=BLUE, label="准确率")
    b2 = ax1.bar(x + w / 2, [c[2] for c in comp], width=w, color=TEAL, label="宏平均 F1")
    for bar in b1:
        v = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v*100:.2f}%", ha="center", va="bottom", fontsize=8.5)
    for bar in b2:
        v = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=8.2, color="#35524F")
    ax1.set_xticks(x)
    ax1.set_xticklabels([c[0] for c in comp], fontsize=9.2)
    ax1.set_ylim(0, 1.14)
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.set_ylabel("首日测试集指标 (128 试次)", fontsize=10.5)
    ax1.set_title("(b) ADS1299 首日留出测试集性能对比", fontsize=11)
    ax1.grid(axis="y", alpha=0.22)
    ax1.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none", fontsize=9)

    save_fig(fig, "Fig_4_31_ADS1299_Day1_Training_and_Test")


def plot_fig_4_32() -> None:
    """图 4-32 ADS1299 跨天独立会话（11月17日 128 试次）各方法表现与逐类召回率对比。"""
    cross_dir = REMOTE_SYNC / "01_ADS1299/crossday_improvement_20261117/results"
    comp_rows = read_csv_rows(cross_dir / "comparison.csv")
    cca_rows = read_csv_rows(RES_DIR / "CCA基线与谐波对比汇总.csv")
    cca_detail = read_json(RES_DIR / "CCA基线详细指标.json")

    name_map = {
        "standard8_aug": "EEGNet-8 (幅度增强)",
        "energy16": "Energy-EEGNet-16",
        "energy8": "Energy-EEGNet-8",
        "hybrid16_spectral": "Hybrid-16 (α=0.1)",
        "hybrid8_spectral": "Hybrid-8 (α=0.1)",
        "baseline": "EEGNet-16,2 (基线)",
        "hybrid8": "Hybrid-8 (α=0.5, 选定)",
    }

    entries = []
    for r in comp_rows:
        entries.append((name_map[r["name"]], float(r["target_accuracy"]), float(r["target_macro_f1"]), "dl"))
    for ds, m, label in [
        ("ADS_day2_all128_1to80Hz", "Standard_CCA_Nh3", "标准 CCA (1~80 Hz)"),
        ("ADS_day2_all128_4to45Hz", "Standard_CCA_Nh2", "标准 CCA (4~45 Hz)"),
        ("ADS_day2_all128_4to45Hz", "FBCCA_3bands_Nh3", "FBCCA (4~45 Hz)"),
    ]:
        row = next(r for r in cca_rows if r["dataset"] == ds and r["method"] == m)
        entries.append((label, float(row["accuracy"]), float(row["macro_f1"]), "cca"))

    entries.sort(key=lambda t: t[1])

    fig = plt.figure(figsize=(13.0, 5.1), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.22, 1.05])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    y_pos = np.arange(len(entries))
    colors = [
        TEAL if "选定" in e[0] else (DEEP_BLUE if "FBCCA" in e[0] else (LIGHT_BLUE if e[3] == "cca" else GRAY))
        for e in entries
    ]
    bars = ax0.barh(y_pos, [e[1] for e in entries], height=0.62, color=colors)
    for bar, e in zip(bars, entries):
        v = bar.get_width()
        ax0.text(v + 0.012, bar.get_y() + bar.get_height() / 2, f"{v*100:.2f}% (F1={e[2]:.3f})", va="center", fontsize=8.8)
    ax0.axvline(0.25, color=GRAY, ls="--", lw=0.9)
    ax0.set_yticks(y_pos)
    ax0.set_yticklabels([e[0] for e in entries], fontsize=9.3)
    ax0.set_xlim(0, 1.15)
    ax0.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax0.set_xlabel("跨天会话准确率 (11月17日, 128 试次)", fontsize=10.5)
    ax0.set_title("(a) 跨天独立会话识别准确率与宏 F1 对比", fontsize=11)
    ax0.grid(axis="x", alpha=0.22)

    base_m = read_json(cross_dir / "baseline/target_metrics.json")
    hyb_m = read_json(cross_dir / "hybrid8/target_metrics.json")
    cca_m = cca_detail["ADS_day2_all128_4to45Hz__CCA_Nh2"]
    fb_m = cca_detail["ADS_day2_all128_4to45Hz__FBCCA_Nh3"]

    cls_labels = ["目标 A\n(6.5 Hz)", "目标 B\n(9.0 Hz)", "目标 C\n(11.5 Hz)", "目标 D\n(14.5 Hz)"]
    xc = np.arange(4)
    bw = 0.205
    series = [
        ("EEGNet-16,2 基线", [p["recall"] for p in base_m["per_class"]], GRAY),
        ("Hybrid-8 (跨天优化)", [p["recall"] for p in hyb_m["per_class"]], TEAL),
        ("标准 CCA (4~45 Hz)", [p["recall"] for p in cca_m["per_class"]], LIGHT_BLUE),
        ("FBCCA (4~45 Hz)", [p["recall"] for p in fb_m["per_class"]], DEEP_BLUE),
    ]
    for idx, (lbl, vals, col) in enumerate(series):
        offset = (idx - 1.5) * bw
        bars_c = ax1.bar(xc + offset, vals, width=bw, color=col, label=lbl)
        for b in bars_c:
            h = b.get_height()
            ax1.text(b.get_x() + b.get_width() / 2, h + 0.015, f"{h*100:.0f}%", ha="center", va="bottom", fontsize=7.0, rotation=45)
    ax1.set_xticks(xc)
    ax1.set_xticklabels(cls_labels, fontsize=9.5)
    ax1.set_ylim(0, 1.32)
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.set_ylabel("逐类召回率", fontsize=10.5)
    ax1.set_title("(b) 跨天独立会话四类目标召回率对比", fontsize=11)
    ax1.grid(axis="y", alpha=0.22)
    ax1.legend(loc="upper center", ncol=2, frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=8.4)

    save_fig(fig, "Fig_4_32_ADS1299_CrossDay_Comparison")


def plot_fig_4_33() -> None:
    """图 4-33 不同时间窗长（1.0–3.0 s）下 CCA 与 FBCCA 准确率及 ITR 对比。"""
    cca_w = read_csv_rows(RES_DIR / "CCA时间窗长度与ITR对比.csv")
    fb_w = read_csv_rows(RES_DIR / "FBCCA时间窗长度与ITR对比.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9), layout="constrained")
    curves = [
        ("g.Nautilus 测试集 (95)", cca_w, "gNautilus_test95", GRAY, "--", "o", "CCA"),
        ("g.Nautilus 测试集 (95)", fb_w, "gNautilus_test95", GRAY, "-", "s", "FBCCA"),
        ("ADS1299 首日测试 (128)", cca_w, "ADS_day1_test128_4to45Hz", BLUE, "--", "o", "CCA"),
        ("ADS1299 首日测试 (128)", fb_w, "ADS_day1_test128_4to45Hz", BLUE, "-", "s", "FBCCA"),
        ("ADS1299 次日跨天 (128)", cca_w, "ADS_day2_128_4to45Hz", TEAL, "--", "o", "CCA"),
        ("ADS1299 次日跨天 (128)", fb_w, "ADS_day2_all128_4to45Hz", TEAL, "-", "s", "FBCCA"),
    ]

    for ds_label, table, ds_key, col, ls, marker, m_name in curves:
        rows = sorted([r for r in table if r["dataset"] == ds_key], key=lambda r: float(r["window_s"]))
        ws = [float(r["window_s"]) for r in rows]
        accs = [float(r["accuracy"]) for r in rows]
        itrs = [float(r["itr_window_bpm"]) for r in rows]
        lbl = f"{ds_label} - {m_name}"
        axes[0].plot(ws, accs, color=col, ls=ls, marker=marker, ms=5, label=lbl)
        axes[1].plot(ws, itrs, color=col, ls=ls, marker=marker, ms=5, label=lbl)

    axes[0].set_xticks([1.0, 1.5, 2.0, 2.5, 3.0])
    axes[0].set_ylim(0.45, 1.03)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].set_xlabel("分析时间窗长 (s)", fontsize=10.5)
    axes[0].set_ylabel("分类准确率", fontsize=10.5)
    axes[0].set_title("(a) 时间窗长对识别准确率的影响", fontsize=11)
    axes[0].grid(alpha=0.22)

    axes[1].set_xticks([1.0, 1.5, 2.0, 2.5, 3.0])
    axes[1].set_ylim(10, 62)
    axes[1].set_xlabel("分析时间窗长 (s)", fontsize=10.5)
    axes[1].set_ylabel("窗内信息传输率 ITR (bits/min)", fontsize=10.5)
    axes[1].set_title("(b) 时间窗长对信息传输率的影响", fontsize=11)
    axes[1].grid(alpha=0.22)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=8.8)

    save_fig(fig, "Fig_4_33_Window_Length_Accuracy_ITR")


def plot_fig_4_34() -> None:
    """图 4-34 跨天会话混淆矩阵对比。"""
    cross_dir = REMOTE_SYNC / "01_ADS1299/crossday_improvement_20261117/results"
    base_m = read_json(cross_dir / "baseline/target_metrics.json")
    hyb_m = read_json(cross_dir / "hybrid8/target_metrics.json")
    cca_detail = read_json(RES_DIR / "CCA基线详细指标.json")
    fb_m = cca_detail["ADS_day2_all128_4to45Hz__FBCCA_Nh3"]

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0), layout="constrained")
    im0 = draw_cm(axes[0], np.array(base_m["confusion_matrix_true_rows"]), f"(a) EEGNet-16,2 基线\n(88/128, 68.75%)", vmax=32, show_ylabel=True)
    im1 = draw_cm(axes[1], np.array(hyb_m["confusion_matrix_true_rows"]), f"(b) Hybrid-8 优化模型\n(93/128, 72.66%)", vmax=32, show_ylabel=False)
    im2 = draw_cm(axes[2], np.array(fb_m["confusion_matrix_true_rows"]), f"(c) FBCCA (4~45 Hz)\n(119/128, 92.97%)", vmax=32, show_ylabel=False)
    fig.colorbar(im2, ax=axes, fraction=0.022, pad=0.02, label="试次数")
    save_fig(fig, "Fig_4_34_CrossDay_Confusion_Comparison")


def main() -> None:
    plot_fig_4_30()
    plot_fig_4_31()
    plot_fig_4_32()
    plot_fig_4_33()
    plot_fig_4_34()
    print("Generated figures:", [p.name for p in sorted(FIG_DIR.glob("*.png"))])


if __name__ == "__main__":
    main()
