"""统一中文绘图：图题、坐标轴、图例与色条均为中文，设备名和类别字母除外。"""

import numpy as np

from ads1299_data import ROOT, CLASSES


def setup_plotting():
    """优先加载目录内附带的中文字体，避免依赖其他机器的系统字体。"""
    import matplotlib
    matplotlib.use("Agg")  # 无显示器时也能保存图像，不弹出交互窗口。
    from matplotlib import font_manager, pyplot as plt
    font_path = ROOT / "assets/fonts/NotoSansCJK-Regular.ttc"
    font_manager.fontManager.addfont(str(font_path))
    family = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"font.family": family, "axes.unicode_minus": False,
                         "font.size": 11, "svg.fonttype": "path"})
    return plt


def save_figure(figure, path):
    """同一张图导出高清 PNG 和字形已转为路径的 SVG，便于查看与论文排版。"""
    figure.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    figure.savefig(path.with_suffix(".svg"), bbox_inches="tight")


def training_curves(directory, history, best_epoch):
    """左图比较训练/验证损失；右图只画验证准确率，测试集不参与逐轮监控。"""
    from matplotlib.ticker import PercentFormatter
    plt = setup_plotting()
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.3))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="训练损失（启用随机失活）")
    axes[0].plot(epochs, [row["validation_loss"] for row in history], label="验证损失（评估模式）")
    axes[0].set(title="ADS1299 训练与验证损失", xlabel="训练轮次", ylabel="交叉熵损失")
    axes[1].plot(epochs, [row["validation_accuracy"] for row in history], label="验证准确率")
    axes[1].set(title="ADS1299 验证准确率", xlabel="训练轮次", ylabel="准确率", ylim=(0, 1.05))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for axis in axes:
        axis.axvline(best_epoch, color="gray", linestyle="--", label=f"选中第 {best_epoch} 轮")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=9)
    figure.tight_layout()
    save_figure(figure, directory / "training_curves")
    plt.close(figure)


def confusion_matrix(directory, result, title, filename):
    """混淆矩阵的行是真实注视目标、列是模型预测；数字是实际试次数。"""
    plt = setup_plotting()
    matrix = np.array(result["confusion_matrix_true_rows"])
    figure, axis = plt.subplots(figsize=(6.2, 5.2))
    color = axis.imshow(matrix, cmap="Blues", vmin=0)
    for row in range(4):
        for column in range(4):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center",
                      color="white" if matrix[row, column] > matrix.max() / 2 else "black")
    labels = [f"目标 {c}" for c in CLASSES]
    axis.set(xticks=range(4), yticks=range(4), xticklabels=labels, yticklabels=labels,
             xlabel="预测注视目标", ylabel="真实注视目标",
             title=f"{title}\n准确率 {result['accuracy']:.2%}（{result['correct']}/{result['count']}）")
    figure.colorbar(color, ax=axis, label="试次数")
    figure.tight_layout()
    save_figure(figure, directory / filename)
    plt.close(figure)


def external_accuracy(directory, summaries, internal_accuracy):
    """将内部会话测试与外部跨设备复测明确分开，避免把两种指标混为一谈。"""
    from matplotlib.ticker import PercentFormatter
    plt = setup_plotting()
    labels = ["ADS1299\n内部留出会话"] + [row["label"] for row in summaries]
    values = [internal_accuracy] + [row["accuracy"] for row in summaries]
    figure, axis = plt.subplots(figsize=(9.5, 5.2))
    bars = axis.bar(labels, values, color=["#3574a3"] + ["#d99143"] * len(summaries), width=0.55)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.2%}", ha="center")
    axis.axhline(0.25, color="gray", linestyle="--", label="四分类机会水平（25%）")
    axis.set(title="冻结 ADS1299 模型的内部测试与跨设备外部复测", ylabel="准确率", ylim=(0, 1.15))
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.legend(loc="upper right")
    figure.tight_layout()
    save_figure(figure, directory / "internal_external_comparison")
    plt.close(figure)
