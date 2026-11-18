"""对三位受试者分别进行 EEGNet 四分类训练；所有数据处理均遵守交接说明。"""

import argparse
import csv
import hashlib
import json
import os
import random
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# 必须在首次使用 CUDA 前设置，保证矩阵运算采用可复现的实现。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
CHANNELS = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]
CLASSES = ["A", "B", "C", "D"]
FREQUENCIES = [6.5, 9.0, 11.5, 14.5]
SPLITS = ("train", "validation", "test")
EXPECTED_COUNTS = {"S01": (77, 10, 32), "S02": (78, 16, 31), "S03": (47, 16, 32)}


def save_json(path, content):
    """以可直接阅读的中文 JSON 保存配置和结果。"""
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_csv(path, rows):
    """保存带表头的记录；使用 UTF-8 BOM，方便 Excel 正确显示中文。"""
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed):
    """固定初始化、批次打乱和 Dropout 的随机性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def load_subject(data_dir, subject):
    """读取唯一的 accepted 4 秒版本，并按 trial_id 一对一关联元数据。"""
    npz_path = data_dir / "NumPy" / f"{subject}_accepted_0to4_uv.npz"
    csv_path = data_dir / "Metadata" / f"{subject}_trials.csv"
    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        metadata = list(csv.DictReader(file))
    by_id = {row["trial_id"]: row for row in metadata}
    if len(by_id) != len(metadata):
        raise ValueError(f"{subject}: 元数据存在重复 trial_id")

    with np.load(npz_path, allow_pickle=False) as data:
        x = data["X"].astype(np.float32)
        y = data["y"].copy()
        trial_ids = data["trial_ids"].astype(str)
        # 先核对单位、通道、窗口等定义；绝不再次乘 1e6，也不追加滤波。
        checks = [
            float(data["sfreq"]) == 250.0,
            data["ch_names"].tolist() == CHANNELS,
            str(data["unit"]) in ("uV", "μV", "µV"),
            data["class_names"].tolist() == CLASSES,
            np.array_equal(data["stimulus_frequencies"], FREQUENCIES),
            float(data["epoch_start"]) == 0.0,
            float(data["epoch_end_exclusive"]) == 4.0,
            bool(data["accepted"].all()),
            bool((data["subject_ids"] == subject).all()),
        ]
        block_ids = data["block_ids"].copy()
    if not all(checks):
        raise ValueError(f"{subject}: 数据定义与交接说明不一致")
    if x.shape != (sum(EXPECTED_COUNTS[subject]), 8, 1000):
        raise ValueError(f"{subject}: X 维度不符合预期：{x.shape}")
    if y.shape != (len(x),) or y.dtype.kind not in "iu" or not np.isin(y, range(4)).all():
        raise ValueError(f"{subject}: 分类标签必须是一维 0～3 整数")
    if not np.isfinite(x).all() or len(set(trial_ids)) != len(x):
        raise ValueError(f"{subject}: 信号非有限，或 trial_id 重复/数量错误")

    # 字典查找保持 NPZ 原有顺序；CSV 即使重排，也不会造成标签或划分错位。
    if not set(trial_ids).issubset(by_id):
        raise ValueError(f"{subject}: 部分 NPZ trial_id 未出现在元数据中")
    rows = [by_id[trial_id] for trial_id in trial_ids]
    for index, row in enumerate(rows):
        block = int(row["block_id"])
        expected_split = "train" if 1 <= block <= 5 else "validation" if block == 6 else "test" if block in (7, 8) else None
        if (
            row["subject_id"] != subject
            or int(row["class_label"]) != int(y[index])
            or row["event_code"] != CLASSES[y[index]]
            or float(row["stimulus_frequency_hz"]) != FREQUENCIES[y[index]]
            or row["accepted"].lower() != "true"
            or block != int(block_ids[index])
            or expected_split is None
            or row["split"] != expected_split
        ):
            raise ValueError(f"{subject}: {row['trial_id']} 的标签、质量标记或区组划分不一致")
    # 只用关联后的 split 建立掩码，绝不使用指向完整 128 行数组的 JSON 索引。
    masks = {split: np.array([row["split"] == split for row in rows]) for split in SPLITS}
    counts = tuple(int(masks[split].sum()) for split in SPLITS)
    if counts != EXPECTED_COUNTS[subject]:
        raise ValueError(f"{subject}: 划分数量错误：{counts}")
    if not np.all(sum(mask.astype(int) for mask in masks.values()) == 1):
        raise ValueError(f"{subject}: 划分重叠或存在遗漏")
    for split, mask in masks.items():
        if set(y[mask]) != set(range(4)):
            raise ValueError(f"{subject}: {split} 中未包含全部四类")
    return x, y.astype(np.int64), trial_ids, rows, masks


def fit_standardizer(train_x):
    """仅在训练试次和时间点上，计算每个通道的均值、标准差。"""
    # 保留 [1,8,1] 形状，方便广播到任意数量的试次；双精度计算统计量。
    mean = train_x.mean(axis=(0, 2), keepdims=True, dtype=np.float64).astype(np.float32)
    std = train_x.std(axis=(0, 2), keepdims=True, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(std, 1e-6)


class EEGNet(nn.Module):
    """EEGNet-8,2：时间卷积 → 深度空间卷积 → 可分离卷积 → 四分类。"""

    def __init__(self, channels=8, samples=1000, classes=4, dropout=0.5):
        super().__init__()
        # 第一层学习时间滤波器。250 Hz 下 125 点对应约 0.5 秒。
        self.temporal = nn.Sequential(
            nn.Conv2d(1, 8, (1, 125), padding=(0, 62), bias=False),
            nn.BatchNorm2d(8, eps=1e-3, momentum=0.01),
        )
        # groups=8：每个时间滤波器独立学习两个跨越全部通道的空间滤波器。
        self.spatial_conv = nn.Conv2d(8, 16, (channels, 1), groups=8, bias=False)
        self.spatial = nn.Sequential(
            nn.BatchNorm2d(16, eps=1e-3, momentum=0.01),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        # 深度时间卷积后接 1×1 卷积，融合不同滤波器的特征。
        # 原实现 128 Hz 下的 16 点核在这里调整为 32 点；显式补零保持长度。
        self.separable = nn.Sequential(
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(16, 16, (1, 32), groups=16, bias=False),
            nn.Conv2d(16, 16, (1, 1), bias=False),
            nn.BatchNorm2d(16, eps=1e-3, momentum=0.01),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(16 * (samples // 4 // 8), classes)
        # 与原实现一致，卷积和全连接使用 Xavier 初始化。
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        """输入 [批量,1,8,1000]，输出四类 logits；交叉熵内部会处理 softmax。"""
        x = self.temporal(x)
        x = self.spatial(self.spatial_conv(x))
        return self.classifier(self.separable(x))

    @torch.no_grad()
    def constrain_weights(self):
        """每次更新后限制空间滤波器和分类器范数，减少小样本过拟合。"""
        for weight, limit in ((self.spatial_conv.weight, 1.0), (self.classifier.weight, 0.25)):
            dimensions = tuple(range(1, weight.ndim))
            norm = weight.norm(p=2, dim=dimensions, keepdim=True).clamp_min(1e-8)
            weight.mul_((limit / norm).clamp(max=1.0))


@torch.no_grad()
def evaluate(model, x, y, device):
    """关闭 Dropout，并固定 BatchNorm 统计量，再计算损失和概率。"""
    model.eval()
    logits = model(x.to(device))
    loss = nn.functional.cross_entropy(logits, y.to(device)).item()
    probabilities = logits.softmax(dim=1).cpu().numpy()
    return loss, probabilities


def classification_metrics(y_true, probabilities):
    """直接用 NumPy 计算混淆矩阵与各类指标，避免额外依赖。"""
    prediction = probabilities.argmax(axis=1)
    matrix = np.zeros((4, 4), dtype=np.int64)
    np.add.at(matrix, (y_true, prediction), 1)
    true_positive = np.diag(matrix)
    precision = true_positive / np.maximum(matrix.sum(axis=0), 1)
    recall = true_positive / np.maximum(matrix.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "accuracy": float((prediction == y_true).mean()),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "confusion_matrix_true_rows_predicted_columns": matrix.tolist(),
        "per_class": [
            {"label": index, "class": CLASSES[index], "frequency_hz": FREQUENCIES[index],
             "precision": float(precision[index]), "recall": float(recall[index]),
             "f1": float(f1[index]), "support": int(matrix[index].sum())}
            for index in range(4)
        ],
    }


def plot_results(output_dir, history, metrics, best_epoch):
    """生成训练曲线和测试混淆矩阵；图中文字使用英文以避免字体依赖。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train (with dropout)")
    axes[0].plot(epochs, [row["validation_loss"] for row in history], label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[1].plot(epochs, [row["validation_accuracy"] for row in history], label="Validation accuracy")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1.05))
    for axis in axes:
        axis.axvline(best_epoch, color="gray", linestyle="--", label="Selected epoch")
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(figure)

    matrix = np.array(metrics["confusion_matrix_true_rows_predicted_columns"])
    figure, axis = plt.subplots(figsize=(4.5, 4))
    image = axis.imshow(matrix, cmap="Blues", vmin=0)
    for row in range(4):
        for column in range(4):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center",
                      color="white" if matrix[row, column] > matrix.max() / 2 else "black")
    axis.set(xticks=range(4), yticks=range(4), xticklabels=CLASSES, yticklabels=CLASSES,
             xlabel="Predicted fixation target", ylabel="True fixation target", title="Test confusion matrix")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_dir / "test_confusion_matrix.png", dpi=160)
    plt.close(figure)


def train_subject(subject, data, args, device, output_dir):
    """每位受试者独立标准化、独立初始化并训练，不混合不同受试者。"""
    set_seed(args.seed)
    subject_dir = output_dir / subject
    subject_dir.mkdir()
    x, y, trial_ids, rows, masks = data
    mean, std = fit_standardizer(x[masks["train"]])
    np.savez(subject_dir / "standardization.npz", mean=mean, std=std, channels=CHANNELS)

    # 验证和测试沿用训练集统计量，禁止各自重新拟合。
    tensors = {}
    for split, mask in masks.items():
        normalized = np.ascontiguousarray((x[mask] - mean) / std)
        tensors[split] = (torch.from_numpy(normalized[:, None]), torch.from_numpy(y[mask]))
    split_rows = [
        {"trial_id": trial_id, "split": row["split"], "block_id": row["block_id"], "label": int(label)}
        for trial_id, row, label in zip(trial_ids, rows, y)
    ]
    save_csv(subject_dir / "split_membership.csv", split_rows)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(TensorDataset(*tensors["train"]), batch_size=args.batch_size,
                        shuffle=True, num_workers=0, generator=generator)
    model = EEGNet().to(device)
    model.constrain_weights()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_function = nn.CrossEntropyLoss()
    best_loss, best_epoch, stale_epochs = float("inf"), 0, 0
    best_state, history = None, []
    print(f"\n{subject}: train/validation/test = "
          f"{'/'.join(str(int(masks[s].sum())) for s in SPLITS)}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_x), batch_y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{subject}: 训练损失非有限")
            loss.backward()
            optimizer.step()
            model.constrain_weights()
            total_loss += loss.item() * len(batch_y)
        validation_loss, validation_probabilities = evaluate(model, *tensors["validation"], device)
        validation_accuracy = float((validation_probabilities.argmax(1) == y[masks["validation"]]).mean())
        history.append({"epoch": epoch, "train_loss": total_loss / int(masks["train"].sum()),
                        "validation_loss": validation_loss, "validation_accuracy": validation_accuracy})
        # 检查点和早停只看验证损失；训练过程中从不计算测试准确率。
        if validation_loss < best_loss:
            best_loss, best_epoch, stale_epochs = validation_loss, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 20 == 0:
            print(f"  epoch {epoch:3d}: train_loss={history[-1]['train_loss']:.4f}, "
                  f"val_loss={validation_loss:.4f}, val_acc={validation_accuracy:.1%}", flush=True)
        if stale_epochs >= args.patience:
            print(f"  验证损失连续 {args.patience} 轮未改善，提前停止。", flush=True)
            break

    # 先固化模型和预处理，再进行一次最终测试；不根据测试结果重选参数。
    model.load_state_dict(best_state)
    checkpoint = {
        "model_state_dict": best_state, "model_config": {"channels": 8, "samples": 1000, "classes": 4, "dropout": 0.5},
        "standardization_mean": torch.from_numpy(mean), "standardization_std": torch.from_numpy(std),
        "subject": subject, "channels": CHANNELS, "classes": CLASSES, "frequencies_hz": FREQUENCIES,
        "sampling_rate_hz": 250, "input_unit": "uV", "seed": args.seed, "best_epoch": best_epoch,
    }
    torch.save(checkpoint, subject_dir / "best_model.pt")
    save_csv(subject_dir / "history.csv", history)
    test_loss, test_probabilities = evaluate(model, *tensors["test"], device)
    metrics = classification_metrics(y[masks["test"]], test_probabilities)
    metrics.update({"subject": subject, "best_epoch": best_epoch, "epochs_run": len(history),
                    "best_validation_loss": best_loss,
                    "selected_validation_accuracy": history[best_epoch - 1]["validation_accuracy"],
                    "test_loss": test_loss,
                    "split_class_counts": {s: np.bincount(y[masks[s]], minlength=4).tolist() for s in SPLITS}})
    save_json(subject_dir / "metrics.json", metrics)
    predictions = []
    for trial_id, label, probability in zip(trial_ids[masks["test"]], y[masks["test"]], test_probabilities):
        predicted = int(probability.argmax())
        predictions.append({"trial_id": trial_id, "true_label": int(label), "predicted_label": predicted,
                            "true_target": CLASSES[label], "predicted_target": CLASSES[predicted],
                            **{f"probability_{name}": float(probability[i]) for i, name in enumerate(CLASSES)}})
    save_csv(subject_dir / "test_predictions.csv", predictions)
    plot_results(subject_dir, history, metrics, best_epoch)
    print(f"  最佳轮次={best_epoch}, test_acc={metrics['accuracy']:.2%}, macro_F1={metrics['macro_f1']:.4f}", flush=True)
    return metrics


def main():
    """读取参数、审计全部输入，再按固定方案依次训练三位受试者。"""
    parser = argparse.ArgumentParser(description="EEGNet 受试者内四分类训练")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "ML_DL_Dataset", help="交接数据目录")
    parser.add_argument("--output-dir", type=Path, default=None, help="新的结果目录，不能已存在")
    parser.add_argument("--subjects", nargs="+", choices=list(EXPECTED_COUNTS), default=list(EXPECTED_COUNTS))
    parser.add_argument("--epochs", type=int, default=300, help="最大训练轮数")
    parser.add_argument("--patience", type=int, default=60, help="验证损失无改善时的等待轮数")
    parser.add_argument("--batch-size", type=int, default=16, help="每批训练试次数")
    parser.add_argument("--lr", type=float, default=0.001, help="Adam 学习率")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1 or args.lr <= 0:
        parser.error("训练轮数、等待轮数、批量大小和学习率必须为正")
    if len(args.subjects) != len(set(args.subjects)):
        parser.error("受试者列表不能重复")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    torch.set_num_threads(2)
    # 开始训练前，先验证所有受试者数据，避免训练到一半才发现划分错误。
    datasets = {subject: load_subject(args.data_dir, subject) for subject in args.subjects}
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or ROOT / "results" / f"eegnet_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    input_hashes = {}
    for subject in args.subjects:
        for relative in (f"NumPy/{subject}_accepted_0to4_uv.npz", f"Metadata/{subject}_trials.csv"):
            input_hashes[relative] = hashlib.sha256((args.data_dir / relative).read_bytes()).hexdigest()
    save_json(output_dir / "config.json", {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "output_dir": str(output_dir.resolve()), "device_used": str(device),
        "torch_version": str(torch.__version__), "numpy_version": np.__version__,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "input_sha256": input_hashes,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model": "EEGNet-8,2; temporal_kernel=125; separable_kernel=32; pooling=4,8; dropout=0.5",
        "selection": "minimum validation cross-entropy; no test-based tuning",
        "preprocessing": "existing 30 Hz acquisition low-pass, 50 Hz notch, offline 4 Hz zero-phase high-pass; train-only channel z-score",
    })
    print(f"设备：{device}；结果目录：{output_dir}", flush=True)
    results = [train_subject(subject, datasets[subject], args, device, output_dir) for subject in args.subjects]
    summary = [{"subject": item["subject"], "best_epoch": item["best_epoch"],
                "test_count": sum(item["split_class_counts"]["test"]),
                "test_accuracy": item["accuracy"], "balanced_accuracy": item["balanced_accuracy"],
                "macro_f1": item["macro_f1"]} for item in results]
    save_csv(output_dir / "summary.csv", summary)
    save_json(output_dir / "aggregate.json", {
        "mean_subject_accuracy": float(np.mean([row["test_accuracy"] for row in summary])),
        "mean_subject_macro_f1": float(np.mean([row["macro_f1"] for row in summary])),
        "pooled_test_accuracy": float(np.average([row["test_accuracy"] for row in summary],
                                                  weights=[row["test_count"] for row in summary])),
        "note": "均为当前固定区组划分和单个随机种子的结果；不代表跨受试者泛化能力。",
    })
    print(f"\n训练与测试完成：{output_dir}", flush=True)


if __name__ == "__main__":
    main()
