"""ADS1299 数据核验、分组划分、训练集预处理及结果保存；不修改原始数据包。"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, resample_poly, sosfiltfilt


ROOT = Path(__file__).resolve().parent
CHANNELS = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]
CLASSES = ["A", "B", "C", "D"]
FREQUENCIES = [6.5, 9.0, 11.5, 14.5]
SPLITS = ("train", "validation", "test")


def digest(path):
    """分块计算哈希，核对数据来源以及代码版本。"""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(2 ** 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_json(path, value):
    """保留中文和缩进，禁止把非有限数写入结果文件。"""
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_csv(path, rows):
    """保存逐试次或逐轮记录，UTF-8 BOM 方便 Excel 阅读。"""
    with Path(path).open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_data(data_dir):
    """只读取合并版的 544 个独立试次；逐 trial_id 关联元数据，不叠加单次版。"""
    paths = list(Path(data_dir).glob("*combined4Recordings*.npz"))
    if len(paths) != 1:
        raise ValueError("应当恰好存在一份 combined4Recordings NPZ")
    path = paths[0]
    metadata_dir = path.parent / "npz_metadata"
    info = json.loads((metadata_dir / f"{path.stem}.metadata.json").read_text(encoding="utf-8"))
    if info["device"] != "ADS1299" or digest(path) != info["sha256"]:
        raise ValueError("设备类型或 NPZ SHA256 核验失败")
    with np.load(path, allow_pickle=False) as source:
        data = {name: source[name].copy() for name in source.files}
    x, y, ids = data["X"], data["y"], data["trial_ids"].astype(str)
    if (x.shape != (544, 8, 1500) or list(x.shape) != info["shape"]
            or y.shape != (544,) or y.dtype.kind not in "iu"
            or not np.isfinite(x).all() or len(set(ids)) != 544
            or not np.array_equal(np.bincount(y, minlength=4), [136] * 4)):
        raise ValueError("信号形状、有限性、标签或 trial_id 核验失败")
    if (str(data["unit"]) != "V" or float(data["sfreq"]) != 500
            or data["channel_names"].tolist() != CHANNELS
            or not np.all(data["split"] == "unsplit")
            or info["epoch_start_s"] != 0.5 or info["epoch_end_exclusive_s"] != 3.5):
        raise ValueError("单位、采样率、通道、窗口或原始划分定义不一致")
    table_path = metadata_dir / f"{path.stem}.trials.csv"
    with table_path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    by_id = {row["trial_id"]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != set(ids):
        raise ValueError("元数据 trial_id 重复、缺失或额外出现")
    rows = [by_id[trial_id] for trial_id in ids]
    for index, row in enumerate(rows):
        label = int(y[index])
        checks = [int(row["label"]) == label, row["class_name"] == CLASSES[label],
                  float(row["frequency_hz"]) == FREQUENCIES[label],
                  float(data["stimulus_frequencies_hz"][index]) == FREQUENCIES[label],
                  row["run_id"] == data["run_ids"][index],
                  row["session_id"] == data["session_ids"][index],
                  row["recording_id"] == data["recording_ids"][index],
                  row["subject_id"] == data["subject_ids"][index] == "001",
                  row["block_id"] == data["block_ids"][index],
                  row["quality_flags"] == data["quality_flags"][index],
                  int(row["epoch_samples"]) == 1500, float(row["epoch_start_s"]) == 0.5,
                  float(row["epoch_end_exclusive_s"]) == 3.5, row["split"] == "unsplit"]
        if not all(checks):
            raise ValueError(f"{ids[index]} 的标签、来源、窗口或质量标记不一致")
    if int(np.count_nonzero(data["quality_flags"] != "")) != 34:
        raise ValueError("质量标记数量与数据交接说明不一致")
    data.update(rows=rows, info=info, path=path, table_path=table_path)
    return data


def make_split(data):
    """事先固定完整会话划分：前两次训练、第三次验证、第四次测试。"""
    sessions = sorted(np.unique(data["session_ids"]).tolist())
    if len(sessions) != 4:
        raise ValueError("应有四个采集会话")
    assignments = {"train": sessions[:2], "validation": sessions[2:3], "test": sessions[3:]}
    masks = {split: np.isin(data["session_ids"], groups) for split, groups in assignments.items()}
    if tuple(int(masks[s].sum()) for s in SPLITS) != (288, 128, 128):
        raise ValueError("分组划分数量不正确")
    if not np.all(sum(mask.astype(int) for mask in masks.values()) == 1):
        raise ValueError("划分存在重叠或遗漏")
    # 第二次采集包含两个 run；整个会话归入训练，因此二者不会分散到其他集合。
    for key in ("session_ids", "run_ids"):
        for group in np.unique(data[key]):
            if sum(bool(np.any(mask & (data[key] == group))) for mask in masks.values()) != 1:
                raise ValueError(f"{key} 的组 {group} 跨集合泄漏")
    return masks, assignments


def signal_transform(x_v, band=None):
    """将 V 转成 μV 一次，再独立重采样每个试次到 250 Hz（750 点）。"""
    if x_v.ndim != 3 or x_v.shape[1:] != (8, 1500) or not np.isfinite(x_v).all():
        raise ValueError("输入应为有限的 [N,8,1500] V 单位信号")
    # 有抗混叠 FIR 的 2 倍降采样，保留 3 秒时长；从不拼接试次来滤波。
    x = resample_poly(x_v.astype(np.float64) * 1e6, up=1, down=2, axis=-1, padtype="line")
    if band is not None:
        # 可选的额外频带为候选超参数，只用训练/验证选择；不改写数据包。
        sos = butter(4, band, btype="bandpass", fs=250, output="sos")
        x = sosfiltfilt(sos, x, axis=-1)
    return x.astype(np.float32)


def spectral_features(x, harmonics=3):
    """对每段信号同时计算四个固定候选频率的谐波信噪比，不接收真实类别。"""
    time = np.arange(x.shape[-1]) / 250
    offsets = np.array([0, -1, -2 / 3, 2 / 3, 1])
    frequencies = np.array(FREQUENCIES)[:, None, None] * np.arange(1, harmonics + 1)[None, :, None] + offsets
    # 直接在指定频率做加窗傅里叶投影；没有把补零误当作提高真实频率分辨率。
    basis = np.exp(-2j * np.pi * frequencies[..., None] * time)
    centered = (x - x.mean(-1, keepdims=True)) * np.hanning(x.shape[-1])
    projection = np.einsum("nct,fhkt->ncfhk", centered, basis, optimize=True)
    power = np.abs(projection) ** 2
    snr = np.log((power[..., 0] + 1e-8) / (power[..., 1:].mean(-1) + 1e-8))
    return snr.transpose(0, 2, 3, 1).reshape(len(x), 4, 8 * harmonics).astype(np.float32)


def fit_preprocessing(train_v, validation_v, config):
    """所有需拟合的通道/频谱统计量只来自训练数据。"""
    train = signal_transform(train_v, config.get("band"))
    validation = signal_transform(validation_v, config.get("band"))
    mean = train.mean(axis=(0, 2), keepdims=True, dtype=np.float64).astype(np.float32)
    std = train.std(axis=(0, 2), keepdims=True, dtype=np.float64).clip(1e-6).astype(np.float32)
    if config["family"] in ("hybrid", "spectral"):
        train_s = spectral_features(train)
        val_s = spectral_features(validation)
    else:
        train_s, val_s = np.zeros((len(train), 4, 1), np.float32), np.zeros((len(validation), 4, 1), np.float32)
    score_mean = train_s.mean(axis=(0, 1), keepdims=True)
    score_std = train_s.std(axis=(0, 1), keepdims=True).clip(1e-6)
    stats = {"mean": torch.from_numpy(mean), "std": torch.from_numpy(std),
             "score_mean": torch.from_numpy(score_mean), "score_std": torch.from_numpy(score_std)}
    pairs = [(torch.from_numpy(np.ascontiguousarray(((x - mean) / std)[:, None])),
              torch.from_numpy(np.ascontiguousarray((s - score_mean) / score_std)))
             for x, s in ((train, train_s), (validation, val_s))]
    return pairs, stats


def transform_saved(x_v, checkpoint):
    """对新信号仅应用固定变换及已保存的训练统计量，不重新拟合。"""
    config = checkpoint["config"]
    x = signal_transform(x_v, config.get("band"))
    s = spectral_features(x) if config["family"] in ("hybrid", "spectral") else np.zeros((len(x), 4, 1), np.float32)
    p = {key: value.numpy() for key, value in checkpoint["statistics"].items()}
    return (torch.from_numpy(np.ascontiguousarray(((x - p["mean"]) / p["std"])[:, None])),
            torch.from_numpy(np.ascontiguousarray((s - p["score_mean"]) / p["score_std"])))


def metrics(y, probabilities):
    """统计准确率、宏 F1、平衡准确率和各类召回；混淆矩阵行是真实类别。"""
    predicted = probabilities.argmax(1)
    matrix = np.zeros((4, 4), dtype=np.int64)
    np.add.at(matrix, (y, predicted), 1)
    precision = np.diag(matrix) / matrix.sum(0).clip(1)
    recall = np.diag(matrix) / matrix.sum(1).clip(1)
    f1 = 2 * precision * recall / (precision + recall).clip(1e-12)
    return {"accuracy": float(np.mean(predicted == y)), "correct": int(np.sum(predicted == y)),
            "count": len(y), "balanced_accuracy": float(recall.mean()), "macro_f1": float(f1.mean()),
            "confusion_matrix_true_rows": matrix.tolist(),
            "per_class": [{"label": i, "target": CLASSES[i], "precision": float(precision[i]),
                           "recall": float(recall[i]), "f1": float(f1[i]), "support": int(matrix[i].sum())}
                          for i in range(4)]}


def plot_results(directory, history, result, best_epoch):
    """导出训练曲线及测试混淆矩阵；英文图注避免系统中文字体依赖。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for key, label in (("train_loss", "Train (dropout on)"), ("validation_loss", "Validation")):
        axes[0].plot(epochs, [row[key] for row in history], label=label)
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy")
    axes[1].plot(epochs, [row["validation_accuracy"] for row in history], label="Validation accuracy")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1.05))
    for axis in axes:
        axis.axvline(best_epoch, color="gray", linestyle="--", label="Selected epoch")
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(directory / "training_curves.png", dpi=160)
    plt.close(figure)
    matrix = np.array(result["confusion_matrix_true_rows"])
    figure, axis = plt.subplots(figsize=(4.5, 4))
    im = axis.imshow(matrix, cmap="Blues", vmin=0)
    for i in range(4):
        for j in range(4):
            axis.text(j, i, str(matrix[i, j]), ha="center", va="center",
                      color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    axis.set(xticks=range(4), yticks=range(4), xticklabels=CLASSES, yticklabels=CLASSES,
             xlabel="Predicted fixation target", ylabel="True fixation target", title="ADS1299: held-out session")
    figure.colorbar(im, ax=axis)
    figure.tight_layout()
    figure.savefig(directory / "test_confusion_matrix.png", dpi=160)
    plt.close(figure)
