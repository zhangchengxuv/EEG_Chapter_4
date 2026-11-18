"""把 S01/S02/S03 当作同一人的三次采集，合并训练并仅用验证集选择模型。"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_eegnet import (
    ROOT, CHANNELS, CLASSES, FREQUENCIES, SPLITS, EEGNet, classification_metrics,
    fit_standardizer, load_subject, plot_results, save_csv, save_json, set_seed,
)


def load_merged(data_dir):
    """先逐文件核对 trial_id 和原有区组划分，再沿试次轴合并；不重新随机划分。"""
    sessions = [load_subject(data_dir, name) for name in ("S01", "S02", "S03")]
    x = np.concatenate([data[0] for data in sessions])
    y = np.concatenate([data[1] for data in sessions])
    ids = np.concatenate([data[2] for data in sessions])
    rows = [row for data in sessions for row in data[3]]
    masks = {split: np.concatenate([data[4][split] for data in sessions]) for split in SPLITS}
    if len(set(ids)) != len(ids):
        raise ValueError("三次采集之间存在重复 trial_id")
    if tuple(int(masks[s].sum()) for s in SPLITS) != (202, 42, 95):
        raise ValueError("合并后的划分数量与预期不一致")
    return x, y, ids, rows, masks


def spectral_scores(x, harmonics=2, width=1):
    """从每个信号同时计算四个候选目标的局部频谱信噪比，不接收标签或元数据。"""
    # 4 秒、250 Hz 的 FFT 分辨率为 0.25 Hz；只使用现有 30 Hz 低通以内的信息。
    # 固定任务频率是所有试次共有的参考定义，不是单个试次的已知刺激类别。
    centered = x - x.mean(axis=-1, keepdims=True)
    power = np.abs(np.fft.rfft(centered * np.hanning(x.shape[-1]), axis=-1)) ** 2
    frequencies = np.fft.rfftfreq(x.shape[-1], 1 / 250)
    features = []
    for target in FREQUENCIES:
        parts = []
        for harmonic in range(1, harmonics + 1):
            index = int(np.argmin(abs(frequencies - target * harmonic)))
            # 邻近背景固定取 ±0.5、±0.75、±1 Hz，避开目标频点及紧邻泄漏频点。
            background = power[:, :, [index - 4, index - 3, index - 2, index + 2, index + 3, index + 4]].mean(-1)
            signal = power[:, :, index - width:index + width + 1].mean(-1)
            parts.append(np.log((signal + 1e-8) / (background + 1e-8)))
        features.append(np.concatenate(parts, axis=1))
    return np.stack(features, axis=1).astype(np.float32)  # [试次,4,通道×谐波]


class SharedSpectralHead(nn.Module):
    """四个候选类别共用一组频谱评分权重，减少小样本下的自由参数。"""

    def __init__(self, dimensions, hidden=0, dropout=0.25):
        super().__init__()
        if hidden:
            self.scorer = nn.Sequential(nn.Linear(dimensions, hidden), nn.ELU(),
                                        nn.Dropout(dropout), nn.Linear(hidden, 1))
        else:
            self.scorer = nn.Linear(dimensions, 1, bias=False)
            nn.init.constant_(self.scorer.weight, 1 / dimensions)
        self.bias = nn.Parameter(torch.zeros(4))

    def forward(self, scores):
        """同一评分器对所有候选频率工作，输出四类可学习的评分。"""
        return self.scorer(scores).squeeze(-1) + self.bias


class TunedEEGNet(nn.Module):
    """可调 EEGNet；energy 选项以平方能量汇聚减轻相位变化的影响。"""

    def __init__(self, filters=16, kernel=251, dropout=0.5, energy=False):
        super().__init__()
        self.energy = energy
        features = filters * 2
        self.temporal = nn.Sequential(
            nn.Conv2d(1, filters, (1, kernel), padding=(0, kernel // 2), bias=False),
            nn.BatchNorm2d(filters, eps=1e-3, momentum=0.1),
        )
        self.spatial_conv = nn.Conv2d(filters, features, (8, 1), groups=filters, bias=False)
        self.norm = nn.BatchNorm2d(features, eps=1e-3, momentum=0.1)
        self.pool = nn.AvgPool2d((1, 4))
        self.drop = nn.Dropout(dropout)
        self.separable = nn.Sequential(
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(features, features, (1, 32), groups=features, bias=False),
            nn.Conv2d(features, features, 1, bias=False),
            nn.BatchNorm2d(features, eps=1e-3, momentum=0.1), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(dropout), nn.Flatten(),
        )
        self.classifier = nn.Linear(features if energy else features * 31, 4)

    def forward(self, x):
        """标准分支保留时间结构；能量分支对时间平方均值取对数。"""
        x = self.norm(self.spatial_conv(self.temporal(x)))
        if self.energy:
            x = torch.log(x.square().mean(dim=(2, 3)).clamp_min(1e-6))
            return self.classifier(self.drop(x))
        return self.classifier(self.separable(self.drop(self.pool(nn.functional.elu(x)))))

    @torch.no_grad()
    def constrain_weights(self):
        """沿用 EEGNet 的空间滤波器范数限制，分类器上限适当放宽。"""
        for weight, limit in ((self.spatial_conv.weight, 1.0), (self.classifier.weight, 0.5)):
            norm = weight.flatten(1).norm(dim=1).reshape((-1,) + (1,) * (weight.ndim - 1)).clamp_min(1e-8)
            weight.mul_((limit / norm).clamp(max=1))


class MergedModel(nn.Module):
    """统一封装时域 EEGNet、频谱评分器和二者融合，便于公平地选择方案。"""

    def __init__(self, config):
        super().__init__()
        kind = config["kind"]
        self.raw = None
        self.spectral = None
        self.raw_scale = config.get("raw_scale", 1.0)
        if kind in ("baseline", "hybrid"):
            self.raw = EEGNet(dropout=config.get("dropout", 0.5))
        elif kind == "tuned":
            self.raw = TunedEEGNet(energy=config.get("energy", False))
        if kind in ("spectral", "hybrid"):
            self.spectral = SharedSpectralHead(8 * config.get("harmonics", 2), hidden=config.get("hidden", 0))

    def forward(self, x, scores):
        """时域分支作为频谱分支的残差；仅频谱模型用来做消融比较。"""
        logits = self.raw(x) * self.raw_scale if self.raw is not None else 0
        if self.spectral is not None:
            logits = logits + self.spectral(scores)
        return logits

    def constrain_weights(self):
        """更新时域分支的最大范数约束。"""
        if self.raw is not None:
            self.raw.constrain_weights()


# 搜索开始前明确候选集合：避免看到测试成绩之后再增加或挑选方案。
CANDIDATES = [
    {"name": "merged_original", "kind": "baseline", "lr": 0.001, "weight_decay": 0.0},
    {"name": "long_kernel", "kind": "tuned", "lr": 0.0005, "weight_decay": 0.01},
    {"name": "energy_eegnet", "kind": "tuned", "energy": True, "lr": 0.001, "weight_decay": 0.01},
    {"name": "spectrum_fundamental", "kind": "spectral", "harmonics": 1, "lr": 0.01, "weight_decay": 0.01},
    {"name": "spectrum_two_harmonics", "kind": "spectral", "harmonics": 2, "lr": 0.01, "weight_decay": 0.01},
    {"name": "spectrum_nonlinear", "kind": "spectral", "harmonics": 2, "hidden": 16, "lr": 0.002, "weight_decay": 0.05},
    {"name": "hybrid_smooth", "kind": "hybrid", "harmonics": 2, "width": 1, "raw_scale": 0.15, "lr": 0.002, "weight_decay": 0.01},
    {"name": "hybrid_exact_bin", "kind": "hybrid", "harmonics": 2, "width": 0, "raw_scale": 0.15, "lr": 0.002, "weight_decay": 0.01},
    {"name": "hybrid_regularized", "kind": "hybrid", "harmonics": 2, "width": 1, "raw_scale": 0.1, "lr": 0.001, "weight_decay": 0.05},
]


def prepare(train_x, validation_x, config, channel_stats):
    """预处理只拟合训练集；候选模型搜索期间不传入测试集。"""
    mean, std = channel_stats
    harmonics, width = config.get("harmonics", 2), config.get("width", 1)
    train_scores = spectral_scores(train_x, harmonics, width)
    validation_scores = spectral_scores(validation_x, harmonics, width)
    # 四个候选频率共用通道/谐波统计量，保留候选频率之间的相对强弱。
    score_mean = train_scores.mean(axis=(0, 1), keepdims=True)
    score_std = train_scores.std(axis=(0, 1), keepdims=True).clip(1e-6)
    preprocessing = {"mean": torch.from_numpy(mean), "std": torch.from_numpy(std),
                     "score_mean": torch.from_numpy(score_mean), "score_std": torch.from_numpy(score_std)}
    pairs = []
    for raw, scores in ((train_x, train_scores), (validation_x, validation_scores)):
        pairs.append((torch.from_numpy(np.ascontiguousarray(((raw - mean) / std)[:, None])),
                      torch.from_numpy(np.ascontiguousarray((scores - score_mean) / score_std))))
    return pairs, preprocessing


@torch.no_grad()
def predict(model, pair, labels, device):
    """评估模式下预测；按批次处理以控制 GPU 显存。"""
    model.eval()
    logits = torch.cat([model(pair[0][i:i + 32].to(device), pair[1][i:i + 32].to(device)).cpu()
                        for i in range(0, len(pair[0]), 32)])
    loss = nn.functional.cross_entropy(logits, torch.from_numpy(labels)).item()
    return loss, logits.softmax(1).numpy()


def fit_candidate(config, seed, train_x, train_y, val_x, val_y, channel_stats, args, output):
    """一个候选独立训练；以验证准确率优先、交叉熵次优选择轮次。"""
    set_seed(seed)
    name = f"{config['name']}_seed{seed}"
    directory = output / "candidates" / name
    directory.mkdir(parents=True)
    pairs, preprocessing = prepare(train_x, val_x, config, channel_stats)
    train_pair, val_pair = pairs
    loader = DataLoader(TensorDataset(*train_pair, torch.from_numpy(train_y)),
                        batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(seed))
    model = MergedModel(config).to(args.device)
    model.constrain_weights()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    best_key, best_state, history = (-1, -float("inf")), None, []
    best_epoch, stale = 0, 0
    print(f"\n候选 {name}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch_x, batch_scores, batch_y in loader:
            batch_x, batch_scores, batch_y = batch_x.to(args.device), batch_scores.to(args.device), batch_y.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(batch_x, batch_scores), batch_y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{name}: 损失非有限")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            model.constrain_weights()
            total += loss.item() * len(batch_y)
        val_loss, probabilities = predict(model, val_pair, val_y, args.device)
        accuracy = float((probabilities.argmax(1) == val_y).mean())
        history.append({"epoch": epoch, "train_loss": total / len(train_y),
                        "validation_loss": val_loss, "validation_accuracy": accuracy})
        key = (accuracy, -val_loss)
        if key > best_key:
            best_key, best_epoch, stale = key, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 40 == 0:
            print(f"  epoch={epoch}, val_acc={accuracy:.2%}, val_loss={val_loss:.4f}", flush=True)
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    val_loss, probabilities = predict(model, val_pair, val_y, args.device)
    checkpoint = {"model_state_dict": best_state, "config": config, "seed": seed,
                  "best_epoch": best_epoch, "preprocessing": preprocessing,
                  "channels": CHANNELS, "classes": CLASSES, "frequencies_hz": FREQUENCIES,
                  "sampling_rate_hz": 250, "input_unit": "uV", "sessions": ["S01", "S02", "S03"]}
    torch.save(checkpoint, directory / "best_model.pt")
    save_csv(directory / "history.csv", history)
    np.save(directory / "validation_probabilities.npy", probabilities)
    result = {"candidate": name, "architecture": config["name"], "kind": config["kind"], "seed": seed,
              "best_epoch": best_epoch, "epochs_run": len(history),
              "validation_accuracy": best_key[0], "validation_loss": val_loss}
    save_json(directory / "validation_metrics.json", result)
    print(f"  选中第 {best_epoch} 轮，验证准确率={best_key[0]:.2%}", flush=True)
    return result


def checkpoint_pair(raw, checkpoint):
    """使用已保存的训练统计量变换新信号；推理时无需标签、区组或采集编号。"""
    p = {name: tensor.numpy() for name, tensor in checkpoint["preprocessing"].items()}
    config = checkpoint["config"]
    scores = spectral_scores(raw, config.get("harmonics", 2), config.get("width", 1))
    return (torch.from_numpy(np.ascontiguousarray(((raw - p["mean"]) / p["std"])[:, None])),
            torch.from_numpy(np.ascontiguousarray((scores - p["score_mean"]) / p["score_std"])))


def main():
    """固定搜索集合→验证集选择→多种子集成比较→固化选择→最终测试。"""
    parser = argparse.ArgumentParser(description="同一人三次采集合并的 EEGNet 优化")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "ML_DL_Dataset")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/eegnet_merged_search")
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if min(args.epochs, args.patience) < 1:
        parser.error("轮数和早停等待轮数必须为正")
    torch.set_num_threads(2)
    x, y, ids, rows, masks = load_merged(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    # 原有 subject_id 仅保留为采集来源 session，不作为模型输入。
    save_csv(args.output_dir / "split_membership.csv", [
        {"trial_id": trial_id, "session": row["subject_id"], "block": row["block_id"],
         "split": row["split"], "label": int(label)} for trial_id, row, label in zip(ids, rows, y)])
    train_x, train_y = x[masks["train"]], y[masks["train"]]
    val_x, val_y = x[masks["validation"]], y[masks["validation"]]
    channel_stats = fit_standardizer(train_x)
    source_files = [Path(__file__), ROOT / "train_eegnet.py"]
    source_files += [args.data_dir / folder / f"{session}{suffix}" for session in ("S01", "S02", "S03")
                     for folder, suffix in (("NumPy", "_accepted_0to4_uv.npz"), ("Metadata", "_trials.csv"))]
    save_json(args.output_dir / "search_plan.json", {
        "candidates": CANDIDATES, "epochs": args.epochs, "patience": args.patience,
        "batch_size": 32, "search_seed": 42, "ensemble_seeds": [42, 43, 44],
        "selection": "validation accuracy descending, then validation cross-entropy ascending",
        "splits": {split: int(mask.sum()) for split, mask in masks.items()},
        "device": args.device, "torch": str(torch.__version__), "numpy": np.__version__,
        "sha256": {str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "session_interpretation": "同一受试者的三次采集；每次保留原区组划分；不把采集编号当作输入",
    })
    print("合并训练/验证/测试 = 202/42/95；搜索期间仅使用训练和验证。", flush=True)
    results = [fit_candidate(config, 42, train_x, train_y, val_x, val_y, channel_stats, args, args.output_dir)
               for config in CANDIDATES]
    save_csv(args.output_dir / "validation_search.csv", results)
    # 原始 EEGNet 与最优 EEGNet 类模型作公平比较；纯频谱模型只作为诊断消融。
    eligible = [row for row in results if row["kind"] != "spectral"]
    best = max(eligible, key=lambda row: (row["validation_accuracy"], -row["validation_loss"]))
    config = next(config for config in CANDIDATES if config["name"] == best["architecture"])
    extra = [fit_candidate(config, seed, train_x, train_y, val_x, val_y, channel_stats, args, args.output_dir)
             for seed in (43, 44)]
    results.extend(extra)
    save_csv(args.output_dir / "validation_search.csv", results)
    seed_names = [best["candidate"]] + [row["candidate"] for row in extra]
    ensemble_probabilities = np.mean([np.load(args.output_dir / "candidates" / name / "validation_probabilities.npy")
                                      for name in seed_names], axis=0)
    ensemble_accuracy = float((ensemble_probabilities.argmax(1) == val_y).mean())
    ensemble_loss = float(-np.log(ensemble_probabilities[np.arange(len(val_y)), val_y].clip(1e-8)).mean())
    ensemble_key = (ensemble_accuracy, -ensemble_loss)
    single_key = (best["validation_accuracy"], -best["validation_loss"])
    selected = seed_names if ensemble_key > single_key else [best["candidate"]]
    # 测试前将所有选择写入磁盘；之后不再挑模型，也不再训练。
    selection = {"selected_members": selected, "best_architecture": best["architecture"],
                 "single_validation_accuracy": best["validation_accuracy"],
                 "ensemble_validation_accuracy": ensemble_accuracy,
                 "ensemble_validation_loss": ensemble_loss,
                 "selection_scope": "仅验证集；纯频谱消融不参与最终 EEGNet 模型选择"}
    save_json(args.output_dir / "selection_before_test.json", selection)
    print(f"\n模型已固化：{selected}；开始最终测试。", flush=True)
    # 只评估合并原始基线与已固化方案，各模型的概率仅计算一次并缓存。
    baseline = "merged_original_seed42"
    needed = list(dict.fromkeys([baseline] + selected))
    probabilities, model_cache = {}, {}
    for name in needed:
        path = args.output_dir / "candidates" / name / "best_model.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = MergedModel(checkpoint["config"]).to(args.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        _, probability = predict(model, checkpoint_pair(x[masks["test"]], checkpoint), y[masks["test"]], args.device)
        probabilities[name] = probability
        model_cache[name] = checkpoint
    final_probabilities = np.mean([probabilities[name] for name in selected], axis=0)
    summary = []
    for name, probability in (("merged_original", probabilities[baseline]), ("selected_model", final_probabilities)):
        directory = args.output_dir / name
        directory.mkdir()
        metrics = classification_metrics(y[masks["test"]], probability)
        metrics["by_session"] = {}
        test_sessions = np.array([row["subject_id"] for row, keep in zip(rows, masks["test"]) if keep])
        for session in ("S01", "S02", "S03"):
            mask = test_sessions == session
            metrics["by_session"][session] = classification_metrics(y[masks["test"]][mask], probability[mask])
        save_json(directory / "metrics.json", metrics)
        prediction_rows = [
            {"trial_id": trial_id, "session": session, "true_label": int(label),
             "predicted_label": int(p.argmax()),
             **{f"probability_{target}": float(p[i]) for i, target in enumerate(CLASSES)}}
            for trial_id, session, label, p in zip(ids[masks["test"]], test_sessions, y[masks["test"]], probability)]
        save_csv(directory / "test_predictions.csv", prediction_rows)
        member = baseline if name == "merged_original" else selected[0]
        with (args.output_dir / "candidates" / member / "history.csv").open(encoding="utf-8-sig", newline="") as file:
            import csv
            history = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(file)]
        plot_results(directory, history, metrics, model_cache[member]["best_epoch"])
        summary.append({"model": name, "test_count": 95, "accuracy": metrics["accuracy"],
                        "balanced_accuracy": metrics["balanced_accuracy"], "macro_f1": metrics["macro_f1"]})
        print(f"{name}: test_acc={metrics['accuracy']:.2%}, macro_F1={metrics['macro_f1']:.4f}", flush=True)
    # 打包最终成员，推理时只需这一个文件，无需原始训练数据或元数据。
    torch.save({"members": [model_cache[name] for name in selected], "selection": selection,
                "combination": "arithmetic mean of softmax probabilities"}, args.output_dir / "selected_model" / "model.pt")
    save_csv(args.output_dir / "test_summary.csv", summary)
    print(f"全部结果：{args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
