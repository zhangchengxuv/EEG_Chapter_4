"""ADS1299 四分类：先固定完整会话划分，再用训练/验证集选 EEGNet 配置。"""

import argparse
import csv
import os
import random
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# CUDA 确定性设置必须放在首次使用 GPU 之前。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import scipy
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ads1299_data import (ROOT, CHANNELS, CLASSES, SPLITS, digest, fit_preprocessing,
                          load_data, make_split, metrics, plot_results, save_csv,
                          save_json, transform_saved)
from ads1299_models import Model


# 在读取任何测试成绩前固定这十组候选；纯频谱模型仅作为消融，不替代 EEGNet。
CANDIDATES = [
    {"name": "eegnet_standard", "family": "standard", "filters": 8, "kernel": 125, "momentum": 0.01, "lr": 0.001, "weight_decay": 0.0},
    {"name": "eegnet_long", "family": "standard", "filters": 16, "kernel": 251, "lr": 0.0005, "weight_decay": 0.01},
    {"name": "energy_previous", "family": "energy", "kernel": 251, "lr": 0.001, "weight_decay": 0.01},
    {"name": "energy_focus", "family": "energy", "kernel": 251, "band": [4, 45], "lr": 0.001, "weight_decay": 0.01},
    {"name": "energy_short", "family": "energy", "kernel": 125, "band": [4, 45], "lr": 0.001, "weight_decay": 0.01},
    {"name": "energy_wide", "family": "energy", "filters": 32, "kernel": 251, "band": [4, 45], "lr": 0.001, "weight_decay": 0.05},
    {"name": "energy_regularized", "family": "energy", "kernel": 251, "band": [4, 45], "dropout": 0.6, "lr": 0.0005, "weight_decay": 0.05},
    {"name": "spectrum_linear", "family": "spectral", "lr": 0.01, "weight_decay": 0.01},
    {"name": "spectrum_nonlinear", "family": "spectral", "hidden": 24, "lr": 0.002, "weight_decay": 0.05},
    {"name": "hybrid_energy", "family": "hybrid", "kernel": 251, "band": [4, 45], "raw_scale": 0.5, "lr": 0.001, "weight_decay": 0.01},
]


def set_seed(seed):
    """固定权重初始化、训练打乱和 Dropout；不同候选使用同一个初始种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


@torch.no_grad()
def probabilities(model, pair, device):
    """分批评估，关闭 Dropout，BatchNorm 只使用训练期间累计的统计量。"""
    model.eval()
    output = [model(pair[0][i:i + 32].to(device), pair[1][i:i + 32].to(device)).softmax(1).cpu().numpy()
              for i in range(0, len(pair[0]), 32)]
    return np.concatenate(output)


def cross_entropy(probability, y):
    """由真实类的预测概率计算交叉熵，只用于验证选轮次及结果记录。"""
    return float(-np.log(probability[np.arange(len(y)), y].clip(1e-12)).mean())


def train_candidate(config, seed, train_v, train_y, val_v, val_y, args, output_dir):
    """仅接收训练和验证数据；这个函数没有测试输入，因此无法用测试来选轮次。"""
    set_seed(seed)
    name = f"{config['name']}_seed{seed}"
    directory = output_dir / "candidates" / name
    directory.mkdir(parents=True)
    pairs, statistics = fit_preprocessing(train_v, val_v, config)
    train_pair, val_pair = pairs
    loader = DataLoader(TensorDataset(*train_pair, torch.from_numpy(train_y)),
                        batch_size=32, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed))
    model = Model(config).to(args.device)
    model.constrain_weights()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    best_key, best_state, best_epoch, stale = (-1, -float("inf")), None, 0, 0
    history = []
    print(f"\n候选 {name}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_spectrum, batch_y in loader:
            batch_x, batch_spectrum, batch_y = batch_x.to(args.device), batch_spectrum.to(args.device), batch_y.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(batch_x, batch_spectrum), batch_y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{name} 出现非有限损失")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            model.constrain_weights()
            total_loss += loss.item() * len(batch_y)
        val_probability = probabilities(model, val_pair, args.device)
        val_loss = cross_entropy(val_probability, val_y)
        accuracy = float(np.mean(val_probability.argmax(1) == val_y))
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_y),
                        "validation_loss": val_loss, "validation_accuracy": accuracy})
        key = (accuracy, -val_loss)
        if key > best_key:
            best_key, best_epoch, stale = key, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 50 == 0:
            print(f"  epoch={epoch}, val_acc={accuracy:.2%}, val_loss={val_loss:.4f}", flush=True)
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    val_probability = probabilities(model, val_pair, args.device)
    checkpoint = {"config": config, "seed": seed, "best_epoch": best_epoch,
                  "model_state_dict": best_state, "statistics": statistics,
                  "channels": CHANNELS, "classes": CLASSES, "input_unit": "V",
                  "input_sampling_rate_hz": 500, "input_samples": 1500,
                  "epoch_seconds": [0.5, 3.5], "internal_sampling_rate_hz": 250}
    torch.save(checkpoint, directory / "best_model.pt")
    save_csv(directory / "history.csv", history)
    np.save(directory / "validation_probabilities.npy", val_probability)
    result = {"candidate": name, "architecture": config["name"], "family": config["family"],
              "seed": seed, "best_epoch": best_epoch, "epochs_run": len(history),
              "validation_accuracy": best_key[0], "validation_loss": cross_entropy(val_probability, val_y)}
    save_json(directory / "validation_metrics.json", result)
    print(f"  最佳轮次={best_epoch}, 验证准确率={best_key[0]:.2%}", flush=True)
    return result


def read_history(path):
    """读取已保存的训练曲线，避免绘图时需要重新计算任何指标。"""
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(file)]


def main():
    """核验→保存会话划分→验证搜索→固化模型→最后一次测试→保存全部产物。"""
    parser = argparse.ArgumentParser(description="同一人的 ADS1299 跨会话 EEGNet 四分类")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "SSVEP_ML_Dataset")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--final-only", action="store_true", help="只复现最近一次实验选好的配置及成员，不重跑搜索")
    args = parser.parse_args()
    if min(args.epochs, args.patience) < 1:
        parser.error("epochs 和 patience 必须为正整数")
    torch.set_num_threads(2)
    previous = None
    if args.final_only:
        previous_dir = ROOT / (ROOT / "latest_run.txt").read_text(encoding="utf-8").strip()
        previous = torch.load(previous_dir / "selected_model/model.pt", map_location="cpu", weights_only=True)
    data = load_data(args.data_dir)
    masks, assignments = make_split(data)
    x, y = data["X"], data["y"]
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    output = args.output_dir or ROOT / "results" / f"ads1299_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    # 必须先写出所有 trial_id 的分组，再开始任何模型训练。
    membership = [{"trial_id": row["trial_id"], "session_id": row["session_id"], "run_id": row["run_id"],
                   "block_id": row["block_id"], "split": next(s for s in SPLITS if masks[s][i]),
                   "label": int(y[i]), "quality_flags": row["quality_flags"]}
                  for i, row in enumerate(data["rows"])]
    save_csv(output / "split_membership.csv", membership)
    source_paths = [ROOT / name for name in ("train_ads1299.py", "ads1299_data.py", "ads1299_models.py")]
    # 所有输入文件也留下哈希；训练结束后可证明数据包没有被改动。
    input_paths = sorted(p for p in args.data_dir.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    save_json(output / "search_plan.json", {
        "mode": "final_only" if previous else "search", "candidates": CANDIDATES if previous is None else [m["config"] for m in previous["members"]],
        "search_seed": 42, "robustness_seeds": [43, 44], "epochs": args.epochs, "patience": args.patience,
        "batch_size": 32, "sessions": assignments, "counts": {s: int(masks[s].sum()) for s in SPLITS},
        "quality_policy": "保留全部试次和全部 34 个质量标记，不以测试结果筛选",
        "selection_rule": "validation accuracy descending; ties resolved by lower validation cross-entropy",
        "device": args.device, "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "torch": str(torch.__version__), "numpy": np.__version__, "scipy": scipy.__version__,
        "source_sha256": {p.name: digest(p) for p in source_paths},
        "input_sha256": {str(p.relative_to(args.data_dir)): digest(p) for p in input_paths},
    })
    train_args = (x[masks["train"]], y[masks["train"]], x[masks["validation"]], y[masks["validation"]], args, output)
    print(f"训练/验证/测试=288/128/128，设备={args.device}；不混入其他设备数据。", flush=True)
    if previous is None:
        results = [train_candidate(config, 42, *train_args) for config in CANDIDATES]
        save_csv(output / "validation_search.csv", results)
        best = max((r for r in results if r["family"] != "spectral"),
                   key=lambda r: (r["validation_accuracy"], -r["validation_loss"]))
        config = next(c for c in CANDIDATES if c["name"] == best["architecture"])
        extra = [train_candidate(config, seed, *train_args) for seed in (43, 44)]
        results.extend(extra)
        save_csv(output / "validation_search.csv", results)
        member_names = [best["candidate"]] + [r["candidate"] for r in extra]
        ensemble_p = np.mean([np.load(output / "candidates" / name / "validation_probabilities.npy") for name in member_names], axis=0)
        ensemble_key = (float(np.mean(ensemble_p.argmax(1) == y[masks["validation"]])), -cross_entropy(ensemble_p, y[masks["validation"]]))
        selected = member_names if ensemble_key > (best["validation_accuracy"], -best["validation_loss"]) else [best["candidate"]]
        selection = {"selected_members": selected, "architecture": best["architecture"],
                     "single_validation_accuracy": best["validation_accuracy"], "ensemble_validation_accuracy": ensemble_key[0],
                     "ensemble_validation_loss": -ensemble_key[1], "rule": "只按验证集比较单模型与固定三种子概率平均"}
        baseline = "eegnet_standard_seed42"
    else:
        results = [train_candidate(m["config"], m["seed"], *train_args) for m in previous["members"]]
        selected, baseline = [r["candidate"] for r in results], None
        selection = {"selected_members": selected, "rule": "直接复现上次选好的配置及种子；仅验证集选轮次"}
        save_csv(output / "validation_search.csv", results)
    # 写入选择文件、固化权重后才取测试信号做预处理与推理。
    save_json(output / "selection_before_test.json", selection)
    final_dir = output / "selected_model"
    final_dir.mkdir()
    needed = list(dict.fromkeys(([baseline] if baseline else []) + selected))
    checkpoints = {name: torch.load(output / "candidates" / name / "best_model.pt", map_location="cpu", weights_only=True) for name in needed}
    torch.save({"members": [checkpoints[name] for name in selected], "selection": selection,
                "combination": "mean of member probabilities"}, final_dir / "model.pt")
    print(f"\n选择已固化：{selected}；现在开始最终测试。", flush=True)
    test_p = {}
    for name, checkpoint in checkpoints.items():
        model = Model(checkpoint["config"]).to(args.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_p[name] = probabilities(model, transform_saved(x[masks["test"]], checkpoint), args.device)
    evaluated = {"selected_model": np.mean([test_p[name] for name in selected], axis=0)}
    if baseline:
        evaluated["standard_baseline"] = test_p[baseline]
    summary = []
    for name, probability in evaluated.items():
        directory = output / name
        directory.mkdir(exist_ok=True)
        result = metrics(y[masks["test"]], probability)
        flagged = data["quality_flags"][masks["test"]] != ""
        result["quality_flags_retained"] = int(flagged.sum())
        result["unflagged_subset_descriptive_only"] = metrics(y[masks["test"]][~flagged], probability[~flagged])
        save_json(directory / "metrics.json", result)
        save_csv(directory / "test_predictions.csv", [
            {"trial_id": trial_id, "true_label": int(label), "predicted_label": int(p.argmax()), "quality_flags": flag,
             **{f"probability_{c}": float(p[i]) for i, c in enumerate(CLASSES)}}
            for trial_id, label, p, flag in zip(data["trial_ids"][masks["test"]], y[masks["test"]], probability, data["quality_flags"][masks["test"]])])
        member = baseline if name == "standard_baseline" else selected[0]
        history = read_history(output / "candidates" / member / "history.csv")
        plot_results(directory, history, result, checkpoints[member]["best_epoch"])
        summary.append({"model": name, **{key: result[key] for key in ("accuracy", "correct", "count", "balanced_accuracy", "macro_f1")}})
        print(f"{name}: test_acc={result['accuracy']:.2%} ({result['correct']}/{result['count']}), macro_F1={result['macro_f1']:.4f}", flush=True)
    save_csv(output / "test_summary.csv", summary)
    (ROOT / "latest_run.txt").write_text(str(output.resolve()), encoding="utf-8")
    print(f"结果目录：{output}", flush=True)


if __name__ == "__main__":
    main()
