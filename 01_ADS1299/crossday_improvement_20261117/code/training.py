"""EEGNet候选配置的训练循环；仅输入训练和验证集合，不接收测试集合。"""

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


def set_seed(seed):
    """固定权重初始化、训练打乱和 Dropout；不同候选使用同一个初始种子。"""
    # 本次所有候选使用 seed=42；设置覆盖 Python、NumPy、CPU 与所有 GPU。
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
    # 每次从相同随机状态开始；这里没有加载历史模型权重的操作。
    set_seed(seed)
    name = f"{config['name']}_seed{seed}"
    directory = output_dir / "candidates" / name
    directory.mkdir(parents=True)
    pairs, statistics = fit_preprocessing(train_v, val_v, config)
    train_pair, val_pair = pairs
    # 训练集打乱批次，验证集保持原试次记录顺序；每批 32 个试次不增加样本数。
    loader = DataLoader(TensorDataset(*train_pair, torch.from_numpy(train_y)),
                        batch_size=32, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed))
    # 创建全新的 EEGNet；权重初始化次序沿用此前选型实验，以便严格复现。
    model = Model(config).to(args.device)
    model.constrain_weights()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    best_key, best_state, best_epoch, stale = (-1, -float("inf")), None, 0, 0
    history = []
    print(f"\n候选 {name}", flush=True)
    for epoch in range(1, args.epochs + 1):
        # 训练模式启用 Dropout，并且只在训练批次更新 BatchNorm 运行统计量。
        model.train()
        total_loss = 0.0
        for batch_x, batch_spectrum, batch_y in loader:
            batch_x, batch_spectrum, batch_y = batch_x.to(args.device), batch_spectrum.to(args.device), batch_y.to(args.device)
            # 清空上一批梯度→计算四分类交叉熵→反向传播→限制梯度→更新参数。
            # 仅对训练批次增强；幅度扰动保持目标频率，验证及推理完全不增强。
            if config.get("augment", False):
                gain = torch.empty((len(batch_x), 1, 8, 1), device=batch_x.device).uniform_(0.8, 1.2)
                batch_x = batch_x * gain + torch.randn_like(batch_x) * 0.03
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(batch_x, batch_spectrum), batch_y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{name} 出现非有限损失")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            # 更新后施加最大范数约束，限制空间滤波器和分类器的权重大小。
            model.constrain_weights()
            total_loss += loss.item() * len(batch_y)
        # probabilities 内部切换为评估模式，验证数据不会更新 BatchNorm。
        val_probability = probabilities(model, val_pair, args.device)
        val_loss = cross_entropy(val_probability, val_y)
        accuracy = float(np.mean(val_probability.argmax(1) == val_y))
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_y),
                        "validation_loss": val_loss, "validation_accuracy": accuracy})
        # Python 元组按顺序比较：先最大化验证准确率，同分时最小化验证损失。
        key = (accuracy, -val_loss)
        if key > best_key:
            best_key, best_epoch, stale = key, epoch, 0
            # clone 保留独立的最佳状态快照，防止后续训练把它一同改写。
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 50 == 0:
            print(f"  epoch={epoch}, val_acc={accuracy:.2%}, val_loss={val_loss:.4f}", flush=True)
        # 持续没有更优验证分数才早停，测试表现始终不参与这个条件。
        if stale >= args.patience:
            break
    model.load_state_dict(best_state)
    val_probability = probabilities(model, val_pair, args.device)
    # 检查点同时保存训练集标准化参数和输入定义，保证推理无需重新拟合。
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

