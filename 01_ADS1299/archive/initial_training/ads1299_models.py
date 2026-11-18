"""EEGNet、对数能量汇聚变体以及可选频谱分支；模型只接收 EEG 派生数据。"""

import torch
from torch import nn


class EEGNet(nn.Module):
    """同一时空卷积骨干，可选择原版可分离卷积头或对数能量头。"""

    def __init__(self, config):
        super().__init__()
        f1 = config.get("filters", 16)
        features = f1 * 2
        kernel, dropout = config.get("kernel", 251), config.get("dropout", 0.5)
        momentum = config.get("momentum", 0.1)
        self.energy = config["family"] in ("energy", "hybrid")
        # 时间卷积学习频率响应；空间深度卷积为每个时间滤波器学习两个空间滤波器。
        self.temporal = nn.Sequential(
            nn.Conv2d(1, f1, (1, kernel), padding=(0, kernel // 2), bias=False),
            nn.BatchNorm2d(f1, eps=1e-3, momentum=momentum),
        )
        self.spatial_conv = nn.Conv2d(f1, features, (8, 1), groups=f1, bias=False)
        self.norm = nn.BatchNorm2d(features, eps=1e-3, momentum=momentum)
        self.drop = nn.Dropout(dropout)
        if not self.energy:
            # 原始 EEGNet 的 ELU、平均池化和可分离卷积，输入重采样后是 750 点。
            self.tail = nn.Sequential(
                nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
                nn.ZeroPad2d((15, 16, 0, 0)),
                nn.Conv2d(features, features, (1, 32), groups=features, bias=False),
                nn.Conv2d(features, features, 1, bias=False),
                nn.BatchNorm2d(features, eps=1e-3, momentum=momentum),
                nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(dropout), nn.Flatten(),
            )
        self.classifier = nn.Linear(features if self.energy else features * (750 // 4 // 8), 4)

    def forward(self, x):
        """能量头不保留绝对时间位置，旨在减轻不同试次相位变化的影响。"""
        x = self.norm(self.spatial_conv(self.temporal(x)))
        if self.energy:
            x = self.drop(torch.log(x.square().mean((2, 3)).clamp_min(1e-6)))
        else:
            x = self.tail(x)
        return self.classifier(x)

    @torch.no_grad()
    def constrain_weights(self):
        """约束空间滤波器范数为 1，分类器为 0.5（能量头）或 0.25（原版头）。"""
        for weight, limit in ((self.spatial_conv.weight, 1.0), (self.classifier.weight, 0.5 if self.energy else 0.25)):
            norm = weight.flatten(1).norm(dim=1).reshape((-1,) + (1,) * (weight.ndim - 1)).clamp_min(1e-8)
            weight.mul_((limit / norm).clamp(max=1))


class SpectralHead(nn.Module):
    """四个目标共用同一频谱评分器；只输入由 EEG 计算的 8 通道×3 谐波特征。"""

    def __init__(self, hidden=0):
        super().__init__()
        if hidden:
            self.scorer = nn.Sequential(nn.Linear(24, hidden), nn.ELU(), nn.Dropout(0.25), nn.Linear(hidden, 1))
        else:
            self.scorer = nn.Linear(24, 1, bias=False)
            nn.init.constant_(self.scorer.weight, 1 / 24)
        self.bias = nn.Parameter(torch.zeros(4))

    def forward(self, spectrum):
        """返回四个候选目标的 logits，不需要单个试次的已知刺激频率。"""
        return self.scorer(spectrum).squeeze(-1) + self.bias


class Model(nn.Module):
    """统一各候选的调用接口；纯频谱模型只用于消融比较。"""

    def __init__(self, config):
        super().__init__()
        self.raw = EEGNet(config) if config["family"] != "spectral" else None
        self.spectral = SpectralHead(config.get("hidden", 0)) if config["family"] in ("hybrid", "spectral") else None
        self.raw_scale = config.get("raw_scale", 1.0)

    def forward(self, x, spectrum):
        """原版/能量头只用时域信号；融合配置再加入频谱评分。"""
        logits = self.raw(x) * self.raw_scale if self.raw is not None else 0
        return logits + self.spectral(spectrum) if self.spectral is not None else logits

    def constrain_weights(self):
        """每次参数更新后调用时域骨干的范数约束。"""
        if self.raw is not None:
            self.raw.constrain_weights()
