"""EEGNet及融合变体：标准时域头、对数能量头和四候选频谱评分器。

网络分支由配置family决定，全部只接收信号衍生特征，不读取真实标签或试次元数据。
以下维度注释以标准16滤波器配置为例；8滤波器配置的特征图数量相应减半。
"""

import torch
from torch import nn


class EEGNet(nn.Module):
    """同一时空卷积骨干，可选择原版可分离卷积头或对数能量头。"""

    def __init__(self, config):
        super().__init__()
        # 示例 F1=16、深度倍数D=2，因此空间卷积后有32个特征图。
        f1 = config.get("filters", 16)
        features = f1 * 2
        kernel, dropout = config.get("kernel", 251), config.get("dropout", 0.5)
        momentum = config.get("momentum", 0.1)
        self.energy = config["family"] in ("energy", "hybrid")
        # 时间卷积学习频率响应；空间深度卷积为每个时间滤波器学习两个空间滤波器。
        # 输入 [N,1,8,750] → [N,16,8,750]，251点时间核约覆盖1秒。
        # 同长度补零保持750点；bias=False避免与后续BatchNorm的偏置重复。
        self.temporal = nn.Sequential(
            nn.Conv2d(1, f1, (1, kernel), padding=(0, kernel // 2), bias=False),
            nn.BatchNorm2d(f1, eps=1e-3, momentum=momentum),
        )
        # groups=16使每个时间滤波器独立学习跨8通道的两个空间滤波器。
        # 这里 [N,16,8,750] → [N,32,1,750]，不混入通道名称或类别元数据。
        self.spatial_conv = nn.Conv2d(f1, features, (8, 1), groups=f1, bias=False)
        self.norm = nn.BatchNorm2d(features, eps=1e-3, momentum=momentum)
        self.drop = nn.Dropout(dropout)
        if not self.energy:
            # 原始 EEGNet 的 ELU、平均池化和可分离卷积，输入重采样后是 750 点。
            # 第一次平均池化：750//4=187；深度卷积只在各特征图内沿时间处理。
            # 32点偶数核前后分别补15和16点，保持187点；1×1卷积再融合特征图。
            # 第二次平均池化：187//8=23，最后展开为32×23=736维。
            self.tail = nn.Sequential(
                nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
                nn.ZeroPad2d((15, 16, 0, 0)),
                nn.Conv2d(features, features, (1, 32), groups=features, bias=False),
                nn.Conv2d(features, features, 1, bias=False),
                nn.BatchNorm2d(features, eps=1e-3, momentum=momentum),
                nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(dropout), nn.Flatten(),
            )
        # 示例标准分支把736维特征映射到4个logits；交叉熵内部再做log-softmax。
        # 不覆盖PyTorch默认初始化，保证与原选型实验的随机初始化严格一致。
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
        # standard/energy配置中 spectral=None；hybrid配置另加频谱评分，raw_scale来自配置；spectrum 只是全零占位，不参与输出。
        logits = self.raw(x) * self.raw_scale if self.raw is not None else 0
        return logits + self.spectral(spectrum) if self.spectral is not None else logits

    def constrain_weights(self):
        """每次参数更新后调用时域骨干的范数约束。"""
        if self.raw is not None:
            self.raw.constrain_weights()
