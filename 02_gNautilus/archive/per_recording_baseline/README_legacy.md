# SSVEP EEGNet 四分类

使用 PyTorch 实现 EEGNet-8,2，分别为 S01、S02、S03 训练独立模型。主代码为 `train_eegnet.py`，各步骤均有中文注释；不依赖 pandas、scikit-learn 或额外 EEG 模型库。

## 数据与任务

- 输入：`ML_DL_Dataset/NumPy/Sxx_accepted_0to4_uv.npz` 中的 `X`，形状为 `[试次,8,1000]`，250 Hz、0～4 秒。
- 固定通道：PO7、PO3、POz、PO4、PO8、O1、Oz、O2。
- 原始输入单位已是 μV，代码不会乘以 `1e6`。
- 标签：0→A→6.5 Hz，1→B→9.0 Hz，2→C→11.5 Hz，3→D→14.5 Hz；表示注视目标类别。
- 通过 NPZ 的 `trial_ids` 与 `Metadata/Sxx_trials.csv` 的 `trial_id` 一对一关联，按元数据 `split` 划分，并检查其与区组号一致。不读取 `Sxx_split_indices.json` 的完整数组索引。
- 区组 1～5 训练、区组 6 验证、区组 7～8 测试。三人的数量分别为 77/10/32、78/16/31、47/16/32。
- 仅使用时序信号作为模型输入；trial_id、标签、刺激频率、区组等只用于校验、划分和记录。
- 不混入 all 版本或 0.5～4 秒版本，不新增质量筛选，不删除测试试次。

数据已有采集端 30 Hz 低通、50 Hz 陷波和离线 4 Hz 零相位高通。这里不重复滤波，也不将数据描述为 4～90 Hz 宽带数据。accepted 来自既有 500 μV 峰峰值规则，不代表人工确认完全无伪迹。

## 运行

本项目已在 `.venv` 中实际训练完成，直接运行：

```bash
.venv/bin/python train_eegnet.py
```

程序默认使用可用 GPU，在 `results/eegnet_日期_时间/` 创建新目录，防止覆盖已有实验。指定受试者或 CPU：

```bash
.venv/bin/python train_eegnet.py --subjects S01 --device cpu
```

本机 conda `pytorch` 环境能识别 GTX 1650，但原有 NumPy 安装导致 `torch.from_numpy` 报 `expected np.ndarray (got numpy.ndarray)`。项目内 `.venv` 复用该环境的 PyTorch，并单独安装干净的 NumPy 1.26.4；没有修改原 conda 环境。若需要重建：

```bash
conda activate pytorch
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install --ignore-installed --no-deps numpy==1.26.4
.venv/bin/python train_eegnet.py
```

在其他环境安装依赖时，可参考 `requirements.txt`。本次实际环境为 Python 3.9.21、PyTorch 2.7.0+cu126、NumPy 1.26.4、Matplotlib 3.9.4。

## 代码流程与模型

1. `load_subject`：核对单位、通道、采样率、窗口、标签、accepted 标记、trial_id 唯一性和区组划分。所有受试者先校验，再开始训练。
2. `fit_standardizer`：仅用训练试次，沿试次与时间维度计算每通道均值和标准差，形状为 `[1,8,1]`。训练、验证和测试均使用这组统计量。
3. `EEGNet`：输入由 `[N,8,1000]` 增加一个维度成为 `[N,1,8,1000]`。8 个时间滤波器、每个时间滤波器 2 个空间滤波器、16 个输出特征图。
4. `train_subject`：训练阶段使用 Adam 和交叉熵；仅打乱训练批次。验证时关闭 Dropout，BatchNorm 使用训练期间累计的统计量。
5. 以最小验证交叉熵选择检查点。连续 60 轮未改善则早停，最多 300 轮。模型固化后进行最终测试，保存全部测试试次的预测和概率。

固定参数：学习率 0.001、批量大小 16、Dropout 0.5、随机种子 42。空间卷积最大范数 1.0，分类器每类权重最大范数 0.25。没有特征选择、数据增强或根据测试准确率调整的步骤。

模型采用 EEGNet 的时间卷积、深度空间卷积和可分离卷积结构。针对 250 Hz，首层时间核使用 125 点，可分离时间核使用 32 点；池化为 4 和 8。这是适配当前采样率的固定基线配置，并非经过搜索的最优配置。

参考：[EEGNet 作者实现](https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py)、[Lawhern 等，EEGNet 论文](https://arxiv.org/abs/1611.08024)。

## 已完成的实验

结果目录：`results/eegnet_baseline_seed42/`。

| 受试者 | 训练/验证/测试 | 实际训练轮数 | 选中轮次 | 测试准确率 | 平衡准确率 | 宏 F1 |
| --- | --- | --- | --- | --- | --- | --- |
| S01 | 77/10/32 | 71 | 11 | 28.12% | 28.12% | 0.2625 |
| S02 | 78/16/31 | 96 | 36 | 25.81% | 27.68% | 0.2049 |
| S03 | 47/16/32 | 84 | 24 | 31.25% | 31.25% | 0.3032 |

三位受试者准确率算术平均为 **28.39%**；合并全部 95 个测试试次为 **27/95=28.42%**。这些值接近四分类均匀随机猜测的 25% 参考水平，当前模型尚未显示可靠分类能力。训练损失下降而验证损失未同步下降，呈现过拟合迹象；仅凭本次结果不能确定其全部原因。

以上是固定区组划分、单个随机种子的受试者内结果，不代表跨受试者泛化能力，也不能据此认为模型可可靠控制机器人。未根据测试结果修改模型参数、筛选阈值或试次集合。

## 输出文件

结果根目录保存 `config.json`（参数、运行版本、输入及代码 SHA256）、`summary.csv`（逐人结果）、`aggregate.json`（汇总）。每位受试者子目录保存：

| 文件 | 内容 |
| --- | --- |
| `best_model.pt` | 最佳权重、模型定义参数、训练集标准化参数、通道和标签定义 |
| `standardization.npz` | 训练集每通道均值和标准差；均值/标准差单位为 μV |
| `split_membership.csv` | 关联后的 trial_id、区组、标签及所属划分，便于审计 |
| `history.csv` | 每轮训练损失、验证损失和验证准确率 |
| `metrics.json` | 选中轮次、各划分类别数量、测试各类指标与混淆矩阵 |
| `test_predictions.csv` | 所有保留测试试次的真实标签、预测标签与四类概率 |
| `training_curves.png` | 训练/验证损失与验证准确率曲线 |
| `test_confusion_matrix.png` | 测试混淆矩阵：行是真实类别，列是预测类别 |

训练损失在启用 Dropout 的训练批次上累计，验证损失在评估模式计算，图中已标明。最佳模型按验证损失选取，因此不一定对应最高验证准确率。

## 加载模型

下面示例中的 `new_x` 必须是同一受试者、相同通道顺序和预处理定义、单位为 μV 的 `[N,8,1000]` 数组。每个受试者使用自己的模型和标准化参数。

```python
import numpy as np
import torch
from train_eegnet import EEGNet

# 读取保存的权重与训练集统计量。
checkpoint = torch.load(
    "results/eegnet_baseline_seed42/S01/best_model.pt",
    map_location="cpu", weights_only=True,
)
model = EEGNet(**checkpoint["model_config"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 这里假设 new_x 已由调用者提供；不重新拟合均值或标准差。
mean = checkpoint["standardization_mean"].numpy()
std = checkpoint["standardization_std"].numpy()
normalized = np.ascontiguousarray((new_x.astype(np.float32) - mean) / std)
with torch.no_grad():
    probabilities = model(torch.from_numpy(normalized[:, None])).softmax(dim=1)
    labels = probabilities.argmax(dim=1)  # 0～3，分别对应注视 A/B/C/D。
```

## 检查

```bash
.venv/bin/python -m unittest discover -s tests -v
```

检查覆盖 CSV 逆序后关联仍正确、重复 trial_id 被拒绝、测试区组误标为训练被拒绝，以及模型前向/反向传播和范数约束。检查对数据的改动仅发生在临时副本中。
