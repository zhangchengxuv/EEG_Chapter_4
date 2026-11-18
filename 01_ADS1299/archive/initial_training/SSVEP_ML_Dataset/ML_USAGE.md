# ADS1299 数据的机器学习使用说明

## 1. 数据版本

本包只含ADS1299数据。研究全部记录时读取 `ADS1299_20261116_combined4Recordings_allTrials_500Hz_0p5to3p5s_V.npz`，得到544个试次；也可分别读取四次采集文件。合并版已经包含四份单次版的全部数据，不能一起叠加。

数据未标准化、未划分、未增强、未训练。`allTrials`表示保留所有完整窗口，不等于无伪迹。四类各136次，其中34次具有显示时序异常标记，当前仍保留。

## 2. 数组与标签

`X.shape=(n_trials,8,1500)`，`y.shape=(n_trials,)`，采样率500 Hz，物理单位V。每窗为软件Marker后[0.5,3.5)秒。MNE与NPZ物理单位一致；显示为μV时乘1e6，不能误将当前数组解释为μV。

通道键为 `channel_names`，固定顺序PO7、PO3、POz、PO4、PO8、O1、Oz、O2。`trial_ids`是跨文件关联主键。NPZ用 `allow_pickle=False` 读取。需要CNN输入轴时可在加载后使用 `X[:,None,:,:]`，这不会增加试次数。

| y | 类别 | 名义频率 |
|---:|---|---:|
| 0 | A | 6.5 Hz |
| 1 | B | 9 Hz |
| 2 | C | 11.5 Hz |
| 3 | D | 14.5 Hz |

标签表示提示注视目标，没有静息或拒识类别，也不是机器人执行结果。频率来自真实刺激配置，未通过光学频率检测独立校准。

## 3. 处理与质量限制

实际采用1–80 Hz FIR零相位处理和50 Hz陷波，保留原参考，无ICA、基线校正、通道插值或自动伪迹剔除。零相位为离线处理，不代表已验证在线因果滤波。

软件同步基于同一接收时钟的批次末样本插值，原Marker索引与重估索引均保留；未知串口传输/缓冲延迟未经校准。窗口裕量不能证明物理屏幕刺激边界准确。原始数据存在较大DC偏置、漂移与工频干扰；处理后PSD/SNR不构成四类稳定SSVEP或识别效果的验证。

每份NPZ的质量标记和来源见 `npz_metadata/<同名>.metadata.json` 和 `.trials.csv`；表格已按该NPZ的trial_ids排列。来源参数、核心库版本和既有核验见provenance/。其中历史绝对路径仅用于追溯，读取本包不依赖旧路径。

## 4. 分组划分与防止泄漏

目前所有split均为unsplit。合并版含5个刺激run，第二次连续采集包含2个run。优先按 `run_ids` 或录制session整组划分。若使用block，应以 `(run_id,block_id)` 组成唯一组键，避免不同run复用block号。

整组划分会使比例近似。先确定并保存训练/验证/测试的trial_ids和组归属，再训练。不能将同一原始试次的窗口、不同版本或高度重叠窗口分散到训练与测试。

日志声明的subject编号均为001，真实归属仍需确认；按run/session划分不能称为跨受试者验证。

标准化、特征选择和降维均只允许在训练集合拟合，验证/测试只应用训练得到的变换。不得使用测试集反复选择滤波、特征、阈值或模型超参数。后续如切窗或增强，应仅作用于训练集合，并保留原trial_id与组信息。

## 5. 读取示例

在本文件夹运行：

```python
from pathlib import Path
import numpy as np
from load_dataset import load_dataset

p = Path('.')/'ADS1299_20261116_combined4Recordings_allTrials_500Hz_0p5to3p5s_V.npz'
d = load_dataset(p)
X, y, trials = d['X'], d['y'], d['trials']
groups = d['run_ids']
trial_ids = d['trial_ids']
print(X.shape, y.shape, d['unit'].item())
print(np.unique(groups, return_counts=True))
# 根据实验方案固定整组划分，再仅在训练集合拟合数据变换。
```

命令行执行 `python -X utf8 load_dataset.py` 可核验5份文件的哈希、维度、单位、标签和试次对应关系；不会自动生成划分或训练模型。

## 6. 后续任务交接文字

> 数据包位于E:/ChenXZ/Desktop/data/SSVEP_ML_Dataset，仅含ADS1299数据。请先阅读README.md、ML_USAGE.md和清单，核对单位、质量标记和试次ID，固定按run/session的分组划分后开展我另行指定的模型任务。不要重复叠加合并版与单次版，不在全体数据上拟合标准化，不用测试集反复调参，不修改数据包或原始文件，不预设或编造准确率。
