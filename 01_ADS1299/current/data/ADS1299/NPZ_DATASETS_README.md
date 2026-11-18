# ADS1299 NPZ 数据清单

目录：`E:\ChenXZ\Desktop\data\SSVEP_ML_Dataset`。按原文件复制并核对SHA256，未重新滤波、转换单位、改标签或划分训练测试集。

| 文件名 | X维度 | 采样率 | 单位 | A/B/C/D试次数 |
|---|---|---:|---|---|
| [ADS1299_20261116_192641_allTrials_500Hz_0p5to3p5s_V.npz](ADS1299_20261116_192641_allTrials_500Hz_0p5to3p5s_V.npz) | (96, 8, 1500) | 500 Hz | V | 24/24/24/24 |
| [ADS1299_20261116_200253_allTrials_500Hz_0p5to3p5s_V.npz](ADS1299_20261116_200253_allTrials_500Hz_0p5to3p5s_V.npz) | (192, 8, 1500) | 500 Hz | V | 48/48/48/48 |
| [ADS1299_20261116_204128_allTrials_500Hz_0p5to3p5s_V.npz](ADS1299_20261116_204128_allTrials_500Hz_0p5to3p5s_V.npz) | (128, 8, 1500) | 500 Hz | V | 32/32/32/32 |
| [ADS1299_20261116_210748_allTrials_500Hz_0p5to3p5s_V.npz](ADS1299_20261116_210748_allTrials_500Hz_0p5to3p5s_V.npz) | (128, 8, 1500) | 500 Hz | V | 32/32/32/32 |
| [ADS1299_20261116_combined4Recordings_allTrials_500Hz_0p5to3p5s_V.npz](ADS1299_20261116_combined4Recordings_allTrials_500Hz_0p5to3p5s_V.npz) | (544, 8, 1500) | 500 Hz | V | 136/136/136/136 |

## 命名与版本

- ADS1299：日期/采集开始时刻、allTrials、500Hz、0p5to3p5s、V；combined4Recordings为四份记录合并版。
- 单次与合并版包含相同试次，不能一起叠加作为新增样本；合并版独立试次总数为544。
- 软件同步尚未光电/硬件校准，34次显示时序异常标记保留。
- 每文件对应的来源、采样率、单位、通道、类别数和处理说明见npz_metadata/*.metadata.json；逐试次对应表见同名*.trials.csv。总清单为npz_datasets_manifest.json。
- 本包未标准化、未自动剔除或划分；开发历史及验证用标准化版保留在原results目录。

## 读取示例

```python
from pathlib import Path
import numpy as np
p = Path('.') / 'ADS1299_20261116_combined4Recordings_allTrials_500Hz_0p5to3p5s_V.npz'  # 在数据包目录运行
with np.load(p, allow_pickle=False) as d:
    X, y = d['X'], d['y']
    print(X.shape, y.shape, d['sfreq'], d['unit'])
```

如需使用原始物理幅值，请按unit字段解释；文件命名不会改变数组单位。训练前的标准化只能使用训练集合拟合。
