# ADS1299 SSVEP 机器学习数据交接包

本文件夹仅包含自制ADS1299八通道采集系统的数据：**4份单次采集＋1份合并版**，以及10份配套元数据/试次表、来源配置和核验记录。可整体复制到其他工作目录。

先读[机器学习使用说明](ML_USAGE.md)，文件名、维度及类别数见[NPZ清单](NPZ_DATASETS_README.md)。

| 记录 | X维度 | 各类试次数 |
|---|---|---:|
| 19:26:41 | (96,8,1500) | 24 |
| 20:02:53 | (192,8,1500) | 48 |
| 20:41:28 | (128,8,1500) | 32 |
| 21:07:48 | (128,8,1500) | 32 |
| 四次合并 | **(544,8,1500)** | **136** |

采样率统一500 Hz，物理单位V，软件Marker后的右开窗口[0.5,3.5)秒，未标准化或划分。合并版已经包含四份单次版全部试次，不能再叠加扩充样本。

通道顺序为PO7、PO3、POz、PO4、PO8、O1、Oz、O2。标签0/1/2/3对应A/B/C/D，名义频率6.5/9/11.5/14.5 Hz。

```text
SSVEP_ML_Dataset/
├── README.md / ML_USAGE.md
├── load_dataset.py
├── ADS1299_*.npz                   # 5份
├── NPZ_DATASETS_README.md
├── npz_datasets_manifest.json
├── package_validation.json
├── npz_metadata/                  # 每份NPZ同名metadata.json及trials.csv
└── provenance/                    # ADS1299配置、环境、核验与来源哈希
```

在本目录使用已有ssvep环境验证全部文件：

```powershell
conda run -n ssvep python -X utf8 load_dataset.py
```

需要读取某份文件时加 `--file 文件名.npz`。脚本只核验与读取，不训练模型。

34个显示时序异常试次已标记并保留；软件同步未经光电/硬件校准，当前PSD/SNR尚不足以确认四类稳定锁定响应。应先复核质量、明确分组划分，再开展后续机器学习。
