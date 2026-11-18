# SSVEP 两种设备的实验总入口

已按设备统一目录命名，并把当前运行工程与早期结果分开。两个设备的数据均按同一个人的不同采集解释；设备内部训练结果与跨设备外部复测结果分别保存。

| 设备 | 当前工程 | 当前模型 | 结果含义 |
| --- | --- | --- | --- |
| 自制 ADS1299 | [01_ADS1299/current](01_ADS1299/current/README.md) | EEGNet-16,2，最大110轮，选中109轮 | 本设备留出会话128/128；对g.Nautilus外部总集114/339 |
| g.Nautilus | [02_gNautilus/current](02_gNautilus/current/README.md) | 时空卷积＋对数能量头，历史选中231轮 | 本设备三次采集合并训练，测试86/95 |

11月17日新增ADS1299会话已用冻结的110轮模型单独测试：**88/128=68.75%，宏F1=0.6723**。全部29个质量标记试次均保留，没有重新训练或重新拟合标准化。见 [新增会话完整报告](01_ADS1299/current/results/run_110epochs/ads1299_new_session_20261117_160423/REPORT.md)。该会话的标签顺序仍与旧run相同，尚不是更换随机刺激顺序后的验证。

## 目录

```text
SSVEP/
├── README.md
├── 目录整理记录_20261117.json        # 每个原文件的新旧位置和SHA256
├── 01_ADS1299/
│   ├── README.md                   # ADS1299导航
│   ├── current/                    # 推荐使用的独立110轮工程
│   │   ├── code/                   # 中文注释代码
│   │   ├── data/                   # ADS数据包及g.Nautilus外部复测副本
│   │   ├── models/                 # 当前110轮及保留的300轮模型
│   │   ├── results/                # 内部测试和外部复测分别保存
│   │   ├── docs/                   # 阅读说明与历史泄漏审计
│   │   └── run_all.py
│   └── archive/
│       └── initial_training/       # 最初十组搜索、审计、原数据包及旧入口
└── 02_gNautilus/
    ├── README.md                   # g.Nautilus导航
    ├── current/                    # 三次采集合并的模型与完整原数据集
    │   ├── ML_DL_Dataset/
    │   ├── results/eegnet_merged_search/
    │   ├── train_merged_final.py
    │   └── predict_merged_eegnet.py
    └── archive/
        └── per_recording_baseline/ # 早期逐文件训练结果，已纠正“三位受试者”解释
```

## 常用操作

在本总目录运行 ADS1299 110轮训练及外部复测，会创建新实验名：

```bash
cd 01_ADS1299/current
.venv/bin/python run_all.py
```

运行 g.Nautilus 本设备合并模型训练：

```bash
cd 02_gNautilus/current
.venv/bin/python train_merged_final.py
```

两个项目的 `.venv` 入口已按新路径修复。当前 ADS1299 默认模型仍为 `run_110epochs`；g.Nautilus原有模型和参数保持原样。本次只整理文件，不重新训练或改变已有模型权重。

## 重要区别

- ADS1299源数据为500Hz、单位V、0.5～3.5秒；g.Nautilus源数据为250Hz、单位μV、0～4秒。不能直接混用单位或当成相同预处理的数据。
- `01_ADS1299/current/results/run_110epochs/g_nautilus_external/` 是 **ADS1299模型对另一设备的复测**，不是g.Nautilus本设备模型的成绩。339个外部试次中的95个历史测试子集不应重复累加。
- 同一刺激顺序跨run重复的审计限制仍然保留。内部满分不等于已证明跨设备或新随机顺序的泛化能力。
- 为保留独立工程的完整性，当前ADS目录中的数据副本和历史原包均保留；没有通过链接合并或删除原数据。

历史日志、来源清单、运行计划中的旧绝对路径作为实验记录保留，不应把它们当成现在的运行路径。新位置以本页及设备导航为准。
