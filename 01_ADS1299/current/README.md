# 选定 EEGNet 的独立重跑与跨设备外部复测

整理后的入口为 `01_ADS1299/current/`，原目录名为 `ADS1299/EEGNet_Selected_20261117/`。本页所有运行命令均在当前目录执行。

本目录已实际完成一次**从随机初始化开始的重新训练**。代码、ADS1299 数据包、g.Nautilus 外部数据、模型、结果、日志、参数、中文字体及历史审计材料均保存在本目录内；运行代码不依赖父工程中的代码、数据或旧模型。

当前选定模型为 **EEGNet-16,2 长时间核版本**，使用种子42，最大训练轮数已按要求降为 **110**，仅以验证准确率优先、交叉熵次优选择轮次。当前实验名为 **run_110epochs**，实际训练110轮，选中第 **109** 轮；默认预测入口已指向这一模型。旧300轮实验 `run_001` 的模型、日志、结果及原配置均保留。

## 当前110轮结果（run_110epochs）

| 评估范围 | 正确/总数 | 准确率 | 宏 F1 |
| --- | ---: | ---: | ---: |
| ADS1299 训练集，评估模式 | 288/288 | 100.00% | 1.0000 |
| ADS1299 验证会话 | 128/128 | 100.00% | 1.0000 |
| ADS1299 内部留出测试会话 | **128/128** | **100.00%** | **1.0000** |
| ADS1299 11月17日新增独立会话 | **88/128** | **68.75%** | **0.6723** |
| g.Nautilus 跨设备历史外部复测总集 | **114/339** | **33.63%** | **0.2281** |
| 其中 g.Nautilus 原历史测试子集 | **34/95** | **35.79%** | **0.2444** |

本轮从随机初始化重新训练；前110轮的原始训练记录与旧300轮实验前110轮逐项一致。轮数由用户指定为110，未依据外部成绩自动选择训练预算。110轮结果位于 `results/run_110epochs/`，模型位于 `models/run_110epochs/model.pt`。

新增的 `ADS1299_20261117_160423_allTrials_500Hz_0p5to3p5s_V.npz` 已单独测试，模型与原训练统计量保持冻结，128个试次及29个质量标记全部保留。详见 [新增会话测试报告](results/run_110epochs/ads1299_new_session_20261117_160423/REPORT.md)。新旧试次和完整信号无重复，但128个标签顺序仍与旧完整run相同，固定刺激顺序的限制尚未解决。

新增会话测试代码为 `code/evaluate_new_session.py`；数据副本独立保存在 `data/independent_tests/`，没有加入原训练集合。已执行的测试名称为 `20261117_160423`，结果目录不会被覆盖。如需复跑，请指定新的测试名称，例如：

```bash
.venv/bin/python code/evaluate_new_session.py \
  --input data/independent_tests/20261117_160423/ADS1299_20261117_160423_allTrials_500Hz_0p5to3p5s_V.npz \
  --session-name 20261117_160423_repeat --run-name run_110epochs
```

## 历史300轮结果（run_001，保留）

旧实验最多300轮，选中第293轮；完整训练历史及状态张量曾与最初选型结果核对一致。以下表格专指旧300轮实验：

| 评估范围 | 正确/总数 | 准确率 | 宏 F1 |
| --- | ---: | ---: | ---: |
| ADS1299 训练集，评估模式 | 288/288 | 100.00% | 1.0000 |
| ADS1299 验证会话 | 128/128 | 100.00% | 1.0000 |
| ADS1299 内部留出测试会话 | **128/128** | **100.00%** | **1.0000** |
| g.Nautilus 跨设备历史外部复测总集 | **114/339** | **33.63%** | **0.2270** |
| 其中 g.Nautilus 原历史测试子集 | **35/95** | **36.84%** | **0.2533** |

95个试次是339个外部试次的子集，不能相加。g.Nautilus 的 S01/S02/S03 均按同一个人的不同采集解释；旧300轮实验分别为36/119、44/125、34/95。

内部满分没有直接迁移到另一设备的数据上：外部总集预测只出现A/B，C/D召回均为0。采集硬件、参考与滤波等分布差异均可能影响结果；本实验没有把差异归因为某一种设备优劣，也没有依据外部成绩继续调参。

外部真实类别数量为A/B/C/D=85/84/88/82，预测数量为237/102/0/0；图的右两列为零是实际预测计数，不是绘图漏列。详见 [外部混淆矩阵解读](results/run_001/g_nautilus_external/结果解读.md)。

**已有实验限制仍然保留：** ADS1299 不同run重复相同刺激顺序，仅用run内试次位置查表也能满分。网络没有输入位置，因此这不是它实际利用顺序的直接证据，但内部100%不能解释为已经排除所有混淆或能泛化到新顺序。g.Nautilus也是已有历史数据，本次属于冻结模型的跨设备复测，并非新采集盲测。历史审计见 `docs/leakage_audit/`。

## 目录结构

```text
current/
├── run_all.py                         # 一键从头训练，然后外部复测
├── selected_config.json               # 已选模型、训练参数及外部协议
├── requirements.txt                  # 实际使用的关键依赖版本
├── code/
│   ├── train.py                      # 七步训练主流程，完整中文说明
│   ├── training.py                   # 原训练循环，保持数学操作一致
│   ├── ads1299_models.py             # 网络与逐层维度注释
│   ├── ads1299_data.py               # 数据核验、划分、预处理和指标
│   ├── test_gnautilus.py             # 仅推理的跨设备外部复测
│   ├── predict.py                    # 对新ADS1299数据无标签预测
│   └── plots.py                      # 中文图题、坐标轴和图例
├── data/
│   ├── ADS1299/                      # 原始交接包的完整复制
│   └── gNautilus/                    # 三份accepted 4秒NPZ及对应元数据
├── models/run_110epochs/             # 当前110轮模型
│   ├── model.pt                     # 推荐加载：权重、统计量及输入定义
│   ├── best_checkpoint.pt           # 本轮最佳检查点
│   ├── weights.pt                   # 单独的state_dict
│   ├── standardization.npz          # 仅ADS1299训练集拟合的统计量
│   └── model_info.json              # 输入、通道、类别及选中轮次
├── models/run_001/                   # 保留的旧300轮模型
├── results/run_110epochs/            # 当前110轮结果
│   ├── ads1299/                     # 分组、历史、指标、逐试次预测和中文图
│   ├── g_nautilus_external/          # 独立外部协议、指标、预测与中文图
│   └── run_info.json
├── results/run_001/                  # 保留的旧300轮结果及重载验证记录
├── logs/                            # 实际训练与外部复测的完整日志
├── docs/                            # 代码阅读顺序和历史审计
├── provenance/                      # 来源、原代码快照、哈希及复现核验
├── assets/fonts/                    # 附带中文字体及许可
└── tests/                           # 五项有针对性的复现检查
```

## 重新运行

在本目录中一键启动新的110轮实验，不覆盖任何已有实验：

```bash
.venv/bin/python run_all.py
```

也可以显式指定一个尚不存在的名称：

```bash
.venv/bin/python run_all.py --run-name run_002 --device cuda
```

没有GPU时改用 `--device cpu`。每轮结果位于 `results/<实验名>/`，模型位于 `models/<实验名>/`，两个步骤的日志分别保存在 `logs/`。一键完整完成后，`latest_run.txt`记录最近实验名称。

如果分步运行，请使用同一个新名称：

```bash
.venv/bin/python code/train.py --run-name run_manual --device cuda
.venv/bin/python code/test_gnautilus.py --run-name run_manual --device cuda
```

分步运行时，预测请显式指定相应模型路径；`latest_run.txt`仅由一键入口更新。

## 对新信号预测

```bash
.venv/bin/python code/predict.py \
  --input 你的ADS1299新信号.npz \
  --model models/run_110epochs/model.pt \
  --output results/new_predictions.csv
```

输入应为同一定义的 `X=[N,8,1500]`、单位V、500Hz、Marker后 `[0.5,3.5)` 秒，通道顺序为PO7、PO3、POz、PO4、PO8、O1、Oz、O2。`trial_ids`可选。真实标签、刺激频率和其他类别元数据均不需要，也不会作为预测输入。

标签仍表示提示注视目标：0→A→6.5Hz，1→B→9Hz，2→C→11.5Hz，3→D→14.5Hz，不是机器人实际动作。

## 数据划分及两个设备的区别

ADS1299只读取 `data/ADS1299/ADS1299_20261116_combined4Recordings_allTrials_500Hz_0p5to3p5s_V.npz` 的544个独立试次，不再叠加同包的四份单次文件。按完整会话固定为：

| 采集开始时间（2026-11-16） | 训练 | 验证 | 内部测试 |
| --- | ---: | ---: | ---: |
| 19:26:41 | 96 | 0 | 0 |
| 20:02:53 | 192 | 0 | 0 |
| 20:41:28 | 0 | 128 | 0 |
| 21:07:48 | 0 | 0 | 128 |
| 合计 | **288** | **128** | **128** |

34个质量标记全部保留，其中测试会话2个。通过 `trial_id` 对齐元数据、按完整session/run隔离，标准化只拟合训练集。保留原始1～80Hz FIR零相位处理和50Hz陷波，不在本轮额外加4～45Hz滤波。

ADS1299输入先从V转换为μV一次，再逐试次独立抗混叠降采样到250Hz、750点。g.Nautilus 已是μV和250Hz，直接从原0～4秒窗口裁剪 `[125:875]`，即0.5～3.5秒750点，不再次乘1e6、不重新拟合标准化。随后应用 **ADS1299训练集** 保存的均值和标准差。

g.Nautilus 数据已有采集端30Hz低通、50Hz陷波及离线4Hz高通，与ADS1299的1～80Hz处理不同，不能通过裁剪补回缺失频段，也不能把两份数据都描述为同一宽带信号。外部数据、单位/窗口适配及模型冻结哈希均写入 `g_nautilus_external/evaluation_protocol.json`。

## 模型及参数

本轮只用 `standard` 时域分支：16个长度251的时间滤波器，每个配两个跨通道空间滤波器，随后深度可分离卷积、平均池化和四分类层。内部输入 `[N,1,8,750]`，池化后为 `[N,32,1,23]`，展开736维；总参数量 **9428**。

固定参数为学习率0.0005、AdamW weight decay 0.01、批量32、Dropout 0.5、BatchNorm momentum 0.1/eps 0.001、梯度范数上限5、空间滤波器最大范数1、分类器最大范数0.25、随机种子42。当前训练最多110轮，早停等待60轮。仅按验证成绩选轮次，不使用ADS1299测试或g.Nautilus外部结果选参数。

逐步中文说明见 [代码阅读顺序](docs/代码阅读顺序.md)。原代码快照仅供追溯，不是运行入口。

## 图像与文件说明

本轮全部新图的图题、坐标轴、图例和色条均为中文，设备名和A/B/C/D类别字母保留。每张图同时导出PNG与SVG，使用目录内的Noto中文字体；SVG中文字形已转路径，便于跨机器查看。

- `results/run_110epochs/ads1299/training_curves.png`：训练/验证损失和验证准确率，未用测试集做逐轮监控。
- `results/run_110epochs/ads1299/test_confusion_matrix.png`：ADS1299内部留出会话，128试次。
- `results/run_110epochs/g_nautilus_external/external_confusion_matrix.png`：跨设备外部总集，339试次。
- `results/run_110epochs/g_nautilus_external/historical_test_confusion_matrix.png`：其中原历史测试子集，95试次。
- `results/run_110epochs/g_nautilus_external/internal_external_comparison.png`：明确区分内部与外部评估。

混淆矩阵行是真实注视目标，列是预测目标。训练损失在启用随机失活的批次计算，验证损失在评估模式计算，两者不能直接作为完全相同条件的损失比较。

## 检查、环境和迁移

五项检查已通过，包括数据复制哈希、会话划分、模型初始化与原代码逐项一致、标准化隔离及外部窗口/单位适配：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

`provenance/reproduction_check_run_110epochs.json`记录110轮预算、选中109轮、前110轮记录一致、代码/数据哈希及外部复测模型未改变。历史 `provenance/reproduction_check.json`仍对应旧300轮实验；其配置和代码快照保存在 `provenance/run_001_selected_config.json`、`provenance/run_001_source_snapshot/`。`provenance/data_copy_manifest.json`记录所有数据副本的原路径及SHA256。

实际环境为Python3.9.21、PyTorch2.7.0+cu126、NumPy1.26.4、SciPy1.13.1、Matplotlib3.9.4，GPU为GTX1650。项目内虚拟环境复用本机conda `pytorch` 的GPU库，并独立安装干净NumPy，未改动原conda环境。

迁移整个目录后，代码和数据均可按相对路径定位；虚拟环境应在目标机器重建：

```bash
conda activate pytorch
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install --ignore-installed --no-deps numpy==1.26.4
.venv/bin/python run_all.py
```

目标环境还需具备 `requirements.txt` 所列依赖。机器/软件版本变化时，不保证与本机逐位一致。
