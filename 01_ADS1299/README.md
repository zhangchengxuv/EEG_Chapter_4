# ADS1299 目录导航

日常使用 [current/README.md](current/README.md)。该目录是此前独立重跑工程，已整体移动，内部代码、数据、模型和结果的相对层级保持原样。

| 需要的内容 | 位置 |
| --- | --- |
| 一键训练入口 | `current/run_all.py` |
| 当前参数 | `current/selected_config.json`：最大110轮、种子42 |
| 当前模型 | `current/models/run_110epochs/model.pt`，验证选中109轮 |
| 当前内部结果 | `current/results/run_110epochs/ads1299/` |
| 当前跨设备外部复测 | `current/results/run_110epochs/g_nautilus_external/` |
| 11月17日新增独立会话测试 | `current/results/run_110epochs/ads1299_new_session_20261117_160423/`，88/128=68.75% |
| 新增测试数据副本 | `current/data/independent_tests/20261117_160423/`，仅测试、不参与训练 |
| 300轮重跑历史 | `current/models/run_001/`、`current/results/run_001/` |
| 代码、数据、日志及中文图 | `current/code/`、`current/data/`、`current/logs/`、对应结果目录 |
| 顺序混淆等历史审计 | `current/docs/leakage_audit/` |
| 最初的十组搜索工程 | `archive/initial_training/` |

在本设备目录下执行：

```bash
cd current
.venv/bin/python run_all.py
```

只对新的同定义ADS1299信号预测：

```bash
.venv/bin/python code/predict.py --input 你的新信号.npz --output results/new_predictions.csv
```

默认读取 `current/latest_run.txt` 指向的110轮模型，输入单位仍为V、500Hz、`[N,8,1500]`。移动目录没有改变模型状态、数据、结果或当前轮数。

`archive/initial_training`保留旧代码、原数据包、首次搜索和审计资料；其运行指针已改为相对路径，历史来源记录中的旧路径不改写。推荐从current开始，避免混淆旧入口与当前110轮流程。

## 11月17日定向跨天改进

[crossday_improvement_20261117/README.md](crossday_improvement_20261117/README.md)：仅11月16日训练，目标日开发准确率由68.75%提高到72.66%；新模型单独保存，原默认模型保持不变。
