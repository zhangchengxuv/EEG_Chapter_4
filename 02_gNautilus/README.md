# g.Nautilus 目录导航

日常使用 [current/README.md](current/README.md)。原g.Nautilus工程整体放入current，原始数据和模型均保留。

| 需要的内容 | 位置 |
| --- | --- |
| 合并训练入口 | `current/train_merged_final.py` |
| 预测入口 | `current/predict_merged_eegnet.py` |
| 完整数据集 | `current/ML_DL_Dataset/` |
| 已选定模型 | `current/results/eegnet_merged_search/selected_model/model.pt` |
| 合并模型结果 | `current/results/eegnet_merged_search/` |
| 早期逐文件训练结果 | `archive/per_recording_baseline/` |

在本设备目录下执行：

```bash
cd current
.venv/bin/python train_merged_final.py
```

本设备模型使用同一人的S01/S02/S03三次采集合并，历史训练/验证/测试数量为202/42/95，测试86/95=90.53%。这里保留的是g.Nautilus自己的历史选定配置，不随ADS1299的110轮调整而改变。

输入采样率250Hz、单位μV、固定八通道、4秒1000点。不要当成ADS1299的500Hz、V单位、3秒输入。ADS1299模型对g.Nautilus的外部复测保存在另一个设备目录的 `01_ADS1299/current/results/`，两类成绩含义不同。

早期把三份文件当成三位受试者训练的结果已归档到 `archive/per_recording_baseline`；归档只保留历史事实，当前解释均为同一人的三次采集。
