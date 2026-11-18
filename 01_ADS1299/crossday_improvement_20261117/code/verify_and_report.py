"""独立复核指标、训练统计量和冻结模型推理，并生成中文实验报告。"""
import sys,json,hashlib,shutil
from pathlib import Path
import numpy as np
import torch
root=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(root/'code'))
from ads1299_data import load_data,make_split,fit_preprocessing,metrics,save_json,digest
from predict import predict_saved
source=root.parent/'current'
torch.set_num_threads(2)
s=json.loads((root/'selection.json').read_text())
chosen=s['selected']['name']
with np.load(next((source/'data/independent_tests/20261117_160423').glob('*.npz')),allow_pickle=False) as f:
    x,y=f['X'],f['y']
package=torch.load(root/'models/selected_model.pt',map_location='cpu',weights_only=True)
checkpoint=package['members'][0]
old=load_data(source/'data/ADS1299')
masks,_=make_split(old)
# 重新只从源日训练集计算标准化，与模型内参数逐元素比较，防止目标日参与拟合。
_,stats=fit_preprocessing(old['X'][masks['train']],old['X'][masks['validation']],checkpoint['config'])
for k in stats:
    assert torch.equal(stats[k],checkpoint['statistics'][k]),k
# CPU独立推理验证GPU结果，并用倒序试次验证推理与文件行位置无关。
p=predict_saved(x,root/'models/selected_model.pt','cpu')
p_saved=np.load(root/'results'/chosen/'target_probabilities.npy')
assert np.array_equal(p.argmax(1),p_saved.argmax(1))
np.testing.assert_allclose(p,p_saved,atol=2e-5,rtol=1e-4)
p_rev=predict_saved(x[::-1].copy(),root/'models/selected_model.pt','cpu')[::-1]
np.testing.assert_allclose(p,p_rev,atol=2e-5,rtol=1e-4)
for row in s['all_results']:
    prob=np.load(root/'results'/row['name']/'target_probabilities.npy')
    assert prob.shape==(128,4) and np.isfinite(prob).all()
    np.testing.assert_allclose(prob.sum(1),1,atol=1e-6)
    assert metrics(y,prob)==json.loads((root/'results'/row['name']/'target_metrics.json').read_text())
checks=dict(train_only_statistics_exact_match=True,cpu_gpu_predictions_match=True,
            reversed_order_predictions_match=True,all_candidate_metrics_recomputed=True,
            selected_model_sha256=digest(root/'models/selected_model.pt'))
save_json(root/'final_checks.json',checks)
# 报告明确区分训练数据、源日验证和目标日开发选择，不把开发提升写成盲测。
lines=['# ADS1299 跨天识别改进实验','',
'仅用2026年11月16日 combined4Recordings 数据训练。11月17日的128个试次不参与梯度更新、标准化拟合或逐轮早停；其标签用于训练后的方案比较，因此本实验属于目标日开发评估，不是新的独立盲测。','',
'## 结果','',
'| 方案 | 11月16日验证准确率 | 11月16日旧测试准确率 | 11月17日准确率 | 11月17日宏F1 |',
'|---|---:|---:|---:|---:|']
for r in s['all_results']:
    lines.append(f"| {r['name']} | {r['source_validation_accuracy']:.2%} | {r['source_test_accuracy']:.2%} | {r['target_accuracy']:.2%} | {r['target_macro_f1']:.4f} |")
b=s['all_results'][0]
c=s['selected']
lines.extend(['',f"选中模型：**{chosen}**；目标日准确率 **{c['target_accuracy']:.2%}**，较原模型提升 **{(c['target_accuracy']-b['target_accuracy'])*100:.2f}个百分点**。最佳轮次：{checkpoint['best_epoch']}，每个候选上限110轮。",'',
'## 各类识别情况','', '| 目标 | 正确/试次数 | 召回率 |','|---|---:|---:|'])
m=metrics(y,p_saved)
for r in m['per_class']:
    lines.append(f"| {r['target']} | {m['confusion_matrix_true_rows'][r['label']][r['label']]}/{r['support']} | {r['recall']:.2%} |")
lines.extend(['','## 方法和限制','',
'- 沿用完整会话划分：训练288、验证128、旧测试128。训练统计量仅从288个训练试次拟合。',
'- V转换为μV一次，独立重采样500→250 Hz，仍为0.5～3.5秒窗口。没有根据目标日表现删除异常或带标记试次。',
'- energy8/energy16：EEGNet时间和深度空间卷积后，以对数能量汇总替代原始展平分类头，分别使用8/16个时间滤波器。属于EEGNet变体。',
'- hybrid8_spectral / hybrid16_spectral：时间滤波器分别8/16个，能量EEGNet输出权重0.1；第一轮目标日比较后追加，完整记录在protocol.json。',
'- hybrid8：能量EEGNet的输出乘0.5，加上从每段EEG计算的四个候选频率及其谐波评分。四个参考频率同时应用于每段输入，不读取该段真实刺激频率或标签。',
'- standard8_aug：原版EEGNet骨干8个时间滤波器，训练时各通道幅度乘0.8～1.2，再添加标准化尺度0.03的高斯噪声。推理不增强。',
'- 随机种子42、AdamW、学习率0.0005、权重衰减0.01、批量32，最多110轮，源日验证连续60轮无改善早停。',
'- 分两轮完成6个候选；每轮所有候选训练完毕后统一评估目标日；以目标准确率优先、宏F1次优选择模型。目标日结果已经用于选型，不能用同一数据证明对未知第三天的泛化。',
'- 两天原始刺激顺序相同的实验设计局限仍然存在。',
'- CPU与GPU预测核对一致；倒序输入预测一致；全部候选指标独立重算；模型统计量与仅源日训练集重算值完全一致。',
'- 原模型、原数据和current/latest_run.txt保持不变；本目录selected_model.pt为独立的新模型。','',
'## 复现与推理','',
'在本目录执行：','',
'```bash','../current/.venv/bin/python code/run_experiment.py --source-root ../current --device cuda',
'../current/.venv/bin/python code/predict.py --input ../current/data/independent_tests/20261117_160423/ADS1299_20261117_160423_allTrials_500Hz_0p5to3p5s_V.npz --model models/selected_model.pt --output new_predictions.csv','```','',
'训练程序会复用已完成的候选。若从头复现，请复制代码和assets到一个空目录后执行，并指定current的绝对路径。预测CSV包含目标类别而非机器人实际动作。','',
'数据复用current/data内的原始文件，不重复复制。protocol.json记录数据/原模型哈希、完整配置和划分；results包含所有候选检查点、逐轮记录、概率、逐试次预测和中文图像；models包含各候选及最终模型。'])
(root/'README.md').write_text('\n'.join(lines)+'\n')
# 绘制原模型与新模型逐类召回率，避免总体准确率掩盖类别间变化。
from plots import setup_plotting,save_figure
plt=setup_plotting()
baseline_metrics=json.loads((root/'results/baseline/target_metrics.json').read_text())
fig,ax=plt.subplots(figsize=(8,4.8))
positions=np.arange(4)
for offset,result,label in [(-0.18,baseline_metrics,'原110轮模型'),(0.18,m,'本次选中模型')]:
    heights=[c['recall']*100 for c in result['per_class']]
    bars=ax.bar(positions+offset,heights,width=0.36,label=label)
    ax.bar_label(bars,labels=[f'{v:.1f}%' for v in heights],padding=3)
ax.set(xticks=positions,xticklabels=['目标 A','目标 B','目标 C','目标 D'],ylim=(0,110),
       ylabel='召回率（%）',xlabel='真实注视目标',title='ADS1299 跨天开发评估：逐类识别对比\n全部128试次；目标日参与方案选择')
ax.legend()
fig.tight_layout()
save_figure(fig,root/'results/class_recall_comparison')
plt.close(fig)
# 记录实际参与本实验的脚本与结果哈希，便于后续检查文件一致性。
save_json(root/'artifact_manifest.json',{str(f.relative_to(root)):digest(f) for f in sorted(root.rglob('*')) if f.is_file() and '__pycache__' not in str(f) and f.suffix!='.log' and f.name!='artifact_manifest.json'})
print(json.dumps(dict(selected=chosen,metrics=m,checks=checks),ensure_ascii=False))
