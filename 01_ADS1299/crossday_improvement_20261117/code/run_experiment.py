"""仅以11月16日训练，定向改善11月17日识别；目标日用于方案选择，不称盲测。"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from ads1299_data import ROOT, load_data, make_split, digest, save_json, save_csv, metrics
from training import train_candidate, read_history
from predict import predict_saved
from plots import confusion_matrix, training_curves


def main():
    """预先记录有限候选，全部训练后再统一读取目标日并比较，保留所有结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--device', default='cpu', choices=['cpu','cuda'])
    args = parser.parse_args()
    args.epochs, args.patience = 110, 60
    torch.set_num_threads(2)
    source = args.source_root.resolve()
    output = ROOT / 'results'
    output.mkdir(exist_ok=True)
    configs = [dict(name='energy8',family='energy',filters=8),
               dict(name='energy16',family='energy',filters=16),
               dict(name='hybrid8',family='hybrid',filters=8,raw_scale=0.5),
               dict(name='standard8_aug',family='standard',filters=8,augment=True),
               dict(name='hybrid8_spectral',family='hybrid',filters=8,raw_scale=0.1),
               dict(name='hybrid16_spectral',family='hybrid',filters=16,raw_scale=0.1)]
    for config in configs:
        config.update(kernel=251,lr=0.0005,weight_decay=0.01)
    # 复用已核验的旧数据加载及完整会话划分；只用288个训练试次拟合统计量。
    data = load_data(source/'data/ADS1299')
    masks, assignments = make_split(data)
    baseline = source/'models/run_110epochs/model.pt'
    target_path = next((source/'data/independent_tests/20261117_160423').glob('*.npz'))
    protected = {str(p):digest(p) for p in [data['path'], target_path, baseline]}
    protocol = dict(configs=configs, seed=42, epochs=110, patience=60, assignments=assignments,
                    source_sha256=digest(data['path']),target_sha256=digest(target_path),
                    baseline_sha256=digest(baseline),
                    epoch_selection='仅11月16日验证准确率优先、交叉熵次优',
                    candidate_selection='11月17日准确率优先、宏F1次优；目标日属于开发评估',
                    target_usage='不参与训练、标准化或早停；所有128试次保留',
                    adaptive_extension='第一轮目标日hybrid8较好，因此追加两个raw_scale=0.1的融合候选；目标日已参与开发',device=args.device)
    protocol_path = ROOT/'protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol, '恢复实验时协议必须不变'
    else:
        save_json(protocol_path,protocol)
    save_csv(output/'source_split.csv',[dict(trial_id=str(t),split=next(k for k,m in masks.items() if m[i]),label=int(data['y'][i])) for i,t in enumerate(data['trial_ids'])])
    for config in configs:
        directory = output/'candidates'/f"{config['name']}_seed42"
        # 已完成的候选允许恢复运行；不覆盖、不重复训练已保存的模型。
        if (directory/'validation_metrics.json').exists():
            continue
        if directory.exists():
            raise RuntimeError(f'发现未完成候选，请检查日志：{directory}')
        train_candidate(config,42,data['X'][masks['train']],data['y'][masks['train']],
                        data['X'][masks['validation']],data['y'][masks['validation']],args,output)
    # 所有候选训练完毕后才加载目标信号和标签；目标标签仅用于最后计算指标。
    with np.load(target_path,allow_pickle=False) as f:
        target = {k:f[k] for k in f.files}
    assert target['X'].shape == (128,8,1500)
    assert str(target['unit']) == 'V' and float(target['sfreq']) == 500
    assert target['channel_names'].tolist() == data['channel_names'].tolist()
    assert not set(target['trial_ids']) & set(data['trial_ids'])
    assert np.bincount(target['y'],minlength=4).tolist() == [32]*4
    old_hashes = {__import__('hashlib').sha256(x.tobytes()).hexdigest() for x in data['X']}
    assert all(__import__('hashlib').sha256(x.tobytes()).hexdigest() not in old_hashes for x in target['X'])
    model_dir = ROOT/'models'
    model_dir.mkdir(exist_ok=True)
    packages = [('baseline',baseline)]
    for config in configs:
        name=config['name']
        checkpoint=torch.load(output/'candidates'/f'{name}_seed42'/'best_model.pt',map_location='cpu',weights_only=True)
        path=model_dir/f'{name}.pt'
        torch.save(dict(members=[checkpoint],combination='single model'),path)
        packages.append((name,path))
    rows=[]
    for name,path in packages:
        directory=output/name
        directory.mkdir(exist_ok=True)
        # 保存全体概率及逐试次预测；原内部测试仅作描述，不参与选择。
        p=predict_saved(target['X'],path,args.device)
        np.save(directory/'target_probabilities.npy',p)
        result=metrics(target['y'],p)
        save_json(directory/'target_metrics.json',result)
        save_csv(directory/'target_predictions.csv',[dict(trial_id=str(t),true_label=int(y),predicted_label=int(pr.argmax()),quality_flags=str(q),**{f'p_{c}':float(pr[j]) for j,c in enumerate('ABCD')}) for t,y,pr,q in zip(target['trial_ids'],target['y'],p,target['quality_flags'])])
        pv=predict_saved(data['X'][masks['validation']],path,args.device)
        pt=predict_saved(data['X'][masks['test']],path,args.device)
        vm,tm=metrics(data['y'][masks['validation']],pv),metrics(data['y'][masks['test']],pt)
        save_json(directory/'source_validation_metrics.json',vm)
        save_json(directory/'source_test_metrics.json',tm)
        rows.append(dict(name=name,target_accuracy=result['accuracy'],target_macro_f1=result['macro_f1'],source_validation_accuracy=vm['accuracy'],source_test_accuracy=tm['accuracy']))
        confusion_matrix(directory,result,'ADS1299 跨天开发评估（11月17日）','target_confusion_matrix')
        if name!='baseline':
            candidate=output/'candidates'/f'{name}_seed42'
            v=json.loads((candidate/'validation_metrics.json').read_text())
            training_curves(directory,read_history(candidate/'history.csv'),v['best_epoch'])
        print(json.dumps(rows[-1],ensure_ascii=False),flush=True)
    # 当前用户要求定向改善目标日；明确记录这种选择已使用目标日标签。
    best=max(rows,key=lambda r:(r['target_accuracy'],r['target_macro_f1']))
    best_path=dict(packages)[best['name']]
    import shutil
    shutil.copy2(best_path,model_dir/'selected_model.pt')
    save_json(ROOT/'selection.json',dict(selected=best,criterion='目标日开发准确率优先、宏F1次优',all_results=rows))
    save_csv(output/'comparison.csv',rows)
    assert all(digest(Path(p))==h for p,h in protected.items())
    save_json(ROOT/'verification.json',dict(protected_files_unchanged=True,protected_sha256=protected,target_trials=128,exact_source_duplicates=0))
    print('实验完成；模型和源文件均已核验。',flush=True)

if __name__=='__main__':
    main()
