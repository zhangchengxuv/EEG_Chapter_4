"""便携的NPZ读取与文件核验示例；不训练模型，不改变单位或数据。"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def load_dataset(path):
    path=Path(path).resolve();metadata=path.parent/'npz_metadata'
    info=json.loads((metadata/(path.stem+'.metadata.json')).read_text(encoding='utf-8'))
    assert info['device']=='ADS1299' and info['unit']=='V' and info['sfreq']==500
    assert digest(path)==info['sha256'],f'文件哈希改变：{path.name}'
    with np.load(path,allow_pickle=False) as z:
        d={k:z[k] for k in z.files}
    x,y=d['X'],d['y'];ids=d['trial_ids'].astype(str)
    ch=d.get('channel_names',d.get('ch_names'))
    assert x.ndim==3 and len(x)==len(y)==len(ids)
    assert list(x.shape)==info['shape'] and x.shape[1]==len(ch)==8
    assert len(np.unique(ids))==len(ids) and np.isfinite(x).all()
    assert np.issubdtype(y.dtype,np.integer) and set(np.unique(y))=={0,1,2,3}
    assert str(d['unit'].item())==info['unit']
    assert float(d['sfreq'])==info['sfreq']
    meta=pd.read_csv(metadata/(path.stem+'.trials.csv'),dtype={'trial_id':str,'subject_id':str},keep_default_na=False)
    assert meta.trial_id.is_unique and len(meta)==len(ids)
    meta=meta.set_index('trial_id').loc[ids].reset_index()
    label='label' if 'label' in meta else 'class_label'
    assert np.array_equal(meta[label].to_numpy(dtype=int),y)
    assert np.array_equal(meta.trial_id.to_numpy(dtype=str),ids)
    d.update(channel_names=ch,trials=meta,metadata=info)
    return d


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='核验EEG数据包或读取指定NPZ，不训练模型')
    parser.add_argument('--file');args=parser.parse_args()
    paths=[ROOT/args.file] if args.file else sorted(ROOT.glob('*.npz'))
    if not paths:raise ValueError('未找到NPZ')
    for path in paths:
        d=load_dataset(path)
        counts={str(k):int((d['y']==k).sum()) for k in np.unique(d['y'])}
        print(path.name,'X=',d['X'].shape,'unit=',d['unit'].item(),'classes=',counts)
    print(f'PASS: {len(paths)} datasets; no training performed.')
