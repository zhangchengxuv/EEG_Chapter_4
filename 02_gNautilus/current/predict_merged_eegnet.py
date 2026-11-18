"""对新的同定义 μV 信号使用合并模型预测注视目标，不需要真实标签。"""

import argparse
from pathlib import Path

import numpy as np
import torch

from train_eegnet import ROOT, CLASSES, save_csv
from train_merged_eegnet import MergedModel, checkpoint_pair


@torch.no_grad()
def predict_merged(x, model_path):
    """输入 [N,8,1000]、单位 μV；返回四类概率，所有统计量从模型文件读取。"""
    if x.ndim != 3 or x.shape[1:] != (8, 1000) or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("输入必须是非空、有限的 [N,8,1000] 数组")
    x = x.astype(np.float32)
    package = torch.load(model_path, map_location="cpu", weights_only=True)
    member_probabilities = []
    for checkpoint in package["members"]:
        model = MergedModel(checkpoint["config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        raw, spectrum = checkpoint_pair(x, checkpoint)
        probabilities = [model(raw[i:i + 32], spectrum[i:i + 32]).softmax(1).numpy()
                         for i in range(0, len(x), 32)]
        member_probabilities.append(np.concatenate(probabilities))
    return np.mean(member_probabilities, axis=0)


def main():
    """从仅需包含 X 的 NPZ 读取信号，将预测写入新文件。"""
    parser = argparse.ArgumentParser(description="合并 EEGNet 模型推理")
    parser.add_argument("--input", type=Path, required=True, help="包含 μV 单位 X 的 NPZ")
    parser.add_argument("--model", type=Path, default=ROOT / "results/eegnet_merged_search/selected_model/model.pt")
    parser.add_argument("--output", type=Path, required=True, help="新的预测 CSV 路径")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"不覆盖已有文件：{args.output}")
    torch.set_num_threads(2)
    # 不读取真实 y、刺激频率或其他元数据；trial_ids 仅用于输出记录。
    with np.load(args.input, allow_pickle=False) as data:
        x = data["X"]
        ids = data["trial_ids"].astype(str) if "trial_ids" in data else np.arange(len(x)).astype(str)
    if ids.shape != (len(x),):
        raise ValueError("trial_ids 与 X 数量不一致")
    probabilities = predict_merged(x, args.model)
    rows = [{"trial_id": trial_id, "predicted_label": int(p.argmax()), "predicted_target": CLASSES[p.argmax()],
             **{f"probability_{name}": float(p[i]) for i, name in enumerate(CLASSES)}}
            for trial_id, p in zip(ids, probabilities)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_csv(args.output, rows)
    print(f"已保存 {len(rows)} 个试次的预测：{args.output}")


if __name__ == "__main__":
    main()
