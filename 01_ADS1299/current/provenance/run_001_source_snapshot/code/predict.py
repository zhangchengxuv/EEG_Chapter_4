"""使用已保存的 ADS1299 模型预测注视目标；无需真实标签或刺激频率元数据。"""

import argparse
from pathlib import Path

import numpy as np
import torch

from ads1299_data import ROOT, CHANNELS, CLASSES, save_csv, transform_saved
from ads1299_models import Model


@torch.no_grad()
def predict_saved(x_v, model_path, device="cpu"):
    """输入单位 V、500 Hz 的 [N,8,1500] 数组，返回四类概率。"""
    if x_v.ndim != 3 or x_v.shape[1:] != (8, 1500) or len(x_v) == 0 or not np.isfinite(x_v).all():
        raise ValueError("输入应为非空、有限的 [N,8,1500] V 单位信号")
    # 检查点仅包含张量和基础类型；先在 CPU 安全加载，再将网络送入指定设备。
    package = torch.load(model_path, map_location="cpu", weights_only=True)
    members = []
    for checkpoint in package["members"]:
        # 模型文件同时包含输入定义与训练统计量，推理时不重新拟合。
        if (checkpoint["input_unit"] != "V" or checkpoint["input_sampling_rate_hz"] != 500
                or checkpoint["channels"] != CHANNELS):
            raise ValueError("模型输入定义与 ADS1299 接口不符")
        model = Model(checkpoint["config"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        # eval 同时关闭随机失活和 BatchNorm 统计更新，同一试次的预测不依赖批次顺序。
        model.eval()
        raw, spectrum = transform_saved(x_v, checkpoint)
        probability = [model(raw[i:i + 32].to(device), spectrum[i:i + 32].to(device)).softmax(1).cpu().numpy()
                       for i in range(0, len(raw), 32)]
        # 按批次拼接恢复原试次顺序，softmax 给出和为1的四类概率。
        members.append(np.concatenate(probability))
    return np.mean(members, axis=0)


def main():
    """从 NPZ 读取 X 和可选 trial_ids，不读取 y、采集编号或刺激标签。"""
    parser = argparse.ArgumentParser(description="ADS1299 EEGNet 无标签预测")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=None, help="默认加载最近训练完成的模型")
    parser.add_argument("--output", type=Path, required=True, help="新建预测 CSV，不覆盖已有文件")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"不覆盖已有文件：{args.output}")
    model_path = args.model
    if model_path is None:
        # latest_run 只保存本目录内的实验名称，移动整个目录后仍能正确定位模型。
        run_name = (ROOT / "latest_run.txt").read_text(encoding="utf-8").strip()
        model_path = ROOT / "models" / run_name / "model.pt"
    torch.set_num_threads(2)
    with np.load(args.input, allow_pickle=False) as source:
        x = source["X"]
        if ("unit" in source and str(source["unit"]) != "V"
                or "sfreq" in source and float(source["sfreq"]) != 500
                or "channel_names" in source and source["channel_names"].tolist() != CHANNELS):
            raise ValueError("输入文件的单位、采样率或通道定义不符合模型要求")
        # trial_id 只是输出索引；缺少它时用行号，不会影响实际预测。
        trial_ids = source["trial_ids"].astype(str) if "trial_ids" in source else np.arange(len(x)).astype(str)
    if trial_ids.shape != (len(x),):
        raise ValueError("trial_ids 数量与 X 不符")
    probability = predict_saved(x, model_path, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_csv(args.output, [{"trial_id": trial_id, "predicted_label": int(p.argmax()),
                           "predicted_target": CLASSES[p.argmax()],
                           **{f"probability_{c}": float(p[i]) for i, c in enumerate(CLASSES)}}
                          for trial_id, p in zip(trial_ids, probability)])
    print(f"已保存 {len(x)} 个试次的预测：{args.output}")


if __name__ == "__main__":
    main()
