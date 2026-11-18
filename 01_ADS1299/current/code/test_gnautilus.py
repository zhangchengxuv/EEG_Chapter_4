"""冻结 ADS1299 模型后，对 g.Nautilus 历史数据进行单独标明的跨设备外部复测。"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from ads1299_data import ROOT, CHANNELS, CLASSES, digest, metrics, save_csv, save_json
from ads1299_models import Model
from plots import confusion_matrix, external_accuracy
from training import probabilities


def load_external():
    """按 trial_id 核对三份 accepted 文件；只取 EEG，不把频率或采集编号当特征。"""
    x_list, y_list, rows = [], [], []
    for session in ("S01", "S02", "S03"):
        path = ROOT / "data/gNautilus/NumPy" / f"{session}_accepted_0to4_uv.npz"
        with np.load(path, allow_pickle=False) as source:
            x, y, ids = source["X"], source["y"], source["trial_ids"].astype(str)
            checks = [float(source["sfreq"]) == 250, str(source["unit"]) in ("uV", "μV", "µV"),
                      source["ch_names"].tolist() == CHANNELS, source["class_names"].tolist() == CLASSES,
                      bool(source["accepted"].all()), x.shape[1:] == (8, 1000), bool(np.isfinite(x).all()),
                      float(source["epoch_start"]) == 0, float(source["epoch_end_exclusive"]) == 4]
        if not all(checks) or y.shape != (len(x),) or len(set(ids)) != len(x):
            raise ValueError(f"{session}: g.Nautilus 输入定义或试次唯一性不符合预期")
        metadata = ROOT / "data/gNautilus/Metadata" / f"{session}_trials.csv"
        with metadata.open(encoding="utf-8-sig", newline="") as file:
            table = list(csv.DictReader(file))
        by_id = {row["trial_id"]: row for row in table}
        if len(by_id) != len(table) or not set(ids).issubset(by_id):
            raise ValueError(f"{session}: 元数据 trial_id 重复或缺失")
        for trial_id, label in zip(ids, y):
            row = by_id[trial_id]
            if int(row["class_label"]) != int(label) or row["accepted"].lower() != "true":
                raise ValueError(f"{trial_id}: 外部数据标签或 accepted 标记不一致")
            rows.append({"trial_id": trial_id, "recording": session, "historical_g_split": row["split"]})
        # 原始 g.Nautilus 已是 μV、250 Hz；0.5～3.5秒对应 [125:875]，正好750点。
        # 不乘1e6、不插值到500Hz，也不在这些外部试次上拟合标准化。
        x_list.append(x[:, :, 125:875].astype(np.float32))
        y_list.append(y.astype(np.int64))
    x, y = np.concatenate(x_list), np.concatenate(y_list)
    if x.shape != (339, 8, 750) or len(set(row["trial_id"] for row in rows)) != 339:
        raise ValueError("外部数据应为339个独立 accepted 试次")
    return x, y, rows


def main():
    """仅推理固定模型，输出外部总集与历史测试子集指标，绝不根据结果调参。"""
    parser = argparse.ArgumentParser(description="g.Nautilus 跨设备外部复测（当前模型训练设备为 ADS1299）")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name:
        parser.error("run-name 只能是目录名称")
    torch.set_num_threads(2)
    model_path = ROOT / "models" / args.run_name / "model.pt"
    output = ROOT / "results" / args.run_name / "g_nautilus_external"
    output.mkdir(parents=True, exist_ok=False)
    package = torch.load(model_path, map_location="cpu", weights_only=True)
    if len(package["members"]) != 1:
        raise ValueError("当前复现实验应当是单个已选定 EEGNet")
    checkpoint = package["members"][0]
    if checkpoint["config"]["family"] != "standard" or checkpoint["config"].get("band") is not None:
        raise ValueError("此跨设备接口仅用于当前不加额外频带滤波的标准 EEGNet")
    x_uv, y, rows = load_external()

    # 在输出任何外部成绩前保存协议和模型哈希，证明使用的是已冻结的同一权重。
    # g.Nautilus 的30Hz低通和4Hz高通与ADS1299的1～80Hz处理不同，无法补回丢失频段。
    settings = json.loads((ROOT / "selected_config.json").read_text(encoding="utf-8"))
    save_json(output / "evaluation_protocol.json", {**settings["external_protocol"],
        "model_sha256_before": digest(model_path), "normalizer_fitted_on": "仅 ADS1299 训练集",
        "current_model_trained_on_gnautilus": False, "label_or_frequency_used_as_input": False,
        "data_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / "data/gNautilus").rglob("*")) if p.is_file()}})

    # 使用 ADS1299 训练集保存的 μV 均值和标准差；这里只做应用，不重新估计任何参数。
    mean, std = checkpoint["statistics"]["mean"].numpy(), checkpoint["statistics"]["std"].numpy()
    normalized = torch.from_numpy(np.ascontiguousarray(((x_uv - mean) / std)[:, None]))
    dummy_spectrum = torch.zeros((len(x_uv), 4, 1))  # 当前时域模型不使用这个占位张量。
    model = Model(checkpoint["config"]).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    probability = probabilities(model, (normalized, dummy_spectrum), args.device)

    # 全部339个旧数据试次是外部复测总集；另列旧划分中的95个测试试次，避免混淆。
    overall = metrics(y, probability)
    historical_test = np.array([row["historical_g_split"] == "test" for row in rows])
    subset = metrics(y[historical_test], probability[historical_test])
    by_recording = {name: metrics(y[mask], probability[mask])
                    for name in ("S01", "S02", "S03")
                    for mask in [np.array([row["recording"] == name for row in rows])]}
    save_json(output / "metrics.json", {"evaluation_type": "跨设备外部历史数据复测",
              "trained_device": "ADS1299", "evaluated_device": "g.Nautilus",
              "all_accepted_339": overall, "historical_test_95": subset, "by_recording": by_recording,
              "note": "这批数据曾用于以前的g.Nautilus研究；不是新采集盲测，未用于本轮ADS1299训练或选轮次。"})
    save_csv(output / "predictions.csv", [
        {**row, "true_label": int(label), "predicted_label": int(p.argmax()),
         **{f"probability_{name}": float(p[i]) for i, name in enumerate(CLASSES)}}
        for row, label, p in zip(rows, y, probability)])
    save_csv(output / "summary.csv", [
        {"范围": name, "正确试次": result["correct"], "总试次": result["count"], "准确率": result["accuracy"], "宏F1": result["macro_f1"]}
        for name, result in [("跨设备历史外部复测总集", overall), ("其中历史测试子集", subset)] + list(by_recording.items())])

    # 外部图题明确写设备和历史数据范围，并与内部留出测试分开保存。
    confusion_matrix(output, overall, "g.Nautilus 跨设备外部复测（全部339试次）", "external_confusion_matrix")
    confusion_matrix(output, subset, "g.Nautilus 跨设备外部复测（历史测试子集95试次）", "historical_test_confusion_matrix")
    internal = json.loads((ROOT / "results" / args.run_name / "ads1299/metrics.json").read_text())
    external_accuracy(output, [{"label": "g.Nautilus\n历史外部总集（339次）", "accuracy": overall["accuracy"]},
                              {"label": "g.Nautilus\n历史测试子集（95次）", "accuracy": subset["accuracy"]}], internal["test"]["accuracy"])
    protocol = json.loads((output / "evaluation_protocol.json").read_text())
    if digest(model_path) != protocol["model_sha256_before"]:
        raise RuntimeError("外部复测期间模型文件发生变化")
    print(f"g.Nautilus 跨设备外部复测总集：{overall['correct']}/339，准确率={overall['accuracy']:.2%}", flush=True)
    print(f"其中历史测试子集：{subset['correct']}/95，准确率={subset['accuracy']:.2%}", flush=True)


if __name__ == "__main__":
    main()
