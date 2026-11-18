"""将新采集的 ADS1299 会话仅用作独立测试，不训练、不调参、不重拟合标准化。

输入必须是500Hz、单位V、Marker后0.5～3.5秒的[N,8,1500]数组。
先保存冻结模型与测试协议，再进行无标签推理，最后才用y计算指标。
"""

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch

from ads1299_data import ROOT, CHANNELS, CLASSES, FREQUENCIES, digest, metrics, save_csv, save_json
from plots import confusion_matrix, save_figure, setup_plotting
from predict import predict_saved


def load_new_session(path):
    """只按文件内容校验输入，不根据幅值、质量标记或测试正确与否筛选试次。"""
    with np.load(path, allow_pickle=False) as source:
        required = {"X", "y", "sfreq", "unit", "channel_names", "trial_ids", "run_ids", "session_ids", "recording_ids"}
        if not required.issubset(source.files):
            raise ValueError(f"新文件缺少必要字段：{sorted(required - set(source.files))}")
        data = {name: source[name].copy() for name in source.files}
    x, y = data["X"], data["y"]
    if x.ndim != 3 or x.shape[1:] != (8, 1500) or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("X必须是非空、有限的[N,8,1500]信号")
    if y.shape != (len(x),) or y.dtype.kind not in "iu" or not np.isin(y, range(4)).all():
        raise ValueError("y必须是一维0～3整数标签")
    if str(data["unit"]) != "V" or float(data["sfreq"]) != 500 or data["channel_names"].tolist() != CHANNELS:
        raise ValueError("新文件的单位、采样率或通道顺序与冻结模型不符")
    for name in ("trial_ids", "run_ids", "session_ids", "recording_ids"):
        if data[name].shape != (len(x),):
            raise ValueError(f"{name}数量与信号试次数不一致")
    if len(set(data["trial_ids"].tolist())) != len(x):
        raise ValueError("新文件中trial_id不唯一")
    data.setdefault("quality_flags", np.full(len(x), "", dtype="U1"))
    if data["quality_flags"].shape != (len(x),):
        raise ValueError("quality_flags数量不一致")
    # 这些字段仅用于核对标签定义；真实频率和事件编码不会传给预测函数。
    if "stimulus_frequencies_hz" in data and not np.array_equal(data["stimulus_frequencies_hz"], np.array(FREQUENCIES)[y]):
        raise ValueError("刺激频率与0～3标签定义不一致")
    if "event_ids" in data and not np.array_equal(data["event_ids"], y + 1):
        raise ValueError("事件编码与标签定义不一致")
    return data


def audit_independence(data):
    """与全部旧544试次检查重合，并比较新旧事件标签顺序；不调整模型或划分。"""
    previous_path = next((ROOT / "data/ADS1299").glob("*combined4Recordings*.npz"))
    with np.load(previous_path, allow_pickle=False) as source:
        previous = {name: source[name] for name in ("X", "y", "trial_ids", "run_ids", "session_ids", "recording_ids")}
    overlaps = {name: sorted(set(data[name].astype(str)) & set(previous[name].astype(str)))
                for name in ("trial_ids", "run_ids", "session_ids", "recording_ids")}
    old_hashes = {hashlib.sha256(x.tobytes()).hexdigest() for x in previous["X"]}
    new_hashes = [hashlib.sha256(x.tobytes()).hexdigest() for x in data["X"]]
    duplicates = int(sum(value in old_hashes for value in new_hashes))
    if any(overlaps.values()) or duplicates:
        raise ValueError("新数据与旧数据的标识或完整信号内容重合，不能作为本次独立测试")
    comparisons = []
    for run in sorted(np.unique(previous["run_ids"])):
        labels = previous["y"][previous["run_ids"] == run]
        if len(labels) == len(data["y"]):
            comparisons.append({"old_run": str(run), "same_label_order": bool(np.array_equal(labels, data["y"])),
                                "matching_positions": int(np.sum(labels == data["y"])), "count": len(labels)})
    return {"identifier_overlaps_with_old_544": overlaps, "exact_signal_duplicates_with_old_544": duplicates,
            "new_unique_signal_contents": len(set(new_hashes)), "label_order_comparisons": comparisons,
            "sequence_confound_still_present": any(row["same_label_order"] for row in comparisons),
            "quality_flags": dict(Counter(data["quality_flags"].astype(str).tolist())),
            "class_counts": np.bincount(data["y"], minlength=4).tolist(),
            "sampling_rate_hz": 500, "unit": "V", "shape": list(data["X"].shape),
            "upstream_filter_parameters_independently_verified": False,
            "window_definition_source": "文件名0p5to3p5s及用户说明；新文件没有配套窗口/滤波metadata.json"}


def main():
    """存档新数据→冻结协议→预测→计算主指标与质量分组描述→保存中文图。"""
    parser = argparse.ArgumentParser(description="冻结ADS1299模型，在新采集会话上独立测试")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--session-name", required=True, help="新测试名称，例如20261117_160423")
    parser.add_argument("--run-name", default="run_110epochs", help="使用事先指定的冻结模型，不比较多个训练轮数")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if any(Path(name).name != name for name in (args.session_name, args.run_name)):
        parser.error("实验名和会话名只能是单个目录名称")
    torch.set_num_threads(2)
    model_path = ROOT / "models" / args.run_name / "model.pt"
    model_hash = digest(model_path)
    source_hash = digest(args.input)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    data = load_new_session(args.input)
    audit = audit_independence(data)
    output = ROOT / "results" / args.run_name / f"ads1299_new_session_{args.session_name}"
    output.mkdir(parents=True, exist_ok=False)

    # 保存真正的数据副本，原文件保留；独立测试数据单独存放，训练入口不会读取这里。
    stored_dir = ROOT / "data/independent_tests" / args.session_name
    stored_dir.mkdir(parents=True, exist_ok=True)
    stored_path = stored_dir / args.input.name
    if stored_path.exists():
        if digest(stored_path) != source_hash:
            raise FileExistsError("目标测试数据副本已存在且内容不同，不覆盖")
    else:
        shutil.copy2(args.input, stored_path)
    if digest(stored_path) != source_hash:
        raise RuntimeError("测试数据复制哈希不一致")

    # 在任何测试成绩计算前明确固定所用模型、预处理规则、输入文件和全试次策略。
    protocol = {"evaluation_type": "新增ADS1299独立采集会话测试", "session_name": args.session_name,
                "started_at_shanghai": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "source_input": str(args.input.resolve()), "stored_input": str(stored_path.relative_to(ROOT)),
                "input_sha256": source_hash, "model": str(model_path.relative_to(ROOT)),
                "model_sha256_before": model_hash, "model_run": args.run_name,
                "selected_model_epochs": [member["best_epoch"] for member in checkpoint["members"]],
                "normalizer": "只应用旧ADS1299训练集统计量，不重新拟合",
                "retraining": False, "test_based_hyperparameter_selection": False,
                "quality_policy": "保留全部试次及全部质量标记，主指标不剔除任何试次",
                "device": args.device, "python": sys.version, "torch": str(torch.__version__),
                "code_sha256": {name: digest(ROOT / "code" / name)
                                for name in ("evaluate_new_session.py", "predict.py", "ads1299_data.py", "ads1299_models.py", "plots.py")}}
    save_json(output / "evaluation_protocol_before_test.json", protocol)
    save_json(output / "input_audit.json", audit)
    print(f"冻结模型={args.run_name}，选中轮次={protocol['selected_model_epochs']}，新会话试次={len(data['X'])}", flush=True)

    # 预测函数只接收X与冻结模型文件，不传入y、刺激频率、试次位置或采集编号。
    probability = predict_saved(data["X"], model_path, args.device)
    np.save(output / "probabilities.npy", probability)
    blind_rows = [{"trial_id": str(trial_id), "predicted_label": int(p.argmax()),
                   "predicted_target": CLASSES[p.argmax()],
                   **{f"probability_{name}": float(p[i]) for i, name in enumerate(CLASSES)}}
                  for trial_id, p in zip(data["trial_ids"], probability)]
    save_csv(output / "predictions_without_true_labels.csv", blind_rows)

    # 无标签预测写出后，再以同一顺序的真实y计算指标；不会反馈到网络或优化器。
    result = metrics(data["y"], probability)
    result.update(evaluation_type=protocol["evaluation_type"], model_run=args.run_name,
                  session_name=args.session_name, flagged_trials_retained=int(np.count_nonzero(data["quality_flags"] != "")))
    rows = [{**row, "true_label": int(label), "true_target": CLASSES[label],
             "quality_flags": str(flag), "session_id": str(session), "run_id": str(run)}
            for row, label, flag, session, run in zip(blind_rows, data["y"], data["quality_flags"], data["session_ids"], data["run_ids"])]
    save_csv(output / "predictions.csv", rows)
    save_json(output / "metrics.json", result)
    save_csv(output / "per_class_metrics.csv", result["per_class"])

    # 质量分组仅作描述，不替代完整128试次主指标，不把小分组宏F1当成可靠结论。
    groups = []
    for flag in sorted(np.unique(data["quality_flags"])):
        mask = data["quality_flags"] == flag
        correct = int(np.sum(probability[mask].argmax(1) == data["y"][mask]))
        groups.append({"质量标记": str(flag) or "无质量标记", "正确试次": correct,
                       "总试次": int(mask.sum()), "准确率": correct / int(mask.sum())})
    save_csv(output / "quality_groups_descriptive.csv", groups)
    confusion_matrix(output, result, f"ADS1299 新采集独立会话测试\n采集：{args.session_name}；冻结110轮模型", "test_confusion_matrix")

    # 逐类召回率图的类别支持数明确标出，四类是否均能识别比单一总准确率更直观。
    from matplotlib.ticker import PercentFormatter
    plt = setup_plotting()
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    values = [row["recall"] for row in result["per_class"]]
    labels = [f"目标 {row['target']}\n{row['support']} 个试次" for row in result["per_class"]]
    bars = axis.bar(labels, values, color="#3574a3", width=0.55)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.2%}", ha="center")
    axis.set(title=f"ADS1299 新采集会话逐类召回率\n采集：{args.session_name}", ylabel="召回率", ylim=(0, 1.12))
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    figure.tight_layout()
    save_figure(figure, output / "class_recall")
    plt.close(figure)

    # 最后核对模型文件和原始新数据均未修改，记录测试的单向信息边界。
    if digest(model_path) != model_hash or digest(args.input) != source_hash:
        raise RuntimeError("评估过程中模型或原始输入文件发生变化")
    save_json(output / "verification.json", {"model_sha256_unchanged": True, "input_sha256_unchanged": True,
              "stored_copy_matches_source": True, "all_trials_retained": True,
              "old_and_new_trial_ids_disjoint": True, "old_and_new_signal_contents_disjoint": True,
              "prediction_received_labels": False, "normalizer_refitted": False, "model_retrained": False})
    print(f"新会话测试准确率：{result['accuracy']:.2%}（{result['correct']}/{result['count']}），宏F1={result['macro_f1']:.4f}", flush=True)
    print(f"结果目录：{output}", flush=True)


if __name__ == "__main__":
    main()
