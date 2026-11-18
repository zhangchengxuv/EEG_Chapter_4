"""简洁训练入口：只复现已通过验证选择的能量汇聚 EEGNet，不再运行九组搜索。"""

import argparse
import csv
import hashlib
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch

from train_eegnet import ROOT, CLASSES, fit_standardizer, classification_metrics, plot_results, save_csv, save_json
from train_merged_eegnet import CANDIDATES, fit_candidate, load_merged
from predict_merged_eegnet import predict_merged


def main():
    """原有区组划分→训练集标准化→训练→验证选择轮次→保存→最终测试。"""
    parser = argparse.ArgumentParser(description="同一人的三次采集合并，复现最终 EEGNet 配置")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "ML_DL_Dataset")
    parser.add_argument("--output-dir", type=Path, default=None, help="新建结果目录，默认自动生成时间戳")
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if min(args.epochs, args.patience) <= 0:
        parser.error("epochs 和 patience 必须大于零")
    torch.set_num_threads(2)
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    args.output_dir = args.output_dir or ROOT / "results" / f"eegnet_merged_final_{timestamp}"
    args.output_dir.mkdir(parents=True, exist_ok=False)

    # 第一步：把文件编号解释为采集来源，保留原始区组划分，不使用采集编号预测。
    x, y, trial_ids, rows, masks = load_merged(args.data_dir)
    save_csv(args.output_dir / "split_membership.csv", [
        {"trial_id": trial_id, "session": row["subject_id"], "block": row["block_id"],
         "split": row["split"], "label": int(label)} for trial_id, row, label in zip(trial_ids, rows, y)])

    # 第二步：只在合并后的 202 个训练试次上拟合标准化，不使用验证或测试统计量。
    train_x, train_y = x[masks["train"]], y[masks["train"]]
    val_x, val_y = x[masks["validation"]], y[masks["validation"]]
    channel_stats = fit_standardizer(train_x)
    config = next(config.copy() for config in CANDIDATES if config["name"] == "energy_eegnet")
    source_paths = [Path(__file__), ROOT / "train_eegnet.py", ROOT / "train_merged_eegnet.py",
                    ROOT / "predict_merged_eegnet.py"]
    data_paths = [args.data_dir / folder / f"{session}{suffix}" for session in ("S01", "S02", "S03")
                  for folder, suffix in (("NumPy", "_accepted_0to4_uv.npz"), ("Metadata", "_trials.csv"))]
    save_json(args.output_dir / "config.json", {
        "model": config, "seed": 42, "epochs": args.epochs, "patience": args.patience,
        "batch_size": 32, "device": args.device, "torch": str(torch.__version__),
        "selection": "validation accuracy, then validation cross-entropy",
        "interpretation": "同一人的三次采集，固定 202/42/95 划分",
        "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths + data_paths},
    })

    # 第三步：固定已选好的模型和随机种子；只用验证集选择轮次和执行早停。
    result = fit_candidate(config, 42, train_x, train_y, val_x, val_y,
                           channel_stats, args, args.output_dir)
    candidate_dir = args.output_dir / "candidates" / result["candidate"]
    checkpoint = torch.load(candidate_dir / "best_model.pt", map_location="cpu", weights_only=True)

    # 第四步：测试前保存最终模型及训练统计量，测试后不再修改模型。
    final_dir = args.output_dir / "selected_model"
    final_dir.mkdir()
    torch.save({"members": [checkpoint], "combination": "single model"}, final_dir / "model.pt")

    # 第五步：推理函数只接收信号和模型路径，不接收标签或任何类别元数据。
    test_y = y[masks["test"]]
    probabilities = predict_merged(x[masks["test"]], final_dir / "model.pt")
    metrics = classification_metrics(test_y, probabilities)
    metrics.update({"validation_accuracy": result["validation_accuracy"], "best_epoch": result["best_epoch"]})
    save_json(final_dir / "metrics.json", metrics)
    save_csv(final_dir / "test_predictions.csv", [
        {"trial_id": trial_id, "true_label": int(label), "predicted_label": int(p.argmax()),
         **{f"probability_{name}": float(p[i]) for i, name in enumerate(CLASSES)}}
        for trial_id, label, p in zip(trial_ids[masks["test"]], test_y, probabilities)])

    # 第六步：保存可视化及汇总；混淆矩阵的行是真实类别，列是预测类别。
    with (candidate_dir / "history.csv").open(encoding="utf-8-sig", newline="") as file:
        history = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(file)]
    plot_results(final_dir, history, metrics, result["best_epoch"])
    print(f"\n测试准确率={metrics['accuracy']:.2%}，宏 F1={metrics['macro_f1']:.4f}")
    print(f"结果目录：{args.output_dir}")


if __name__ == "__main__":
    main()
