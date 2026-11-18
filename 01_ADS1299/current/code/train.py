"""从随机初始化重跑当前选定的 EEGNet；只用 ADS1299，不重新搜索模型。"""

import argparse
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import scipy
import torch

from ads1299_data import (ROOT, CLASSES, SPLITS, digest, load_data, make_split,
                          metrics, plot_results, save_csv, save_json, transform_saved)
from ads1299_models import Model
from training import probabilities, read_history, train_candidate


def prediction_rows(data, mask, probability):
    """把概率与 trial_id 一对一对应，仅用于输出；这些元数据不送入网络。"""
    return [{"trial_id": trial_id, "session_id": session, "true_label": int(label),
             "predicted_label": int(p.argmax()), "true_target": CLASSES[label],
             "predicted_target": CLASSES[p.argmax()], "quality_flags": flag,
             **{f"probability_{name}": float(p[i]) for i, name in enumerate(CLASSES)}}
            for trial_id, session, label, p, flag in zip(data["trial_ids"][mask], data["session_ids"][mask],
                                                        data["y"][mask], probability, data["quality_flags"][mask])]


def main():
    """固定配置→核验数据→训练→验证选轮次→固化模型→测试→保存中文图。"""
    parser = argparse.ArgumentParser(description="在独立目录重新训练已选定的 EEGNet")
    parser.add_argument("--run-name", required=True, help="新实验名称；已有名称不会被覆盖")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name:
        parser.error("run-name 只能是目录名称，不能包含路径")
    torch.set_num_threads(2)

    # 第一步：读取事先固定的配置。这里没有读取旧模型文件，更不会继续搜索超参数。
    settings = json.loads((ROOT / "selected_config.json").read_text(encoding="utf-8"))
    config, training = settings["model"], settings["training"]
    if config["name"] != "eegnet_long" or config["family"] != "standard" or training["batch_size"] != 32:
        raise ValueError("本入口固定复现已选定的长时间核 EEGNet，批量大小为 32")
    output = ROOT / "results" / args.run_name / "ads1299"
    model_dir = ROOT / "models" / args.run_name
    output.mkdir(parents=True, exist_ok=False)
    model_dir.mkdir(parents=True, exist_ok=False)

    # 第二步：只读取本目录内合并版的 544 个试次，按来源会话固定为 288/128/128。
    data = load_data(ROOT / "data/ADS1299")
    masks, assignments = make_split(data)
    save_csv(output / "split_membership.csv", [
        {"trial_id": row["trial_id"], "session_id": row["session_id"], "run_id": row["run_id"],
         "block_id": row["block_id"], "split": next(s for s in SPLITS if masks[s][i]),
         "label": int(data["y"][i]), "quality_flags": row["quality_flags"]}
        for i, row in enumerate(data["rows"])])

    # 第三步：在训练前保存参数、分组和代码/数据哈希，确保本轮可追溯。
    # g.Nautilus 的协议在配置里事先声明，但此阶段不加载它来训练或选择轮次。
    code_paths = sorted((ROOT / "code").glob("*.py"))
    data_paths = sorted(p for p in (ROOT / "data/ADS1299").rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    plan = {"started_at_shanghai": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "settings": settings, "sessions": assignments,
            "split_counts": {s: int(masks[s].sum()) for s in SPLITS},
            "initialization": "从随机初始化重新训练，未加载历史权重", "external_data_used_for_training": False,
            "code_sha256": {str(p.relative_to(ROOT)): digest(p) for p in code_paths},
            "data_sha256": {str(p.relative_to(ROOT)): digest(p) for p in data_paths},
            "python": sys.version, "platform": platform.platform(), "torch": str(torch.__version__),
            "numpy": np.__version__, "scipy": scipy.__version__, "device": args.device,
            "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    save_json(output / "run_plan.json", plan)
    print(f"ADS1299 从头训练；训练/验证/测试=288/128/128；设备={args.device}", flush=True)

    # 第四步：核心训练函数只接收训练集与验证集，测试信号不传入训练循环。
    # 它按训练集计算每通道统计量，按配置设定的最大轮数训练，仅按验证成绩选择检查点。
    core_args = SimpleNamespace(epochs=training["epochs"], patience=training["patience"], device=args.device)
    selected = train_candidate(config, training["seed"], data["X"][masks["train"]], data["y"][masks["train"]],
                               data["X"][masks["validation"]], data["y"][masks["validation"]], core_args, output)
    candidate_dir = output / "candidates" / selected["candidate"]
    checkpoint = torch.load(candidate_dir / "best_model.pt", map_location="cpu", weights_only=True)

    # 第五步：先固化本轮新模型和预处理参数，再开展测试；全程未借用旧权重。
    selection = {**selected, "architecture_fixed_before_training": True,
                 "selected_members": [selected["candidate"]], "test_based_selection": False}
    save_json(output / "selection_before_test.json", selection)
    torch.save({"members": [checkpoint], "selection": selection, "combination": "single model"}, model_dir / "model.pt")
    torch.save(checkpoint, model_dir / "best_checkpoint.pt")
    torch.save(checkpoint["model_state_dict"], model_dir / "weights.pt")
    np.savez(model_dir / "standardization.npz", **{k: v.numpy() for k, v in checkpoint["statistics"].items()})
    save_json(model_dir / "model_info.json", {"config": config, "seed": training["seed"],
              "best_epoch": checkpoint["best_epoch"], "input_unit": "V", "input_sampling_rate_hz": 500,
              "input_shape": ["试次数", 8, 1500], "channels": checkpoint["channels"], "classes": CLASSES,
              "internal_sampling_rate_hz": 250, "internal_samples": 750})

    # 第六步：加载本轮选出的状态，在评估模式下分别统计训练、验证和测试结果。
    # 这里是最终汇总，不会把这些结果反馈给优化器；测试会话只推理一次。
    model = Model(config).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    evaluation = {}
    for split in SPLITS:
        probability = probabilities(model, transform_saved(data["X"][masks[split]], checkpoint), args.device)
        result = metrics(data["y"][masks[split]], probability)
        result["device"] = "ADS1299"
        result["split"] = split
        result["quality_flagged_trials_retained"] = int(np.count_nonzero(data["quality_flags"][masks[split]] != ""))
        evaluation[split] = result
        save_csv(output / f"{split}_predictions.csv", prediction_rows(data, masks[split], probability))
    save_json(output / "metrics.json", evaluation)
    save_csv(output / "summary.csv", [
        {"评估集合": name, "正确试次": evaluation[split]["correct"], "总试次": evaluation[split]["count"],
         "准确率": evaluation[split]["accuracy"], "宏F1": evaluation[split]["macro_f1"]}
        for split, name in zip(SPLITS, ["训练集（评估模式）", "验证会话", "内部测试会话"])])

    # 第七步：保存完整历史及中文图；测试曲线不会出现在逐轮训练监控中。
    history = read_history(candidate_dir / "history.csv")
    save_csv(output / "training_history.csv", history)
    plot_results(output, history, evaluation["test"], checkpoint["best_epoch"])
    save_json(ROOT / "results" / args.run_name / "run_info.json", {
        "run_name": args.run_name, "model": str((model_dir / "model.pt").relative_to(ROOT)),
        "ads1299_results": str(output.relative_to(ROOT)), "best_epoch": checkpoint["best_epoch"]})
    print(f"ADS1299 内部测试：{evaluation['test']['correct']}/128，准确率={evaluation['test']['accuracy']:.2%}", flush=True)
    print(f"本轮模型：{model_dir / 'model.pt'}", flush=True)


if __name__ == "__main__":
    main()
