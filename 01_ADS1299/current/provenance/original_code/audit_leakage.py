"""审查满分结果：核验常见泄漏、固定刺激顺序、推理负对照及打乱标签训练。

仅用于审计，不选择新超参数，不改动生产模型、latest_run 或原始数据。
打乱标签实验只训练一次，固定原模型选中的轮数，不以对照验证成绩选轮次。
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ads1299_data import ROOT, digest, fit_preprocessing, load_data, make_split, save_csv, save_json, signal_transform
from ads1299_models import Model
from predict_ads1299 import predict_saved
from train_ads1299 import probabilities, set_seed


def sequence_control(data, masks, output):
    """不看 EEG，仅用训练会话的 run 内位置建立查表，检验标签与顺序的混淆。"""
    labels = defaultdict(list)
    for row, label, keep in zip(data["rows"], data["y"], masks["train"]):
        if keep:
            labels[int(float(row["protocol_trial"]))].append(int(label))
    lookup = {index: int(np.bincount(values, minlength=4).argmax()) for index, values in labels.items()}
    save_csv(output / "position_lookup_from_training.csv", [
        {"position_in_run": index, "training_examples": len(labels[index]),
         "unique_training_labels": len(set(labels[index])), "predicted_label": lookup[index]}
        for index in sorted(lookup)])
    results = {}
    for split in ("validation", "test"):
        indices = np.flatnonzero(masks[split])
        prediction = np.array([lookup.get(int(float(data["rows"][i]["protocol_trial"])), -1) for i in indices])
        correct = int(np.sum(prediction == data["y"][indices]))
        results[split] = {"correct": correct, "count": len(indices), "accuracy": correct / len(indices)}
    return {"method": "training position-in-run majority label, without EEG",
            "all_training_positions_have_unique_class": all(len(set(v)) == 1 for v in labels.values()),
            "results": results,
            "interpretation": "表明刺激顺序与类别完全混淆；EEGNet 没有输入序号，故不能仅由此证明它实际利用了顺序。"}


def similarity_control(data, masks):
    """检查完全重复及零时移波形近重复；不能替代对原始连续采集文件的审查。"""
    import hashlib
    hashes = [hashlib.sha256(trial.tobytes()).hexdigest() for trial in data["X"]]
    x = signal_transform(data["X"]).astype(np.float64)
    x -= x.mean(-1, keepdims=True)
    flat = x.reshape(len(x), -1)
    flat /= np.linalg.norm(flat, axis=1, keepdims=True).clip(1e-15)
    results = {"unique_signal_contents": len(set(hashes)), "total_trials": len(x)}
    for split in ("validation", "test"):
        correlation = flat[masks[split]] @ flat[masks["train"]].T
        closest = np.abs(correlation).argmax(1)
        strength = np.abs(correlation).max(1)
        results[split] = {
            "median_max_absolute_zero_lag_correlation": float(np.median(strength)),
            "largest_absolute_zero_lag_correlation": float(strength.max()),
            "zero_lag_nearest_waveform_label_accuracy": float(np.mean(data["y"][masks["train"]][closest] == data["y"][masks[split]])),
        }
    return results


def frozen_model_controls(data, masks, model_path):
    """不重新训练：打乱输入顺序、零输入、独立白噪声输入，均只评估验证会话。"""
    rng = np.random.default_rng(20261117)
    val_x, val_y = data["X"][masks["validation"]], data["y"][masks["validation"]]
    original = predict_saved(val_x, model_path)
    order = rng.permutation(len(val_x))
    shuffled = predict_saved(val_x[order], model_path)[np.argsort(order)]
    zeros = predict_saved(np.zeros_like(val_x), model_path)
    # 噪声幅度仅由训练集估计，不使用验证/测试类别信息。
    train_x = data["X"][masks["train"]]
    scale = train_x.std(axis=(0, 2), keepdims=True)
    noise_x = (rng.normal(size=val_x.shape) * scale).astype(np.float32)
    noise = predict_saved(noise_x, model_path)
    return {"original_validation_accuracy": float(np.mean(original.argmax(1) == val_y)),
            "input_order_max_probability_difference": float(np.max(abs(original - shuffled))),
            "input_order_labels_identical": bool(np.array_equal(original.argmax(1), shuffled.argmax(1))),
            "zero_input_accuracy": float(np.mean(zeros.argmax(1) == val_y)),
            "independent_noise_accuracy": float(np.mean(noise.argmax(1) == val_y)),
            "interpretation": "输入序号未被显式使用；零/噪声对照只能检查部分程序错误，不能证明所有隐蔽混淆都不存在。"}


def shuffled_label_control(data, masks, checkpoint, device, output):
    """训练标签独立打乱，固定训练轮数，最终只评估一次真实验证标签。"""
    config = checkpoint["config"]
    epochs = int(checkpoint["best_epoch"])
    train_x, train_y = data["X"][masks["train"]], data["y"][masks["train"]]
    val_x, val_y = data["X"][masks["validation"]], data["y"][masks["validation"]]
    pairs, _ = fit_preprocessing(train_x, val_x, config)
    shuffled_y = np.random.default_rng(271828).permutation(train_y)
    np.save(output / "shuffled_training_labels.npy", shuffled_y)
    set_seed(42)
    model = Model(config).to(device)
    model.constrain_weights()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    loader = DataLoader(TensorDataset(*pairs[0], torch.from_numpy(shuffled_y)), batch_size=32,
                        shuffle=True, generator=torch.Generator().manual_seed(42))
    history = []
    print(f"打乱训练标签负对照：固定 {epochs} 轮；训练过程中不计算验证成绩。", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for x, spectrum, label in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x.to(device), spectrum.to(device)), label.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            model.constrain_weights()
            total += loss.item() * len(label)
        history.append({"epoch": epoch, "training_loss_shuffled_labels": total / len(train_y)})
        if epoch == 1 or epoch % 100 == 0:
            print(f"  epoch={epoch}, shuffled_train_loss={total / len(train_y):.4f}", flush=True)
    train_p = probabilities(model, pairs[0], device)
    val_p = probabilities(model, pairs[1], device)
    save_csv(output / "shuffled_label_training_history.csv", history)
    np.save(output / "shuffled_label_validation_probabilities.npy", val_p)
    return {"label_permutation_seed": 271828, "initialization_seed": 42, "epochs_fixed": epochs,
            "validation_based_epoch_selection": False,
            "training_accuracy_against_shuffled_labels": float(np.mean(train_p.argmax(1) == shuffled_y)),
            "validation_accuracy_against_real_labels": float(np.mean(val_p.argmax(1) == val_y)),
            "validation_correct": int(np.sum(val_p.argmax(1) == val_y)), "validation_count": len(val_y),
            "interpretation": "这是一次固定随机置换负对照，不是多次置换检验，也不能排除固定顺序或刺激伪迹混淆。"}


def generalization_gap(data, masks, run_dir):
    """用同一评估模式测训练表现，并从既有测试记录读取测试表现，不重跑测试。"""
    train_y = data["y"][masks["train"]]
    p = predict_saved(data["X"][masks["train"]], run_dir / "selected_model/model.pt")
    with (run_dir / "selected_model/test_predictions.csv").open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    test_p = np.array([[float(row["probability_" + c]) for c in "ABCD"] for row in rows])
    test_y = np.array([int(row["true_label"]) for row in rows])
    return {"train_correct_eval_mode": int(np.sum(p.argmax(1) == train_y)), "train_count": len(train_y),
            "train_accuracy_eval_mode": float(np.mean(p.argmax(1) == train_y)),
            "train_cross_entropy_eval_mode": float(-np.log(p[np.arange(len(train_y)), train_y].clip(1e-12)).mean()),
            "test_accuracy_from_saved_records": float(np.mean(test_p.argmax(1) == test_y)),
            "test_cross_entropy_from_saved_records": float(-np.log(test_p[np.arange(len(test_y)), test_y].clip(1e-12)).mean())}


def main():
    """输出独立审计报告，不修改既有训练结果或继续选择生产模型。"""
    parser = argparse.ArgumentParser(description="ADS1299 满分结果的泄漏与混淆审计")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/ads1299_session_search")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/leakage_audit_20261117")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    data = load_data(ROOT / "SSVEP_ML_Dataset")
    masks, _ = make_split(data)
    package = torch.load(args.run_dir / "selected_model/model.pt", map_location="cpu", weights_only=True)
    import json
    plan = json.loads((args.run_dir / "search_plan.json").read_text())
    report = {"dataset_file": data["path"].name, "trial_count": len(data["X"]),
              "audit_script_sha256": digest(Path(__file__)),
              "source_hashes_match_original_run": all(digest(ROOT / p) == h for p, h in plan["source_sha256"].items()),
              "input_hashes_match_original_run": all(digest(ROOT / "SSVEP_ML_Dataset" / p) == h for p, h in plan["input_sha256"].items()),
              "session_and_run_groups_disjoint": True,
              "model_does_not_take_trial_id_position_frequency_or_label_as_input": True,
              "raw_continuous_recordings_present_in_package": False,
              "upstream_export_pipeline_independently_audited": False,
              "test_used_to_tune_new_model": False}
    report["position_only_control"] = sequence_control(data, masks, args.output_dir)
    report["signal_similarity"] = similarity_control(data, masks)
    report["frozen_model_controls"] = frozen_model_controls(data, masks, args.run_dir / "selected_model/model.pt")
    report["frozen_model_generalization_gap"] = generalization_gap(data, masks, args.run_dir)
    save_json(args.output_dir / "audit.json", report)
    print("固定刺激序列、重复信号及冻结模型对照已完成。", flush=True)
    report["shuffled_label_control"] = shuffled_label_control(data, masks, package["members"][0], args.device, args.output_dir)
    save_json(args.output_dir / "audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
