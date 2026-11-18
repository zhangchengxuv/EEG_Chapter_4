"""检查跨会话泄漏、数据单位、频谱含义和模型训练/保存接口。"""

import csv
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ads1299_data import (ROOT, FREQUENCIES, fit_preprocessing, load_data, make_split,
                          signal_transform, spectral_features, transform_saved)
from ads1299_models import Model


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """核验原始包后只读共享；破坏性检查在临时元数据副本上进行。"""
        cls.data = load_data(ROOT / "SSVEP_ML_Dataset")

    def test_whole_session_and_run_are_disjoint(self):
        """每个 session/run 都只能属于一个集合，不能随机拆散。"""
        masks, _ = make_split(self.data)
        self.assertEqual(tuple(int(masks[s].sum()) for s in ("train", "validation", "test")), (288, 128, 128))
        poisoned = dict(self.data)
        poisoned["run_ids"] = self.data["run_ids"].copy()
        poisoned["run_ids"][np.flatnonzero(masks["validation"])[0]] = self.data["run_ids"][0]
        with self.assertRaisesRegex(ValueError, "跨集合泄漏"):
            make_split(poisoned)

    def test_csv_reordering_and_duplicate_detection(self):
        """元数据逆序仍正确对齐，重复 trial_id 必须报错；原始包不改动。"""
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory / "npz_metadata").mkdir()
            source = self.data["path"]
            (directory / source.name).symlink_to(source)
            info_name = source.stem + ".metadata.json"
            shutil.copyfile(source.parent / "npz_metadata" / info_name, directory / "npz_metadata" / info_name)
            table = directory / "npz_metadata" / (source.stem + ".trials.csv")
            rows = list(reversed(self.data["rows"]))
            for duplicate in (False, True):
                with table.open("w", encoding="utf-8-sig", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows + [rows[0]] if duplicate else rows)
                if duplicate:
                    with self.assertRaisesRegex(ValueError, "trial_id"):
                        load_data(directory)
                else:
                    loaded = load_data(directory)
                    self.assertEqual([r["trial_id"] for r in loaded["rows"]], self.data["trial_ids"].tolist())

    def test_no_cross_trial_filtering_and_correct_unit(self):
        """微伏幅度的 V 信号经处理仍为约 1 μV；批处理与单试次处理相同。"""
        time = np.arange(1500) / 500
        x = np.stack([np.tile(np.sin(2 * np.pi * 9 * time) * 1e-6 * scale, (8, 1)) for scale in (1, 20)])
        result = signal_transform(x)
        self.assertEqual(result.shape, (2, 8, 750))
        self.assertAlmostEqual(float(result[0, 0, 20:-20].std()), 2 ** -0.5, delta=0.02)
        for i in range(2):
            np.testing.assert_allclose(result[i], signal_transform(x[i:i + 1])[0], rtol=0, atol=0)

    def test_validation_cannot_change_training_statistics(self):
        """验证幅度和偏置改变不能影响通道或频谱的任何训练统计量。"""
        train = self.data["X"][:5]
        validation = self.data["X"][300:304]
        config = {"family": "hybrid", "band": [4, 45]}
        _, first = fit_preprocessing(train, validation, config)
        _, second = fit_preprocessing(train, validation * 100 + 1, config)
        for key in first:
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)

    def test_frequency_features_need_no_labels(self):
        """纯正弦输入在自己的候选频率处最强；函数没有标签或元数据参数。"""
        time = np.arange(750) / 250
        x = np.array([np.tile(np.sin(2 * np.pi * f * time), (8, 1)) for f in FREQUENCIES])
        result = spectral_features(x, harmonics=1)
        np.testing.assert_array_equal(result.mean(-1).argmax(1), np.arange(4))


class ModelTests(unittest.TestCase):
    def test_all_families_forward_backward_and_saved_transform(self):
        """四类模型均可训练，保存的统计量能精确复现同一输入变换。"""
        torch.set_num_threads(2)
        rng = np.random.default_rng(12)
        x = rng.normal(size=(4, 8, 1500)).astype(np.float32) * 1e-6
        for family in ("standard", "energy", "spectral", "hybrid"):
            config = {"family": family, "kernel": 125, "filters": 8}
            pairs, statistics = fit_preprocessing(x[:2], x[2:], config)
            model = Model(config)
            logits = model(*pairs[0])
            self.assertEqual(tuple(logits.shape), (2, 4))
            torch.nn.functional.cross_entropy(logits, torch.tensor([0, 3])).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            model.constrain_weights()
            restored = transform_saved(x[2:], {"config": config, "statistics": statistics})
            for expected, actual in zip(pairs[1], restored):
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
