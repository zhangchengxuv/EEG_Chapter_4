"""检查容易导致标签错位和测试泄漏的边界，并验证模型可正常反向传播。"""

import csv
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from train_eegnet import ROOT, EEGNet, load_subject


class DataSafetyTests(unittest.TestCase):
    def setUp(self):
        """在临时目录复制 S01，所有破坏性检查都不会修改原始数据。"""
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        (self.directory / "NumPy").mkdir()
        (self.directory / "Metadata").mkdir()
        self.metadata_path = self.directory / "Metadata/S01_trials.csv"
        shutil.copyfile(ROOT / "ML_DL_Dataset/NumPy/S01_accepted_0to4_uv.npz",
                        self.directory / "NumPy/S01_accepted_0to4_uv.npz")
        with (ROOT / "ML_DL_Dataset/Metadata/S01_trials.csv").open(encoding="utf-8-sig", newline="") as file:
            self.rows = list(csv.DictReader(file))
        self.write_metadata(self.rows)

    def tearDown(self):
        """检查完成后自动清理临时副本。"""
        self.temporary.cleanup()

    def write_metadata(self, rows):
        """将待检验的元数据写入临时文件。"""
        with self.metadata_path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_metadata_order_does_not_change_trial_alignment(self):
        """CSV 完全逆序后，标签与三个划分必须仍与 NPZ 试次一一对应。"""
        original = load_subject(self.directory, "S01")
        self.write_metadata(list(reversed(self.rows)))
        reordered = load_subject(self.directory, "S01")
        np.testing.assert_array_equal(original[1], reordered[1])
        np.testing.assert_array_equal(original[2], reordered[2])
        for split in original[4]:
            np.testing.assert_array_equal(original[4][split], reordered[4][split])

    def test_duplicate_trial_id_is_rejected(self):
        """重复主键不能静默覆盖，否则一对一关联会失效。"""
        self.write_metadata(self.rows + [self.rows[0]])
        with self.assertRaisesRegex(ValueError, "重复 trial_id"):
            load_subject(self.directory, "S01")

    def test_test_trial_cannot_be_moved_into_training(self):
        """如果测试区组被标成训练，必须在训练开始前报错。"""
        next(row for row in self.rows if row["split"] == "test")["split"] = "train"
        self.write_metadata(self.rows)
        with self.assertRaisesRegex(ValueError, "区组划分不一致"):
            load_subject(self.directory, "S01")


class ModelTests(unittest.TestCase):
    def test_forward_backward_and_constraints(self):
        """使用合成信号验证四分类形状、有限梯度与最大范数限制。"""
        torch.set_num_threads(2)
        model = EEGNet()
        logits = model(torch.randn(2, 1, 8, 1000))
        self.assertEqual(tuple(logits.shape), (2, 4))
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 3])).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        model.constrain_weights()
        self.assertLessEqual(float(model.classifier.weight.norm(dim=1).max()), 0.250001)
        self.assertLessEqual(float(model.spatial_conv.weight.flatten(1).norm(dim=1).max()), 1.000001)


if __name__ == "__main__":
    unittest.main()
