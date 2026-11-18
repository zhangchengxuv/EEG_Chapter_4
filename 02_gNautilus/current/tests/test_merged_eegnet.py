"""验证合并划分、预处理隔离及频谱分支的物理含义。"""

import unittest

import numpy as np
import torch

from train_eegnet import ROOT, FREQUENCIES, fit_standardizer
from train_merged_eegnet import MergedModel, load_merged, prepare, spectral_scores


class MergedTests(unittest.TestCase):
    def test_original_blocks_are_preserved(self):
        """合并只增加试次数，三个集合的 trial_id 不能交叉。"""
        x, y, ids, rows, masks = load_merged(ROOT / "ML_DL_Dataset")
        self.assertEqual(x.shape, (339, 8, 1000))
        self.assertEqual(tuple(int(masks[s].sum()) for s in ("train", "validation", "test")), (202, 42, 95))
        groups = [set(ids[masks[s]]) for s in ("train", "validation", "test")]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        self.assertEqual(len(y), len(rows))

    def test_validation_cannot_change_fitted_statistics(self):
        """把验证信号放大并加上巨大偏置，所有训练统计量仍应完全相同。"""
        rng = np.random.default_rng(10)
        train = rng.normal(size=(8, 8, 1000)).astype(np.float32)
        validation = rng.normal(size=(4, 8, 1000)).astype(np.float32)
        config = {"harmonics": 2, "width": 1}
        stats = fit_standardizer(train)
        _, first = prepare(train, validation, config, stats)
        _, second = prepare(train, validation * 100 + 1000, config, stats)
        for name in first:
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)

    def test_synthetic_frequencies_match_the_correct_target(self):
        """四个无噪声正弦信号无需提供标签，频谱分支就应在相应候选处最强。"""
        time = np.arange(1000) / 250
        signals = np.array([np.tile(np.sin(2 * np.pi * f * time), (8, 1)) for f in FREQUENCIES], dtype=np.float32)
        scores = spectral_scores(signals, harmonics=1)
        np.testing.assert_array_equal(scores.mean(-1).argmax(1), np.arange(4))

    def test_hybrid_uses_both_branches(self):
        """合成数据反向传播时，两个分支的参数均应有有限梯度。"""
        torch.set_num_threads(2)
        model = MergedModel({"kind": "hybrid", "harmonics": 2, "raw_scale": 0.15})
        logits = model(torch.randn(3, 1, 8, 1000), torch.randn(3, 4, 16))
        self.assertEqual(tuple(logits.shape), (3, 4))
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 3])).backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
