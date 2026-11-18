"""在独立目录内检查可复现性、训练统计量隔离和跨设备窗口适配。"""

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

# 只添加本实验目录内的 code，不依赖原 ADS1299 项目所在的 Python 路径。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from ads1299_data import digest, fit_preprocessing, load_data, make_split
from ads1299_models import Model
from test_gnautilus import load_external
from training import set_seed


class ReproductionTests(unittest.TestCase):
    def test_copied_files_match_manifest(self):
        """对复制后的真实文件核对哈希，防止链接错文件或复制时发生改动。"""
        manifest = json.loads((ROOT / "provenance/data_copy_manifest.json").read_text())
        for row in manifest["files"]:
            path = ROOT / row["copy"]
            self.assertFalse(path.is_symlink())
            self.assertEqual(digest(path), row["sha256"])

    def test_original_session_split_is_preserved(self):
        """仍是544个唯一试次，完整会话/run固定为288/128/128。"""
        data = load_data(ROOT / "data/ADS1299")
        masks, _ = make_split(data)
        self.assertEqual(tuple(int(masks[s].sum()) for s in ("train", "validation", "test")), (288, 128, 128))
        self.assertEqual(len(set(data["trial_ids"])), 544)

    def test_initialization_exactly_matches_selected_architecture(self):
        """与本目录中归档的原模型代码比较初始化，确保只改注释和组织方式。"""
        settings = json.loads((ROOT / "selected_config.json").read_text())
        spec = importlib.util.spec_from_file_location("archived_models", ROOT / "provenance/original_code/ads1299_models.py")
        archived = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(archived)
        set_seed(42)
        first = archived.Model(settings["model"])
        set_seed(42)
        second = Model(settings["model"])
        self.assertEqual(first.state_dict().keys(), second.state_dict().keys())
        for key in first.state_dict():
            torch.testing.assert_close(first.state_dict()[key], second.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in second.parameters()), 9428)

    def test_validation_cannot_affect_training_statistics(self):
        """验证信号改变100倍并增加偏置，训练统计量必须完全不变。"""
        rng = np.random.default_rng(10)
        train = rng.normal(size=(5, 8, 1500)).astype(np.float32) * 1e-6
        val = rng.normal(size=(3, 8, 1500)).astype(np.float32) * 1e-6
        config = json.loads((ROOT / "selected_config.json").read_text())["model"]
        _, first = fit_preprocessing(train, val, config)
        _, second = fit_preprocessing(train, val * 100 + 1, config)
        for key in first:
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)

    def test_external_window_and_units(self):
        """g.Nautilus直接裁剪125:875，不再缩放单位或把旧窗口当新样本。"""
        x, y, rows = load_external()
        self.assertEqual(x.shape, (339, 8, 750))
        self.assertEqual(len(y), 339)
        self.assertEqual(sum(row["historical_g_split"] == "test" for row in rows), 95)
        with np.load(ROOT / "data/gNautilus/NumPy/S01_accepted_0to4_uv.npz") as source:
            np.testing.assert_array_equal(x[:119], source["X"][:, :, 125:875])


if __name__ == "__main__":
    unittest.main()
