"""Small CPU integration tests; no model downloads or experiment-store writes."""
import json
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch
from datasets import Dataset
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src import oracle, train
from src.data import collate
from src.sentence import SentenceEncoder


ROOT = Path(__file__).resolve().parents[1]


class TinyEncoder(SentenceEncoder):
    def __init__(self):
        super().__init__(normalize=False)
        self.register_buffer("states", torch.randn(20, 6, generator=torch.Generator().manual_seed(7)))

    def token_embeddings(self, ids, attention_mask):
        states = self.states[ids]
        context = self.pool(states, attention_mask)
        return states + context[:, None, :]


def tiny_data(*args, **kwargs):
    rows = []
    for ids in ([1, 2, 3], [4, 5], [6, 7, 8, 9]):
        rows.append({
            "ids": ids, "attn_mask": [1] * len(ids),
            "tokens": [str(i) for i in ids], "word_ids": list(range(len(ids))),
            "labels": [str(i % 3) for i in ids],
        })
    loader = DataLoader(Dataset.from_list(rows), batch_size=2, collate_fn=collate)
    return loader, loader, TinyEncoder(), None, {"0", "1", "2"}, None


class OracleFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(os.chdir, Path.cwd())
        self.cfg = OmegaConf.load(ROOT / "conf/config.yaml")
        self.cfg.task = "oracle"
        self.cfg.forge.store = self.temp.name
        self.cfg.forge.verbose = False
        self.cfg.runtime.threads = 1
        self.cfg.runtime.eval.short_log = True
        self.cfg.runtime.oracle.max_combinations = 2
        self.cfg.runtime.oracle.samples = 3
        self.cfg.model.loss.sweep_range = [0.5, 1.0, 2]
        self.cfg.train.epochs = 1

    def run_flow(self, entry=None):
        with (
            patch.object(train, "initialize_data", side_effect=tiny_data),
            patch.object(train, "get_logger", return_value=logging.getLogger("test")),
            patch.object(train, "save_eval_plots", return_value={}),
            patch.object(train, "save_train_eval_loss_plot"),
            patch.object(train, "run_stsb_sweep", return_value=(0.8, {0.5: 0.7}, {0.5: 0.6})) as stsb,
        ):
            self.assertEqual((entry or train.main)(self.cfg), 0)
        return Path.cwd(), stsb.call_count

    def test_oracle_uses_shared_evaluation_without_training(self):
        self.cfg.train.no_train = True
        self.cfg.train["continue"] = True
        self.cfg.runtime.compile = True
        # Historically these training switches do not suppress oracle search.
        self.cfg.runtime.eval.skip = True
        with patch.object(torch.optim, "AdamW") as optimizer, patch.object(torch, "compile") as compile_model:
            run, stsb_calls = self.run_flow()
        optimizer.assert_not_called()
        compile_model.assert_not_called()
        self.assertEqual(stsb_calls, 0)
        self.assertFalse((run / "state").exists())
        self.assertEqual(json.loads((run / "meta.json").read_text())["status"], "done")
        summary = json.loads((run / "data/oracle_summary.json").read_text())
        self.assertEqual(summary["sentences"], 3)
        self.assertEqual(summary["searched_pairs"], 3)
        self.assertAlmostEqual(summary["exhaustive_fraction"], 1 / 3)
        metrics = json.loads((run / "metrics.json").read_text())
        self.assertEqual(metrics["eval_loss"], summary["reconstruction_loss"])
        self.assertEqual(metrics["epochs_completed"], 0)
        details = json.loads((run / "metrics_details.json").read_text())
        self.assertEqual(details["training"]["epochs_target"], 0)
        with np.load(run / "data/oracle_masks.npz") as masks:
            np.testing.assert_array_equal(np.diff(masks["offsets_0"]), [2, 1, 2])
            np.testing.assert_array_equal(np.diff(masks["offsets_1"]), [3, 2, 4])
        self.assertTrue((run / "data/selection_rate_curves.json").exists())
        self.assertTrue((run / "data/selection_logreg.json").exists())

    def test_legacy_entry_point_and_optional_stsb(self):
        self.cfg.runtime.oracle.stsb = True
        run, stsb_calls = self.run_flow(oracle.main)
        self.assertEqual(stsb_calls, 1)
        self.assertTrue((run / "data/spearman_curves.json").exists())

    def test_selector_still_trains_and_saves_checkpoints(self):
        self.cfg.task = "rationale"
        run, stsb_calls = self.run_flow()
        self.assertEqual(stsb_calls, 1)
        self.assertTrue((run / "state/models/model_1.pth").exists())
        self.assertFalse((run / "data/oracle_masks.npz").exists())
        self.assertEqual(json.loads((run / "metrics.json").read_text())["epochs_completed"], 1)

    def test_wrong_task_rejected_before_creating_run(self):
        self.cfg.task = "ner"
        with self.assertRaises(ValueError):
            train.main(self.cfg)
        with self.assertRaises(ValueError):
            oracle.main(self.cfg)
        self.assertFalse((Path(self.temp.name) / "xps").exists())

    def test_tf32_restored_after_search_failure(self):
        selector = oracle.OracleSelector(TinyEncoder(), device="cuda")
        before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        with self.assertRaisesRegex(RuntimeError, "search failed"):
            with selector.search_precision():
                self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
                raise RuntimeError("search failed")
        self.assertEqual(before, (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32))


if __name__ == "__main__":
    unittest.main()
