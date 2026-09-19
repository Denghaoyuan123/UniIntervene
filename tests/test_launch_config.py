from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run
from unittest import mock


class PaperConfigTest(unittest.TestCase):
    def setUp(self):
        self.args = argparse.Namespace(
            raw_dir=Path("/tmp/raw"), work_dir=Path("/tmp/work"),
            siglip=Path("/tmp/siglip"), gemma=Path("/tmp/gemma"),
            qwen=Path("/tmp/qwen"), vq1_ckpt=Path("/tmp/vq1_epoch30"),
            fast_tokenizer=Path("/tmp/physical-intelligence-fast"),
            gpu="0", dry_run=True,
        )

    def test_paper_numbers(self):
        q = run.CFG["vq1"]
        m = run.CFG["memory"]
        self.assertEqual((q["history_window"], q["epochs"]), (8, 30))
        self.assertEqual((m["span"], m["progress_bins"], m["items_per_bin"], m["target_items"]),
                         (32, 20, 12, 240))
        self.assertEqual((run.CFG["vq2"]["recovery_horizon"], run.CFG["vq2"]["action_dim"]), (8, 7))

    def test_vf_env_forwards_objective(self):
        _, env, _ = run.commands(self.args, "vf")
        self.assertEqual(env["LAMBDA_LABEL"], "0.3")
        self.assertEqual(env["LAMBDA_CONS"], "0.05")
        self.assertEqual(env["WARMUP_PROGRESS_ONLY"], "1")

    def test_vq1_full_future_focal_and_tvr(self):
        cmd, _, _ = run.commands(self.args, "vq1")
        self.assertIn("model.variant=full", cmd)
        self.assertIn("model.future_target.require_future_loss=true", cmd)
        self.assertIn("model.intervention.loss_type=focal", cmd)
        self.assertIn("training.loss_weights.tvr=1.0", cmd)

    def test_bank_and_vq2_gate(self):
        cmd, _, _ = run.commands(self.args, "memory")
        self.assertEqual(cmd[cmd.index("--target_items") + 1], "240")
        self.assertEqual(cmd[cmd.index("--split") + 1], "train")
        self.assertEqual(cmd[cmd.index("--lora_path") + 1], "/tmp/vq1_epoch30")
        self.assertIn("--query_embedding_output_dir", cmd)
        cmd, _, _ = run.commands(self.args, "vq2")
        self.assertIn("use_fast_decoder_v2=true", cmd)
        self.assertIn("model.variant=vq2_fast", cmd)
        self.assertIn("round2_bridge_steps=0", cmd)
        self.assertIn("round2_sample_boost=1.0", cmd)
        self.assertIn("round2_ce_only=true", cmd)
        self.assertIn("use_twin_aux=false", cmd)
        self.assertIn("fast_tokenizer_path=/tmp/physical-intelligence-fast", cmd)

    def test_all_runs_vq2_and_uses_new_vq1_checkpoint(self):
        argv = [
            "run.py", "--stage", "all", "--raw-dir", "/tmp/raw",
            "--work-dir", "/tmp/work", "--siglip", "/tmp/siglip",
            "--gemma", "/tmp/gemma", "--qwen", "/tmp/qwen",
            "--fast-tokenizer", "/tmp/physical-intelligence-fast", "--dry-run",
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch("builtins.print") as printed:
            run.main()
        rendered = "\n".join(" ".join(map(str, call.args)) for call in printed.call_args_list)
        self.assertIn("[vq2]", rendered)
        self.assertIn("vq1_epoch30", rendered)


if __name__ == "__main__":
    unittest.main()
