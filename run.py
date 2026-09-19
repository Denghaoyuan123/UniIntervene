#!/usr/bin/env python3
"""Fold Towel offline training launcher; never starts a robot or online RL."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "research_impl"
CFG = json.loads((ROOT / "configs/fold_towel.json").read_text())
ORDER = ("nttg", "vf", "annotate", "mining", "vq1", "memory", "vq2")


def require(path: Path | None, label: str, dry_run: bool) -> None:
    if not dry_run and (path is None or not path.exists()):
        raise SystemExit(f"Missing {label}: {path}")


def overrides(items: dict[str, object]) -> list[str]:
    return [f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in items.items()]


def commands(a: argparse.Namespace, stage: str) -> tuple[list[str], dict[str, str], Path]:
    work = a.work_dir.resolve()
    nttg = work / "buffer_nttg"
    vf = work / "vf"
    scored = work / "buffer_nttg_vf"
    mined = work / "buffer_nttg_vf_mining_3rules"
    embedded = work / "buffer_nttg_vf_mining_3rules_emb"
    bank = work / "memory_bank/fold_towel_delta_bank.pt"
    cache = vf / "fold_towel_siglip_cache.pt"
    py = sys.executable
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    env["PYTHONPATH"] = os.pathsep.join((str(SRC), env.get("PYTHONPATH", "")))

    if stage == "nttg":
        require(a.raw_dir, "raw buffer", a.dry_run)
        return [py, str(SRC / "nttg_label_buffer.py"), "--buffer_dir", str(a.raw_dir),
                "--output_dir", str(nttg), "--fail_value", "-500"], env, nttg

    if stage == "vf":
        require(nttg, "NTTG buffer", a.dry_run)
        require(a.siglip, "SigLIP", a.dry_run)
        require(a.gemma, "Gemma", a.dry_run)
        v = CFG["value_function"]
        env.update({
            "PYTHON_BIN": py, "TRAIN_VF_SCRIPT": str(SRC / "JEPA/train_vf_stitched.py"),
            "VISION_MODEL": str(a.siglip), "LANGUAGE_MODEL": str(a.gemma),
            "BUFFER_DIR": str(nttg), "OUTPUT_DIR": str(vf), "CACHE_FILE": str(cache),
            "PROMPT": CFG["instruction"], "LOG_FILE": str(vf / "train.log"),
            "EPOCHS": str(v["epochs"]), "BATCH_SIZE": str(v["batch_size"]),
            "GAMMA": str(v["gamma"]), "LAMBDA_TD": str(v["lambda_td"]),
            "LAMBDA_LABEL": str(v["lambda_label"]), "LAMBDA_PROG": str(v["lambda_prog"]),
            "LAMBDA_CONS": str(v["lambda_cql"]), "USE_CONSERVATIVE": "1",
            "PROGRESS_WARMUP_EPOCHS": str(v["progress_warmup_epochs"]),
            "WARMUP_PROGRESS_ONLY": "1", "BALANCE_SUCCESS_FAIL": "1",
            "FAIL_SAMPLE_RATIO": str(v["fail_sample_ratio"]),
            "EARLY_STOP_PATIENCE": str(v["early_stop_patience"]),
            # Periodic checkpoints are ~1.6 GB each; the best model is saved separately.
            "SAVE_EVERY_EPOCHS": str(v.get("save_every_epochs", 0)),
            "USE_EXTERNAL_WRIST_KEYS": "1", "USE_WANDB": "0", "USE_CALIBRATION": "0",
        })
        return ["bash", str(SRC / "scripts/run_vf_stitched.sh")], env, vf

    if stage == "annotate":
        require(nttg, "NTTG buffer", a.dry_run)
        require(vf / "stitched_value_model_best.pt", "VF checkpoint", a.dry_run)
        return [py, str(SRC / "scripts/annotate_buffer_vf_value_pred.py"),
                "--buffer_dir", str(nttg), "--vf_dir", str(vf),
                "--output_dir", str(scored), "--cache_file", str(cache),
                "--device", "cuda" if a.gpu != "cpu" else "cpu"], env, scored

    if stage == "mining":
        require(scored, "VF-scored buffer", a.dry_run)
        m = CFG["mining"]
        cmd = [py, str(SRC / "scripts/mining_multitask_full_3rules.py"),
               "--buffer_dir", str(scored), "--output_dir", str(mined)]
        for key in ("window", "min_drop", "min_down_ratio", "slope_threshold", "flat_ratio",
                    "delta", "abrupt_min_drop", "recover_horizon", "recover_frac"):
            cmd += [f"--{key}", str(m[key])]
        return cmd, env, mined

    if stage == "vq1":
        require(mined, "mined VQ1 buffer", a.dry_run)
        require(vf / "stitched_value_model_best.pt", "VF checkpoint", a.dry_run)
        require(a.qwen, "Qwen3-VL-2B-Instruct", a.dry_run)
        q = CFG["vq1"]
        out = work / "vq1"
        ov = {
            "data.buffer_dir": mined, "data.use_external_wrist_keys": True,
            "data.memory_bank_path": bank,
            "data.proxy_vf.checkpoint_path": vf / "stitched_value_model_best.pt",
            "data.proxy_vf.source_name": "stitched_siglip_gemma_vf_fold_towel",
            "data.proxy_vf.is_latent_based": False,
            "model.backbone.model_path": a.qwen, "model.variant": q["variant"],
            "model.backbone.lora.r": q["lora_rank"],
            "model.backbone.lora.alpha": q["lora_alpha"],
            "model.backbone.lora.dropout": q["lora_dropout"],
            "model.future_latent_mode": q["future_latent_mode"],
            "model.future_target.source": "online_vjepa2",
            "model.future_target.vjepa_path": q["future_target"],
            "model.future_target.require_future_loss": True,
            "model.intervention.head_type": "tvr_history",
            "model.intervention.loss_type": "focal",
            "model.intervention.focal.alpha": q["focal_alpha"],
            "model.intervention.focal.gamma": q["focal_gamma"],
            "model.intervention.threshold": q["threshold"],
            "model.intervention.tvr.window": q["history_window"],
            "model.intervention.tvr.value_history_embed_dim": q["history_embed_dim"],
            "model.intervention.tvr.epsilon": q["epsilon"],
            "model.intervention.tvr.gamma": q["gamma_r"],
            "training.loss_weights.future": q["lambda_future"],
            "training.loss_weights.q": q["lambda_q"],
            "training.loss_weights.intervention": q["lambda_intervention"],
            "training.loss_weights.tvr": q["lambda_tvr"],
            "training.loss_weights.v_curr": q["lambda_v_curr"],
            "training.output_dir": out, "training.epochs": q["epochs"],
            "training.batch_size": q["batch_size"],
            "training.lr.backbone": q["lr_backbone"],
            "training.lr.heads": q["lr_heads"],
            "training.max_train_steps": 0,
            "training.wandb.enabled": False,
        }
        return [py, str(SRC / "hil_vlm_train.py"), "--config",
                str(SRC / "configs/vq1/full_tvr_history.yaml"), "--instr",
                CFG["instruction"], "--override", *overrides(ov)], env, out

    if stage == "memory":
        require(mined, "mined episode buffer", a.dry_run)
        require(a.qwen, "Qwen3-VL-2B-Instruct", a.dry_run)
        if a.vq1_ckpt is None:
            raise SystemExit("Memory construction requires --vq1-ckpt")
        require(a.vq1_ckpt, "VQ1 checkpoint", a.dry_run)
        m = CFG["memory"]
        cmd = [py, str(SRC / "scripts/build_vf_delta_bank_fast.py"),
               "--buffer_dir", str(mined), "--out", str(bank),
               "--report_json", str(bank.with_name("fold_towel_delta_bank_report.json")),
               "--base_model_path", str(a.qwen), "--lora_path", str(a.vq1_ckpt),
               "--instr", CFG["instruction"], "--split", "train",
               "--train_ratio", "0.8", "--val_ratio", "0.1", "--test_ratio", "0.1",
               "--split_seed", "42", "--query_embedding_output_dir", str(embedded),
               "--fixed_span", str(m["span"]), "--delta_thr", str(m["delta_threshold"]),
               "--progress_bins", str(m["progress_bins"]),
               "--target_items", str(m["target_items"]),
               "--max_per_episode", str(m["max_per_episode"]),
               "--target_vf_min", str(m["target_vf_min"]),
               "--action_chunk_length", str(CFG["vq2"]["recovery_horizon"])]
        return cmd, env, bank.parent

    if stage == "vq2":
        require(embedded, "query-embedded episode buffer", a.dry_run)
        require(bank, "memory bank", a.dry_run)
        require(a.qwen, "Qwen3-VL-2B-Instruct", a.dry_run)
        if a.vq1_ckpt is None:
            raise SystemExit("VQ2 requires --vq1-ckpt; checkpoint selection must be explicit")
        require(a.vq1_ckpt, "VQ1 checkpoint", a.dry_run)
        require(a.fast_tokenizer, "official FAST processor", a.dry_run)
        out = work / "vq2_fast"
        ov = {
            "model.backbone.model_path": a.qwen,
            "model.variant": "vq2_fast", "data.buffer_dir": embedded,
            "data.memory_bank_path": bank, "data.use_external_wrist_keys": True,
            "vq2.adapter_mode": "shared_residual",
            "vq2.vq1_shared_lora_path": a.vq1_ckpt,
            "vq2.residual_lora.r": CFG["vq2"]["lora_rank"],
            "vq2.residual_lora.alpha": CFG["vq2"]["lora_alpha"],
            "use_fast_decoder": True, "use_fast_decoder_v2": True,
            "fast_tokenizer_path": a.fast_tokenizer,
            "fast_decoder_v2_max_T": CFG["vq2"]["recovery_horizon"],
            "fast_decoder_v2_max_tokens": 128,
            "fast_decoder_v2_label_smoothing": 0.0,
            "round2_target_space": "action",
            "round2_bridge_steps": 0,
            "round2_sample_boost": 1.0,
            "round2_ce_only": True,
            "use_twin_aux": False,
            "disable_stage1_ce": True,
            "training.output_dir": out,
            "training.wandb.enabled": False,
        }
        return [py, str(SRC / "hil_vlm_train.py"), "--config",
                str(SRC / "configs/vq2/shared_residual.yaml"),
                "--instr", CFG["instruction"], "--override", *overrides(ov)], env, out
    raise AssertionError(stage)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("all", *ORDER), required=True)
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--siglip", type=Path)
    p.add_argument("--gemma", type=Path)
    p.add_argument("--qwen", type=Path)
    p.add_argument("--fast-tokenizer", type=Path,
                   help="local physical-intelligence/fast processor directory")
    p.add_argument("--vq1-ckpt", type=Path)
    p.add_argument("--gpu", default="0")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.stage == "all":
        if a.vq1_ckpt is None:
            a.vq1_ckpt = a.work_dir.resolve() / "vq1" / f"vq1_epoch{CFG['vq1']['epochs']}"
        stages = ORDER
    else:
        stages = (a.stage,)
    for stage in stages:
        cmd, env, output = commands(a, stage)
        print(f"[{stage}] {shlex.join(map(str, cmd))}", flush=True)
        if stage == "vf":
            print("[vf env] " + " ".join(f"{k}={env[k]}" for k in
                  ("GAMMA", "LAMBDA_TD", "LAMBDA_LABEL", "LAMBDA_PROG", "LAMBDA_CONS",
                   "PROGRESS_WARMUP_EPOCHS", "BALANCE_SUCCESS_FAIL", "FAIL_SAMPLE_RATIO")))
        if a.dry_run:
            continue
        output.mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd, check=True, env=env, cwd=SRC)


if __name__ == "__main__":
    main()
