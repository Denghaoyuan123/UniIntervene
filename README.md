<div align="center">

# UniIntervene: Agentic Intervention for Efficient Real-World Reinforcement Learning

[![CoRL 2026](https://img.shields.io/badge/CoRL_2026-Accepted-success)](https://denghaoyuan123.github.io/UniIntervene-project/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.57%2B-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/docs/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Haoyuan Deng](https://github.com/Denghaoyuan123), Yitong Gao, Yudong Lin, [Haichao Liu](https://henryhcliu.github.io/), Zhenyu Wu, [Ziwei Wang](https://ziweiwangthu.github.io/)<sup>†</sup>

Nanyang Technological University · Beijing University of Posts and Telecommunications

**Accepted to the Conference on Robot Learning (CoRL) 2026**

**[[Project Page](https://denghaoyuan123.github.io/UniIntervene-project/)] | [[Paper](https://arxiv.org/pdf/2606.12372)] | [[arXiv](https://arxiv.org/abs/2606.12372)] | [[Code](https://github.com/Denghaoyuan123/UniIntervene)]**

</div>

**UniIntervene** is an agentic intervention framework for efficient real-world reinforcement learning. It learns when intervention is needed, retrieves a relevant recovery trajectory from memory, and generates corrective actions through a vision-language-action policy. Across the reported real-world tasks, UniIntervene improves average success by **8.6%**, reduces human interventions by **57%**, and reaches **88%** average task success with a **14.6%** intervention rate.

This repository provides the UniIntervene offline training pipeline: NTTG labeling, proxy value training, intervention mining, VQ1, recovery memory, and VQ2. Robot deployment and HIL-SERL integration are outside this package.

The release contains one Fold Towel example configuration that exercises the full pipeline end to end. Memory settings (span, bins, thresholds) are task- and data-dependent; the per-task values used in the paper are listed in its appendix. The release does not include robot trajectories, checkpoints, memory banks, credentials, or machine-specific configuration.

## Pipeline

| Stage | Input | Output |
|---|---|---|
| NTTG | episode pickle files | `raw_value`, `raw_norm` |
| VF | NTTG episodes and observations | frozen proxy value function |
| Annotation | frozen VF | `vf_value_pred` per transition |
| Mining | value-scored episodes | three-rule intervention labels |
| VQ1 | mined episodes and frozen V-JEPA2 | future, twin-Q, risk, and current-value heads |
| Memory | train split and VQ1 checkpoint | verified recovery targets and query-embedded episodes |
| VQ2 | train memory and query-embedded episodes | goal-conditioned FAST recovery policy |

Episode files are `list[dict]` pickle files. Each transition contains `observations`, `next_observations`, a 7D `actions` vector, reward/termination fields, and task metadata. The observation dictionaries contain two RGB views and robot state. Only load pickle files from trusted sources.

The memory builder uses the deterministic episode-level train split. A fixed-span candidate is retained only when both conditions hold:

- `end_vf - start_vf >= delta_threshold`
- `end_vf >= target_vf_min`

For Fold Towel the example uses 20 progress bins, 12 items per bin, span 32, and a 0.40 value-improvement threshold. Query embeddings are generated with the same VQ1 checkpoint used to encode memory keys. Validation queries use the train memory and never use validation or test episodes as memory entries.

VQ2 sends the retrieved raw 8-step, 7D recovery trajectory directly to the `physical-intelligence/fast` processor: orthonormal DCT, scale-and-round quantization, BPE, EOS/PAD masking, and per-token cross entropy. The seventh action dimension is the gripper command.

## Usage

Inspect commands without launching training:

```bash
python run.py --stage all --raw-dir /path/to/fold_towel/buffer \
  --work-dir /path/to/output --qwen /path/to/Qwen3-VL-2B-Instruct \
  --siglip /path/to/SigLIP-SO400M --gemma /path/to/Gemma-3-270M \
  --fast-tokenizer /path/to/physical-intelligence/fast --dry-run
```

With `--stage all`, the memory and VQ2 stages use the `vq1_epoch30` checkpoint produced under the selected work directory. For an individual memory or VQ2 run, pass `--vq1-ckpt` explicitly.

Run stages in order:

```bash
python run.py --stage nttg --raw-dir /path/to/buffer --work-dir /path/to/output
python run.py --stage vf --raw-dir /path/to/buffer --work-dir /path/to/output \
  --siglip /path/to/SigLIP-SO400M --gemma /path/to/Gemma-3-270M
python run.py --stage annotate --raw-dir /path/to/buffer --work-dir /path/to/output
python run.py --stage mining --raw-dir /path/to/buffer --work-dir /path/to/output
python run.py --stage vq1 --raw-dir /path/to/buffer --work-dir /path/to/output \
  --qwen /path/to/Qwen3-VL-2B-Instruct
python run.py --stage memory --raw-dir /path/to/buffer --work-dir /path/to/output \
  --qwen /path/to/Qwen3-VL-2B-Instruct --vq1-ckpt /path/to/vq1_epoch30
python run.py --stage vq2 --raw-dir /path/to/buffer --work-dir /path/to/output \
  --qwen /path/to/Qwen3-VL-2B-Instruct --vq1-ckpt /path/to/vq1_epoch30 \
  --fast-tokenizer /path/to/physical-intelligence-fast
```

Validate input schema before training:

```bash
python validate_data.py --raw-dir /path/to/buffer --max-episodes 20
```

The FAST smoke test uses a small set of action chunks to verify tokenizer round-trip and decoder optimization:

```bash
python tools/smoke_fast_overfit.py --buffer-dir /path/to/mined_buffer \
  --fast-tokenizer /path/to/physical-intelligence/fast --chunks 16 --steps 300
```

The included smoke-test record reports token CE `7.6259 -> 0.00351`, token accuracy `1.0`, and tokenizer reconstruction MAE `0.00404`. It is a component test, not a held-out policy evaluation.

## Reproducibility and safety

- Use local, reviewed model snapshots and record their revisions and checksums. The FAST revision used by this release is recorded in `third_party_revisions.json`.
- W&B is disabled by default in every launcher.
- Remote model code is disabled by default for the VF/VQA path.
- Memory reports contain basenames and aggregate statistics, not absolute data paths.
- Build the memory from the train split and keep validation/test episodes out of it.
- Memory thresholds interact with value-function calibration: if the bank comes out empty or short of `target_items`, lower `delta_threshold`/`target_vf_min` or raise `span` for your data. Verify `selected_stats.min_vf_delta`, `selected_stats.min_target_vf`, item count, bin counts, and query embedding count before starting VQ2.

The data contract is documented in [`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md), and release provenance is documented in [`CODE_PROVENANCE.md`](CODE_PROVENANCE.md).

# 🔗 Citation

If you find this repository helpful, please consider citing:

```bibtex
@inproceedings{deng2026uniintervene,
  title = {UniIntervene: Agentic Intervention for Efficient Real-World Reinforcement Learning},
  author = {Deng, Haoyuan and Gao, Yitong and Lin, Yudong and Liu, Haichao and Wu, Zhenyu and Wang, Ziwei},
  booktitle = {Conference on Robot Learning (CoRL)},
  year = {2026}
}
```
