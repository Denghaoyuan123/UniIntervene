# Fold Towel data contract

```text
raw/buffer/transitions_<episode>.pkl
work/buffer_nttg/transitions_<episode>_nttg.pkl
work/vf/stitched_value_model_best.pt
work/buffer_nttg_vf/transitions_<episode>_nttg_vf.pkl
work/buffer_nttg_vf_mining_3rules/transitions_<episode>_nttg_vf.pkl
work/vq1/vq1_epoch<N>/
work/memory_bank/fold_towel_delta_bank.pt
work/memory_bank/fold_towel_delta_bank_report.json
work/buffer_nttg_vf_mining_3rules_emb/transitions_<episode>_nttg_vf.pkl
work/vq2_fast/
```

Each episode file is a non-empty `list[dict]`. Required transition fields:

| Key | Contract |
|---|---|
| `observations`, `next_observations` | dictionaries containing `external`, `wrist`, and finite state values |
| `actions` | finite 7D vector; dimension 7 is the gripper command |
| `rewards` or `reward` | scalar reward |
| `dones`, `masks` | optional termination and validity fields |
| `task_name` or task text | task identity used for stratification and memory routing |

NTTG adds `raw_value` and `raw_norm`. VF annotation adds finite `vf_value_pred`. Mining adds `vf_mining_decline`, `vf_mining_plateau`, `vf_mining_abrupt_drop`, and their union `vf_mining_intervention_label`.

Memory construction uses only the deterministic train split. Each retained entry contains task identity, VQ1 failure and target embeddings, source indices, an 8-step 7D recovery action chunk, proxy-value statistics, and robot-state metadata.

The memory stage also writes an episode directory in which every positive intervention query has `failure_emb_1d`. VQ2 requires this field and does not synthesize retrieval embeddings from class labels.

For Fold Towel, the memory segment span is 32 and the decoded recovery horizon is 8. The retrieved goal is the span endpoint; the supervised action chunk is the first 8 raw 7D actions directed toward that goal. The seventh dimension carries the gripper command and is tokenized together with the arm motion.

Before VQ2, verify the report contains exactly 240 items, the expected bin allocation, `min_vf_delta >= 0.40`, `min_target_vf >= 0.75`, and a nonzero query embedding count. Raw pickle and PyTorch artifacts must come from trusted sources.
