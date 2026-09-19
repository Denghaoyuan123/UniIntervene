#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONDA_ENV_PREFIX="${CONDA_PREFIX:-$(dirname "$(dirname "$PYTHON_BIN")")}" 
SANITIZE_RUNTIME_ENV="${SANITIZE_RUNTIME_ENV:-1}"
if [[ "$SANITIZE_RUNTIME_ENV" == "1" ]]; then
  unset LD_PRELOAD || true
  export LD_LIBRARY_PATH="${CONDA_ENV_PREFIX}/lib"
fi

BUFFER_DIR="${BUFFER_DIR:-new_data/multitask_vf/buffer_nttg}"
BUFFER_BASENAME="$(basename "$BUFFER_DIR")"
if [[ "$BUFFER_BASENAME" == buffer* ]]; then
  TASK_NAME_RAW="$(basename "$(dirname "$BUFFER_DIR")")"
else
  TASK_NAME_RAW="$BUFFER_BASENAME"
fi
TASK_NAME="$(printf '%s' "$TASK_NAME_RAW" | tr -cs 'A-Za-z0-9._-' '_')"
OUTPUT_DIR="${OUTPUT_DIR:-runs/vf_stitched_${TASK_NAME}}"
CACHE_FILE="${CACHE_FILE:-${OUTPUT_DIR}/${TASK_NAME}_siglip_cache_v7.pt}"
VISION_MODEL="${VISION_MODEL:-google/siglip-so400m-patch14-384}"
LANGUAGE_MODEL="${LANGUAGE_MODEL:-google/gemma-3-270m-it}"
PROMPT="${PROMPT:-}"
VQA_DATASET="HuggingFaceM4/VQAv2"
VQA_MANIFEST="${VQA_MANIFEST:-}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
ALLOW_VQA_FALLBACK="${ALLOW_VQA_FALLBACK:-0}"

EPOCHS="${EPOCHS:-20}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-24}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
PHASE1_BATCH_SIZE="${PHASE1_BATCH_SIZE:-256}"
PROJECTOR_MICRO_BATCH_SIZE="${PROJECTOR_MICRO_BATCH_SIZE:-0}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
VAL_RATIO="${VAL_RATIO:-0.1}"
TEST_RATIO="${TEST_RATIO:-0.1}"
GAMMA="${GAMMA:-0.99}"
PROGRESS_WARMUP_EPOCHS="${PROGRESS_WARMUP_EPOCHS:-5}"
LAMBDA_TD="${LAMBDA_TD:-1.0}"
LAMBDA_LABEL="${LAMBDA_LABEL:-0.3}"
LAMBDA_MC="${LAMBDA_MC:-0.0}"
LAMBDA_PROG="${LAMBDA_PROG:-1.0}"
ALPHA_PROG="${ALPHA_PROG:-1.0}"
TERMINAL_SUCCESS_REWARD="${TERMINAL_SUCCESS_REWARD:-1.0}"
TERMINAL_FAILURE_REWARD="${TERMINAL_FAILURE_REWARD:-0.0}"
PROGRESS_ON_SUCCESS_ONLY="${PROGRESS_ON_SUCCESS_ONLY:-1}"
BETA="${BETA:-1.0}"
USE_CONSERVATIVE="${USE_CONSERVATIVE:-1}"
LAMBDA_CONS="${LAMBDA_CONS:-0.05}"
USE_CALIBRATION="${USE_CALIBRATION:-0}"
LAMBDA_CAL="${LAMBDA_CAL:-0.1}"
LAMBDA_VQA="${LAMBDA_VQA:-0.0}"
SUCCESS_WEIGHT="${SUCCESS_WEIGHT:-1.0}"
FAIL_WEIGHT="${FAIL_WEIGHT:-1.0}"
BALANCE_SUCCESS_FAIL="${BALANCE_SUCCESS_FAIL:-1}"
FAIL_SAMPLE_RATIO="${FAIL_SAMPLE_RATIO:-0.5}"
WARMUP_PROGRESS_ONLY="${WARMUP_PROGRESS_ONLY:-0}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-6}"
EVAL_EVERY_EPOCHS="${EVAL_EVERY_EPOCHS:-5}"
VQA_NUM_SAMPLES="${VQA_NUM_SAMPLES:-4096}"
VQA_BATCH_SIZE="${VQA_BATCH_SIZE:-1}"
VQA_MAX_LENGTH="${VQA_MAX_LENGTH:-192}"
VQA_STREAMING="${VQA_STREAMING:-1}"
VQA_TRUST_REMOTE_CODE="${VQA_TRUST_REMOTE_CODE:-0}"
USE_ACTION_COND="${USE_ACTION_COND:-0}"
USE_DELTA_ACTION="${USE_DELTA_ACTION:-0}"
ACTION_DIM="${ACTION_DIM:-0}"
ACTION_HIDDEN_DIM="${ACTION_HIDDEN_DIM:-256}"
USE_EXTERNAL_WRIST_KEYS="${USE_EXTERNAL_WRIST_KEYS:-0}"
USE_WANDB="${USE_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-vf_stitched}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"

LOG_FILE="${LOG_FILE:-logs/train_vf_stitched.log}"
mkdir -p logs "$OUTPUT_DIR"

echo "[run_vf_stitched] starting stitched value-learning training..."
echo "  buffer_dir  : $BUFFER_DIR"
echo "  task_name   : $TASK_NAME"
echo "  output_dir  : $OUTPUT_DIR"
if [[ -n "$CACHE_FILE" ]]; then
  echo "  cache_file  : $CACHE_FILE"
fi
echo "  backbone    : SigLIP + Gemma + Value Head"
echo "  prompt      : $PROMPT"
echo "  epochs      : $EPOCHS  batch_size=$BATCH_SIZE  eval_batch_size=$EVAL_BATCH_SIZE"
echo "  projector   : micro_batch=$PROJECTOR_MICRO_BATCH_SIZE  phase1_batch_size=$PHASE1_BATCH_SIZE"
echo "  save_every  : $SAVE_EVERY_EPOCHS  eval_every=$EVAL_EVERY_EPOCHS"
echo "  lr          : $LR  weight_decay=$WEIGHT_DECAY  grad_clip=$GRAD_CLIP_NORM"
echo "  gamma       : $GAMMA  warmup_epochs=$PROGRESS_WARMUP_EPOCHS"
echo "  lambdas     : td=$LAMBDA_TD mc=$LAMBDA_MC prog=$LAMBDA_PROG alpha_prog=$ALPHA_PROG cons=$LAMBDA_CONS cal=$LAMBDA_CAL vqa=$LAMBDA_VQA"
echo "  rewards     : success=$TERMINAL_SUCCESS_REWARD failure=$TERMINAL_FAILURE_REWARD progress_on_success_only=$PROGRESS_ON_SUCCESS_ONLY"
echo "  balancing   : success_w=$SUCCESS_WEIGHT fail_w=$FAIL_WEIGHT balance=$BALANCE_SUCCESS_FAIL fail_ratio=$FAIL_SAMPLE_RATIO warmup_only=$WARMUP_PROGRESS_ONLY patience=$EARLY_STOP_PATIENCE"
echo "  vqa         : ${VQA_MANIFEST:-$VQA_DATASET}  num_samples=$VQA_NUM_SAMPLES batch_size=$VQA_BATCH_SIZE streaming=$VQA_STREAMING trust_remote_code=$VQA_TRUST_REMOTE_CODE fallback=$ALLOW_VQA_FALLBACK"
echo "  action_cond : use=$USE_ACTION_COND delta=$USE_DELTA_ACTION action_dim=$ACTION_DIM action_hidden_dim=$ACTION_HIDDEN_DIM"
echo "  obs_keys    : use_external_wrist_keys=$USE_EXTERNAL_WRIST_KEYS (0=side_policy_256/wrist_1, 1=external/wrist)"
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  echo "  resume      : $RESUME_CHECKPOINT"
fi
echo "  wandb       : $USE_WANDB  project=$WANDB_PROJECT"
echo "  CUDA        : $CUDA_VISIBLE_DEVICES"
echo "  sanitize    : $SANITIZE_RUNTIME_ENV  conda_env=$CONDA_ENV_PREFIX"
echo "  hf_hub      : disable_xet=$HF_HUB_DISABLE_XET  hf_transfer=$HF_HUB_ENABLE_HF_TRANSFER"
echo "  log         : $LOG_FILE"

TRAIN_VF_SCRIPT="${TRAIN_VF_SCRIPT:-$PWD/JEPA/train_vf_stitched.py}"
if [[ ! -f "$TRAIN_VF_SCRIPT" ]]; then
  echo "[run_vf_stitched] error: TRAIN_VF_SCRIPT not found: $TRAIN_VF_SCRIPT" >&2
  exit 2
fi

CMD=("$PYTHON_BIN" -u "$TRAIN_VF_SCRIPT"
    --buffer_dir "$BUFFER_DIR"
    --vision_model "$VISION_MODEL"
    --language_model "$LANGUAGE_MODEL"
    --prompt "$PROMPT"
    --output_dir "$OUTPUT_DIR"
    --cache_file "$CACHE_FILE"
    --epochs $EPOCHS
    --save_every_epochs $SAVE_EVERY_EPOCHS
    --batch_size $BATCH_SIZE
    --eval_batch_size $EVAL_BATCH_SIZE
    --phase1_batch_size $PHASE1_BATCH_SIZE
    --projector_micro_batch_size $PROJECTOR_MICRO_BATCH_SIZE
    --lr $LR
    --weight_decay $WEIGHT_DECAY
    --grad_clip_norm $GRAD_CLIP_NORM
    --val_ratio $VAL_RATIO
    --test_ratio $TEST_RATIO
    --gamma $GAMMA
    --progress_warmup_epochs $PROGRESS_WARMUP_EPOCHS
    --lambda_td $LAMBDA_TD
    --lambda_label $LAMBDA_LABEL
    --lambda_mc $LAMBDA_MC
    --lambda_prog $LAMBDA_PROG
    --alpha_prog $ALPHA_PROG
    --terminal_success_reward $TERMINAL_SUCCESS_REWARD
    --terminal_failure_reward $TERMINAL_FAILURE_REWARD
    --progress_on_success_only $PROGRESS_ON_SUCCESS_ONLY
    --beta $BETA
    --lambda_cons $LAMBDA_CONS
    --lambda_cal $LAMBDA_CAL
    --lambda_vqa $LAMBDA_VQA
    --success_weight $SUCCESS_WEIGHT
    --fail_weight $FAIL_WEIGHT
    --balance_success_fail $BALANCE_SUCCESS_FAIL
    --fail_sample_ratio $FAIL_SAMPLE_RATIO
    --warmup_progress_only $WARMUP_PROGRESS_ONLY
    --early_stop_patience $EARLY_STOP_PATIENCE
    --eval_every_epochs $EVAL_EVERY_EPOCHS
    --vqa_dataset "$VQA_DATASET"
    --vqa_num_samples $VQA_NUM_SAMPLES
    --vqa_batch_size $VQA_BATCH_SIZE
    --vqa_max_length $VQA_MAX_LENGTH
    --vqa_streaming $VQA_STREAMING
    --vqa_trust_remote_code $VQA_TRUST_REMOTE_CODE
    --use_action_cond $USE_ACTION_COND
    --use_delta_action $USE_DELTA_ACTION
    --action_dim $ACTION_DIM
    --action_hidden_dim $ACTION_HIDDEN_DIM
    --use_external_wrist_keys $USE_EXTERNAL_WRIST_KEYS
    --allow_vqa_fallback $ALLOW_VQA_FALLBACK
    --wandb_project $WANDB_PROJECT)

if [[ -n "$VQA_MANIFEST" ]]; then
  CMD+=(--vqa_manifest "$VQA_MANIFEST")
fi

if [[ -n "$RESUME_CHECKPOINT" ]]; then
  CMD+=(--resume_checkpoint "$RESUME_CHECKPOINT")
fi

if [[ "$USE_CONSERVATIVE" == "1" ]]; then
  CMD+=(--use_conservative)
fi

if [[ "$USE_CALIBRATION" == "1" ]]; then
  CMD+=(--use_calibration)
fi

if [[ "$USE_WANDB" == "1" ]]; then
  CMD+=(--use_wandb)
fi

if [[ -n "$WANDB_RUN_NAME" ]]; then
  CMD+=(--wandb_run_name "$WANDB_RUN_NAME")
fi

"${CMD[@]}" 2>&1 | tee "$LOG_FILE"

echo ""
echo "[run_vf_stitched] training finished. Outputs are in $OUTPUT_DIR/"
echo "Next stage: python run.py --stage annotate ..."
