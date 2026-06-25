#!/usr/bin/env bash
set -euo pipefail
NODE_RANK=$1
export http_proxy="${http_proxy:-http://oversea-squid1.jp.txyun:11080}"
export https_proxy="${https_proxy:-http://oversea-squid1.jp.txyun:11080}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com}"

export PATH="/opt/conda/envs/py312/bin:$PATH"
export LD_PRELOAD=/nlp_group/chenjiapeng/spec_test/libnccl.so.2.27.7.ubuntu-cuda128.fix7
SPECFORGE_DIR="${SPECFORGE_DIR:-/mmu_mllm_hdd_3/renjunchi/SpecForge-private}"
KSYNC_DIR="${KSYNC_DIR:-/mmu_mllm_hdd_3/renjunchi/ksync-dev}"
DATASETS_DIR="${DATASETS_DIR:-/mmu_mllm_hdd_3/renjunchi/datasets}"
MODEL_DIR="${MODEL_DIR:-/mmu_mllm_hdd_3/renjunchi/models}"

TARGET_MODEL_PATH="${TARGET_MODEL_PATH:-$MODEL_DIR/Qwen2.5-VL-7B-Instruct}"
DRAFT_MODEL_CONFIG="${DRAFT_MODEL_CONFIG:-$SPECFORGE_DIR/configs/qwen2-5-vl-7b-128k-eagle3-full-vocab.json}"
DATA_PATH="${DATA_PATH:-$DATASETS_DIR/spec_jsonl/allava4v_infra_perfectblend_mix_all_valid_images_plus_en_long_0430_len128000_truncated_seed42.jsonl}"
HIDDEN_PATH="${HIDDEN_PATH:-$DATASETS_DIR/specforge_runs/duanwu_qwen2_5_vl_7b_mix_plus_en_long_truncated_fresh_20260619_150833/hidden_states_qwen2_5_vl_7b_mix_plus_en_long_truncated_fresh_len128000_tp2}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$SPECFORGE_DIR/outputs}"
LOG_ROOT="${LOG_ROOT:-$KSYNC_DIR/logs/train/qwen2_5_vl_7b}"
TRIAL_NAME="${TRIAL_NAME:-h800-qwen2_5_VL_7b-eagle3-P_A_I_128-full_vocab-sp16-len128000}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$TRIAL_NAME}"
TRAIN_CACHE="${TRAIN_CACHE:-/mmu_mllm_hdd_3/renjunchi/SpecForge-private/cache/allava4v_mix_plus_en_long_full_vocab_sp_len128000_cache/train_cache_len128000}"
COMPILED_KERNELS_CACHE="${COMPILED_KERNELS_CACHE:-/mmu_mllm_hdd_3/renjunchi/SpecForge-private/cache/allava4v_mix_plus_en_long_full_vocab_sp_len128000_cache/compiled_kernels}"

NNODES="${NNODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-10.82.113.27}"
MASTER_PORT="${MASTER_PORT:-29543}"
NODE_RANK="${NODE_RANK:?set NODE_RANK=0 on master and NODE_RANK=1 on worker}"

MAX_LENGTH="${MAX_LENGTH:-128000}"
MIN_PIXELS="${MIN_PIXELS:-50176}"
MAX_PIXELS="${MAX_PIXELS:-802816}"
DIST_TIMEOUT="${DIST_TIMEOUT:-720}"

SP_ULYSSES_SIZE="${SP_ULYSSES_SIZE:-4}"
SP_RING_SIZE="${SP_RING_SIZE:-4}"

TRAIN_BUILD_DATASET_NUM_PROC="${TRAIN_BUILD_DATASET_NUM_PROC:-128}"
TRAIN_DATALOADER_NUM_WORKERS="${TRAIN_DATALOADER_NUM_WORKERS:-16}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
TTT_LENGTH="${TTT_LENGTH:-7}"
DRAFT_ACCUMULATION_STEPS="${DRAFT_ACCUMULATION_STEPS:-1}"
REPORT_TO="${REPORT_TO:-none}"
PROFILE="${PROFILE:-0}"
PROFILE_START_STEP="${PROFILE_START_STEP:-1}"
PROFILE_NUM_STEPS="${PROFILE_NUM_STEPS:-2}"
MAX_NUM_STEPS="${MAX_NUM_STEPS:-}"
VOCAB_MAPPING_PATH="${VOCAB_MAPPING_PATH:-/mmu_mllm_hdd_3/renjunchi/SpecForge-private/outputs/full_vocab_mapping.pt}"

. "$KSYNC_DIR/scripts/lib/load_hf_env.sh"

mkdir -p "$OUTPUT_DIR" "$LOG_ROOT" "$TRAIN_CACHE" "$COMPILED_KERNELS_CACHE"

export HF_HOME="${HF_HOME:-$DATASETS_DIR/.hf-cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$DATASETS_DIR/.hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$COMPILED_KERNELS_CACHE}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
#export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-1}"
#export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
#export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-ALL}"
export PYTHONPATH="$SPECFORGE_DIR${PYTHONPATH:+:$PYTHONPATH}"

for required_path in "$TARGET_MODEL_PATH" "$DRAFT_MODEL_CONFIG" "$DATA_PATH" "$HIDDEN_PATH"; do
  if [ ! -e "$required_path" ]; then
    echo "Required path does not exist: $required_path" >&2
    exit 1
  fi
done

LOG_FILE="$LOG_ROOT/${TRIAL_NAME}_node${NODE_RANK}.log"

EXTRA_ARGS=()
if [ -n "${VOCAB_MAPPING_PATH:-}" ]; then
  EXTRA_ARGS+=(--vocab-mapping-path "$VOCAB_MAPPING_PATH")
fi
if [ -n "${MAX_NUM_STEPS:-}" ]; then
  EXTRA_ARGS+=(--max-num-steps "$MAX_NUM_STEPS")
fi
if [ "${RESUME:-0}" = "1" ]; then
  EXTRA_ARGS+=(--resume)
fi
if [ -n "${CKPT_DIR:-}" ]; then
  EXTRA_ARGS+=(--ckpt-dir "$CKPT_DIR")
fi

cmd=(
  torchrun
  --nnodes "$NNODES"
  --nproc_per_node "$GPUS_PER_NODE"
  --node_rank "$NODE_RANK"
  --master_addr "$MASTER_ADDR"
  --master_port "$MASTER_PORT"
  scripts/train_eagle3.py
  --target-model-path "$TARGET_MODEL_PATH"
  --trust-remote-code
  --draft-model-config "$DRAFT_MODEL_CONFIG"
  --train-data-path "$DATA_PATH"
  --train-hidden-states-path "$HIDDEN_PATH"
  --build-dataset-num-proc "$TRAIN_BUILD_DATASET_NUM_PROC"
  --dataloader-num-workers "$TRAIN_DATALOADER_NUM_WORKERS"
  --output-dir "$OUTPUT_DIR"
  --num-epochs "$NUM_EPOCHS"
  --batch-size "$BATCH_SIZE"
  --learning-rate "$LEARNING_RATE"
  --max-length "$MAX_LENGTH"
  --chat-template qwen2-vl
  --target-model-backend sglang
  --cache-dir "$TRAIN_CACHE"
  --embedding-key model.embed_tokens.weight
  --lm-head-key lm_head.weight
  --tp-size 1
  --attention-backend usp
  --sp-ulysses-size "$SP_ULYSSES_SIZE"
  --sp-ring-size "$SP_RING_SIZE"
  --ttt-length "$TTT_LENGTH"
  --draft-accumulation-steps "$DRAFT_ACCUMULATION_STEPS"
  --is-vlm
  --min-pixels "$MIN_PIXELS"
  --max-pixels "$MAX_PIXELS"
  --dist-timeout "$DIST_TIMEOUT"
  --save-interval "$SAVE_INTERVAL"
  --log-interval "$LOG_INTERVAL"
  --report-to "$REPORT_TO"
)

if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
  cmd+=("${EXTRA_ARGS[@]}")
fi

case "$PROFILE" in
  1|true|TRUE|yes|YES|on|ON)
    RUN_MODE="profile train"
    cmd+=(
      --profile
      --profile-start-step "$PROFILE_START_STEP"
      --profile-num-steps "$PROFILE_NUM_STEPS"
    )
    ;;
  0|false|FALSE|no|NO|off|OFF)
    RUN_MODE="train"
    ;;
  *)
    echo "Invalid PROFILE=$PROFILE; expected 0/1, true/false, yes/no, or on/off." >&2
    exit 1
    ;;
esac

cd "$SPECFORGE_DIR"
{
  date '+[%Y-%m-%d %H:%M:%S %Z] Starting full Qwen2.5-VL-7B SpecForge training'
  echo "trial_name=$TRIAL_NAME"
  echo "node_rank=$NODE_RANK"
  echo "nnodes=$NNODES"
  echo "gpus_per_node=$GPUS_PER_NODE"
  echo "master_addr=$MASTER_ADDR"
  echo "master_port=$MASTER_PORT"
  echo "data_path=$DATA_PATH"
  echo "hidden_path=$HIDDEN_PATH"
  echo "output_dir=$OUTPUT_DIR"
  echo "train_cache=$TRAIN_CACHE"
  echo "sp_ulysses_size=$SP_ULYSSES_SIZE"
  echo "sp_ring_size=$SP_RING_SIZE"
} | tee -a "$LOG_FILE"
"${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"
