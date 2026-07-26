#!/usr/bin/env bash
set -euo pipefail

# DSpark online training for Qwen3-8B.
#
# DSpark = DFlash block-diffusion drafter + EAGLE-style Markov & confidence heads,
# trained with cross-entropy + L1 distribution distillation + confidence BCE.
# The L1 / confidence losses need the target model's FINAL hidden state, so the
# target backend must surface it. The 'hf' backend (default below) always does;
# the 'sglang' backend does when its runner returns both the captured aux stream
# and the final hidden state. To train CE-only (no target final hidden state),
# pass: --l1-loss-alpha 0 --no-confidence-head --ce-loss-alpha 1.0

if [[ -z "${PYTHON_ENV_PREFIX:-}" ]]; then
  echo "Set PYTHON_ENV_PREFIX to a Python/Conda/venv environment prefix." >&2
  exit 2
fi
PYTHON_BIN="${PYTHON_ENV_PREFIX%/}/bin/python"
[[ -x "${PYTHON_BIN}" ]] || { echo "Python is not executable: ${PYTHON_BIN}" >&2; exit 1; }
export PYTHON_ENV_PREFIX PYTHON_BIN
PYTHON_BIN_DIR="$(dirname -- "${PYTHON_BIN}")"
export PATH="${PYTHON_BIN_DIR}:${PATH:-/usr/local/bin:/usr/bin:/bin}"

# Node rank is provided by mpirun.sh as the first positional arg
# (OMPI_COMM_WORLD_RANK). Falls back to env override, then 0.
NODE_RANK="${NODE_RANK:-${1:-0}}"

# Hostfile shared by mpirun; its line order matches the rank assignment
# (-npernode 1), so the first entry is rank 0 == the rendezvous master.
HOSTFILE="${HOSTFILE:-/etc/mpi/mpi-hostfile}"

# Auto-detect MASTER_ADDR (first host in the hostfile) and NNODES (line count)
# unless the caller overrides them explicitly.
if [[ -z "${MASTER_ADDR:-}" || -z "${NNODES:-}" ]]; then
  if [[ -r "$HOSTFILE" ]]; then
    mapfile -t _hosts < <(awk 'NF && $1 !~ /^#/ {print $1}' "$HOSTFILE")
    if (( ${#_hosts[@]} > 0 )); then
      MASTER_ADDR="${MASTER_ADDR:-${_hosts[0]}}"
      NNODES="${NNODES:-${#_hosts[@]}}"
    fi
  fi
fi

if [[ -z "${MASTER_ADDR:-}" ]]; then
  echo "Could not auto-detect MASTER_ADDR from hostfile '$HOSTFILE'; set MASTER_ADDR explicitly." >&2
  exit 1
fi
NNODES="${NNODES:-2}"

export http_proxy="${http_proxy:-http://oversea-squid1.jp.txyun:11080}"
export https_proxy="${https_proxy:-http://oversea-squid1.jp.txyun:11080}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com}"

export LD_PRELOAD="${LD_PRELOAD:-/nlp_group/chenjiapeng/spec_test/libnccl.so.2.27.7.ubuntu-cuda128.fix7}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SPECFORGE_DIR="${SPECFORGE_DIR:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
KSYNC_DIR="${KSYNC_DIR:-$(dirname -- "$SPECFORGE_DIR")/ksync-dev}"

TARGET_MODEL_PATH="${TARGET_MODEL_PATH:-Qwen/Qwen3-14B}"
DRAFT_CONFIG_PATH="${DRAFT_CONFIG_PATH:-$SPECFORGE_DIR/configs/qwen3-14b-dspark-block7.json}"
DATA_PATH="${DATA_PATH:-}"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29544}"

TRIAL_NAME="${TRIAL_NAME:-qwen3-14b-dspark-perfectblend}"
OUTPUT_DIR="${OUTPUT_DIR:-$SPECFORGE_DIR/outputs/qwen3-14b-dspark}"
LOG_DIR="${LOG_DIR:-$SPECFORGE_DIR/logs/train}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${TRIAL_NAME}.rank${NODE_RANK}.log}"

NUM_EPOCHS="${NUM_EPOCHS:-6}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-6e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.04}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-qwen}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flex_attention}"
LOSS_DECAY_GAMMA="${LOSS_DECAY_GAMMA:-4.0}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-specforge-qwen3-14b-dspark}"
TARGET_BACKEND="${TARGET_BACKEND:-sglang}"
BLOCK_SIZE="${BLOCK_SIZE:-7}"
NUM_ANCHORS="${NUM_ANCHORS:-512}"
MARKOV_RANK="${MARKOV_RANK:-256}"
CE_LOSS_ALPHA="${CE_LOSS_ALPHA:-0.1}"
L1_LOSS_ALPHA="${L1_LOSS_ALPHA:-0.9}"
CONFIDENCE_HEAD_ALPHA="${CONFIDENCE_HEAD_ALPHA:-1.0}"
WANDB_NAME="${WANDB_NAME:-qwen3-14b-dspark-perfectblend}"

# shellcheck disable=SC1091
. "$KSYNC_DIR/scripts/lib/load_hf_env.sh"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
export HF_HOME="${HF_HOME:-$SPECFORGE_DIR/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$SPECFORGE_DIR/cache/compiled_kernels}"
export SPECFORGE_DATA_NUM_PROC="${SPECFORGE_DATA_NUM_PROC:-32}"
export PYTHONPATH="$SPECFORGE_DIR${PYTHONPATH:+:$PYTHONPATH}"

cmd=(
  "${PYTHON_BIN}" -m torch.distributed.run
  --nnodes "$NNODES"
  --node_rank "$NODE_RANK"
  --nproc_per_node "$GPUS_PER_NODE"
  --master_addr "$MASTER_ADDR"
  --master_port "$MASTER_PORT"
  scripts/train_dspark.py
  --target-model-path "$TARGET_MODEL_PATH"
  --draft-config-path "$DRAFT_CONFIG_PATH"
  --train-data-path "$DATA_PATH"
  --output-dir "$OUTPUT_DIR"
  --num-epochs "$NUM_EPOCHS"
  --batch-size "$BATCH_SIZE"
  --learning-rate "$LEARNING_RATE"
  --warmup-ratio "$WARMUP_RATIO"
  --max-grad-norm "$MAX_GRAD_NORM"
  --max-length "$MAX_LENGTH"
  --chat-template "$CHAT_TEMPLATE"
  --attention-backend "$ATTENTION_BACKEND"
  --loss-decay-gamma "$LOSS_DECAY_GAMMA"
  --log-interval "$LOG_INTERVAL"
  --save-interval "$SAVE_INTERVAL"
  --report-to "$REPORT_TO"
  --wandb-project "$WANDB_PROJECT"
  --target-model-backend "$TARGET_BACKEND"
  --block-size "$BLOCK_SIZE"
  --num-anchors "$NUM_ANCHORS"
  --markov-rank "$MARKOV_RANK"
  --enable-confidence-head
  --confidence-head-with-markov
  --ce-loss-alpha "$CE_LOSS_ALPHA"
  --l1-loss-alpha "$L1_LOSS_ALPHA"
  --confidence-head-alpha "$CONFIDENCE_HEAD_ALPHA"
  --wandb-name "$WANDB_NAME"
)

cd "$SPECFORGE_DIR"
{
  date "+[%Y-%m-%d %H:%M:%S %Z] Starting Qwen3-14B DSpark 4K online train"
  echo "trial_name=$TRIAL_NAME"
  echo "nnodes=$NNODES node_rank=$NODE_RANK master_addr=$MASTER_ADDR master_port=$MASTER_PORT"
  echo "target_backend=$TARGET_BACKEND attention_backend=$ATTENTION_BACKEND"
  echo "target_model_path=$TARGET_MODEL_PATH"
  echo "draft_config_path=$DRAFT_CONFIG_PATH"
  echo "data_path=$DATA_PATH"
  echo "output_dir=$OUTPUT_DIR"
  echo "mode=online (target model generates hidden states on-the-fly)"
  printf 'command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
} | tee "$LOG_FILE"
"${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"
