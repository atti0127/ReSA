#!/bin/bash

set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/workspace/ldh/data/FSL}
SAVE_PATH=${SAVE_PATH:-output_cross_dataset_ori}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
TARGET_DATASETS=${TARGET_DATASETS:-cross_dataset}
read -r -a TARGET_DATASET_ARGS <<< "${TARGET_DATASETS}"

SHOTS=${SHOTS:-16}
FORCE_EVAL=${FORCE_EVAL:-0}

echo "Eval-only protocol: original CLIP-LoRA ImageNet source adapters."
echo "Target group: ${TARGET_DATASETS}"

for seed in 1 2 3; do
  RUN_DIR="${SAVE_PATH}/vitb16/imagenet/${SHOTS}shots/seed${seed}/cross_dataset"
  CHECKPOINT="${RUN_DIR}/lora_weights.pt"
  EVAL_LOG="${RUN_DIR}/eval_log.jsonl"

  if [ ! -f "${CHECKPOINT}" ]; then
    echo "Missing checkpoint for seed=${seed}: ${CHECKPOINT}"
    echo "Skipping this seed."
    continue
  fi

  if [ "${FORCE_EVAL}" != "1" ] && [ -f "${EVAL_LOG}" ]; then
    echo "Found existing eval log for seed=${seed}: ${EVAL_LOG}"
    echo "Skipping eval. Set FORCE_EVAL=1 to append a fresh eval."
    continue
  fi

  echo "Evaluating original CLIP-LoRA seed=${seed}: ${CHECKPOINT}"
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" main.py \
    --root_path "${DATA_ROOT}" \
    --dataset imagenet \
    --shots "${SHOTS}" \
    --seed "${seed}" \
    --setting cross_dataset \
    --save_path "${SAVE_PATH}" \
    --eval_only \
    --checkpoint_dataset imagenet \
    --target_datasets "${TARGET_DATASET_ARGS[@]}"
done

python parse_transfer_results.py "${SAVE_PATH}"
