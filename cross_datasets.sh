#!/bin/bash

set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/workspace/ldh/data/FSL}
SAVE_PATH=${SAVE_PATH:-output_cross_dataset}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
TARGET_DATASETS=${TARGET_DATASETS:-cross_dataset}
read -r -a TARGET_DATASET_ARGS <<< "${TARGET_DATASETS}"

SHOTS=${SHOTS:-16}
echo "Protocol: train one ImageNet source adapter, then evaluate transfer targets."
echo "Target group: ${TARGET_DATASETS}"
echo "  cross_dataset          -> Caltech101, DTD, EuroSAT, FGVC, Food101, Flowers, Pets, Cars, SUN397, UCF101"
echo "  domain_generalization  -> ImageNetV2, ImageNet-Sketch, ImageNet-A, ImageNet-R"
echo "  all_transfer           -> both groups"
echo "Results are printed to stdout and eval-only results are appended to:"
echo "  ${SAVE_PATH}/vitb16/imagenet/${SHOTS}shots/seed*/cross_dataset/eval_log.jsonl"

for seed in 2 3; do
  CHECKPOINT="${SAVE_PATH}/vitb16/imagenet/${SHOTS}shots/seed${seed}/cross_dataset/lora_weights.pt"
  if [ -f "${CHECKPOINT}" ]; then
    echo "Found existing ImageNet source adapter for seed=${seed}: ${CHECKPOINT}"
    echo "Skipping training."
  else
    echo "Training ImageNet source adapter for transfer evaluation, seed=${seed}"
    torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" main.py \
      --root_path "${DATA_ROOT}" \
      --dataset imagenet \
      --shots "${SHOTS}" \
      --seed "${seed}" \
      --setting cross_dataset \
      --save_path "${SAVE_PATH}"
  fi

  echo "Evaluating ImageNet source adapter on ${TARGET_DATASETS} targets, seed=${seed}"
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
