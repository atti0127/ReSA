#!/bin/bash

set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/workspace/ldh/data/FSL}
SAVE_PATH=${SAVE_PATH:-output_domain_generalization}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}

for seed in 1 2 3; do
  echo "Training ImageNet source adapter for domain generalization, seed=${seed}"
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" main.py \
    --root_path "${DATA_ROOT}" \
    --dataset imagenet \
    --shots 16 \
    --seed "${seed}" \
    --setting domain_generalization \
    --save_path "${SAVE_PATH}"

  echo "Evaluating ImageNet source adapter on ImageNet domain shifts, seed=${seed}"
  torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" main.py \
    --root_path "${DATA_ROOT}" \
    --dataset imagenet \
    --shots 16 \
    --seed "${seed}" \
    --setting domain_generalization \
    --save_path "${SAVE_PATH}" \
    --eval_only \
    --checkpoint_dataset imagenet \
    --target_datasets domain_generalization
done
