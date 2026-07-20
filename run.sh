#!/bin/bash

set -euo pipefail

datasets=(imagenet dtd fgvc eurosat food101 
caltech101 oxford_flowers oxford_pets stanford_cars sun397 ucf101)
shots_list=(16)
settings=(standard base2new)

for dataset in "${datasets[@]}"; do
  for shots in "${shots_list[@]}"; do
    for seed in 1 2 3; do
      for setting in "${settings[@]}"; do
        echo "dataset=${dataset}, shots=${shots}, seed=${seed}, setting=${setting}"

        torchrun --standalone --nproc_per_node=4 main.py \
          --root_path /workspace/ldh/data/FSL \
          --dataset "${dataset}" \
          --shots "${shots}" \
          --seed "${seed}" \
          --backbone ViT-B/16 \
          --setting "${setting}" \
          --image_anchor_weight 1.0 \
          --text_anchor_weight 1.0 \
          --prototype_anchor_weight 1.0 \
          --v_rpr \
          --mrsa \
          --save_path "results/output_mrsa2"
      done
    done
  done
done
