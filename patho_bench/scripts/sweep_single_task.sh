#!/bin/bash

python sweep_single_task.py \
    --experiment_type linprobe \
    --model_name threads \
    --task_code bcnb--her2 \
    --combine_slides_per_patient True \
    --saveto ../../artifacts/experiments/single_task_example \
    --hyperparams_yaml "../../advanced_usage/configs/linprobe/linprobe.yaml" \
    --pooled_dirs_yaml "../../advanced_usage/configs/pooled_embeddings_paths.yaml" \
    --patch_dirs_yaml "../../advanced_usage/configs/patch_embeddings_paths.yaml" \
    --splits_root "../../artifacts/splits" \
    --gpu 0