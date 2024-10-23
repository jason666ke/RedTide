#!/bin/bash

# Define the base paths
base_path="/root/lhq/data/data_processed_cls2"
dest_path="all_datasets/AllRedTide"

# Create destination folder if it doesn't exist
mkdir -p "$dest_path"

# Iterate over all subdirectories in /root/lhq/data/data_processed_cls2
for dir in "$base_path"/*/; do
    if [ -d "$dir" ]; then
        # Get the folder name without the full path
        folder_name=$(basename "$dir")
        
        # Print the folder name
        echo "Processing folder: $folder_name"
        
        # Define source paths for current folder
        train_file="$dir/AllRedTide_TRAIN.ts"
        test_file="$dir/AllRedTide_TEST.ts"
        
        # Check if both files exist before copying
        if [ -f "$train_file" ] && [ -f "$test_file" ]; then
            # Copy the relevant files
            cp "$train_file" "$dest_path"
            cp "$test_file" "$dest_path"
            
            # Run the Python script
            python -u run.py \
              --task_name classification \
              --is_training 1 \
              --root_path "./$dest_path/" \
              --data UEA \
              --model_id AllRedTide \
              --model ModernTCN \
              --seq_len 24 \
              --ffn_ratio 1 \
              --patch_size 1 \
              --patch_stride 1 \
              --num_blocks 1 1 \
              --large_size 21 19 \
              --small_size 5 5 \
              --dims 256 512 \
              --head_dropout 0.0 \
              --class_dropout 0.0 \
              --dropout 0.5 \
              --itr 1 \
              --learning_rate 0.0001 \
              --batch_size 32 \
              --train_epochs 100 \
              --patience 20 \
              --des Exp \
              --use_multi_scale False
        else
            echo "Skipping folder: $folder_name (missing files)"
        fi
    fi
done