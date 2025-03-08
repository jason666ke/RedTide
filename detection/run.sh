#!/bin/bash

for dir in ../data_final/*/; do
    PREFIX=$(basename "$dir")
    MODEL_ID="${PREFIX}"
    ROOT_PATH="../data_final/${PREFIX}"
    OUTPUT_PATH="../results/${PREFIX}.csv"

    echo "Processing folder: $PREFIX"
    echo "MODEL_ID: $MODEL_ID"
    echo "ROOT_PATH: $ROOT_PATH"
    echo "OUTPUT_PATH: $OUTPUT_PATH"
    
    CUDA_VISIBLE_DEVICES=0 python -u run.py \
    --is_training 1 \
    --root_path "$ROOT_PATH" \
    --model_id "$MODEL_ID" \
    --output_path "$OUTPUT_PATH" \
    --model TimesNet \
    --data RedTide \
    --features M \
    --seq_len 24 \
    --pred_len 0 \
    --d_model 64 \
    --d_ff 64 \
    --e_layers 2 \
    --enc_in 9 \
    --c_out 9 \
    --top_k 3 \
    --anomaly_ratio 1 \
    --batch_size 64 \
    --train_epochs 1

  echo "-----------------------------------"
done

