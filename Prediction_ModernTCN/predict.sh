#!/bin/bash
MODEL = "大鹏湾下沙"
ROOT_PATH="../data_drop/dpwxs"
PICS_BASE_PATH="./pics"
RESULTS_PATH="./results"
RESULTS_FILE="prediction_2021_2022.csv"
DATA_FILE="../data_drop/大鹏湾下沙.csv"

start_date = "2022-12-31"


python - <<END
import pandas as pd
df = pd.read_csv("$DATA_FILE")
df['date'] = pd.to_datetime(df['date'])
df = df[df['date'] < pd.to_datetime("$start_date")]
output_file = f"$ROOT_PATH/$start_date.csv"
df.to_csv(output_file, index=False)
print(f"数据已保存到: {output_file}")
END

python -u run.py \
    --is_training 1 \
    --model_id "MODEL" \
    --model ModernTCN \
    --root_path "$ROOT_PATH" \
    --data_path "${start_date}.csv" \
    --result_path "$RESULTS_PATH" \
    --result_file "$RESULTS_FILE" \
    --data custom \
    --features S \
    --seq_len 336 \
    --pred_len 72 \
    --ffn_ratio 16 \
    --patch_size 16 \
    --patch_stride 4 \
    --num_blocks 2 \
    --large_size 51 \
    --small_size 5 \
    --dims 64 64 64 64 \
    --head_dropout 0.0 \
    --enc_in 1 \
    --dropout 0.1 \
    --itr 1 \
    --target "chlorophyll" \
    --train_epochs 25 \
    --batch_size 512 \
    --patience 20 \
    --inverse \
    --scale True \
    --learning_rate 0.0001 \
    --des Exp \
    --lradj type3 \
    --do_predict
