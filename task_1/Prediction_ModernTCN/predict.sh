#!/bin/bash
MODEL="大鹏湾下沙"
ROOT_PATH="../data_9features"
PICS_BASE_PATH="./pics"
# RESULTS_PATH="../predict_results"
RESULTS_PATH="../../task_2/data_final"
RESULTS_FILE="prediction_2021_2022.csv"
DATA_FILE="../data_9features/大鹏湾下沙.csv"
start_date="2022-12-31"

# 检查数据文件是否存在
if [ ! -f "$DATA_FILE" ]; then
    echo "错误：数据文件 $DATA_FILE 不存在"
    exit 1
fi

# 创建必要的目录
mkdir -p "$ROOT_PATH"
mkdir -p "$PICS_BASE_PATH"
mkdir -p "$RESULTS_PATH"

# 数据预处理
python - <<END
import pandas as pd
import sys

try:
    df = pd.read_csv("$DATA_FILE")
    if df.empty:
        print("错误：数据文件为空")
        sys.exit(1)
        
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] < pd.to_datetime("$start_date")]
    
    if df.empty:
        print("错误：筛选后的数据为空")
        sys.exit(1)
        
    output_file = f"$ROOT_PATH/$start_date.csv"
    df.to_csv(output_file, index=False)
    print(f"数据已保存到: {output_file}")
except Exception as e:
    print(f"错误：数据处理失败 - {str(e)}")
    sys.exit(1)
END

# 检查Python脚本的执行结果
if [ $? -ne 0 ]; then
    echo "数据预处理失败"
    exit 1
fi

python -u run.py \
    --is_training 1 \
    --model_id "$MODEL" \
    --model ModernTCN \
    --root_path "$ROOT_PATH" \
    --data_path "${start_date}.csv" \
    --result_path "$RESULTS_PATH" \
    --result_file "$RESULTS_FILE" \
    --data custom \
    --features M \
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
    --enc_in 9 \
    --dropout 0.1 \
    --itr 1 \
    --train_epochs 25 \
    --batch_size 512 \
    --patience 20 \
    --inverse \
    --scale True \
    --learning_rate 0.0001 \
    --des Exp \
    --lradj type3 \
    --checkpoints "./checkpoints_M/" \
    --do_predict
