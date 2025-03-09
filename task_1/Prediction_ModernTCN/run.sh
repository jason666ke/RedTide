# CUDA_VISIBLE_DEVICES=0 python -u run.py \
#     --is_training 1 \
#     --model_id "大亚湾东山" \
#     --model ModernTCN \
#     --root_path "../data_drop/" \
#     --data_path "大亚湾东山_2023_2024_9.csv" \
#     --data custom \
#     --features S \
#     --seq_len 336 \
#     --pred_len 72 \
#     --ffn_ratio 16 \
#     --patch_size 16 \
#     --patch_stride 4 \
#     --num_blocks 2 \
#     --large_size 51 \
#     --small_size 5 \
#     --dims 64 64 64 64 \
#     --head_dropout 0.0 \
#     --enc_in 1 \
#     --dropout 0.1 \
#     --itr 1 \
#     --target "chlorophyll" \
#     --train_epochs 25 \
#     --batch_size 512 \
#     --patience 20 \
#     --inverse \
#     --scale True \
#     --learning_rate 0.0001 \
#     --des Exp \
#     --lradj type3

# CUDA_VISIBLE_DEVICES=0 python -u run.py \
#     --is_training 1 \
#     --model_id "大鹏湾南澳" \
#     --model ModernTCN \
#     --root_path "../data_drop/" \
#     --data_path "大鹏湾南澳_2017_2020.csv" \
#     --data custom \
#     --features S \
#     --seq_len 336 \
#     --pred_len 72 \
#     --ffn_ratio 16 \
#     --patch_size 16 \
#     --patch_stride 4 \
#     --num_blocks 2 \
#     --large_size 51 \
#     --small_size 5 \
#     --dims 64 64 64 64 \
#     --head_dropout 0.0 \
#     --enc_in 1 \
#     --dropout 0.1 \
#     --itr 1 \
#     --target "chlorophyll" \
#     --train_epochs 25 \
#     --batch_size 512 \
#     --patience 20 \
#     --inverse \
#     --scale True \
#     --learning_rate 0.0001 \
#     --des Exp \
#     --lradj type3 \
#     --decomposition 1

# CUDA_VISIBLE_DEVICES=0 python -u run.py \
#     --is_training 1 \
#     --model_id "大鹏湾大梅沙" \
#     --model ModernTCN \
#     --root_path "../data_drop/" \
#     --data_path "大鹏湾大梅沙_2017_2020.csv" \
#     --data custom \
#     --features S \
#     --seq_len 336 \
#     --pred_len 72 \
#     --ffn_ratio 16 \
#     --patch_size 16 \
#     --patch_stride 4 \
#     --num_blocks 2 \
#     --large_size 51 \
#     --small_size 5 \
#     --dims 64 64 64 64 \
#     --head_dropout 0.0 \
#     --enc_in 1 \
#     --dropout 0.1 \
#     --itr 1 \
#     --target "chlorophyll" \
#     --train_epochs 25 \
#     --batch_size 512 \
#     --patience 20 \
#     --inverse \
#     --scale True \
#     --learning_rate 0.0001 \
#     --des Exp \
#     --lradj type3


# CUDA_VISIBLE_DEVICES=0 python -u run.py \
#     --is_training 1 \
#     --model_id "大鹏湾下沙" \
#     --model ModernTCN \
#     --root_path "../data_drop/" \
#     --data_path "大鹏湾下沙_2023_2024.csv" \
#     --data custom \
#     --features S \
#     --seq_len 336 \
#     --pred_len 72 \
#     --ffn_ratio 16 \
#     --patch_size 16 \
#     --patch_stride 4 \
#     --num_blocks 2 \
#     --large_size 51 \
#     --small_size 5 \
#     --dims 64 64 64 64 \
#     --head_dropout 0.0 \
#     --enc_in 1 \
#     --dropout 0.1 \
#     --itr 1 \
#     --target "chlorophyll" \
#     --train_epochs 25 \
#     --batch_size 512 \
#     --patience 20 \
#     --inverse \
#     --scale True \
#     --learning_rate 0.0001 \
#     --des Exp \
#     --lradj type3

# CUDA_VISIBLE_DEVICES=0 python -u run.py \
#     --is_training 1 \
#     --model_id "大鹏湾湾口" \
#     --model ModernTCN \
#     --root_path "../data_drop/" \
#     --data_path "大鹏湾湾口_2023_2024.csv" \
#     --data custom \
#     --features S \
#     --seq_len 336 \
#     --pred_len 72 \
#     --ffn_ratio 16 \
#     --patch_size 16 \
#     --patch_stride 4 \
#     --num_blocks 2 \
#     --large_size 51 \
#     --small_size 5 \
#     --dims 64 64 64 64 \
#     --head_dropout 0.0 \
#     --enc_in 1 \
#     --dropout 0.1 \
#     --itr 1 \
#     --target "chlorophyll" \
#     --train_epochs 25 \
#     --batch_size 512 \
#     --patience 20 \
#     --inverse \
#     --scale True \
#     --learning_rate 0.0001 \
#     --des Exp \
#     --lradj type3

# CUDA_VISIBLE_DEVICES=0 python -u run.py \
#     --is_training 1 \
#     --model_id "大鹏湾沙头角" \
#     --model ModernTCN \
#     --root_path "../data_drop/" \
#     --data_path "大鹏湾沙头角_2023_2024.csv" \
#     --data custom \
#     --features S \
#     --seq_len 336 \
#     --pred_len 72 \
#     --ffn_ratio 16 \
#     --patch_size 16 \
#     --patch_stride 4 \
#     --num_blocks 2 \
#     --large_size 51 \
#     --small_size 5 \
#     --dims 64 64 64 64 \
#     --head_dropout 0.0 \
#     --enc_in 1 \
#     --dropout 0.1 \
#     --itr 1 \
#     --target "chlorophyll" \
#     --train_epochs 25 \
#     --batch_size 512 \
#     --patience 25 \
#     --inverse \
#     --scale True \
#     --learning_rate 0.0001 \
#     --des Exp \
#     --lradj type3

# CUDA_VISIBLE_DEVICES=0 python -u run.py \
#     --is_training 1 \
#     --model_id "珠江口矾石" \
#     --model ModernTCN \
#     --root_path "../data_drop/" \
#     --data_path "珠江口矾石_2023_2024.csv" \
#     --data custom \
#     --features S \
#     --seq_len 336 \
#     --pred_len 72 \
#     --ffn_ratio 16 \
#     --patch_size 16 \
#     --patch_stride 4 \
#     --num_blocks 2 \
#     --large_size 51 \
#     --small_size 5 \
#     --dims 64 64 64 64 \
#     --head_dropout 0.0 \
#     --enc_in 1 \
#     --dropout 0.1 \
#     --itr 1 \
#     --target "chlorophyll" \
#     --train_epochs 25 \
#     --batch_size 512 \
#     --patience 20 \
#     --inverse \
#     --scale True \
#     --learning_rate 0.0001 \
#     --des Exp \
#     --lradj type3


# CUDA_VISIBLE_DEVICES=0 python -u run.py \
#     --is_training 1 \
#     --model_id "大亚湾东涌" \
#     --model ModernTCN \
#     --root_path "../data_drop/" \
#     --data_path "大亚湾东涌_2023_2024.csv" \
#     --data custom \
#     --features S \
#     --seq_len 336 \
#     --pred_len 72 \
#     --ffn_ratio 16 \
#     --patch_size 16 \
#     --patch_stride 4 \
#     --num_blocks 2 \
#     --large_size 51 \
#     --small_size 5 \
#     --dims 64 64 64 64 \
#     --head_dropout 0.0 \
#     --enc_in 1 \
#     --dropout 0.1 \
#     --itr 1 \
#     --target "chlorophyll" \
#     --train_epochs 25 \
#     --batch_size 512 \
#     --patience 20 \
#     --inverse \
#     --scale True \
#     --learning_rate 0.0001 \
#     --des Exp \
#     --lradj type3

# CUDA_VISIBLE_DEVICES=0 python -u run.py \
#     --is_training 1 \
#     --model_id "大亚湾坝光" \
#     --model ModernTCN \
#     --root_path "../data_drop/" \
#     --data_path "大亚湾坝光_2023_2024.csv" \
#     --data custom \
#     --features S \
#     --seq_len 336 \
#     --pred_len 72 \
#     --ffn_ratio 16 \
#     --patch_size 16 \
#     --patch_stride 4 \
#     --num_blocks 2 \
#     --large_size 51 \
#     --small_size 5 \
#     --dims 64 64 64 64 \
#     --head_dropout 0.0 \
#     --enc_in 1 \
#     --dropout 0.1 \
#     --itr 1 \
#     --target "chlorophyll" \
#     --train_epochs 25 \
#     --batch_size 512 \
#     --patience 20 \
#     --inverse \
#     --scale True \
#     --learning_rate 0.0001 \
#     --des Exp \
#     --lradj type3

# CUDA_VISIBLE_DEVICES=0 python -u run.py \
#     --is_training 1 \
#     --model_id "大亚湾长湾" \
#     --model ModernTCN \
#     --root_path "../data_drop/" \
#     --data_path "大亚湾长湾_2023_2024.csv" \
#     --data custom \
#     --features S \
#     --seq_len 336 \
#     --pred_len 72 \
#     --ffn_ratio 16 \
#     --patch_size 16 \
#     --patch_stride 4 \
#     --num_blocks 2 \
#     --large_size 51 \
#     --small_size 5 \
#     --dims 64 64 64 64 \
#     --head_dropout 0.0 \
#     --enc_in 1 \
#     --dropout 0.1 \
#     --itr 1 \
#     --target "chlorophyll" \
#     --train_epochs 25 \
#     --batch_size 512 \
#     --patience 20 \
#     --inverse \
#     --scale True \
#     --learning_rate 0.0001 \
#     --des Exp \
#     --lradj type3

# CUDA_VISIBLE_DEVICES=0 python -u run.py \
#     --is_training 1 \
#     --model_id "珠江口内伶仃以南" \
#     --model ModernTCN \
#     --root_path "../data_drop/" \
#     --data_path "珠江口内伶仃以南_2023_2024.csv" \
#     --data custom \
#     --features S \
#     --seq_len 336 \
#     --pred_len 72 \
#     --ffn_ratio 16 \
#     --patch_size 16 \
#     --patch_stride 4 \
#     --num_blocks 2 \
#     --large_size 51 \
#     --small_size 5 \
#     --dims 64 64 64 64 \
#     --head_dropout 0.0 \
#     --enc_in 1 \
#     --dropout 0.1 \
#     --itr 1 \
#     --target "chlorophyll" \
#     --train_epochs 25 \
#     --batch_size 512 \
#     --patience 20 \
#     --inverse \
#     --scale True \
#     --learning_rate 0.0001 \
#     --des Exp \
#     --lradj type3

# CUDA_VISIBLE_DEVICES=0 python -u run.py \
#     --is_training 1 \
#     --model_id "深圳湾蛇口" \
#     --model ModernTCN \
#     --root_path "../data_drop/" \
#     --data_path "深圳湾蛇口_2023_2024.csv" \
#     --data custom \
#     --features S \
#     --seq_len 336 \
#     --pred_len 72 \
#     --ffn_ratio 16 \
#     --patch_size 16 \
#     --patch_stride 4 \
#     --num_blocks 2 \
#     --large_size 51 \
#     --small_size 5 \
#     --dims 64 64 64 64 \
#     --head_dropout 0.0 \
#     --enc_in 1 \
#     --dropout 0.1 \
#     --itr 1 \
#     --target "chlorophyll" \
#     --train_epochs 25 \
#     --batch_size 512 \
#     --patience 20 \
#     --inverse \
#     --scale True \
#     --learning_rate 0.0001 \
#     --des Exp \
#     --lradj type3

# CUDA_VISIBLE_DEVICES=0 python -u run.py \
#     --is_training 1 \
#     --model_id long_term_forecast \
#     --model ModernTCN \
#     --root_path "../data_drop/" \
#     --data_path "珠江口沙井_2023_2024.csv" \
#     --data custom \
#     --features S \
#     --seq_len 336 \
#     --pred_len 72 \
#     --ffn_ratio 16 \
#     --patch_size 16 \
#     --patch_stride 4 \
#     --num_blocks 2 \
#     --large_size 51 \
#     --small_size 5 \
#     --dims 64 64 64 64 \
#     --head_dropout 0.0 \
#     --enc_in 1 \
#     --dropout 0.1 \
#     --itr 1 \
#     --target "chlorophyll" \
#     --train_epochs 25 \
#     --batch_size 512 \
#     --patience 20 \
#     --inverse \
#     --scale True \
#     --learning_rate 0.0001 \
#     --des Exp \
#     --lradj type3

# CUDA_VISIBLE_DEVICES=1 python -u run.py \
#     --is_training 1 \
#     --checkpoints "./checkpoints_M/" \
#     --model_id "珠江口伶仃以南" \
#     --model ModernTCN \
#     --root_path "../data_9features/" \
#     --data_path "珠江口伶仃以南_2023_2024.csv" \
#     --data custom \
#     --features M \
#     --seq_len 336 \
#     --pred_len 24 \
#     --ffn_ratio 16 \
#     --patch_size 16 \
#     --patch_stride 4 \
#     --num_blocks 2 \
#     --large_size 51 \
#     --small_size 5 \
#     --dims 64 64 64 64 \
#     --head_dropout 0.0 \
#     --enc_in 9 \
#     --dropout 0.1 \
#     --itr 1 \
#     --train_epochs 25 \
#     --batch_size 512 \
#     --patience 20 \
#     --inverse \
#     --scale True \
#     --learning_rate 0.0001 \
#     --des Exp \
#     --lradj type3

CUDA_VISIBLE_DEVICES=1 python -u run.py \
    --is_training 1 \
    --checkpoints "./checkpoints_M/" \
    --model_id "大鹏湾沙头角" \
    --model ModernTCN \
    --root_path "../data_9features/" \
    --data_path "大鹏湾沙头角_2023_2024.csv" \
    --data custom \
    --features M \
    --seq_len 336 \
    --pred_len 24 \
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
    --train_epochs 10 \
    --batch_size 512 \
    --patience 20 \
    --inverse \
    --scale True \
    --learning_rate 0.0001 \
    --des Exp \
    --lradj type3

CUDA_VISIBLE_DEVICES=1 python -u run.py \
    --is_training 1 \
    --checkpoints "./checkpoints_M/" \
    --model_id "大鹏湾沙头角" \
    --model ModernTCN \
    --root_path "../data_9features/" \
    --data_path "大鹏湾沙头角_2023_2024.csv" \
    --data custom \
    --features M \
    --seq_len 336 \
    --pred_len 48 \
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
    --train_epochs 10 \
    --batch_size 512 \
    --patience 20 \
    --inverse \
    --scale True \
    --learning_rate 0.0001 \
    --des Exp \
    --lradj type3