export CUDA_VISIBLE_DEVICES=1

model_name=TimeXer

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path "../data_drop/" \
  --data_path "大亚湾东山_2023_2024_9.csv" \
  --model_id ETTh_96_72 \
  --model $model_name \
  --data custom \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 72 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 9 \
  --dec_in 9 \
  --c_out 9 \
  --target "temperature" \
  --inverse \
  --scale True \
  --d_model 256 \
  --batch_size 4 \
  --des 'exp' \
  --itr 1
