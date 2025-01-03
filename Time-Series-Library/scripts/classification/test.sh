export CUDA_VISIBLE_DEVICES=3

python -u run.py \
  --task_name classification \
  --is_training 0 \
  --root_path ./dataset/RedTide/ \
  --model_id RedTide \
  --model TimesNet \
  --data UEA \
  --e_layers 2 \
  --batch_size 16 \
  --d_model 64 \
  --d_ff 64 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 30 \
  --patience 10 \
