cp /root/lhq/data/data_processed_cls2/风向/AllRedTide_TRAIN.ts all_datasets/AllRedTide
cp /root/lhq/data/data_processed_cls2/风向/AllRedTide_TEST.ts all_datasets/AllRedTide
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./all_datasets/AllRedTide/ \
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
  --patience 20\
  --des Exp \
  --use_multi_scale False
