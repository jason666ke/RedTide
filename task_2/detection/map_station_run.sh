#!/bin/bash

# 导入站点映射关系
source ../config/station_mapping.sh

# 设置数据目录
DATA_DIR="../data_final"

# 遍历data_final目录下的所有文件夹
for dir in "$DATA_DIR"/*/ ; do
    if [ ! -d "$dir" ]; then
        continue
    fi

    # 获取文件夹名称并提取站点名称（第一个_前的内容）
    folder_name=$(basename "$dir")
    station_name=${folder_name%%_*}
    
    # 获取对应的编码名称
    encoded_name="${STATION_MAPPING[$station_name]}"
    
    if [ -z "$encoded_name" ]; then
        echo "警告：无法找到站点 '$station_name' 的映射编码，跳过"
        continue
    fi

    echo "处理站点: $station_name -> $encoded_name"
    
    # 检查预测结果文件是否存在
    PRED_FILE="${dir}/prediction_2021_2022.csv"
    if [ ! -f "$PRED_FILE" ]; then
        echo "警告：预测文件不存在 $PRED_FILE"
        continue
    fi

    # 转换CSV到NPY
    python - <<END
import pandas as pd
import numpy as np

try:
    # 读取CSV文件
    df = pd.read_csv("$PRED_FILE")
    
    # 将数据转换为numpy数组
    data = df.values
    
    # 保存为npy文件
    npy_file = "$dir/predict.npy"
    np.save(npy_file, data)
    print(f"已将CSV转换为NPY: {npy_file}")
except Exception as e:
    print(f"转换失败: {str(e)}")
    exit(1)
END

    # 检查Python脚本执行结果
    if [ $? -ne 0 ]; then
        echo "CSV转NPY失败，跳过该站点"
        continue
    fi

    # 设置输出路径
    OUTPUT_PATH="../results/${encoded_name}.csv"
    
    # 运行异常检测模型
    CUDA_VISIBLE_DEVICES=0 python -u run.py \
        --is_training 0 \
        --root_path "$dir" \
        --model_id "$encoded_name" \
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

    echo "完成站点 $station_name 的处理"
    echo "-----------------------------------"
done