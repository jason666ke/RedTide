import pandas as pd
from tabulate import tabulate
import os
import argparse
import re
import datetime

def print_csv_xlsx(file_path, num_rows=50):
    # 获取文件扩展名
    file_ext = os.path.splitext(file_path)[-1].lower()

    # 根据文件类型读取数据
    if file_ext == ".csv":
        df = pd.read_csv(file_path)
    elif file_ext == ".xlsx":
        df = pd.read_excel(file_path)
    else:
        print(f"Unsupported file format: {file_ext}")
        return
    
    df_rows, df_columns = df.shape
    print(f"文件的行数: {df_rows}，列数: {df_columns}")
    # 打印前num_rows行
    print(tabulate(df.head(num_rows), headers='keys', tablefmt='grid'))


import numpy as np

def inspect_npy_file(file_path, num_samples=5):

    try:
        data = np.load(file_path)
        
        print(f"文件路径: {file_path}")
        print(f"数据类型: {type(data)}")
        print(f"数据形状: {data.shape}")
        print(f"数据类型 (dtype): {data.dtype}")
        
        # 打印部分数据
        print("\n部分数据预览:")
        if isinstance(data, np.ndarray):
            print(data[:num_samples])  # 打印前 num_samples 条数据
        else:
            print(data)  # 如果不是数组，直接打印
        
        return data
    
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 未找到。")
    except Exception as e:
        print(f"加载文件时发生错误: {e}")


# 使用argparse解析命令行参数
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print the file.")
    parser.add_argument("file_path", type=str, help="Path to the file (csv or xlsx).")
    parser.add_argument("--rows", type=int, default=50, help="Number of rows to display (default is 50).")
    
    args = parser.parse_args()

    # 打印文件内容 csv, xlsx
    # print_csv_xlsx(args.file_path, args.rows)

    # 打印内容 npy
    inspect_npy_file(args.file_path)

