import pandas as pd
from tabulate import tabulate
import os
import argparse
import re
import datetime
import numpy as np

def print_csv_xlsx(file_path, num_rows=5):
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
    
    # 打印前num_rows行
    print(tabulate(df.head(num_rows), headers='keys', tablefmt='grid'))
    print(f"文件的行数: {df_rows}，列数: {df_columns}")
    
    return df

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
        
def get_cols_stats(df):
    # 处理时间数据，提取年份
    df['date'] = pd.to_datetime(df['date'])
    df['年份'] = df['date'].dt.year

    # 获取数据年份信息
    years = sorted(df['年份'].unique())
    print(f"数据包含年份：{years}")
    df.drop(columns=['年份'], inplace=True)

    # 获取数据的时间范围
    min_date = df['date'].min().strftime('%Y-%m-%d')
    max_date = df['date'].max().strftime('%Y-%m-%d')
    print(f"数据时间范围：从 {min_date} 到 {max_date}\n")
    
    # df['监测时间'] = pd.to_datetime(df['监测时间'])
    # df['年份'] = df['监测时间'].dt.year

    # # 获取数据年份信息
    # years = sorted(df['年份'].unique())
    # print(f"数据包含年份：{years}")
    # df.drop(columns=['年份'], inplace=True)

    # # 获取数据的时间范围
    # min_date = df['监测时间'].min().strftime('%Y-%m-%d')
    # max_date = df['监测时间'].max().strftime('%Y-%m-%d')
    # print(f"数据时间范围：从 {min_date} 到 {max_date}\n")

    # 初始化统计表
    stats = []
    for col in df.columns:
        data_type = df[col].dtype
        non_null_count = df[col].notnull().sum()
        missing_values = df[col].isnull().sum()
        unique_values = df[col].nunique()
        min_value = df[col].min() if pd.api.types.is_numeric_dtype(df[col]) else "-"
        max_value = df[col].max() if pd.api.types.is_numeric_dtype(df[col]) else "-"

        stats.append([col, data_type, non_null_count, missing_values, unique_values, min_value, max_value])

    # 转换为 DataFrame 方便打印
    stats_df = pd.DataFrame(stats, columns=[
        'Column', 'Data Type', 'Non-Null Count', 'Missing Values', 'Category', 'Min Value', 'Max Value'
    ])

    # 整齐打印数据统计信息
    print("\n数据统计信息：\n")
    print(tabulate(stats_df, headers='keys', tablefmt='pretty', showindex=False))

    return stats_df

# 使用argparse解析命令行参数
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print the file.")
    parser.add_argument("file_path", type=str, help="Path to the file (csv or xlsx).")
    parser.add_argument("--rows", type=int, default=50, help="Number of rows to display (default is 50).")
    
    args = parser.parse_args()

    # 打印文件内容 csv, xlsx
    df = print_csv_xlsx(args.file_path, args.rows)
    # get_cols_stats(df)

    # 打印内容 npy
    # inspect_npy_file(args.file_path)

