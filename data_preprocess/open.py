import pandas as pd
from tabulate import tabulate
import os
import argparse
import re
import datetime

def print_file_head(file_path, num_rows=50):
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


# 使用argparse解析命令行参数
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print the file.")
    parser.add_argument("file_path", type=str, help="Path to the file (csv or xlsx).")
    parser.add_argument("--rows", type=int, default=50, help="Number of rows to display (default is 50).")
    
    args = parser.parse_args()

    # 打印文件内容
    print_file_head(args.file_path, args.rows)