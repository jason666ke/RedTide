import pandas as pd
from tabulate import tabulate
import os
import argparse
import re

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

    null_counts = df.isnull().sum()
    print("每一列的空值数量：")
    print(null_counts)

    # 打印前num_rows行
    print(tabulate(df.head(num_rows), headers='keys', tablefmt='grid'))
 
    # col_drop = ['硝酸盐','亚硝酸盐','磷酸盐','氨氮',
    #         '水深','流速','流向','有效波高','平均波向','有效波周期','碘131','钴60','铊208',
    #         '钾40','铋214','铅214','镭226','钍232','铯134','铯137','站点名称','海域','日期']
    # df = df.drop(columns=col_drop)

    # df.loc[:, '年份'] = pd.to_datetime(df['监测时间']).dt.year

    value_counts = df['赤潮类型'].value_counts()

    print("‘赤潮类别’列中每个类型的数量如下：")
    print(value_counts)

    # df = df.drop(columns=['年份'])

# 定义表头清理函数
def clean_column_name(col_name):
    return re.sub(r'[\(（][^)]*[\)）]', '', col_name).strip()

# 处理文件函数
def process_file(file_path):
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        return

    # 正则化表头
    cleaned_columns = [clean_column_name(col) for col in df.columns]
    
    # 输出文件名和正则化的表头
    print(f"File: {file_path}")
    print("Cleaned Headers:", cleaned_columns)

# 遍历指定目录下的文件
def process_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            process_file(file_path)

# 使用argparse解析命令行参数
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print the file.")
    parser.add_argument("file_path", type=str, help="Path to the file (csv or xlsx).")
    parser.add_argument("--rows", type=int, default=50, help="Number of rows to display (default is 50).")
    
    args = parser.parse_args()

    # 打印文件内容
    print_file_head(args.file_path, args.rows)
    # process_directory(args.file_path)