import pandas as pd
from tabulate import tabulate
import os
import argparse

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

    # 时间数据跨度
    df = df.sort_values(by='监测时间')
    print(f"最新数据时间: ", df['监测时间'].max())
    print(f"最原始数据时间: ", df['监测时间'].min())
    time_span = df['监测时间'].max() - df['监测时间'].min()
    print("时间跨度: ", time_span)

    # 打印前num_rows行
    print(tabulate(df.head(num_rows), headers='keys', tablefmt='grid'))
    redtide_count = df['赤潮类型'].value_counts()
    print(redtide_count)

    # col_drop = ['硝酸盐','亚硝酸盐','磷酸盐','氨氮',
    #         '水深','流速','流向','有效波高','平均波向','有效波周期','碘131','钴60','铊208',
    #         '钾40','铋214','铅214','镭226','钍232','铯134','铯137','站点名称','海域','日期']
    # df = df.drop(columns=col_drop)

    # df.loc[:, '年份'] = pd.to_datetime(df['观测时间']).dt.year

    # value_counts = df['年份'].value_counts()

    # print("‘年份’列中每个类别的数量如下：")
    # print(value_counts)

    # df = df.drop(columns=['年份'])

    return df

# 使用argparse解析命令行参数
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print the file.")
    parser.add_argument("file_path", type=str, help="Path to the file (csv or xlsx).")
    parser.add_argument("--rows", type=int, default=50, help="Number of rows to display (default is 50).")
    
    args = parser.parse_args()

    # 打印文件内容
    print_file_head(args.file_path, args.rows)
