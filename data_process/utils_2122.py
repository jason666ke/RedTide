import pandas as pd
import numpy as np
from tabulate import tabulate
import os
import sys

# 换列的位置
def changecol():
    # 读取 /root/lhq/data/join_2122.csv 文件
    file_path = '/root/lhq/data/join_2122.csv'
    df = pd.read_csv(file_path)

    # 查看列的顺序
    columns = list(df.columns)

    # 找到‘赤潮发生’和‘最大成灾面积（平方千米）’的索引位置
    index_occurrence = columns.index('赤潮发生')
    index_area = columns.index('最大成灾面积（平方千米）')

    # 交换两列的位置
    columns[index_occurrence], columns[index_area] = columns[index_area], columns[index_occurrence]

    # 重新排列列的顺序
    df = df[columns]

    # 保存修改后的 DataFrame 到新的 CSV 文件，或者覆盖原文件
    df.to_csv(file_path, index=False)

    print("列的位置已经成功调换，并保存到 join_2122.csv 文件中。")

# 相关性矩阵
def corr(df):
    # col_drop = ['站点名称','监测时间','海域','日期']
    # df = df.drop(columns=col_drop)

    # correlation
    corr_matrix = df.corr(method='spearman')
    # # print(corr_matrix)
    redtide_occur = corr_matrix['赤潮发生']
    # # redtide_area = corr_matrix['最大成灾面积（平方千米）']

    redtide_occur_sorted = redtide_occur.abs().sort_values(ascending=True)
    # # redtide_area_sorted = redtide_area.abs().sort_values(ascending=True)

    print(redtide_occur_sorted)
    # print(redtide_area_sorted)

def downsampling(df):
    df['监测时间'] = pd.to_datetime(df['监测时间'])
    df.set_index('监测时间', inplace=True)
    df_resampled = df.resample('h').mean()
    df_resampled.reset_index(inplace=True)

    # 删除空值
    null_rows = df_resampled[df_resampled['赤潮发生'].isna()]
    null_days = null_rows['监测时间'].dt.date
    all_days = df_resampled['监测时间'].dt.date
    df_resampled = df_resampled[~all_days.isin(null_days)]

    # 删除不足24小时的日期
    df_resampled['日期'] = df_resampled['监测时间'].dt.date
    hours_per_day = df_resampled.groupby('日期').size()
    full_day_dates = hours_per_day[hours_per_day == 24].index
    df_resampled = df_resampled[df_resampled['日期'].isin(full_day_dates)]
    df_resampled.drop(columns=['日期'], inplace=True)

    return df_resampled

# 缺失值处理
def nan_process(df):
    window = 72
    for column in df.columns:
        if not pd.api.types.is_float_dtype(df[column]):
            continue
        print(f"Processing column: {column}")

        mean = df[column].mean()
        std = df[column].std()
        lower_limit = mean - 3 * std
        upper_limit = mean + 3 * std
        df[column].apply(lambda x: x if lower_limit <= x <= upper_limit else np.nan)
        
        if df[df[column].isnull()].empty:
            continue  
        
        moving_avg = df[column].rolling(window=window, min_periods=1).mean().round(3)
        df[column] = df[column].fillna(moving_avg)
    
    numeric_columns = df.select_dtypes(include=['float64']).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean().round(3))
    return df.round(3)

def sort_drop(df): 
    df = df.sort_values(by='监测时间')

    col_drop = ['硝酸盐','亚硝酸盐','磷酸盐','氨氮',
            '水深','流速','流向','有效波高','平均波向','有效波周期','碘131','钴60','铊208',
            '钾40','铋214','铅214','镭226','钍232','铯134','铯137','站点名称','海域','日期']
    df = df.drop(columns=col_drop)

    return df

def pos_neg(df):
    df_pos = df[df['赤潮发生'] == 1.0]
    df_neg = df[df['赤潮发生'] == 0.0]

    return df_pos, df_neg

def split_dataset(df): 
    # 使用 .loc 修改列
    df.loc[:, '赤潮发生'] = df['赤潮发生'].astype(int)
    df.loc[:, '年份'] = pd.to_datetime(df['监测时间']).dt.year

    # 使用 .copy() 来显式创建副本，避免 SettingWithCopyWarning
    df_2021 = df[df['年份'] == 2021].copy()
    df_2022 = df[df['年份'] == 2022].copy()

    # 使用 inplace=False 的方式删除列，这样也避免了潜在的副本问题
    df_2021 = df_2021.drop(columns=['年份'])
    df_2022 = df_2022.drop(columns=['年份'])
    
    return df_2021, df_2022

def to_ts(df):
    df = df.drop(columns=['监测时间', '最大成灾面积（平方千米）'])

    window_size = 24  
    sequences = []
    labels = []

    for i in range(0, len(df) - window_size + 1, window_size):
        window = df.iloc[i:i + window_size]
        sequence = window.drop('赤潮发生', axis=1).values.T.tolist()
        label = int(round(window['赤潮发生'].mean()))  

        sequences.append(sequence)
        labels.append(label)

    sequences = np.array(sequences)
    labels = np.array(labels)
    print(len(labels))
    
    return sequences, labels

def save_ts(sequences, labels, path, dataset_name):
    with open(path, 'w') as f:
        f.write(f'@problemName {dataset_name}\n')
        f.write('@timeStamps false\n')
        f.write('@missing false\n')
        f.write(f'@univariate false\n')
        f.write(f'@dimensions {sequences.shape[1]}\n')
        f.write('@equalLength true\n')
        f.write(f'@seriesLength {sequences.shape[2]}\n') 
        f.write('@classLabel true 0 1\n')
        f.write('@data\n')

        for i in range(len(sequences)):
            sequence = sequences[i]
            label = labels[i]
            sequence_str = ':'.join([','.join(map(str, seq)) for seq in sequence])
            f.write(f'{sequence_str}:{label}\n')

def null_class(df):
    null_counts = df.isnull().sum()

    print("每一列的空值数量：")
    print(null_counts)

    value_counts = df['赤潮发生'].value_counts()

    print("‘赤潮发生’列中每个类别的数量如下：")
    print(value_counts)

    value_counts = df['最大成灾面积（平方千米）'].value_counts()

    print("‘最大成灾面积（平方千米）’列中每个类别的数量如下：")
    print(value_counts)
    

def view_file(df):
    df_rows, df_columns = df.shape
    print(f"文件的行数: {df_rows}，列数: {df_columns}")

    # print(tabulate(df.head(50), headers='keys', tablefmt='pretty', showindex=False))

    # 提取‘赤潮发生’列中值为 1.0 的数据
    filtered_df = df[df['赤潮发生'] == 1.0]

    # 打印前 20 行数据
    print(tabulate(filtered_df.head(50), headers='keys', tablefmt='pretty', showindex=False))

def load_file(path, flag):
    if flag == 'excel':
        df = pd.read_excel(path)
    elif flag == 'csv':
        df = pd.read_csv(path)
    return df

def save_csv(df, path):
    df.to_csv(path, index=False)
    print(f"数据已保存到{path}")

def print_bug(df):
    df_null_red_tide = df[df['赤潮发生'].isnull()]
    df_rows, df_columns = df_null_red_tide.shape
    print(f"文件的行数: {df_rows}，列数: {df_columns}")
    
    if not df_null_red_tide.empty:
        print(tabulate(df_null_red_tide, headers='keys', tablefmt='grid', showindex=False))
    else:
        print("没有空值的 '赤潮发生' 数据。")


if __name__ == "__main__":
    
    input_dir = '/root/lhq/data/data_grouped_2122/'
    train_pos_path = '/root/lhq/data/data_processed/RedTide_pos_TRAIN.ts'
    train_neg_path = '/root/lhq/data/data_processed/RedTide_neg_TRAIN.ts'
    train_path = '/root/lhq/data/data_processed/RedTide_TRAIN.ts'
    test_path = '/root/lhq/data/data_processed/RedTide_TEST.ts'

    train_pos_tmp_path = '/root/lhq/data/data_processed/RedTide_pos_tmp_TRAIN.ts'
    train_neg_tmp_path = '/root/lhq/data/data_processed/RedTide_neg_tmp_TRAIN.ts'
    train_tmp_path = '/root/lhq/data/data_processed/RedTide_tmp_TRAIN.ts'
    test_tmp_path = '/root/lhq/data/data_processed/RedTide_tmp_TEST.ts'

    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    df_tmp_label = []
    df_tmp_feature = []
    df_tmp = []

    for csv_file in csv_files:

        input_path = os.path.join(input_dir, csv_file)
        base_name = os.path.splitext(csv_file)[0]

        if base_name in ['珠江口内伶仃以南', '大鹏湾沙头角', '珠江口沙井']:
            continue


        print('################# 原始数据 ##################')
        print(base_name)
        df = load_file(input_path, 'csv')

        print('################# 排序并删除列 ##################')
        df = sort_drop(df)
        null_class(df)
        # view_file(df)

        print('################# 以小时为单位进行采样 ##################')
        df = downsampling(df)
        null_class(df)
        # view_file(df)

        print('################# 缺失值处理 ##################')
        df = nan_process(df)
        null_class(df)
        view_file(df)

        print('################# 合并 ##################')
        df_tmp.append(df)

    print('################# 数据集制作 ##################')
    df_new = pd.concat(df_tmp)
    df_2021, df_test = split_dataset(df_new)

    df_pos, df_neg = pos_neg(df_2021)
    null_class(df_pos)
    null_class(df_neg)

    df_train = pd.concat([df_pos, df_neg])

    df_pos_tmp = df_pos[:480]
    df_neg_tmp = df_neg[:22080]
    df_train_tmp = pd.concat([df_pos_tmp, df_neg_tmp])
    df_test_tmp = df_test[24480:24480+18240]

    print('################# 生成ts文件并保存 ##################')
    data_dict = {
        'pos_train': (df_pos, train_pos_path),
        'neg_train': (df_neg, train_neg_path),
        'train': (df_train, train_path),
        'test': (df_test, test_path),
        'pos_train_tmp' : (df_pos_tmp, train_pos_tmp_path),
        'neg_train_tmp' : (df_neg_tmp, train_neg_tmp_path),
        'train_tmp': (df_train_tmp, train_tmp_path),
        'test_tmp': (df_test_tmp, test_tmp_path)
    }

    # 生成ts文件并保存
    for _, (df, path) in data_dict.items():
        sequence, label = to_ts(df)
        save_ts(sequence, label, path, 'RedTide')


