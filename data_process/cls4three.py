import os
import pandas as pd
import numpy as np
import utils_all

import pandas as pd
import numpy as np
from tabulate import tabulate
import os
import sys

def load_file(path, flag):
    if flag == 'excel':
        df = pd.read_excel(path)
    elif flag == 'csv':
        df = pd.read_csv(path)
    return df

def view_df(df,num_rows=30):
    df_rows, df_columns = df.shape
    print(f"文件的行数: {df_rows}，列数: {df_columns}")

    null_counts = df.isnull().sum()
    print("每一列的空值数量：")
    print(null_counts)

    # 打印前num_rows行
    # print(tabulate(df.head(num_rows), headers='keys', tablefmt='grid'))
    redtide_count = df['赤潮类型'].value_counts()
    print(redtide_count)

def remove_nulls(df):
    has_nulls = df.isnull().any(axis=1)
    no_red_tide = df['赤潮类型'] == 0
    df_cleaned = df[~(has_nulls & no_red_tide)]
    return df_cleaned


def downsampling(df):
    df['监测时间'] = pd.to_datetime(df['监测时间'])
    df.set_index('监测时间', inplace=True)
    df_resampled = df.resample('h').mean()
    df_resampled.reset_index(inplace=True)

    # 删除空值
    null_rows = df_resampled[df_resampled['赤潮类型'].isna()]
    null_days = null_rows['监测时间'].dt.date
    all_days = df_resampled['监测时间'].dt.date
    df_resampled = df_resampled[~all_days.isin(null_days)]

    # 删除不足24小时的日期
    df_resampled['日期'] = df_resampled['监测时间'].dt.date
    hours_per_day = df_resampled.groupby('日期').size()
    full_day_dates = hours_per_day[hours_per_day == 24].index
    df_resampled = df_resampled[df_resampled['日期'].isin(full_day_dates)]
    df_resampled.drop(columns=['日期'], inplace=True)

    return df_resampled.round(3)

# 修改为适合三分类的情况
def forward_fill(row, df, column, category):
    if pd.notnull(row[column]):
        return row[column]
    
    # 根据传入的类别（1或2），查找符合条件的行
    if category == 1:
        # 类别 1 赤潮：夜光藻、红色中缢虫、棕囊藻，球形棕囊藻
        fill_rows = df[(df['赤潮类型'] == 1) & df[column].notnull()]
    elif category == 2:
        # 类别 2 赤潮：其他藻类
        fill_rows = df[(df['赤潮类型'] == 2) & df[column].notnull()]
    else:
        # 如果类别不在1或2中，说明未发生赤潮(why??)
        return row[column]
    
    # fill_rows = df[(df['赤潮发生'] == 1) & df[column].notnull()]
    fill_rows['时间差'] = abs(pd.to_datetime(fill_rows['监测时间']) - pd.to_datetime(row['监测时间']))
    fill_rows = fill_rows.sort_values(by='时间差')
    
    if not fill_rows.empty:
        fill_value = fill_rows[column].iloc[0]
        return fill_value
    return row[column]

def nan_process(df):
    window = 24
    
    # 处理每一列的缺失值
    for column in df.columns:
        if not pd.api.types.is_float_dtype(df[column]):
            continue
        print(f"开始处理列: {column}")
        
        if df[column].notnull().all():
            continue  
        
        # 滑动平均并填充缺失值
        moving_avg = df[column].rolling(window=window, min_periods=1).mean().round(3)
        df[column] = df[column].fillna(moving_avg)
    
    cols_to_check = df.columns[1:-2]
    half_cols = len(cols_to_check) / 2 

    # 删除非空值列数小于一半的行
    df = df[df[cols_to_check].notnull().sum(axis=1) >= half_cols]

    # 前向填充
    if df.isnull().values.any():
        for column in df.columns:
            if df[column].isnull().any():
                print(f"前向填充列: {column}")
                df.loc[:, column] = df.apply(lambda row: forward_fill(row, df, column, row['赤潮类型']), axis=1)
    
    return df.round(3)

def analysis_df(df):
    df['日期'] = df['监测时间'].dt.date
    df['年份'] = df['监测时间'].dt.year
    
    redtide_day = df.groupby('日期')['赤潮类型'].max()
    redtide_year = redtide_day.value_counts().sort_index()
    
    print("‘赤潮类型’的天数统计：")
    print(f"未发生赤潮的天数: {redtide_year.get(0, 0)}")
    print(f"发生赤潮类型1的天数: {redtide_year.get(1, 0)}")
    print(f"发生赤潮类型2的天数: {redtide_year.get(2, 0)}")

    # redtide_year = df[df['赤潮发生'] == 1].groupby('年份')['日期'].nunique()
    redtide_year = df[df['赤潮类型'] > 0].groupby('年份')['日期'].nunique()
    print("每年发生赤潮的天数统计：")
    print(redtide_year)

    area_day = df.groupby('日期')['最大成灾面积（平方千米）'].max()
    area_day = area_day.value_counts().sort_index()

    print("‘最大成灾面积（平方千米）’类别的天数统计：")
    print(area_day)

    df.drop(columns=['日期','年份'], inplace=True)

def split_dataset(df):
    df['年份'] = df['监测时间'].dt.year
    
    # redtide_years = df[df['赤潮发生'] == 1]['年份'].unique()
    redtide_years = df[df['赤潮类型'] > 0]['年份'].unique()
    redtide_years.sort()

    test_years = redtide_years[-2:] # 选择最后的两年作为测试集
    print(f"测试集年份: {test_years}")
    
    df_test = df[df['年份'].isin(test_years)].copy()
    df_train = df[~df['年份'].isin(test_years)].copy()
    
    df_train.drop(columns=['年份'], inplace=True)
    df_test.drop(columns=['年份'], inplace=True)
    
    return df_train, df_test

def to_ts(df):
    df = df.drop(columns=['监测时间', '最大成灾面积（平方千米）'])

    window_size = 24  
    sequences = []
    labels = []

    for i in range(0, len(df) - window_size + 1, window_size):
        window = df.iloc[i:i + window_size]
        sequence = window.drop('赤潮类型', axis=1).values.T.tolist()
        # label = int(round(window['赤潮发生'].mean()))  # round 将均值四舍五入，代表多数时间点发生了赤潮，则判定为发生赤潮

        # 计算标签，取众数作为判断的主要赤潮类型
        label = window['赤潮类型'].mode().iloc[0]

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
        f.write('@classLabel true 0 1 2\n')
        f.write('@data\n')

        for i in range(len(sequences)):
            sequence = sequences[i]
            label = labels[i]
            sequence_str = ':'.join([','.join(map(str, seq)) for seq in sequence])
            f.write(f'{sequence_str}:{label}\n')




if __name__ == "__main__":
    # redtide_path = '/root/lhq/data/data_raw/赤潮整理.xlsx'
    # input_dir = '/root/lhq/data/data_raw/资源表数据提取'
    # output_dir = '/root/lhq/data/data_grouped_cls3'
    input_dir = '/root/lhq/data/data_grouped_cls3/'
    # sample_data_dir = '/root/lhq/data/data_grouped_cls3/data_grouped_afterSampling/'
    # train_path = '/root/lhq/data/data_processed/AllRedTide_TRAIN.ts'
    # test_path = '/root/lhq/data/data_processed/AllRedTide_TEST.ts'

    train_path = '/root/lhq/data/data_processed_cls3/AllRedTide_TRAIN.ts'
    test_path = '/root/lhq/data/data_processed_cls3/AllRedTide_TEST.ts'
    
    xlsx_files = [f for f in os.listdir(input_dir) if f.endswith('.xlsx')]

    df_tmp_train = []
    df_tmp_test = []

    for xlsx_file in xlsx_files:
        input_path = os.path.join(input_dir, xlsx_file)
        base_name = os.path.splitext(xlsx_file)[0]

        print('################# 数据名称 ##################')
        print(base_name)
        
        # 加载数据
        df = load_file(input_path,'excel')
        df['赤潮类型'] = df['赤潮类型'].astype(int)
        
        # 如果是指定的数据列，删除蓝绿藻列
        print('################# 删除蓝绿藻列 ################')
        if base_name in ['大鹏湾南澳', '大鹏湾大梅沙', '大亚湾东涌','大亚湾坝光','大亚湾东山','大鹏湾下沙']:
            df = df.drop(columns=['蓝绿藻'])
        
        print('################# 按照时间顺序排序 ##################')
        df = df.sort_values(by='监测时间')
        # view_df(df)
        
        print('################# 删除空值 ##################')
        df = remove_nulls(df)
        # view_df(df)
        
        print('################# 缺失值处理 ##################')
        df = nan_process(df)
        # view_df(df)
        
        print('################# 以小时为单位进行采样 ##################')
        df = downsampling(df)
        # view_df(df)
        
        print('################# 打印处理结果 ##################')
        analysis_df(df)
        
        print('################# 切分数据集 ##################')
        df_train, df_test = split_dataset(df)
        
        print('训练集：')
        analysis_df(df_train)
        view_df(df_train)
        
        print('测试集')
        analysis_df(df_test)
        view_df(df_test)
        
        print('################# 合并 ##################')
        df_tmp_train.append(df_train)
        df_tmp_test.append(df_test)

    print('################# 生成ts文件并保存 ##################')
    final_train = pd.concat(df_tmp_train)
    final_test = pd.concat(df_tmp_test)

    data_dict = {
        'train': (final_train, train_path),
        'test': (final_test, test_path)
    }
    for _, (df, path) in data_dict.items():
        sequence, label = to_ts(df)
        save_ts(sequence, label, path, 'AllRedTide')
    

