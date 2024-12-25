import pandas as pd
import numpy as np
from tabulate import tabulate
import os
import sys
from sklearn.ensemble import RandomForestRegressor
from collections import Counter

def load_file(path, flag):
    if flag == 'excel':
        df = pd.read_excel(path)
    elif flag == 'csv':
        df = pd.read_csv(path)
    return df

def view_df(df,num_rows=30):
    df_rows, df_columns = df.shape
    print(f"文件的行数: {df_rows}，列数: {df_columns}")

    # null_counts = df.isnull().sum()
    # print("每一列的空值数量：")
    # print(null_counts)

    # 打印前num_rows行
    print(tabulate(df.head(num_rows), headers='keys', tablefmt='grid'))

def remove_nulls(df):
    has_nulls = df.isnull().any(axis=1)
    no_red_tide = df['赤潮类型'] == 0
    df_cleaned = df[~(has_nulls & no_red_tide)]
    return df_cleaned

def nan_process(df):
    window = 24
    
    # 处理每一列的缺失值
    for column in df.columns:
        if not pd.api.types.is_float_dtype(df[column]):
            continue
        # print(f"滑动窗口处理列: {column}")
        
        if df[column].notnull().all():
            continue  
        
        # 滑动平均并填充缺失值
        moving_avg = df[column].rolling(window=window, min_periods=1).mean().round(3)
        df[column] = df[column].fillna(moving_avg)
    
    cols_to_check = df.columns[1:-2]
    half_cols = len(cols_to_check) / 2 

    # 删除非空值列数小于一半的行
    df = df[df[cols_to_check].notnull().sum(axis=1) >= half_cols]

    # # 前向填充
    # if df.isnull().values.any():
    #     for column in df.columns:
    #         if df[column].isnull().any():
    #             print(f"前向填充列: {column}")
    #             df.loc[:, column] = df.apply(lambda row: forward_fill(row, df, column), axis=1)

    # 随机森林填充
    df = random_forest_predict(df)
    
    # return df.round(3)
    return df

def random_forest_predict(df):
    # Filling in missing values
    df_copy = df.copy()
    numeric_columns = df.select_dtypes(include=['float64']).columns
    df_copy[numeric_columns] = df_copy[numeric_columns].fillna(df[numeric_columns].median())

    for column in numeric_columns:
        # print(f"随机森林处理列: {column}")
        if df[column].isnull().sum() == 0:
            continue

        features = df_copy[numeric_columns].drop(columns=[column])
        target = df[column]

        X_train = features[target.notnull()]
        y_train = target[target.notnull()]
        X_test = features[target.isnull()]

        if len(X_train) == 0 or len(X_test) == 0:
            print(f"跳过列 {column} 因其数据不足.")
            continue

        rf = RandomForestRegressor(n_estimators=80, n_jobs=-1, random_state=0)
        rf.fit(X_train, y_train)

        predicted_values = rf.predict(X_test)
        df.loc[df[column].isnull(), column] = predicted_values
        
    return df

def forward_fill(row, df, column):
    if pd.notnull(row[column]):
        return row[column]
    else:
        fill_rows = df[(df['赤潮类型'] == 1) & df[column].notnull()]
        fill_rows['时间差'] = abs(pd.to_datetime(fill_rows['监测时间']) - pd.to_datetime(row['监测时间']))
        fill_rows = fill_rows.sort_values(by='时间差')
        
        if not fill_rows.empty:
            fill_value = fill_rows[column].iloc[0]
            return fill_value
        return row[column]
    
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

    # return df_resampled.round(3)
    return df_resampled

def split_dataset(df):
    df['年份'] = df['监测时间'].dt.year
    
    redtide_years = df[df['赤潮类型'] == 1]['年份'].unique()
    redtide_years.sort()

    test_years = redtide_years[-2:] # 选择最后的两年作为测试集
    # print(f"测试集年份: {test_years}")
    
    df_test = df[df['年份'].isin(test_years)].copy()
    df_train = df[~df['年份'].isin(test_years)].copy()
    
    df_train.drop(columns=['年份'], inplace=True)
    df_test.drop(columns=['年份'], inplace=True)
    
    return df_train, df_test

def to_csv(df):
    # 重命名
    # save_cols = [
    #     'timestamp', '风速', '风向', '气温', '相对湿度', 
    #     '气压', '雨量', '水温', '电导率', '盐度', 'ph', 
    #     '氧化还原电位', '溶解氧', '浊度', '叶绿素a', '赤潮类型'
    # ]
    # 扰动分析
    # save_cols = [
    #     'timestamp', '气温', '相对湿度', '气压', '水温', '电导率', '盐度', 'ph',  '溶解氧', '叶绿素a', '赤潮类型'
    # ]
    # 只有叶绿素
    save_cols = [
        'timestamp', '叶绿素a', '赤潮类型'
    ]
    df = df[save_cols]
    
    new_cols = {'timestamp': 'timestamp', '赤潮类型': 'label'}
    features_cols = [col for col in save_cols if col not in ['timestamp', '赤潮类型']]
    new_cols.update({col: f"feature_{i}" for i, col in enumerate(features_cols)})
    
    df.rename(columns=new_cols, inplace=True)

    # 切分
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    label_col = 'label'

    feature_df = df[['timestamp'] + feature_cols]
    label_df = df[['timestamp', label_col]]

    return feature_df, label_df

def to_npy(df):

    # 任务一npy
    save_cols = [
        '风速', '气温', '相对湿度', '气压', '水温', '电导率', '盐度', 'ph',  '溶解氧', '叶绿素a', '赤潮类型'
    ]
    
    df = df[save_cols]

    new_cols = {'赤潮类型': 'label'}
    features_cols = [col for col in save_cols if col not in ['赤潮类型']]
    new_cols.update({col: f"feature_{i}" for i, col in enumerate(features_cols)})

    df.rename(columns=new_cols, inplace=True)

    # 切分
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    label_col = 'label'

    feature_df = df[feature_cols]
    label_df = df[label_col]

    return feature_df, label_df

# 单个站点
# if __name__ == "__main__":
#     input_path = '/root/lhq/data/data_grouped_cls2/大亚湾东山.xlsx'
#     output_dir = '/root/lhq/data/data_processed_detection/all_features'

#     timestamp_start = 0

#     print('################# 数据名称 ##################')
#     print(input_path)
#     df = load_file(input_path,'excel')
#     print('################# 按照时间顺序排序 ##################')
#     df = df.sort_values(by='监测时间')
#     # view_df(df)
#     print('################# 删除空值 ##################')
#     df = remove_nulls(df)
#     # view_df(df)
#     print('################# 缺失值处理 ##################')
#     df = nan_process(df)
#     # view_df(df)
#     print('################# 以小时为单位进行采样 ##################')
#     df = downsampling(df)
#     # view_df(df)
#     print('################# 添加时间戳 ##################')
#     # view_df(df)
#     num_rows = len(df)
#     df.insert(0, "timestamp", range(timestamp_start, timestamp_start + num_rows))
#     timestamp_start += num_rows
#     # view_df(df)
#     print('################# 切分数据集 ##################')
#     df_train, df_test = split_dataset(df)
#     # print('训练集：')
#     # view_df(df_train)
#     # print('测试集')
#     # view_df(df_test)
#     print('################# 生成文件并保存 ##################')
#     train, _ = to_csv(df_train)
#     test, test_label = to_csv(df_test)
#     view_df(train)

# 所有站点
if __name__ == "__main__":
    input_dir = '/root/lhq/data/data_grouped_cls2'
    output_dir = '/root/lhq/data/data_processed_detection/onlyChla'

    files = {
        'train_csv': 'train.csv',
        'test_csv': 'test.csv',
        'test_label_csv': 'test_label.csv',
        'train_npy': 'train.npy',
        'test_npy': 'test.npy',
        'test_label_npy': 'test_label.npy'
    }

    paths = {key: os.path.join(output_dir, filename) for key, filename in files.items()}

    df_tmp_train = []
    df_tmp_test = []
    
    xlsx_files = [f for f in os.listdir(input_dir) if f.endswith('.xlsx')]

    timestamp_start = 0

    for xlsx_file in xlsx_files:
        input_path = os.path.join(input_dir, xlsx_file)
        base_name = os.path.splitext(xlsx_file)[0]

        print('################# 数据名称 ##################')
        print(input_path)
        df = load_file(input_path,'excel')
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
        # print('################# 添加时间戳 ##################')
        # view_df(df)
        # num_rows = len(df)
        # df.insert(0, "timestamp", range(timestamp_start, timestamp_start + num_rows))
        # timestamp_start += num_rows
        # view_df(df)
        print('################# 切分数据集 ##################')
        df_train, df_test = split_dataset(df)
        # print('训练集：')
        # view_df(df_train)
        # print('测试集')
        # view_df(df_test)
        print('################# 合并 ##################')
        df_tmp_train.append(df_train)
        df_tmp_test.append(df_test)

    print('################# 生成文件并保存 ##################')
    final_train = pd.concat(df_tmp_train)
    final_test = pd.concat(df_tmp_test)
    # train, _ = to_csv(final_train)
    train, _ = to_npy(final_train)
    # test, test_label = to_csv(final_test)
    test, test_label = to_npy(final_test)
    view_df(train)
    # train.to_csv(paths['train_csv'], index=False)
    # test.to_csv(paths['test_csv'], index=False)
    # test_label.to_csv(paths['test_label_csv'], index=False)
    np.save(paths['train_npy'], train.to_numpy())
    np.save(paths['test_npy'], test.to_numpy())
    np.save(paths['test_label_npy'], test_label.to_numpy())
