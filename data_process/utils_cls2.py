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

    null_counts = df.isnull().sum()
    print("每一列的空值数量：")
    print(null_counts)

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

def analysis_df(df):
    df['日期'] = df['监测时间'].dt.date
    df['年份'] = df['监测时间'].dt.year
    
    redtide_day = df.groupby('日期')['赤潮类型'].max()
    redtide_year = redtide_day.value_counts()
    
    print("‘赤潮类型’的天数统计：")
    print(f"未发生赤潮的天数: {redtide_year.get(0, 0)}")
    print(f"发生赤潮的天数: {redtide_year.get(1, 0)}")
    
    redtide_year = df[df['赤潮类型'] == 1].groupby('年份')['日期'].nunique()
    print("每年发生赤潮的天数统计：")
    print(redtide_year)

    area_day = df.groupby('日期')['最大成灾面积（平方千米）'].max()
    area_day = area_day.value_counts()

    print("‘最大成灾面积（平方千米）’类别的天数统计：")
    print(area_day)

    df.drop(columns=['日期','年份'], inplace=True)

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

def to_ts(df):
    df = df.drop(columns=['监测时间', '最大成灾面积（平方千米）'])

    window_size = 24  
    sequences = []
    labels = []

    for i in range(0, len(df) - window_size + 1, window_size):
        window = df.iloc[i:i + window_size]
        sequence = window.drop('赤潮类型', axis=1).values.T.tolist()
        label = int(round(window['赤潮类型'].mean()))  # round 将均值四舍五入，代表多数时间点发生了赤潮，则判定为发生赤潮

        sequences.append(sequence)
        labels.append(label)

    sequences = np.array(sequences)
    labels = np.array(labels)

    # 打印labels的总数量
    print(f"标签总数: {len(labels)}")
    
    # 统计每个类别的数量
    label_counts = Counter(labels)
    print(f"类别及其对应的个数: {dict(label_counts)}")

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

if __name__ == "__main__":
    input_train = '/root/lhq/data/data_processed_cls2/all_features/AllRedTide_TRAIN.csv'
    input_test = '/root/lhq/data/data_processed_cls2/all_features//AllRedTide_TEST.csv'
    df_train = load_file(input_train, 'csv')
    df_test = load_file(input_test, 'csv')

    output_dir = '/root/lhq/data/data_processed_cls2/'
    train_ts = 'AllRedTide_TRAIN.ts'
    test_ts = 'AllRedTide_TEST.ts'

    # features_to_remove = ['风速', '风向', '气温', '相对湿度', '气压', '雨量', '水温', '电导率', '盐度', 'ph', '氧化还原电位', '溶解氧', '浊度', '叶绿素a']

    # for feature in features_to_remove:

    #     # jump_list = ['风速', '风向', '气温', '相对湿度', '气压', '雨量', '水温', '电导率', '盐度', 'ph', '氧化还原电位', '溶解氧', '浊度']
    #     # if feature in jump_list: 
    #     #     continue

    #     # 为每个特征创建新文件夹
    #     feature_dir = os.path.join(output_dir, feature)
    #     os.makedirs(feature_dir, exist_ok=True)

    #     train_path = os.path.join(feature_dir, train_ts)
    #     test_path = os.path.join(feature_dir, test_ts)

    #     print('################# 去掉特征 ##################')
    #     print(feature)
    #     df_train = df_train.drop(columns=[feature])
    #     df_test = df_test.drop(columns=[feature])

    #     print('################# 生成ts文件并保存 ##################')

    #     data_dict = {
    #         'train': (df_train, train_path),
    #         'test': (df_test, test_path)
    #     }
    #     for _, (df, path) in data_dict.items():
    #         print(f"已保存到{path}")
    #         sequence, label = to_ts(df)
    #         save_ts(sequence, label, path, 'AllRedTide')

    features_to_remove = ['雨量', '水温', '浊度', '氧化还原电位', '风速']

    feature_remove = '雨量水温浊度氧化还原电位风速'
    feature_dir = os.path.join(output_dir, feature_remove)
    os.makedirs(feature_dir, exist_ok=True)

    train_path = os.path.join(feature_dir, train_ts)
    test_path = os.path.join(feature_dir, test_ts)

    print('################# 去掉特征 ##################')
    print(features_to_remove)
    df_train = df_train.drop(columns=features_to_remove)
    df_test = df_test.drop(columns=features_to_remove)

    print('################# 生成ts文件并保存 ##################')

    data_dict = {
        'train': (df_train, train_path),
        'test': (df_test, test_path)
    }
    for _, (df, path) in data_dict.items():
        print(f"已保存到{path}")
        sequence, label = to_ts(df)
        save_ts(sequence, label, path, 'AllRedTide')

# if __name__ == "__main__":
#     input_dir = '/root/lhq/data/data_grouped_cls2/'
#     output_dir = '/root/lhq/data/data_processed_cls2/'

    
#     train_ts = 'AllRedTide_TRAIN.ts'
#     test_ts = 'AllRedTide_TEST.ts'
    
#     xlsx_files = [f for f in os.listdir(input_dir) if f.endswith('.xlsx')]
#     df_train = load_file()

#     features_to_remove = ['风速', '风向', '气温', '相对湿度', '气压', '雨量', '水温', '电导率', '盐度', 'ph', '氧化还原电位', '溶解氧', '浊度', '叶绿素a']

#     for feature in features_to_remove:

#         # jump_list = ['风速', '风向', '气温', '相对湿度', '气压', '雨量', '水温', '电导率', '盐度', 'ph', '氧化还原电位', '溶解氧', '浊度']
#         # if feature in jump_list: 
#         #     continue

#         # 为每个特征创建新文件夹
#         feature_dir = os.path.join(output_dir, feature)
#         os.makedirs(feature_dir, exist_ok=True)

#         train_path = os.path.join(feature_dir, train_ts)
#         test_path = os.path.join(feature_dir, test_ts)

#         df_tmp_train = []
#         df_tmp_test = []

#         print('################# 去掉特征 ##################')
#         print(feature)

#         for xlsx_file in xlsx_files:
#             input_path = os.path.join(input_dir, xlsx_file)
#             base_name = os.path.splitext(xlsx_file)[0]

#             # print('################# 数据名称 ##################')
#             # print(base_name)
#             df = load_file(input_path,'excel')
#             # 去掉指定特征
#             df = df.drop(columns=[feature])
#             # print('################# 按照时间顺序排序 ##################')
#             df = df.sort_values(by='监测时间')
#             # view_df(df)
#             # print('################# 删除空值 ##################')
#             df = remove_nulls(df)
#             # view_df(df)
#             # print('################# 缺失值处理 ##################')
#             df = nan_process(df)
#             # view_df(df)
#             # print('################# 以小时为单位进行采样 ##################')
#             df = downsampling(df)
#             # view_df(df)
#             # print('################# 打印处理结果 ##################')
#             # analysis_df(df)
#             # print('################# 切分数据集 ##################')
#             df_train, df_test = split_dataset(df)
#             # print('训练集：')
#             # analysis_df(df_train)
#             # view_df(df_train)
#             # print('测试集')
#             # analysis_df(df_test)
#             # view_df(df_test)
#             # print('################# 合并 ##################')
#             df_tmp_train.append(df_train)
#             df_tmp_test.append(df_test)

#         print('################# 生成ts文件并保存 ##################')
#         final_train = pd.concat(df_tmp_train)
#         final_test = pd.concat(df_tmp_test)

#         data_dict = {
#             'train': (final_train, train_path),
#             'test': (final_test, test_path)
#         }
#         for _, (df, path) in data_dict.items():
#             print(f"已保存到{path}")
#             sequence, label = to_ts(df)
#             save_ts(sequence, label, path, 'AllRedTide')


# if __name__ == "__main__":
#     input_dir = '/root/lhq/data/data_grouped_cls2'
#     output_dir = '/root/lhq/data/data_processed_cls2/all_features'

#     files = {
#         'train_ts': 'AllRedTide_TRAIN.ts',
#         'test_ts': 'AllRedTide_TEST.ts',
#         'train_csv': 'AllRedTide_TRAIN.csv',
#         'test_csv': 'AllRedTide_TEST.csv'
#     }

#     paths = {key: os.path.join(output_dir, filename) for key, filename in files.items()}

#     df_tmp_train = []
#     df_tmp_test = []
    
#     xlsx_files = [f for f in os.listdir(input_dir) if f.endswith('.xlsx')]

#     for xlsx_file in xlsx_files:
#             input_path = os.path.join(input_dir, xlsx_file)
#             base_name = os.path.splitext(xlsx_file)[0]

#             print('################# 数据名称 ##################')
#             print(base_name)
#             df = load_file(input_path,'excel')
#             print('################# 按照时间顺序排序 ##################')
#             df = df.sort_values(by='监测时间')
#             view_df(df)
#             print('################# 删除空值 ##################')
#             df = remove_nulls(df)
#             view_df(df)
#             print('################# 缺失值处理 ##################')
#             df = nan_process(df)
#             view_df(df)
#             print('################# 以小时为单位进行采样 ##################')
#             df = downsampling(df)
#             # view_df(df)
#             print('################# 打印处理结果 ##################')
#             analysis_df(df)
#             print('################# 切分数据集 ##################')
#             df_train, df_test = split_dataset(df)
#             print('训练集：')
#             analysis_df(df_train)
#             view_df(df_train)
#             print('测试集')
#             analysis_df(df_test)
#             view_df(df_test)
#             print('################# 合并 ##################')
#             df_tmp_train.append(df_train)
#             df_tmp_test.append(df_test)

#     print('################# 生成文件并保存 ##################')
#     final_train = pd.concat(df_tmp_train)
#     final_test = pd.concat(df_tmp_test)

#     # 保存csv文件
#     final_train.to_csv(paths['train_csv'])
#     final_test.to_csv(paths['test_csv'])

#     # 保存ts文件
#     data_dict = {
#         'train': (final_train, paths['train_ts']),
#         'test': (final_test, paths['test_ts'])
#     }
#     for _, (df, path) in data_dict.items():
#         sequence, label = to_ts(df)
#         save_ts(sequence, label, path, 'AllRedTide')
