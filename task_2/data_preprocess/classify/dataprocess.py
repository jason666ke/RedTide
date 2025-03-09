import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import re
import os
from draw import *

def sliding_windows(df):
    """
    该函数对DataFrame中的数值型列应用滑动窗口法（window=96），
    使用滚动平均法填补缺失值。对每一列单独处理，将该列中的缺失值替换为其前96个数据点的平均值。
    """
    window = 96
    numeric_columns = df.select_dtypes(include=['float64']).columns
    for column in numeric_columns:
        if df[column].isnull().sum() == 0:
            continue
        
        # mean = df[column].mean()
        # std = df[column].std()
        # lower_limit = mean - 3 * std
        # upper_limit = mean + 3 * std
        # df[column].apply(lambda x: x if lower_limit <= x <= upper_limit else np.nan)
        
        moving_avg = df[column].rolling(window=window, min_periods=1).mean() # 计算每个位置的前96个数据点的均值，并用这个均值来填补当前数据点的缺失值
        df[column] = df[column].fillna(moving_avg)
    
    return df

# 数据预处理，首先应用滑动窗口方法填补缺失值，然后用随机森林预测缺失值
def data_process(df):
    """
    该函数对传入的DataFrame应用滑动窗口法填补缺失值，并使用随机森林回归填补其余缺失值。
    """
    df = sliding_windows(df)
    # draw_plot_df(df)
    df = random_forest_predict(df)
    return df
    
    
def random_forest_predict(df):
    """
    该函数使用随机森林回归模型填补DataFrame中数值型列的缺失值。
    每一列中的缺失值将通过其余特征进行回归预测，并用预测值填充。
    """
    # Filling in missing values
    df_copy = df.copy()
    numeric_columns = df.select_dtypes(include=['float64']).columns
    df_copy[numeric_columns] = df_copy[numeric_columns].fillna(df[numeric_columns].median()) # 中位数填充
    # df[numeric_columns].fillna(method='ffill', inplace=True)  
    # df[numeric_columns].fillna(method='bfill', inplace=True)  
    # df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    # df[numeric_columns] = df[numeric_columns].interpolate(method='polynomial', order=3)

    for column in numeric_columns:
        print(f"Processing column: {column}")
        if df[column].isnull().sum() == 0:
            continue

        features = df_copy[numeric_columns].drop(columns=[column]) # 除了当前列之外的所有其他列作为特征
        target = df[column] # 包含缺失值的当前列

        X_train = features[target.notnull()]    # 特征集中的非空值作为训练集
        y_train = target[target.notnull()]    # 目标集中的非空值作为训练集
        X_test = features[target.isnull()]  # 特征集中的空值作为测试集

        if len(X_train) == 0 or len(X_test) == 0:
            print(f"Skipping column {column} due to insufficient data.")
            continue

        rf = RandomForestRegressor(n_estimators=80, n_jobs=-1, random_state=0)
        rf.fit(X_train, y_train)

        predicted_values = rf.predict(X_test)
        df.loc[df[column].isnull(), column] = predicted_values
        
    return df

# 测试不同的填补方法，包括均值填充、多项式回归和支持向量回归（SVR）
def test_fill_method(df):
    """
    该函数用于测试不同的数据填充方法，首先使用均值填充缺失值，
    然后对比使用多项式回归和支持向量回归（SVR）进行交叉验证，评估不同方法的均方误差（MSE）。
    """
    df.fillna(df.mean(), inplace=True)
    X = df.drop(columns=['叶绿素a']).values
    y = df['叶绿素a'].values
    
    # polynomial regression
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    poly_model = LinearRegression()
    poly_scores = cross_val_score(poly_model, X_poly, y, cv=5, scoring='neg_mean_squared_error')
    print("Polynomial Regression CV MSE: ", -poly_scores.mean())

    # SVM
    svr_model = SVR()
    svr_scores = cross_val_score(svr_model, X, y, cv=5, scoring='neg_mean_squared_error')
    print("SVR CV MSE: ", -svr_scores.mean())

# 选择特定的特征并重命名
def select_feature(df, file_path=None):
    """
    该函数从DataFrame中选择数值型和日期型列，并对这些列进行重命名，
    返回重新命名后的DataFrame。该函数还提供了重命名后的列名映射。
    """
    df['监测时间'] = pd.to_datetime(df['监测时间'])
    # cols_with_digits = [col for col in df.columns if re.search(r'\d', col)]
    # df = df.drop(cols_with_digits, axis=1)
    selected_columns = df.select_dtypes(include=['float64', 'datetime'])
    columns_mapping = {
        '监测时间': 'date',
        '风速（m/s）': 'wind_speed',
        '风向（°）': 'wind_direction',
        '气温（℃）': 'temperature',
        '相对湿度（%）': 'relative_humidity',
        '气压（hpa）': 'pneumatic',
        '雨量（mm）': 'rainfall',
        '水温（℃）': 'water_temperature',
        '电导率（mS/cm）': 'conductivity',
        '盐度': 'salinity',
        'ph': 'ph', 
        '氧化还原电位（mV）': 'redox_potential',
        '溶解氧（mg/L）': 'DO',
        '浊度': 'turbidity',
        '叶绿素a（μg/L）': 'chlorophyll',
        '流向（°）':'liuxiang',
        '有效波高（m）':'bogao',
        '有效波周期（s）': 'bozhouqi', 
        '蓝绿藻（cell/μL）': 'cyanobacteria',
        '硝酸盐（μg/L）': 'nitrate',
        '亚硝酸盐（μg/L）': 'nitrite',
        '磷酸盐（μg/L）': 'phosphates',
        '氨氮（μg/L）': 'ammonia_nitrogen'
    }
    selected_columns = selected_columns.rename(columns=columns_mapping)
    # selected_columns.to_csv(file_path, index=False)
    print(f"Seclect Process Down")
    return selected_columns

# 根据给定的截止日期移除数据
def remove_data(df, cutoff_date):
    """
    该函数根据给定的截止日期（cutoff_date），过滤DataFrame中日期晚于该日期的数据。
    """
    df_filtered = df[df['date'] >= cutoff_date]
    return df_filtered

# 移除缺失值超过指定阈值的列
def remove_columns_with_nan(df, threshold):
    """
    该函数移除DataFrame中缺失值超过给定阈值的列，只保留缺失值数量小于或等于阈值的列。
    """
    nan_counts = df.isna().sum()
    cols_to_keep = nan_counts[nan_counts <= threshold].index
    df_filtered = df[cols_to_keep]
    return df_filtered

# 对数据进行下采样，将数据从分钟级别重采样到小时级别
def downsampling(df):
    """
    该函数将DataFrame中的数据从分钟级别重采样到小时级别，采用每小时数据的均值。
    """
    # df = pd.read_csv(file_path, parse_dates=['监测时间'])
    df = df.set_index('date') 
    df_resampled = df.resample('h').mean()
    df_resampled.reset_index(inplace=True)
    return df_resampled

# 处理赤潮数据，过滤指定年份并保存为CSV
def process_algal_bloom_data(df):
    """
    该函数过滤赤潮数据中只包含2021年和2022年的数据，并去除一些无关列，
    最后将处理后的数据保存为CSV文件。
    """
    df = df[df['年份'].isin([2021, 2022])]
    df = df.sort_values(by='日期')
    # df_new = df.tail(53)
    df = df.drop(columns=['有毒/有害', '序号', '次数', '年份'])
    df.to_csv('./data_processed/algal_bloom.csv', index=False)

# 合并同一站点的多个CSV文件数据，并按日期求均值
def combine_same_site(folder_path, output_file):
    """
    该函数将指定文件夹中的所有CSV文件读取并合并，按“监测时间”列求均值后保存为新的CSV文件。
    """
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    dataframes = []
    
    for file in files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        dataframes.append(df)
    
    if not dataframes:
        raise ValueError("没有找到符合条件的文件！")
    
    combined_df = pd.concat(dataframes, ignore_index=True, sort=False)
    combined_df['监测时间'] = pd.to_datetime(combined_df['监测时间'])
    df_mean = combined_df.groupby('检测时间').mean(numeric_only=True)
    
    df_mean.to_csv(output_file, index=False)

