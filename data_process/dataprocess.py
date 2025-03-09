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
        
        moving_avg = df[column].rolling(window=window, min_periods=1).mean()
        df[column] = df[column].fillna(moving_avg)
    
    return df
        
def data_process(df):
    df = sliding_windows(df)
    # draw_plot_df(df)
    df = random_forest_predict(df)
    return df
    
    
def random_forest_predict(df):
    # Filling in missing values
    df_copy = df.copy()
    numeric_columns = df.select_dtypes(include=['float64']).columns
    df_copy[numeric_columns] = df_copy[numeric_columns].fillna(df[numeric_columns].median())
    # df[numeric_columns].fillna(method='ffill', inplace=True)  
    # df[numeric_columns].fillna(method='bfill', inplace=True)  
    # df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    # df[numeric_columns] = df[numeric_columns].interpolate(method='polynomial', order=3)

    for column in numeric_columns:
        print(f"Processing column: {column}")
        if df[column].isnull().sum() == 0:
            continue

        features = df_copy[numeric_columns].drop(columns=[column])  
        target = df[column]

        X_train = features[target.notnull()]
        y_train = target[target.notnull()]
        X_test = features[target.isnull()]

        if len(X_train) == 0 or len(X_test) == 0:
            print(f"Skipping column {column} due to insufficient data.")
            continue

        rf = RandomForestRegressor(n_estimators=80, n_jobs=-1, random_state=0)
        rf.fit(X_train, y_train)

        predicted_values = rf.predict(X_test)
        df.loc[df[column].isnull(), column] = predicted_values
        
    return df
    
def test_fill_method(df):
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


def select_feature(df, file_path=None):
    if '监测时间' in df.columns:
        df['监测时间'] = pd.to_datetime(df['监测时间'])
    else: 
        df['观测时间'] = pd.to_datetime(df['观测时间'])
    # cols_with_digits = [col for col in df.columns if re.search(r'\d', col)]
    # df = df.drop(cols_with_digits, axis=1)
    selected_columns = df.select_dtypes(include=['float64', 'datetime'])
    columns_mapping = {
        '监测时间': 'date',
        '观测时间': 'date',
        '检测时间': 'date',
        '风速（m/s）': 'wind_speed',
        '风向（°）': 'wind_direction',
        '气温（℃）': 'temperature',
        '气温(℃)': 'temperature',
        '相对湿度（%）': 'relative_humidity',
        '相对湿度(%)': 'relative_humidity',
        '气压（hpa）': 'pneumatic',
        '气压(hpa)': 'pneumatic',
        '气压（hPa）': 'pneumatic',
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
    columns_to_keep = ['date', 'temperature', 'relative_humidity', 'pneumatic', 
                       'water_temperature', 'conductivity', 'salinity', 'ph', 'DO', 'chlorophyll']
    selected_columns = selected_columns[columns_to_keep]

    # selected_columns.to_csv(file_path, index=False)
    print(f"Seclect Process Down")
    return selected_columns

def remove_data(df, cutoff_date):
    df_filtered = df[df['date'] >= cutoff_date]
    return df_filtered

def remove_columns_with_nan(df, threshold):
    nan_counts = df.isna().sum()
    cols_to_keep = nan_counts[nan_counts <= threshold].index
    df_filtered = df[cols_to_keep]
    return df_filtered

def downsampling(df):
    # df = pd.read_csv(file_path, parse_dates=['监测时间'])
    df = df.set_index('date') 
    df_resampled = df.resample('h').mean()
    df_resampled.reset_index(inplace=True)
    return df_resampled

def process_algal_bloom_data(df):
    df = df[df['年份'].isin([2021, 2022])]
    df = df.sort_values(by='日期')
    # df_new = df.tail(53)
    df = df.drop(columns=['有毒/有害', '序号', '次数', '年份'])
    df.to_csv('./data_processed/algal_bloom.csv', index=False)

def combine_same_site(folder_path, output_file):
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

