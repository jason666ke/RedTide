import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
import os

# load .xlsx
def load_excel(file_path): 
    df = pd.read_excel(file_path, engine='openpyxl')
    df = df.sort_values(by='日期')
    print(df.tail(60))
    return df

# load .csv
def load_csv(file_path): 
    df = pd.read_csv(file_path)
    # df = pd.read_csv(file_path, parse_dates=['监测时间'])
    # df = df.sort_values(by='监测时间')
    # print(df.iloc[50:100, :])
    return df

# statistics of each column
def get_column_statistics(dataframe):
    stats = pd.DataFrame(columns=[
        'Column', 'Data Type', 'Non-Null Count', 
        'Missing Values', 'Category', 
        'Min Value', 'Max Value'
    ])
    for col in dataframe.columns:
        data_type = dataframe[col].dtype
        non_null_count = dataframe[col].notnull().sum()
        missing_values = dataframe[col].isnull().sum()
        unique_values = dataframe[col].nunique()
        min_value = dataframe[col].min() if pd.api.types.is_numeric_dtype(dataframe[col]) else None
        max_value = dataframe[col].max() if pd.api.types.is_numeric_dtype(dataframe[col]) else None
        stats = pd.concat([stats, pd.DataFrame([{
            'Column': col,
            'Data Type': data_type,
            'Non-Null Count': non_null_count,
            'Missing Values': missing_values,
            'Category': unique_values,
            'Min Value': min_value,
            'Max Value': max_value
        }])], ignore_index=True)
    
    # print message
    chunk_size = 10
    num_chunks = len(stats) // chunk_size + (1 if len(stats) % chunk_size != 0 else 0)
    for i in range(num_chunks):
        chunk = stats[i * chunk_size : (i + 1) * chunk_size]
        print(f"数据统计第 {i*chunk_size+1} 到 {(i+1)*chunk_size} 条:\n\n{chunk}\n\n")
    
    return stats

def get_time_intervals(df):
    df['Timestamp'] = pd.to_datetime(df['监测时间'])
    df['Time_Diff'] = df['Timestamp'].diff()
    print(df['Time_Diff'])
    
    time_diff_mean = df['Time_Diff'].mean()
    time_diff_median = df['Time_Diff'].median()
    time_diff_min = df['Time_Diff'].min()
    time_diff_max = df['Time_Diff'].max()
    print(f"Average time interval: {time_diff_mean}")
    print(f"Median time interval: {time_diff_median}")
    print(f"Minimum time interval: {time_diff_min}")
    print(f"Maximum time interval: {time_diff_max}")

def get_correlation(df, target):
    df = df.select_dtypes(include=['float64'])
    correlation_matrix = df.corr()
    chlorophyll_corr = correlation_matrix[target].sort_values(ascending=False)
    print(chlorophyll_corr) 

def data_processed(df, save_path=None):
    # Filling in missing values
    window = 72
    for column in df.columns:
        if not pd.api.types.is_float_dtype(df[column]):
            continue
        print(f"Processing column: {column}")
        if column == "叶绿素a":
            continue
        else:
            mean = df[column].mean()
            std = df[column].std()
            lower_limit = mean - 3 * std
            upper_limit = mean + 3 * std
            df[column].apply(lambda x: x if lower_limit <= x <= upper_limit else np.nan)
        
        if df[df[column].isnull()].empty:
            continue  
        
        moving_avg = df[column].rolling(window=window, min_periods=1).mean()
        df[column] = df[column].fillna(moving_avg)
    
    numeric_columns = df.select_dtypes(include=['float64']).columns
    # df[numeric_columns] = df[numeric_columns].interpolate(method='polynomial', order=3)
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    # df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

    # df['水深'] = df['水深'].fillna(df['水深'].mean())
    # df['流速'] = df['流速'].fillna(df['流速'].mean())
    # df['流向'] = df['流向'].fillna(df['流向'].mean())
    # df['有效波周期'] = df['有效波周期'].fillna(df['有效波周期'].mean())
    
    # rename
    df.rename(columns={'监测时间': 'date'}, inplace=True)
    df.rename(columns={'叶绿素a': 'OT'}, inplace=True)
    
    # change type of time
    # df['date'] = pd.to_datetime(df['date'])
    
    # Move the 'OT' column to the last column
    df = df[[col for col in df.columns if col != 'OT'] + ['OT']]
    
    # df.to_csv(save_path, index=False)
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
    selected_columns = df.iloc[:, :27].select_dtypes(include=['float64', 'datetime'])
    # selected_columns.drop(columns=['水温', '硝酸盐', '亚硝酸盐', '磷酸盐', '氨氮', '水深', '流速', '流向', '有效波高', '平均波向', '有效波周期'], inplace=True)
    # selected_columns.drop(columns=['水温', '硝酸盐', '亚硝酸盐', '磷酸盐', '氨氮', '流速', '流向', '有效波高', '平均波向', '有效波周期', '蓝绿藻'], inplace=True)
    # selected_columns.drop(columns=['硝酸盐', '亚硝酸盐', '磷酸盐', '氨氮', '蓝绿藻', '水深'], inplace=True)
    # selected_columns.drop(columns=['有效波高', '平均波向', '有效波周期'], inplace=True)
    # selected_columns.drop(columns=['硝酸盐', '亚硝酸盐', '磷酸盐', '氨氮', '水深', '流速', '流向', '有效波高', '平均波向', '有效波周期'], inplace=True)
    # selected_columns.drop(columns=['水深', '流速', '流向', '有效波高', '平均波向', '有效波周期'], inplace=True)
    # selected_columns.drop(columns=['硝酸盐', '亚硝酸盐', '磷酸盐', '氨氮', '水深', '流速', '流向', '有效波高', '平均波向', '有效波周期'], inplace=True)
    # selected_columns.drop(columns=['氧化还原电位', '硝酸盐', '亚硝酸盐', '磷酸盐', '氨氮', '水深', '流速', '流向', '有效波周期'], inplace=True)
    # selected_columns.drop(columns=['硝酸盐', '亚硝酸盐', '磷酸盐', '氨氮', '水深', '流速', '流向', '有效波高', '平均波向', '有效波周期'], inplace=True)
    # selected_columns.drop(columns=['硝酸盐', '亚硝酸盐', '磷酸盐', '氨氮', '水深', '蓝绿藻'], inplace=True)
    # selected_columns.drop(columns=['氧化还原电位', '蓝绿藻', '水深', '流速', '流向', '有效波高', '平均波向', '有效波周期'], inplace=True)
    # selected_columns.drop(columns=['硝酸盐', '亚硝酸盐', '磷酸盐', '氨氮', '水深', '流速', '流向', '有效波高', '平均波向', '有效波周期'], inplace=True)
    # selected_columns.drop(columns=['硝酸盐', '亚硝酸盐', '磷酸盐', '氨氮', '水深', '流速', '流向', '有效波高', '平均波向', '有效波周期'], inplace=True)
    selected_columns.drop(columns=['硝酸盐', '亚硝酸盐', '磷酸盐', '氨氮', '水深', '流速', '流向', '有效波高', '平均波向', '有效波周期'], inplace=True)
    
    # selected_columns.to_csv(file_path, index=False)
    print(f"Seclect Process Down")
    return selected_columns
    
def downsampling(df):
    # df = pd.read_csv(file_path, parse_dates=['监测时间'])
    df.set_index('监测时间', inplace=True)
    df_resampled = df.resample('h').mean()
    df_resampled.reset_index(inplace=True)
    # df_resampled.to_csv(save_path, index=False)
    # df['监测时间'] = pd.to_datetime(df['监测时间'])
    # df_30min = df[(df['监测时间'].dt.minute == 0) | (df['监测时间'].dt.minute == 30)]
    # df_30min.to_csv(save_path, index=False)
    return df_resampled

def std_err(df):
    ot_mean = df['OT'].mean()
    ot_std = df['OT'].std()

    # Print the results
    print("Mean of OT column:", ot_mean)
    print("Standard Deviation of OT column:", ot_std)   

def process_algal_bloom_data():
    df = load_excel('../赤潮整理.xlsx')
    df = df[df['年份'].isin([2021, 2022])]
    df = df.sort_values(by='日期')
    # df_new = df.tail(53)
    df = df.drop(columns=['有毒/有害', '序号', '次数', '年份'])
    df.to_csv('./data_processed/algal_bloom.csv', index=False)
    # get_column_statistics(df)

# process_algal_bloom_data()
# df = load_csv('./data_processed/algal_bloom.csv')
# print(df)
# df = load_file('./20240614haicuzhongxintiqushuju/浮标参数单位.xlsx')
# df = load_file('./data_raw/海促中心数据信息表-20240614-提取数据说明.xlsx')
# df = load_excel('./data_raw/海促中心数据信息表-20240614-2021-2022年环境浮标数据.xlsx')
# df = load_excel('./data_raw/海促中心数据信息表-20240614-2021-2022年环境浮标数据.xlsx')
df = load_csv('./data_processed/data.csv')
print(df)
# get_column_statistics(df)



# folder_path='./data_split/'
# file_list = os.listdir(folder_path)
# print(file_list)
# # for file_name in os.listdir(folder_path):
# file_name = file_list[13]
# if file_name.endswith('.csv'):
#     # get name
#     if '_data.csv' in file_name:
#         name = file_name.split('_data.csv')[0]
#     else:
#         name = file_name 
#     print(name)

#     file_path = os.path.join(folder_path, file_name)
#     df = load_csv(file_path)
    
#     get_column_statistics(df)
#     get_correlation(df, '叶绿素a')
#     df = select_feature(df)
#     df = downsampling(df)
#     # get_column_statistics(df)
#     df = data_processed(df)
    
#     output_path = f'./data_14classes_processed/{name}_processed.csv'
#     get_column_statistics(df)
#     get_correlation(df, 'OT')
#     df.to_csv(output_path, index=False)

#     print("{} is successfully processed!".format(name))

