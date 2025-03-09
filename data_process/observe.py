import os
import pandas as pd

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
        mean_value = dataframe[col].mean() if pd.api.types.is_numeric_dtype(dataframe[col]) else None
        std_dev = dataframe[col].std() if pd.api.types.is_numeric_dtype(dataframe[col]) else None
        stats = pd.concat([stats, pd.DataFrame([{
            'Column': col,
            'Data Type': data_type,
            'Non-Null Count': non_null_count,
            'Missing Values': missing_values,
            'Category': unique_values,
            'Min Value': min_value,
            'Max Value': max_value,
            'Mean': mean_value,
            'Std Dev': std_dev
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


def std_err(df):
    ot_mean = df['OT'].mean()
    ot_std = df['OT'].std()

    # Print the results
    print("Mean of OT column:", ot_mean)
    print("Standard Deviation of OT column:", ot_std)