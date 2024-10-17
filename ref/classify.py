import os
import pandas as pd

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

def merge_features():
    directory='大鹏湾/'
    merged_df = pd.DataFrame()
    num=0
    for filename in os.listdir(directory):
        print(filename)
        if filename.endswith(".csv"):  
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)

            if 'OT' in df.columns and 'date' in df.columns:
                if num == 0:
                    extracted_df = df[['date', 'OT']]
                else:
                    extracted_df = df[['OT']]
                
                merged_df = pd.concat([merged_df, extracted_df], axis=1)

        num += 1

    merged_df.columns = ['date', 'feature_1', 'feature_2', 'feature_3',  'feature_4']
    merged_df.to_csv('./大鹏湾/merged_output.csv', index=False)


def add_catagory(place=None):
    df = pd.read_csv('./data_processed/algal_bloom.csv')
    daya_bay_dates = df[df['海域'] == '大鹏湾'][['日期', '最大成灾面积（平方千米）']] 
    daya_bay_dates['日期'] = pd.to_datetime(daya_bay_dates['日期'])

    merged_output = pd.read_csv('./大鹏湾/merged_output.csv')
    merged_output['date'] = pd.to_datetime(merged_output['date'])
    merged_output['date_only'] = merged_output['date'].dt.date
    merged_output['label'] = 0
    merged_output['OT'] = 0
    
    for idx, row in daya_bay_dates.iterrows():
        matched_indices = merged_output['date_only'] == row['日期'].date()
        merged_output.loc[matched_indices, 'OT'] = row['最大成灾面积（平方千米）']
        merged_output.loc[matched_indices, 'label'] = 1

    merged_output.drop(columns=['date_only'], inplace=True)
    
    merged_output.to_csv('./大鹏湾/merged_output_labeled.csv', index=False)

def count(place):
    df = pd.read_csv('./data_processed/algal_bloom.csv')
    df['日期'] = pd.to_datetime(df['日期'])
    filtered_df = df[(df['海域'] == place) & (df['日期'].dt.year.isin([2021, 2022]))]
    print(filtered_df)
    count = filtered_df.shape[0]
    print(f"海域为{place}且年份为2021或2022的数据有 {count} 条")

# merge_features()
# add_catagory()
df = pd.read_csv('./大鹏湾/merged_output_labeled.csv')
# num_ones = df['label'].sum()
# print(f"Label列中有 {num_ones} 个值为1。")
get_column_statistics(df)