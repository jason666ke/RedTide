import os
import pandas as pd

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