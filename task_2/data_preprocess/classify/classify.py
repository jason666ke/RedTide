import os
import pandas as pd

# 合并多个CSV文件中的特征，生成一个包含日期和特征的合并数据集
def merge_features():
    """
    该函数遍历指定文件夹中的所有CSV文件，提取每个文件中的"date"和"OT"列，OT为赤潮成灾面积
    将它们合并成一个包含日期和多个特征的数据集，最终保存为一个新的CSV文件。
    文件合并后，列名将被重新命名为['date', 'feature_1', 'feature_2', 'feature_3', 'feature_4']。(为什么是4个特征呢？)
    """

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
    """
    该函数用于为合并后的数据集添加标签，通过与赤潮数据中的日期匹配，
    判断是否发生赤潮，并根据匹配结果为每一条记录标记"label"（1表示发生赤潮，0表示未发生赤潮）。
    还会将赤潮的最大成灾面积（OT值）添加到合并后的数据中。
    最终，生成带有标签的数据集并保存为新的CSV文件。
    """
    # 读取赤潮数据（例如赤潮日期和最大成灾面积）
    df = pd.read_csv('./data_processed/algal_bloom.csv')
    daya_bay_dates = df[df['海域'] == '大鹏湾'][['日期', '最大成灾面积（平方千米）']] 
    daya_bay_dates['日期'] = pd.to_datetime(daya_bay_dates['日期'])

    merged_output = pd.read_csv('./大鹏湾/merged_output.csv')
    merged_output['date'] = pd.to_datetime(merged_output['date'])
    merged_output['date_only'] = merged_output['date'].dt.date # 提取日期（去掉时间部分）
    merged_output['label'] = 0
    merged_output['OT'] = 0
    
    for idx, row in daya_bay_dates.iterrows():
        matched_indices = merged_output['date_only'] == row['日期'].date()
        merged_output.loc[matched_indices, 'OT'] = row['最大成灾面积（平方千米）']
        merged_output.loc[matched_indices, 'label'] = 1

    merged_output.drop(columns=['date_only'], inplace=True)
    
    merged_output.to_csv('./大鹏湾/merged_output_labeled.csv', index=False)

def count(place):
    """
    该函数用于统计指定海域（如大鹏湾）在2021年和2022年的数据条目数量。
    它会筛选出符合条件的数据并打印出该数据的详细内容以及数量。
    """
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