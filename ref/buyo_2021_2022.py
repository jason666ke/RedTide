import pandas as pd
from tabulate import tabulate
import os

# read data
red_tide_path = './red_tide_analysis/赤潮整理_2021_2022.csv'
red_tide_data = pd.read_csv(red_tide_path)
buoy_data_path= './data_raw/海促中心数据信息表-20240614-2021-2022年环境浮标数据.xlsx'
buoy_data = pd.read_excel(buoy_data_path)

# get columns name
# print("列名: ")
# print(buoy_data.columns)

# # see data head
# print("浮标数据预览: ")
# print(tabulate(buoy_data.head(), headers='keys', tablefmt='pretty'))

# # get station list
# stations = buoy_data['站点名称'].unique()
# print("浮标站点序列： ", stations)

# print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

# print("海域数据预览： ")
# print(tabulate(red_tide_data.head(), headers='keys', tablefmt='pretty'))

# print("海域序列： ", seas)

# transfer time to datetime
buoy_data['监测时间'] = pd.to_datetime(buoy_data['监测时间'])
red_tide_data['日期'] = pd.to_datetime(red_tide_data['日期'])

# add new column shows whether redtide happens
# red_tide_data['赤潮发生'] = red_tide_data['最大成灾面积（平方千米）'].apply(lambda x: 1 if pd.notna(x) and x > 0 else 0)
red_tide_data['赤潮发生'] = 1
print(tabulate(red_tide_data, headers='keys', tablefmt='pretty'))

# match site and sea
# get seas list
seas = red_tide_data['海域'].unique()
def find_sea(site_name):
    for sea in seas:
        if sea in site_name:
            return sea
    return None

buoy_data['海域'] = buoy_data['站点名称'].apply(find_sea)

# check if match correctly
cols_to_view = ['站点名称', '监测时间', '风速', '风向', '海域']
print(tabulate(buoy_data[cols_to_view].head(), headers='keys', tablefmt='pretty'))
print("站点名称与海域匹配情况: ")
print(buoy_data[['站点名称', '海域']].drop_duplicates())


# preprocess the data
buoy_data['日期'] = buoy_data['监测时间'].dt.date

# dict involves redtide happens time and area
red_tide_dict = {
    (row['日期'].date(), row['海域']): row['最大成灾面积（平方千米）'] for _, row in red_tide_data.iterrows()
}

def map_red_tide_info(row):
    key = (row['日期'], row['海域'])
    if key in red_tide_dict:
        return pd.Series([1, red_tide_dict[key]])
    else:
        return pd.Series([0, -1])

print("buoy dataframe shape: ", buoy_data.shape)
print("red tide data shape: ", red_tide_data.shape)

buoy_data[['赤潮发生', '最大成灾面积（平方千米）']] = buoy_data.apply(map_red_tide_info, axis=1)

# merge_data = pd.merge(buoy_data, red_tide_data[['日期', '海域', '赤潮发生', '最大成灾面积（平方千米）']], 
#                       left_on=['监测时间', '海域'], right_on=['日期', '海域'], how='left')

print(tabulate(buoy_data[['站点名称', '监测时间', '赤潮发生', '最大成灾面积（平方千米）']].head(), headers='keys', tablefmt='pretty'))

# 统计赤潮发生列中非零值的数量
red_tide_occurrence_non_zero_count = buoy_data['赤潮发生'].astype(bool).sum()
# 统计最大成灾面积列中非零值的数量
max_affected_area_non_zero_count = (buoy_data['最大成灾面积（平方千米）'] >= 0).sum()
print(f"赤潮发生列中非零的个数: {red_tide_occurrence_non_zero_count}")
print(f"最大成灾面积列中非零的个数: {max_affected_area_non_zero_count}")

# 随机抽取50条数据测试
sample_data = buoy_data[buoy_data['赤潮发生'] != 0].sample(n=50, random_state=1)
# 打印抽取的数据
print(tabulate(sample_data[['站点名称', '监测时间', '风速', '风向', '海域', '赤潮发生', '最大成灾面积（平方千米）']]))

output_file = 'buoy_2021_2022_with_red_tide.csv'
output_folder = 'buyo_analysis'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

buoy_data.to_csv(os.path.join(output_folder, output_file), index=False, encoding='utf-8-sig')
print(f"\n处理后的数据已移动到文件夹: {output_folder}")