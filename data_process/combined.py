import os
import pandas as pd
from glob import glob
import re

input_dir = '/root/lhq/data/data_raw/资源表数据提取'
output_dir = '/root/lhq/data/data_grouped_cls3'
red_tide_path = '/root/lhq/data/data_raw/赤潮整理.xlsx'

red_tide_data = pd.read_excel(red_tide_path)
red_tide_data['日期'] = pd.to_datetime(red_tide_data['日期'])
red_tide_data['赤潮发生'] = 1

# 指定藻类的名称
specified_algae = ['夜光藻', '红色中缢虫', '棕囊藻', '球形棕囊藻']

# 增加新列以区分赤潮种类
red_tide_data['赤潮类型'] = red_tide_data['赤潮生物'].apply(
    lambda x: 1 if x in specified_algae else 2
)

# 记录赤潮信息的字典
red_tide_dict = {
    (row['日期'].date(), row['海域']): (row['最大成灾面积（平方千米）'], row['赤潮类型']) for _, row in red_tide_data.iterrows()
}

# 查找站点对应的海域
seas = red_tide_data['海域'].unique()
def find_sea(site_name):
    for sea in seas:
        if sea in site_name:
            return sea
    return None

def map_red_tide_info(row):
    key = (row['日期'], row['海域'])
    if key in red_tide_dict:
        return pd.Series([red_tide_dict[key][0], red_tide_dict[key][1]])
    else:
        return pd.Series([-1, 0])   # 0表示未发生赤潮
    
# 清理列名括号
def clean_column_name(col_name):
    return re.sub(r'[\(（][^)]*[\)）]', '', col_name).strip()

for filepath in glob(os.path.join(input_dir, '*.xlsx')):
    filename = os.path.basename(filepath)
    print('############################')
    print(filename)

    buoy_data = pd.read_excel(filepath)

    # 统一列名
    if '观测时间' in buoy_data.columns:
        buoy_data.rename(columns={'观测时间': '监测时间'}, inplace=True)
    if '站点名' in buoy_data.columns:
        buoy_data.rename(columns={'站点名': '站点名称'}, inplace=True)

    # 只保留指定的列
    # print(buoy_data.columns)
    cols_to_keep = ['站点名称', '监测时间', '风速', '风向', '气温', '相对湿度', '气压', '雨量', '水温', '电导率', '盐度', 'ph', '氧化还原电位', '溶解氧', '浊度', '叶绿素a', '蓝绿藻']
    buoy_data = buoy_data[[col for col in buoy_data.columns if clean_column_name(col) in cols_to_keep]]

    # 清理列名中的括号及内容
    buoy_data.columns = [clean_column_name(col) for col in buoy_data.columns]

    buoy_data['监测时间'] = pd.to_datetime(buoy_data['监测时间'])
    buoy_data['日期'] = buoy_data['监测时间'].dt.date
    buoy_data['海域'] = buoy_data['站点名称'].apply(find_sea)

    buoy_data[['最大成灾面积（平方千米）', '赤潮类型']] = buoy_data.apply(map_red_tide_info, axis=1)

    col_drop = ['站点名称', '海域', '日期']
    buoy_data = buoy_data.drop(columns=col_drop)

    # red_tide_occurrence_non_zero_count = buoy_data['赤潮发生'].astype(bool).sum()
    # max_affected_area_non_zero_count = (buoy_data['最大成灾面积（平方千米）'] >= 0).sum()
    # print(f"{os.path.basename(filepath)} - 赤潮发生列中非零的个数: {red_tide_occurrence_non_zero_count}")
    # print(f"{os.path.basename(filepath)} - 最大成灾面积列中非零的个数: {max_affected_area_non_zero_count}")

    df_rows, df_columns = buoy_data.shape
    print(f"{os.path.basename(filepath)} - 文件的行数: {df_rows}，列数: {df_columns}")
    
    null_counts = buoy_data.isnull().sum()
    print("每一列的空值数量：")
    print(null_counts)

    output_filepath = os.path.join(output_dir, os.path.basename(filepath))
    buoy_data.to_excel(output_filepath, index=False)
    print(f"{os.path.basename(filepath)} 已保存到 {output_filepath}")
