import pandas as pd

file_path = './data_raw/海促中心数据信息表-20240614-2021-2022年环境浮标数据.xlsx'
df = pd.read_excel(file_path)

grouped = df.groupby('站点名称')

for name, group in grouped:
    file_name = f'{name}_data.csv'
    print(group)
    group.to_csv(file_name, index=False, encoding='utf-8-sig')

print("success!")
