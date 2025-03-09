import os
import pandas as pd
from observe import *
from dataprocess import *
# from draw import *

# load .xlsx
def load_excel(file_path): 
    df = pd.read_excel(file_path, engine='openpyxl')
    # df = df.sort_values(by='日期')
    # print(df.head(50))
    return df

# load .csv
def load_csv(file_path): 
    df = pd.read_csv(file_path)
    # df = pd.read_csv(file_path, parse_dates=['监测时间'])
    # df = df.sort_values(by='监测时间')
    # print(df.head(50))
    # print(df.tail(50))
    return df


# ---------------- observe ----------------##
# root_dir = './data_drop/珠江口沙井.csv'

# for file in os.listdir(root_dir):
#     if '.xlsx' in file:
#        file_name = file.split('.xlsx')[0]
#     else:
#        file_name = file_name 
#     print(file_name)

#     file_path = os.path.join(root_dir, file)
#     df = load_csv(file_path)
#     get_column_statistics(df)

# df = load_csv(root_dir)
# get_column_statistics(df)

# first_10_columns = df.iloc[:, :10]
# print(first_10_columns)

# output_file = "./data_drop/大亚湾东山_2023_2024_9.csv"
# first_10_columns.to_csv(output_file, index=False)
# print(f"前10列数据已保存到: {output_file}")



########################### total ############################
root_dir = '/root/kochi/redtide_detect_whole_process/data_raw/data_10years'

for file in os.listdir(root_dir):
    if '.xlsx' in file:
       file_name = file.split('.xlsx')[0]
    else:
       file_name = file_name 
    print(file_name)

    file_path = os.path.join(root_dir, file)
    df = load_excel(file_path)
    get_column_statistics(df)
    df = select_feature(df)
    df = downsampling(df)
    
    df = df[df['date'] >= '2015-01-01 00:00:00']
    df = df[df['date'] <= '2024-08-25 23:00:00']
    df = data_process(df)
    get_column_statistics(df)

    save_path = f'/root/kochi/redtide_detect_whole_process/task_1/data_9features/{file_name}.csv'
    df.to_csv(save_path, index=False)


# root_dir = '/root/cyli/data_9features'

# for file in os.listdir(root_dir):
#     if '.csv' in file:
#        file_name = file.split('.csv')[0]
#     else:
#        file_name = file_name 
#     print(file_name)

#     file_path = os.path.join(root_dir, file)
#     df = load_csv(file_path)
#     get_column_statistics(df)
    
#     df = df[df['date'] >= '2023-01-01 00:00:00']
#     df = df[df['date'] <= '2024-08-25 23:00:00']
#     df = data_process(df)
#     get_column_statistics(df)

#     save_path = f'./data_9features/{file_name}_2023_2024.csv'
#     df.to_csv(save_path, index=False)