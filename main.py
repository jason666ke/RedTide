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


'''
############################ downsampling ############################
root_dir = './data_split/'

for file in os.listdir(root_dir):
    if '_data.csv' in file:
       file_name = file.split('_data.csv')[0]
    else:
       file_name = file_name 
    print(file_name)

    file_path = os.path.join(root_dir, file)
    df = load_csv(file_path)
    df = select_feature(df)
    df = downsampling(df)
    get_column_statistics(df)

    save_path = f'./data_downsample/{file_name}_downsample.csv'
    df.to_csv(save_path, index=False)
'''

'''
############################ data process ############################
select_path = './data_downsample/'

for file in os.listdir(select_path):
    file_name = file.split('_downsample.csv')[0]
    file_path = os.path.join(select_path, file)
    df = load_csv(file_path)
    df = remove_columns_with_nan(df, 8000)
    df = fill_in_missing_values(df)
    get_column_statistics(df)

    save_path = f'./data_processed/{file_name}_processed.csv'
    df.to_csv(save_path, index=False)
    print("{} is successfully processed!".format(file_name))
'''

# file_path = './data_10years/大鹏湾湾口.xlsx'
# # temp_path = './data_downsample/大亚湾东山.csv'
# # save_path = './data_processed/大鹏湾湾口.csv'
# save2_path = './data_drop/大鹏湾湾口.csv'

# df = load_excel(file_path)
# df = df.iloc[:, :-6]

# get_column_statistics(df)
# df = select_feature(df)
# df = downsampling(df)
# df = remove_data(df, '2014-07-11 14:00:00')
# print(df.head(50))
# print(df.tail(50))
# df = data_process(df)
# get_column_statistics(df)
# # # df.to_csv(save_path, index=False)

# # df = load_csv(save_path)
# columns_mapping = {
#     '风向（°)': 'wind_direction',
#     '气温(℃)': 'temperature',
#     '相对湿度(%)': 'relative_humidity',
#     '气压(hPa)': 'pneumatic',
# }
# df = df.rename(columns=columns_mapping)


# df.drop(columns=['wind_speed'], inplace=True)
# df.drop(columns=['wind_direction'], inplace=True)
# df.drop(columns=['rainfall'], inplace=True)
# df.drop(columns=['redox_potential'], inplace=True)
# df.drop(columns=['turbidity'], inplace=True)
# get_column_statistics(df)
# df.to_csv(save2_path, index=False)
# # # df.drop(columns=['水深（米）'], inplace=True)
# # # df.drop(columns=['流速（cm/s）'], inplace=True)
# # # df.drop(columns=['流向（°）'], inplace=True)
# # # # # df = load_csv(save2_path)


# file_path = './data_drop/大亚湾东山.csv'
# save_path = './data_drop/珠江口内伶仃以南_2023_2024.csv'
# df = load_csv(file_path)
# df = df[df['date'] >= '2023-01-01 00:00:00']
# df = df[df['date'] <= '2024-08-25 23:00:00']
# get_column_statistics(df)

# df.to_csv(save_path, index=False)


############################ total ############################
# root_dir = '/root/cyli/data_10years'

# for file in os.listdir(root_dir):
#     if '.xlsx' in file:
#        file_name = file.split('.xlsx')[0]
#     else:
#        file_name = file_name 
#     print(file_name)

#     file_path = os.path.join(root_dir, file)
#     df = load_excel(file_path)
#     get_column_statistics(df)
#     df = select_feature(df)
#     df = downsampling(df)
    
    
#     df = df[df['date'] >= '2015-01-01 00:00:00']
#     df = df[df['date'] <= '2024-08-25 23:00:00']
#     df = data_process(df)
#     get_column_statistics(df)

#     save_path = f'./data_9features/{file_name}.csv'
#     df.to_csv(save_path, index=False)


root_dir = '/root/cyli/data_9features'

for file in os.listdir(root_dir):
    if '.csv' in file:
       file_name = file.split('.csv')[0]
    else:
       file_name = file_name 
    print(file_name)

    file_path = os.path.join(root_dir, file)
    df = load_csv(file_path)
    get_column_statistics(df)
    
    df = df[df['date'] >= '2023-01-01 00:00:00']
    df = df[df['date'] <= '2024-08-25 23:00:00']
    # df = data_process(df)
    get_column_statistics(df)

    save_path = f'./data_9features/{file_name}_2023_2024.csv'
    df.to_csv(save_path, index=False)