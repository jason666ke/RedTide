import pandas as pd
import numpy as np
import os

# 单个站点添加标签

##########################
station = "zjkfs"
seas = "珠江口"
feature_num = 9 # 特征数

xlsx_path = '/root/lhq/data/data_raw/赤潮整理.xlsx'

# output_dir = '/root/lhq/data/data_predict/zjknld/pred_Chl'
# train_path = '/root/lhq/data/data_predict/zjknld/pred_Chl/prediction_2015_2020.csv'
# test_path ='/root/lhq/data/data_predict/zjknld/pred_Chl/prediction_2021_2022.csv'

output_dir = "/root/lhq/data/predict_1422_9"
train_path = f"/root/cyli/Prediction_ModernTCN/results/old_results/{station}/prediction.csv"
test_path = f"/root/cyli/Prediction_ModernTCN/results/old_results/{station}/prediction_2021_2022.csv"

train_output = f"{output_dir}/train/{station}.csv"
test_output = f"{output_dir}/test/{station}.csv"
label_output = f"{output_dir}/label/{station}.csv"

files = {
        'train_csv': 'train.csv',
        'test_csv': 'test.csv',
        'test_label_csv': 'test_label.csv',
        'train_npy': 'train.npy',
        'test_npy': 'test.npy',
        'test_label_npy': 'test_label.npy'
    }

paths = {key: os.path.join(output_dir, filename) for key, filename in files.items()}

# 标签范围：2021-01-01 往后729天
label_start = '2021-01-01'
label_len = 728

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

df_train = df_train.iloc[:, :feature_num]
# df_train = df_train.tail(61344).iloc[:, :feature_num].reset_index(drop=True) # 大鹏湾南澳特殊情况
df_test = df_test.iloc[:, :feature_num]

rows_train, cols_train = df_train.shape
print(f"行数：{rows_train/24}, 列数：{cols_train}")
rows_test, cols_test = df_test.shape
print(f"行数：{rows_test/24}, 列数：{cols_test}")

df_xlsx = pd.read_excel(xlsx_path)
filtered_data = df_xlsx[(df_xlsx['日期'].dt.year.isin([2021,2022])) & 
                        (df_xlsx['海域'] == seas)]
print(filtered_data)

start_date = pd.Timestamp(label_start)
end_date = start_date + pd.Timedelta(days=label_len)
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

label_data = []
times_stamp = []

cnt = 0
for date in date_range:
    for hour in range(24):
        current_time = date + pd.Timedelta(hours=hour)
        times_stamp.append(current_time)
        if date in filtered_data['日期'].values:
            label_data.append(1)
            cnt += 1
        else:
            label_data.append(0)
            
# for date in date_range:
#     times_stamp.append(date)
#     if date in filtered_data['日期'].values:
#         label_data.append(1)
#         cnt += 1
#     else:
#         label_data.append(0)
                       
print(pd.DataFrame({'time':times_stamp, 'label':label_data}).head(10))

print("发生赤潮数目")
print(cnt)

df_label = pd.DataFrame(label_data)
# print(df_label[df_label[0]==1].head(50))

# df_train.to_csv(paths['train_csv'], index=False)
# df_test.to_csv(paths['test_csv'], index=False)
# df_label.to_csv(paths['test_label_csv'], index=False)
# np.save(paths['train_npy'], df_train.to_numpy())
# np.save(paths['test_npy'], df_test.to_numpy())
# np.save(paths['test_label_npy'], df_label.to_numpy())

df_train.to_csv(train_output, index=False)
df_test.to_csv(test_output, index=False)
df_label.to_csv(label_output, index=False)
##########################

# 合并所有站点
'''
##########################
# base_dir = "/root/lhq/data/predict_1422_9"
# categories = ["train", "test", "label"]

# for category in categories:
#     input_dir = os.path.join(base_dir, category)
#     csv_path = os.path.join(input_dir, f"{category}.csv")
#     npy_path = os.path.join(input_dir, f"{category}.npy")

#     csv_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith('.csv')]

#     df_list = [pd.read_csv(file) for file in csv_files]
#     final_df = pd.concat(df_list, ignore_index=True)

#     rows, cols = final_df.shape
#     print(f"{category} - 行数：{rows/24}, 列数：{cols}")

#     final_df.to_csv(csv_path, index=False)
#     data_arr = final_df.to_numpy()
#     np.save(npy_path, data_arr)

##########################
'''

input_csv = "/root/lhq/data_final/dpwxs/test_label.csv"
output_npy = "/root/lhq/data_final/dpwxs/test_label.npy"
df = pd.read_csv(input_csv)
data_arr = df.to_numpy()
np.save(output_npy, data_arr)