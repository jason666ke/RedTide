import pandas as pd
import numpy as np

xlsx_path = '/root/lhq/data/data_raw/赤潮整理.xlsx'
csv_path = '/root/cyli/Prediction_ModernTCN/results/dyw_dongshan/real_prediction.csv'
test_path = '/root/lhq/data/data_processed_detection/dyw_dongshan/test.npy'
test_label_path = '/root/lhq/data/data_processed_detection/dyw_dongshan/test_label.npy'

# 读取 Excel 文件
df_xlsx = pd.read_excel(xlsx_path)

# 转换日期列为 datetime 格式
df_xlsx['日期'] = pd.to_datetime(df_xlsx['日期'])

# 筛选时间为 2022 年且海域为“大亚湾”的数据
# filtered_data = df_xlsx[(df_xlsx['日期'].dt.year == 2022) & 
#                         (df_xlsx['日期'].dt.month == 1) & 
#                         (df_xlsx['海域'] == '大亚湾')]
filtered_data = df_xlsx[(df_xlsx['日期'].dt.year == 2022) & 
                        (df_xlsx['海域'] == '大亚湾')]
print(filtered_data)

# 标签范围：2022-01-01 往后300天
start_date = pd.Timestamp('2022-01-01')
end_date = start_date + pd.Timedelta(days=299)
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

# 创建空的 DataFrame 来存储标签数据
label_data = []

cnt = 0
# 遍历每一天，生成24小时的标签
for date in date_range:
    day_labels = []
    for hour in range(24):
        # 当前小时的日期时间
        current_time = date + pd.Timedelta(hours=hour)
        # 如果当前时间在筛选的数据中，则标签为1，否则为0
        if current_time in filtered_data['日期'].values:
            day_labels.append(1)
            cnt += 1
        else:
            day_labels.append(0)
    label_data.extend(day_labels)

print("发生赤潮数目")
print(cnt)

# 将标签数据转换为 NumPy 数组并保存
label_data = np.array(label_data)
np.save(test_label_path, label_data)

# 打印标签数据的形状和部分内容
print(f"生成的标签数据形状: {label_data.shape}")
print(f"部分标签数据: {label_data[:24]}")  # 打印前24个标签

# 读取 CSV 文件并保存为 npy 文件
df_csv = pd.read_csv(csv_path)
test_arr = df_csv.to_numpy()
np.save(test_path, test_arr)

print(f"数据已保存为 .npy 文件: {test_path}")
