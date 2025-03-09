import pandas as pd
import os
from observe import *
import numpy as np
from scipy.interpolate import interp1d


###################### 数据处理 #######################

file_path = "/root/cyli/data_drop/大亚湾东山.csv"
df = pd.read_csv(file_path)

df['date'] = pd.to_datetime(df['date'])

start_date = pd.to_datetime("2014-01-01 00:00:00")
end_date = pd.to_datetime("2024-01-01 00:00:00")
filtered_df = df[(df['date'] >= start_date) & (df['date'] < end_date)]

filtered_df = filtered_df[['date', 'chlorophyll']]

output_dir = "./task3"
output_file = os.path.join(output_dir, "dywds_task3.csv")
filtered_df.to_csv(output_file, index=False)
get_column_statistics(filtered_df)

print(f"数据已保存到: {output_file}")




###################### 重新采样 #######################

file_path = "/root/cyli/task3/dywds_task3.csv"
df = pd.read_csv(file_path)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

daily_df = df.resample('D').max()   # 数据按天进行重采样
daily_df.reset_index(inplace=True)  

output_dir = "./task3"
output_file = os.path.join(output_dir, "dywds_task3_D.csv")
daily_df.to_csv(output_file, index=False)
get_column_statistics(daily_df)

print(f"数据已保存到: {output_file}")




##################### 取均值 #######################

file_path = "/root/cyli/task3/dywds_task3_D.csv"
save_path = "/root/cyli/task3/dywds_task3_avg365.csv"

df = pd.read_csv(file_path)
df['date'] = pd.to_datetime(df['date'])

# 去掉2016年2月29日和2020年2月29日的数据
df = df[~((df['date'].dt.year == 2016) & (df['date'].dt.month == 2) & (df['date'].dt.day == 29))]
df = df[~((df['date'].dt.year == 2020) & (df['date'].dt.month == 2) & (df['date'].dt.day == 29))]

# 提取月份和日期作为分组依据
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

daily_stats = df.groupby(['month', 'day'])['chlorophyll'].agg(['mean', 'std']).reset_index()    # 提取月份和日期信息并按照月份和日期分组，计算每组的 chlorophyll 的均值和标准差
daily_stats.rename(columns={'mean': 'chlorophyll_mean', 'std': 'chlorophyll_std'}, inplace=True)
daily_stats[['chlorophyll_mean', 'chlorophyll_std']].to_csv(save_path, index=False)
get_column_statistics(daily_stats)
print(f"平均值和标准差已保存到 {save_path}")



###################### 平滑曲线 #######################

file_path = "/root/cyli/task3/dywds_task3_avg365.csv"
save_path = "/root/cyli/task3/dywds_task3_smooth_8760.csv"

daily_avg = pd.read_csv(file_path)

x = np.arange(len(daily_avg))  
y = daily_avg['chlorophyll'].values  

# 创建插值函数
interp_func = interp1d(x, y, kind='cubic', fill_value="extrapolate")

# 生成更细的插值点（365个点）
x_new = np.linspace(0, len(daily_avg) - 1, 365)
y_new = interp_func(x_new)

# 在每两个点之间插入23个点
# 总共需要 365 * 24 = 8760 个点
x_final = np.linspace(0, len(daily_avg) - 1, 365 * 24)
y_final = interp_func(x_final)

# 创建新的 DataFrame
smoothed_df = pd.DataFrame({
    'chlorophyll': y_final
})

# 保存结果到文件
smoothed_df.to_csv(save_path, index=False)
get_column_statistics(smoothed_df)


print(f"平滑后的曲线已保存到 {save_path}")


###################### 拼接 #######################
file_path = "/root/cyli/task3/dywds_task3.csv"
smooth_path = "/root/cyli/task3/dywds_task3_smooth_8760.csv"
save_path = "/root/cyli/task3/dywds_final.csv"

df = pd.read_csv(file_path)
df['date'] = pd.to_datetime(df['date'])
df_smooth = pd.read_csv(smooth_path)
chlorophyll_repeated = np.tile(df_smooth['chlorophyll'].values, 10) # 通过 np.tile 将平滑数据重复以匹配原始数据的长度

get_column_statistics(df)
get_column_statistics(df_smooth)

# 去掉2016年2月29日和2020年2月29日的数据
df = df[~((df['date'].dt.year == 2016) & (df['date'].dt.month == 2) & (df['date'].dt.day == 29))]
df = df[~((df['date'].dt.year == 2020) & (df['date'].dt.month == 2) & (df['date'].dt.day == 29))]

if len(df) != len(chlorophyll_repeated):
    raise ValueError("df 和 df_smooth 的行数不一致，请检查数据！")

df['chlorophyll_smoothed'] = chlorophyll_repeated
df.to_csv(save_path, index=False)
get_column_statistics(df)

print(f"平滑后的曲线已保存到 {save_path}")


###################### 取上下界限 #######################

file_path = "/root/cyli/task3/dywds_final.csv"
df = pd.read_csv(file_path)
df['date'] = pd.to_datetime(df['date'])


# df['chlorophyll_up'] = df['chlorophyll_smoothed'] + 1.345
# df['chlorophyll_down'] = df['chlorophyll_smoothed'] - 1.345


df['chlorophyll_up'] = df['chlorophyll_smoothed'] + daily_stats['chlorophyll_std'].min()
df['chlorophyll_down'] = df['chlorophyll_smoothed'] - daily_stats['chlorophyll_std'].min()

save_path = "/root/cyli/task3/dywds_final_with_up_down.csv"
df.to_csv(save_path, index=False)
get_column_statistics(df)

print(f"数据已更新并保存到 {save_path}")



###################### 计算异常区域 #######################
from scipy.integrate import trapezoid
file_path = "/root/cyli/task3/dywds_final_with_up_down.csv"
save_path = "/root/cyli/task3/dywds_anomaly_area.csv"

df = pd.read_csv(file_path)
window_size = 7 * 24

def calculate_anomaly_area(chlorophyll, chlorophyll_up, chlorophyll_down, window_size):
    anomaly_areas = []
    
    for i in range(len(chlorophyll)):
        end_idx = min(i + window_size, len(chlorophyll))
        window_chlorophyll = chlorophyll[i:end_idx]
        window_up = chlorophyll_up[i:end_idx]
        window_down = chlorophyll_down[i:end_idx]
        
        # 计算高于上限的面积
        above_up = np.maximum(window_chlorophyll - window_up, 0)
        area_above_up = trapezoid(above_up, dx=1) if len(above_up) > 1 else above_up[0]
        
        # 计算低于下限的面积
        below_down = np.maximum(window_down - window_chlorophyll, 0)
        area_below_down = trapezoid(below_down, dx=1) if len(below_down) > 1 else below_down[0]
        
        # 总异常面积
        anomaly_areas.append(area_above_up + area_below_down)
    
    return np.array(anomaly_areas)

# 计算每个点开始往后7×24个点的异常区域面积

df['anomaly_area'] = calculate_anomaly_area(
    df['chlorophyll'].values,
    df['chlorophyll_up'].values,
    df['chlorophyll_down'].values,
    window_size
)

# 保存结果
df.to_csv(save_path, index=False)
# print(df.head(50))
# print(df.iloc[50:100])
# print(df.tail(50))
get_column_statistics(df)

print(f"数据已更新并保存到 {save_path}")



###################### 判断不稳定的阈值范围 #######################
file_path = "/root/cyli/task3/dywds_anomaly_area.csv"
save_path = "/root/cyli/task3/dywds_anomaly_exceed_300_dates.csv"
df = pd.read_csv(file_path)
if not pd.api.types.is_datetime64_any_dtype(df['date']):
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
anomaly_exceed_300 = df[df['anomaly_area'] > 1000]
anomaly_exceed_300['year'] = anomaly_exceed_300['date'].dt.year
anomaly_exceed_300['month'] = anomaly_exceed_300['date'].dt.month
anomaly_exceed_300['day'] = anomaly_exceed_300['date'].dt.day

unique_dates = anomaly_exceed_300.drop_duplicates(subset=['year', 'month', 'day'])

# 打印这些日期的年月日信息
# print(unique_dates[['year', 'month', 'day']])

# 保存年月日信息到新的 CSV 文件
unique_dates[['date', 'year', 'month', 'day']].to_csv(save_path, index=False)


file_path = "/root/cyli/task3/dywds_anomaly_exceed_300_dates.csv"
df = pd.read_csv(file_path)
print(df.head(50))
print(df.iloc[50:100])