import pandas as pd
import matplotlib.pyplot as plt

def draw_plot(pth):
    df = pd.read_csv(pth)
    df = df.iloc[-17520:]
    df = df.iloc[::24]
    columns = df.columns[1:] 

    for col in columns:
        plt.figure(figsize=(30, 10))
        plt.plot(df['time'], df[col], label=col, color='red')
        plt.ylabel(col, fontsize=16)
        plt.xlabel('Date', fontsize=16)
        xticks = range(0, len(df), 30) 
        plt.xticks(xticks, df['time'].iloc[xticks], rotation=45)
        plt.yticks(fontsize=14)
        plt.title(f'{col} over the past two years')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'./pics/{col}.pdf')  
        plt.close()  
        
def draw_plot_df(df):
    df = df.iloc[-17520:]
    df = df.iloc[::24]
    columns = df.columns[1:] 

    for col in columns:
        plt.figure(figsize=(30, 10))
        plt.plot(df['time'], df[col], label=col, color='red')
        plt.ylabel(col, fontsize=16)
        plt.xlabel('Date', fontsize=16)
        xticks = range(0, len(df), 30) 
        plt.xticks(xticks, df['time'].iloc[xticks], rotation=45)
        plt.yticks(fontsize=14)
        plt.title(f'{col} over the past two years')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'./pics/{col}_withNan.pdf')  
        plt.close() 

# 使用时调用 draw_plot('your_file.csv')

# def draw_plot(pth):
#     df = pd.read_csv(pth)
#     # plt.figure(figsize=(10, 6))

#     # plt.plot(df['date'], df['HUFL'], label='HUFL', marker='o')
#     # plt.plot(df['date'], df['HULL'], label='HULL', marker='x')
#     # plt.plot(df['date'], df['MUFL'], label='MUFL', marker='^')
#     # plt.plot(df['date'], df['MULL'], label='MULL', marker='s')
#     # plt.plot(df['date'], df['OT'], label='OT', marker='*')

#     # plt.xlabel('Date')
#     # plt.ylabel('Values')
#     # plt.title('Data Plot')

#     # plt.xticks(rotation=45)
#     # plt.savefig('./pics.png')
    
#     df['date'] = pd.to_datetime(df['date'])
#     df = df.iloc[:1000]
#     fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)  

#     axs[0].plot(df['date'][:], df['叶绿素a'], label='yelvsu')
#     axs[0].set_ylabel('yelvsu')
#     axs[0].legend()

#     axs[1].plot(df['date'], df['蓝绿藻'], label='lanlvzao', color='orange')
#     axs[1].set_ylabel('lanlvzao')
#     axs[1].legend()

#     axs[2].plot(df['date'], df['ph'], label='ph',color='green')
#     axs[2].set_ylabel('ph')
#     axs[2].legend()

#     axs[3].plot(df['date'], df['氧化还原电位'], label='yanghuahuanyuandianwei', color='red')
#     axs[3].set_ylabel('yanghuahuanyuandianwei')
#     axs[3].legend()

#     axs[4].plot(df['date'], df['风向'], label='fengxiang', color='purple')
#     axs[4].set_ylabel('fengxiang')
#     axs[4].legend()

#     plt.xlabel('Date')

#     plt.xticks(rotation=45)

#     plt.tight_layout() 
#     plt.savefig('./pics.png')

import pandas as pd
import matplotlib.pyplot as plt
from observe import *

# df1 = pd.read_csv('/root/cyli/data_drop/大鹏湾南澳.csv')
# df2 = pd.read_csv('/root/cyli/Prediction_ModernTCN/results/大鹏湾南澳_ModernTCN_custom_ftS_sl336_pl72_dim64_nb2_lk51_sk5_ffr16_ps16_str4_multiTrue_mergedFalse_Exp_0/prediction_2021_2022.csv')

df1 = pd.read_csv('/root/cyli/data_drop/珠江口内伶仃以南.csv')
df2 = pd.read_csv('/root/cyli/Prediction_ModernTCN/results/珠江口内伶仃以南_ModernTCN_custom_ftS_sl336_pl72_dim64_nb2_lk51_sk5_ffr16_ps16_str4_multiTrue_mergedFalse_Exp_0/prediction_2021_2022.csv')

# df1 = pd.read_csv('/root/cyli/data_drop/珠江口矾石.csv')
# df2 = pd.read_csv('/root/cyli/Prediction_ModernTCN/results/珠江口矾石_ModernTCN_custom_ftS_sl336_pl72_dim64_nb2_lk51_sk5_ffr16_ps16_str4_multiTrue_mergedFalse_Exp_0/prediction_2021_2022.csv')

df1 = df1[(df1['date'] >= '2021-01-01') & (df1['date'] <= '2022-12-31')]

get_column_statistics(df1)
get_column_statistics(df2)

y_values_df1 = df1['chlorophyll']  # 替换 'y_value' 为df1中实际的y轴列名
y_values_df2 = df2.iloc[:,0]  # 替换 'prediction' 为df2中实际的y轴列名

rows_above_30 = df2.index[y_values_df2 > 10].tolist()
print("y_values_df2 中大于 30 的值所在的行号：")
print(rows_above_30)

min_length = min(len(y_values_df1), len(y_values_df2))

plt.figure(figsize=(20, 6))
# plt.plot(range(min_length), y_values_df1[:min_length], label='truth', color='blue')
# plt.plot(range(min_length), y_values_df2[:min_length], label='prediction', color='red')
# plt.plot(range(408), y_values_df1[4682:5090], label='truth', color='blue')
# plt.plot(range(336, 408), y_values_df2[5018:5090], label='prediction', color='red')

# plt.plot(range(1000), y_values_df1[16400:17400], label='truth', color='blue')
# plt.plot(range(1000), y_values_df2[16400:17400], label='prediction', color='red')

plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
# plt.grid(True)

# 自动调整日期标签格式
plt.gcf().autofmt_xdate()
plt.savefig('./output.pdf', dpi=300, bbox_inches='tight')

# plt.show()