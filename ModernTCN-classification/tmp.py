from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import pandas as pd
import numpy as np

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader
}

args_data = 'UEA'
args_root_path = '/root/lhq/RedTide/ModernTCN-classification/all_datasets/RedTide/'
flag = 'TRAIN'
batch_size = 32
shuffle_flag = False
args_num_workers = 0
args_seq_len = 24


Data = data_dict[args_data]

drop_last = False
data_set = Data(
    root_path=args_root_path,
    flag=flag,
)

features = data_set.feature_df
labels = data_set.labels_df

labels_list = labels.values.squeeze()
class_sample_count = pd.value_counts(labels_list)

# 假设标签为 0（负样本）和 1（正样本）
# 计算每个类别的权重，使得正样本的采样概率增加
desired_pos_ratio = 1 / 3  # 正样本希望占每个 batch 的 1/3
desired_neg_ratio = 2 / 3  # 负样本占 2/3

# 总样本数
num_samples = len(labels_list)

# 计算每个类别的采样权重
pos_ratio = 3 / 4  
neg_ratio = 1 / 4 

weights = { 
    0: neg_ratio / class_sample_count[0],  
    1: pos_ratio / class_sample_count[1], 
}
# 为每个样本分配权重
samples_weight = torch.tensor(np.array([weights[label] for label in labels_list]), dtype=torch.float32)

# 进行采样
sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)

# weights = 1. / class_sample_count
# samples_weight = torch.tensor(np.array(weights[labels_list]), dtype=torch.float32)
# sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)

data_loader = DataLoader(
    data_set,
    batch_size=batch_size,
    shuffle=shuffle_flag,
    num_workers= args_num_workers,
    drop_last=drop_last,
    collate_fn=lambda x: collate_fn(x, max_len=args_seq_len),
    sampler=sampler
)

for i, (batch_x, label, padding_mask) in enumerate(data_loader):
    pos_samples = (label == 1).sum().item()
    neg_samples = (label == 0).sum().item()
    print(f"Batch {i}: Positive samples: {pos_samples}, Negative samples: {neg_samples}")


# return data_set, data_loader


# import numpy as np

# def read_uea_format(file_path):
#     data = []
#     labels = []
#     series_length = None  # Initialize variable to store series length
    
#     with open(file_path, 'r') as file:
#         is_data_section = False
#         for line in file:
#             line = line.strip()
#             # Check for metadata lines
#             if line.startswith("@"):
#                 if line.startswith("@seriesLength"):
#                     # Extract series length value
#                     series_length = int(line.split(" ")[1])
#                 if line.startswith("@data"):
#                     # Begin reading the data section
#                     is_data_section = True
#                 continue
            
#             if is_data_section:
#                 # Split the line at the ':' to separate features and labels
#                 features, label = line.split(":")
#                 # Convert the features to a list of floats
#                 features = [float(x) for x in features.split(",")]
#                 data.append(features)
#                 labels.append(label)
    
#     # Convert the data and labels to NumPy arrays for easier processing
#     data = np.array(data)
#     labels = np.array(labels)
    
#     return data, labels, series_length

# # Example usage
# file_path = "/Users/hlam/mytask/code/RedTide/ModernTCN-classification/all_datasets/EthanolConcentration/EthanolConcentration_TRAIN.ts"
# data, labels, series_length = read_uea_format(file_path)
# print("Data shape:", data.shape)
# print("Labels:", labels)
# print("Series Length:", series_length)
