import os 
import numpy as np
from utils_era5 import extract_subgrid, find_nearest_index, dms_to_decimal, construct_output_path, save_to_file
import logging


# 数据目录
sst_base_dir = '/mnt/f/era5_subgrid_output/sst_2015_2023'
# 日志目录
log_file = '/mnt/f/era5_subgrid_output/sst_2015_2023.log'
# 设置日志配置
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode="w")

coord_dict = {
    '大亚湾坝光': ((114, 33, 18, 'E'), (22, 39, 33.12, 'N')),
    '大亚湾长湾（核电站附近）': ((114, 34, 48.90, 'E'), (22, 36, 36.89, 'N')),
    '大亚湾东山': ((114, 30, 48.96, 'E'), (22, 34, 14.16, 'N')),
    '大亚湾东冲': ((114, 34, 10.20, 'E'), (22, 28, 30.72, 'N')),
    '大鹏湾沙头角': ((114, 14, 30.48, 'E'), (22, 33, 13.68, 'N')),
    '大鹏湾大梅沙': ((114, 18, 47.88, 'E'), (22, 35, 34.08, 'N')),
    '大鹏湾下沙': ((114, 24, 52.92, 'E'), (22, 36, 7.56, 'N')),
    '大鹏湾南澳': ((114, 28, 39.72, 'E'), (22, 31, 32.16, 'N')),
    '大鹏湾湾口': ((114, 28, 19.2, 'E'), (22, 28, 344.4, 'N')),
    '珠江口沙井': ((113, 44, 3.84, 'E'), (22, 41, 23.64, 'N')),
    '深圳湾蛇口': ((113, 56, 48.48, 'E'), (22, 28, 55.56, 'N')),
    '珠江口矾石': ((113, 48, 3.60, 'E'), (22, 29, 35.52, 'N')),
    '珠江口内伶仃南': ((113, 48, 48.18, 'E'), (22, 22, 48.44, 'N')),
}
coord_dict_decimal = {
    key: (dms_to_decimal(*lon), dms_to_decimal(*lat)) for key, (lon, lat) in coord_dict.items()
}

# 检查单天数据完整性
def check_and_fill_data(path):
    if not os.path.exists(path):
        logging.error(f"文件不存在: {path}")
        return False
    
    try:
        # 读取数据
        data = np.load(path, allow_pickle=True).item()
        for station, station_data in data.items():
            subgrid = station_data['subgrid']

            # 检查station_data的完整性
            if subgrid.shape != (24, 5, 5):
                logging.error(f"数据维度错误: {subgrid.shape}, 数据路径: {path}")
                return False
            
            # 检查是否有有效数据
            if np.all(subgrid == 9999.0):
                logging.error(f"{station} 数据全为无效值: 数据路径：{path}")
                return False
    
    except Exception as e:
        logging.error(f"文件加载失败: {path}, 读取数据时出现异常: {e}")
        return False
    
    logging.info(f"文件完整性检查通过: {path}")
    return True

# 遍历文件夹检查文件
def traverse_and_check(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                check_and_fill_data(file_path)

# 主函数
if __name__ == "__main__":
    logging.info(f"开始检查数据完整性: {sst_base_dir}")
    traverse_and_check(sst_base_dir)
    logging.info(f"数据完整性检查结束。日志保存在: {log_file}")



