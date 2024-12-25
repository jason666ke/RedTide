import os
import glob
import numpy as np
import netCDF4 as nc

def dms2dec(deg, min, sec, dir):
    
    dec = deg + (min / 60) + (sec / 3600)
    if dir in ['S', 'W']:
        dec = -dec
    return dec

def conv_coords(coord_dict):
    dec_coords = {}
    for key, coord in coord_dict.items():
        lon_deg, lon_min, lon_sec, lon_dir = coord[0]
        lat_deg, lat_min, lat_sec, lat_dir = coord[1]
        
        lon = dms2dec(lon_deg, lon_min, lon_sec, lon_dir)
        lat = dms2dec(lat_deg, lat_min, lat_sec, lat_dir)
        
        # 经度范围调整到[0,360]
        if lon < 0:
            lon += 360 
        if lat < -90 or lat > 90:
            raise ValueError("纬度超出有效范围 (-90 到 90)")
        
        dec_coords[key] = (lat, lon)
    
    return dec_coords

def extract_patch(data, lat, lon, buoy_lat, buoy_lon, grid=4):
    lat_idx = np.abs(buoy_lat - lat).argmin()
    lon_idx = np.abs(buoy_lon - lon).argmin()

    half_grid = grid // 2

    lat_range = slice(lat_idx - half_grid, lat_idx + half_grid + 1)
    lon_range = slice(lon_idx - half_grid, lon_idx + half_grid + 1)

    patch = data[lat_range, lon_range]
    patch = patch.filled(np.nan)

    return patch

def process_single(input_path):
    print(f"处理文件: {input_path}")

    data = nc.Dataset(input_path)
    cholra = data.variables['chlor_a'][:]
    lat = data.variables['latitude'][:]
    lon = data.variables['longitude'][:]
    
    for _, (buoy_lat, buoy_lon) in dec_coords.items():
        cholra_patch = extract_patch(cholra, lat, lon, buoy_lat, buoy_lon)
    
    # 空间上空值排查
    if np.all(np.isnan(cholra_patch)):
        return "process_nan", cholra_patch
    else:
        return "process_CHL", cholra_patch

def process_files(dir, coord_dict):
    # 只处理以 '02401_02401.nc' 结尾的文件
    file_pattern = os.path.join(dir, "*02401_02401.nc")
    
    files = glob.glob(file_pattern)
    
    # 时间上空值排查
    # if len(files) == 24:
    #     print(f"文件夹 {dir} 符合条件。")
    #     for file in files:
    #         for key in coord_dict:
    #             relative_path = os.path.relpath(file, '/mnt/g/CHL')
    #             output_root, data = process_single(file)
    #             output_dir = os.path.join('/mnt/f', output_root, key, os.path.dirname(relative_path))
    #             os.makedirs(output_dir, exist_ok=True)
    #             output_file = os.path.join(output_dir, os.path.basename(file).replace('.nc', '.npy'))
    #             np.save(output_file, data)
    # else:
    #     print(f"文件夹 {dir} 中文件数量为 {len(files)}，跳过该文件夹。")

    # 处理daily文件夹
    if os.path.basename(dir) == 'daily':
        print(f"文件夹 {dir} 符合条件。")
        for file in files:
            for key in coord_dict:
                relative_path = os.path.relpath(file, '/mnt/g/CHL')
                output_root, data = process_single(file)
                output_dir = os.path.join('/mnt/f', output_root, key, os.path.dirname(relative_path))
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, os.path.basename(file).replace('.nc', '.npy'))
                np.save(output_file, data)
    else:
        print(f"文件夹 {dir}不符合条件，跳过该文件夹。")

def process_multi_folders(root_dir, coord_dict):
    cnt = 0
    for subdir, _, _ in os.walk(root_dir):
        # cnt += 1
        # if cnt < 1993:
        #     continue
        process_files(subdir, coord_dict)

def cnt_files(dir):
    file_pattern = os.path.join(dir, "*02401_02401.npy")
    
    files = glob.glob(file_pattern)

    cnt = 0
    
    # if len(files) == 24:
    #     for _ in files:
    #         cnt += 1
    # else:
    #     print(f"文件夹 {dir} 中文件数量为 {len(files)}，跳过该文件夹。")

    # 处理daily文件夹
    if os.path.basename(dir) == 'daily':
        for _ in files:
            cnt += 1
    else:
        print(f"文件夹 {dir} 不合要求，跳过该文件夹。")

    return cnt

def cnt_multi_folders(root_dir):
    cnt = 0
    for subdir, _, _ in os.walk(root_dir):
        # print(subdir)
        cnt += 1

        if subdir == '/mnt/g/CHL/2018/201812/27':
            print(subdir)
            break
        # cnt_tmp = cnt_files(subdir)
        # cnt += cnt_tmp

    print(cnt)

# 2015-2017
#'/mnt/f/process_nan/' daily: 2392, hours: 52104
# daily: 9503, hours: 624
root_directory = '/mnt/g/CHL/' 

# 浮标坐标
coord_dict = {
    '大亚湾坝光':((114, 33, 18, 'E'), (22, 39, 33.12, 'N')),
    '大亚湾长湾（核电站附近）':((114, 34, 48.90, 'E'), (22, 36, 36.89, 'N')),     
    '大亚湾东山':((114, 30, 48.96, 'E'), (22, 34, 14.16, 'N')),
    '大亚湾东冲':((114, 34, 10.20, 'E'), (22, 28, 30.72, 'N')),
    '大鹏湾沙头角':((114, 14, 30.48, 'E'), (22, 33, 13.68, 'N')),
    '大鹏湾大梅沙':((114, 18, 47.88, 'E'), (22, 35, 34.08, 'N')),
    '大鹏湾下沙':((114, 24, 52.92, 'E'), (22, 36, 7.56, 'N')), 
    '大鹏湾南澳':((114, 28, 39.72, 'E'), (22, 31, 32.16, 'N')),
    '大鹏湾湾口':((114, 28, 19.2, 'E'), (22, 28, 344.4, 'N')),
    '珠江口沙井':((113, 44, 3.84, 'E'), (22, 41, 23.64, 'N')),
    '深圳湾蛇口':((113, 56, 48.48, 'E'), (22, 28, 55.56, 'N')),
    '珠江口矾石':((113, 48, 3.60, 'E'), (22, 29, 35.52, 'N')),
    '珠江口内伶仃南':((113, 48, 48.18, 'E'), (22, 22, 48.44, 'N')), 
    # '大亚湾东冲':((114, 34, 10.20, 'E'), (22, 28, 30.72, 'N'))
}
# 转换所有浮标坐标
dec_coords = conv_coords(coord_dict)

# 开始处理
process_multi_folders(root_directory, coord_dict)

# 计数
# cnt_multi_folders(root_directory)
