import pygrib
import numpy as np

# 将度分秒转换为十进制
def dms_to_decimal(d, m, s, direction):
    decimal = d + m / 60 + s / 3600
    if direction in ['S', 'W']:
        decimal = -decimal
    return decimal

# 找到最近的网格点索引
def find_nearest_index(lat_grid, lon_grid, lat, lon):
    lat_idx = np.abs(lat_grid - lat).argmin()
    lon_idx = np.abs(lon_grid - lon).argmin()
    return lat_idx, lon_idx

# 提取子图时的边界处理
def extract_subgrid(lat_idx, lon_idx, grid_shape, size=5):
    half_size = size // 2
    lat_start = max(lat_idx - half_size, 0)
    lat_end = min(lat_idx + half_size + 1, grid_shape[0])
    lon_start = max(lon_idx - half_size, 0)
    lon_end = min(lon_idx + half_size + 1, grid_shape[1])
    return lat_start, lat_end, lon_start, lon_end

print("开始处理数据...")
# 浮标坐标（原始格式）
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

print("转换浮标坐标...")
# 转换所有浮标坐标
coord_dict_decimal = {
    key: (dms_to_decimal(*lon), dms_to_decimal(*lat)) for key, (lon, lat) in coord_dict.items()
}

print("使用index方法读取并打开grib文件...")
# 读取并打开grib文件
file_path = "/mnt/f/sst_2015-2023/2022Data/01/01.grib"
# grbs = pygrib.open(file_path)
grib_index = pygrib.index(file_path, 'name')

# 打印前10条信息
selected_grbs = grib_index.select(name="Sea surface temperature")
print(selected_grbs[:10])

# 获取经纬度信息
grb = selected_grbs[0]
lats, lons = grb.latlons()
print("经纬度信息:")
print("纬度:", lats)
print("经度:", lons)

# 获取时间步信息
time_steps = [grb.validDate for grb in selected_grbs]
print("时间步信息:", time_steps)

# 获取网格数据
data = grb.values

# 获取温度单位
temperature_unit = grb.units  # 读取单位

# 提取浮标周围5*5的区域
daily_subgrids = {}
for buoy, (lon, lat) in coord_dict_decimal.items():
    print(f"正在处理 {buoy} 浮标周围 5x5 的海表温度...")

    # 初始化每日数据
    daily_data = {}

    # 找到最近的网格点索引
    lat_idx, lon_idx = find_nearest_index(lats[:, 0], lons[0, :], lat, lon)

    # 提取子图时的边界处理
    lat_start, lat_end, lon_start, lon_end = extract_subgrid(lat_idx, lon_idx, lats.shape)

    # 遍历时间步和对应的Grib信息
    for grb, time_step in zip(selected_grbs, time_steps):
        # 获取温度数据并转换为摄氏度
        data = grb.values - 273.15  # 转换为摄氏度
        data = np.where(data == 9999 - 273.15, np.nan, data)  # 处理无效值

        # 提取子图数据
        subgrid = data[lat_start:lat_end, lon_start:lon_end]

        # 存储每日数据
        data_key = time_step.date() # 按日期分组
        if data_key not in daily_data:
            daily_data[data_key] = []
        
        daily_data[data_key].append({
            "time": time_step,
            "subgrid": subgrid
        })

    # 整理每日数据
    for data_key, entries in daily_data.items():
        # 合并24h的数据
        combined_subgrid = np.stack([entry["subgrid"] for entry in entries], axis=0)
        daily_subgrids[f"{buoy}_{data_key}"] = {
            "subgrid": combined_subgrid,    # 24 * 5 * 5 的数据格式
            "lat_range": lats[lat_start:lat_end, lon_start:lon_end],
            "lon_range": lons[lat_start:lat_end, lon_start:lon_end],
        }
    
    # # 存储子图数据
    # subgrids[buoy] = {
    #     "subgrid": subgrid,
    #     "lat_range": lats[lat_start:lat_end, lon_start:lon_end],
    #     "lon_range": lons[lat_start:lat_end, lon_start:lon_end],
    # }

# 打印结果
# for buoy, info in subgrids.items():
#     print(f"{buoy} 浮标位置周围 5x5 的海表温度:")
#     print("温度数据:")
#     print(info["subgrid"])
#     print("纬度范围:")
#     print(info["lat_range"])
#     print("经度范围:")
#     print(info["lon_range"])
#     print()
# 打印结果
for buoy_date, info in daily_subgrids.items():
    # 合并纬度和经度为网格
    lat_range = info["lat_range"]
    lon_range = info["lon_range"]
    lat_lon_grid = np.array([[f"({lat:.2f}, {lon:.2f})" for lon, lat in zip(lon_row, lat_row)] for lat_row, lon_row in zip(lat_range, lon_range)])

    print(f"{buoy_date} 的 5x5 子图数据:")
    print("每天的 24 小时温度数据 (形状: {}):".format(info["subgrid"].shape))
    print(info["subgrid"])
    print("经纬度范围 (网格):")
    for row in lat_lon_grid:
        print(" ".join(row))
    print()
    
    # print("纬度范围:")
    # print(info["lat_range"])
    # print("经度范围:")
    # print(info["lon_range"])
    # print()

# 关闭 GRIB 文件索引
grib_index.close()