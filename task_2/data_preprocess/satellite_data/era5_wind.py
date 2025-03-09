import numpy as np
import pygrib
from utils_era5 import extract_subgrid, find_nearest_index, dms_to_decimal, construct_output_path, save_to_file

# 文件路径模板和输出目录
wind_file_path_template = '/mnt/f/wind_2015-2023/{}_10m_{}_component_of_wind.grib'  # U 和 V 分量
output_base_dir = '/mnt/f/wind_2015_2023/era_output'

# 浮标坐标字典
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

# 通用处理函数
def process_wind_grib(file_path, data_type, year):
    try:
        grbs = pygrib.open(file_path)
        print(f"[INFO] 成功打开文件: {file_path}")

        # 获取纬度和经度网格
        lats, lons = grbs[1].latlons()

        # 预计算浮标点的索引
        buoy_indices = {}
        for buoy, (lon, lat) in coord_dict_decimal.items():
            lat_idx, lon_idx = find_nearest_index(lats[:, 0], lons[0, :], lat, lon)
            lat_start, lat_end, lon_start, lon_end = extract_subgrid(lat_idx, lon_idx, lats.shape)
            buoy_indices[buoy] = {
                "lat_range": lats[lat_start:lat_end, lon_start:lon_end],
                "lon_range": lons[lat_start:lat_end, lon_start:lon_end],
                "lat_start": lat_start,
                "lat_end": lat_end,
                "lon_start": lon_start,
                "lon_end": lon_end,
            }
        print(f"[INFO] 已完成浮标点索引的预计算。")

        # 初始化每日数据存储
        daily_data = {buoy: [] for buoy in coord_dict_decimal.keys()}

        # 遍历记录和浮标点
        for record_cnt, grb in enumerate(grbs, start=1):
            # 当前记录日期
            current_date = grb.validDate
            year, month, day = current_date.year, current_date.month, current_date.day
            print(f"[INFO] 当前处理时间: {current_date}")

            # 获取当前记录的数据
            data = grb.values  # 风速数据无需单位转换

            # 遍历浮标点
            for buoy, indices in buoy_indices.items():
                subgrid = data[indices["lat_start"]:indices["lat_end"], indices["lon_start"]:indices["lon_end"]]
                daily_data[buoy].append(subgrid)

            # 如果记录达到一天（24小时）将数据保存到文件
            if record_cnt % 24 == 0:
                print(f"[INFO] 已处理完 {year}-{month:02d}-{day:02d} 的24小时数据，准备保存。")
                combined_data = {}
                for buoy, indices in buoy_indices.items():
                    # 提取数据、纬度范围和经度范围
                    combined_data[buoy] = {
                        "subgrid": np.stack(daily_data[buoy], axis=0),  # 合并为(24, 5, 5)的数据
                        "lat_range": indices["lat_range"],
                        "lon_range": indices["lon_range"],
                    }

                # 构造保存路径并保存
                output_file = construct_output_path(f"{year}_{data_type}", f"{month:02d}", f"{day:02d}", output_base_dir)
                save_to_file(combined_data, output_file)
                print(f"[INFO] 数据已保存至路径: {output_file}")

                # 清空每日数据
                daily_data = {buoy: [] for buoy in coord_dict_decimal.keys()}

        # 关闭文件
        grbs.close()
        print(f"[INFO] {year} 年 {data_type} 数据处理完成。")

    except FileNotFoundError:
        print(f"[WARNING] 文件 {file_path} 未找到，跳过处理。")
    except Exception as e:
        print(f"[ERROR] 处理文件 {file_path} 时发生错误: {e}")

# 主函数
def main():
    years_to_process = range(2015, 2024)

    # 处理风速数据
    for year in years_to_process:
        print(f"[INFO] 开始处理 {year} 年的风速数据...")
        u_file = wind_file_path_template.format(year, "u")
        v_file = wind_file_path_template.format(year, "v")

        print(f"[INFO] 开始处理 {year} 年的 U 分量风速数据...")
        process_wind_grib(u_file, "wind_u", year)
        print(f"[INFO] {year} 年的 U 分量风速数据处理完成。\n")

        print(f"[INFO] 开始处理 {year} 年的 V 分量风速数据...")
        process_wind_grib(v_file, "wind_v", year)
        print(f"[INFO] {year} 年的 V 分量风速数据处理完成。\n")

if __name__ == "__main__":
    main()
