# test for NumPy NPY-file operations(read,write,save......) and contour

# 读取NPY文件

import glob
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import xarray as xr


# idx: "00  01  02  04  05  07  08  09  11  12  14  15  20  23  24  25  29  30  98  99"
def load_wind_or_rader_single(patten,scale=10.0):
    files = glob.glob(patten)
    times_line = [os.path.basename(f).split('-')[0][-4:] + os.path.basename(f).split('-')[1][:4] for f in files]
    data = [np.load(f) / scale for f in tqdm(files,desc="processing files", leave=False)]
    data = np.array(data)
    
    # === 处理缺失值 ===
    if "RADAR" in patten:
        data[data == -32768] = np.nan  # 雷达数据中，-32768视为缺失
    elif "WA" in patten:
        data[data == -9] = np.nan   # 风场数据中，-9视为缺失

    # === 将时间字符串转为时间戳 ===
    times_line = pd.to_datetime(['2000' + t for t in times_line], format='%Y%m%d%H%M')
    # === 四舍五入到最近6分钟 ===
    times_line = times_line.round('6min')
    
    # 去重: 保留最后一个出现的时间点
    rev_idx = np.arange(len(times_line))[::-1]
    _, unique_idx = np.unique(times_line[::-1], return_index=True)
    # 还原到原始顺序
    keep_idx = rev_idx[unique_idx][::-1]
    times_line = times_line[keep_idx]
    data = data[keep_idx]
    
    
    # === 构建xarray对象 ===
    data = xr.DataArray(
        data,
        coords={'time': times_line},
        dims=['time', 'x', 'y']
    )

    return data, times_line

# radar_type: "CR  R05  RG1  RG2  V05  V15  VIL"
def load_wind_radar_data(idx: str, radar_type: str):
    """
    加载指定索引的风场与雷达数据
    返回:
        dict: {time_station_id: (x坐标, y坐标, 风场数据, 雷达数据)}
    """
    base_dir = "/home/dataset-assist-1/SevereWeather_AI_2025"
    src = os.path.join(base_dir, "TSW", "TrainSet", idx)
    result = {}

    for dirs in tqdm(os.listdir(src), desc="processing dirs"):
        # 读取 WA 文件
        pattern = os.path.join(src, dirs, "LABEL", "WA", "*.npy")
        wind_data, wa_times_line = load_wind_or_rader_single(pattern, scale=10.0)

        # 读取雷达文件
        pattern = os.path.join(src, dirs, "RADAR", radar_type, "*.npy")
        radar_data, radar_times_line = load_wind_or_rader_single(pattern, scale=1.0)

        # === 创建统一时间网格 ===
        start = min(wa_times_line.min(), radar_times_line.min()).floor('6min')
        end = max(wa_times_line.max(), radar_times_line.max()).ceil('6min')
        time_grid = pd.date_range(start, end, freq='6min')

        # === 同步对齐两者时间 ===
        da_wind_aligned = wind_data.reindex(time=time_grid)
        da_radar_aligned = radar_data.reindex(time=time_grid)

        # === 缺失帧填充 ===
        # da_wind_aligned = da_wind_aligned.fillna(0)
        # da_radar_aligned = da_radar_aligned.fillna(0)

        # 解析目录信息
        stanum = dirs[-5:]
        time_info = dirs[-15:-5]

        result[f"{time_info}{stanum}"] = xr.Dataset({
            'wind': da_wind_aligned,
            'radar': da_radar_aligned
        })

    return result


def load_wind_data(idx: str):
    """
    加载指定索引的风场数据（无异常处理版本）
    返回:
        dict: {time_station_id: (x坐标, y坐标, 风场数据)}
    """
    base_dir = "/home/dataset-assist-1/SevereWeather_AI_2025"
    src = os.path.join(base_dir, "TSW", "TrainSet", idx)
    result = {}

    for dirs in tqdm(os.listdir(src), desc="processing dirs"):
        # 读取 WA 文件
        pattern = os.path.join(src, dirs, "LABEL", "WA", "*.npy")
        files = glob.glob(pattern)
        files = sorted(files)
        start_time = files[0].split('/')[-1].split('-')[1][:4]  # 提取起始时间,6min间隔
        data = [np.load(f) / 10.0 for f in tqdm(files,
                                                desc="processing files", leave=False)]
        data = np.array(data)

        # 解析目录信息
        stanum = dirs[-5:]
        time_info = dirs[-15:-5]

        # 读取格式文件
        formatfile = os.path.join(
            base_dir, "FORMAT", "Label_Format_DOC", f"Label_Format_{stanum}.txt")
        labelfmt = pd.read_csv(formatfile)

        # 提取经纬度范围
        def get_value(row):
            return round(float(labelfmt.iloc[row, 0].split('=')[1]), 4)

        lons, lats = get_value(0), get_value(1)
        lone, late = get_value(2), get_value(3)

        # 坐标轴生成
        x1 = np.linspace(lons, lone, 301)
        y1 = np.linspace(late, lats, 301)[::-1]

        result[f"{time_info}{stanum}"] = (x1, y1, data, start_time)

    return result
