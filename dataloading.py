import glob
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import xarray as xr
from concurrent.futures import ThreadPoolExecutor, as_completed


# idx: "00  01  02  04  05  07  08  09  11  12  14  15  20  23  24  25  29  30  98  99"
def load_wind_or_rader_single(pattern, scale=10.0, num_workers=8):
    files = glob.glob(pattern)
    if not files:
        return xr.DataArray(), pd.to_datetime([])

    # 提前生成 times_line（不重复做字符串操作）
    basenames = [os.path.basename(f) for f in files]
    times_line = ['2000' + b.split('-')[0][-4:] +
                  b.split('-')[1][:4] for b in basenames]
    times_line = pd.to_datetime(times_line, format='%Y%m%d%H%M').round('6min')

    # 并行加载数据
    def _load_file(f):
        return np.load(f) / scale

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        data = list(ex.map(_load_file, files))

    data = np.array(data, dtype=np.float32)

    # 缺失值处理
    if "RADAR" in pattern:
        data[data == -32768] = np.nan
    elif "WA" in pattern:
        data[data == -9] = np.nan

    # 去重: 保留最后一个时间点
    _, unique_idx = np.unique(times_line[::-1], return_index=True)
    keep_idx = np.sort(len(times_line) - 1 - unique_idx)
    times_line = times_line[keep_idx]
    data = data[keep_idx]

    # 构建 xarray
    da = xr.DataArray(data, coords={'time': times_line}, dims=[
                      'time', 'x', 'y'])
    return da, times_line

# radar_type: "CR  R05  RG1  RG2  V05  V15  VIL"


def load_wind_radar_data(idx: str, radar_type: str, num_workers=4):
    base_dir = "/home/dataset-assist-1/SevereWeather_AI_2025"
    src = os.path.join(base_dir, "TSW", "TrainSet", idx)
    dirs_list = os.listdir(src)
    result = {}

    def process_dir(dirs):
        wa_pattern = os.path.join(src, dirs, "LABEL", "WA", "*.npy")
        radar_pattern = os.path.join(src, dirs, "RADAR", radar_type, "*.npy")
        wind_data, wa_times = load_wind_or_rader_single(wa_pattern, scale=10.0)
        radar_data, radar_times = load_wind_or_rader_single(
            radar_pattern, scale=1.0)

        if len(wa_times) == 0 or len(radar_times) == 0:
            return None

        start = min(wa_times.min(), radar_times.min()).floor('6min')
        end = max(wa_times.max(), radar_times.max()).ceil('6min')
        time_grid = pd.date_range(start, end, freq='6min')

        da_wind = wind_data.reindex(time=time_grid)
        da_radar = radar_data.reindex(time=time_grid)

        stanum = dirs[-5:]
        time_info = dirs[-15:-5]
        key = f"{time_info}{stanum}"
        return key, xr.Dataset({'wind': da_wind, 'radar': da_radar})

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(process_dir, d) for d in dirs_list]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="processing dirs"):
            res = fut.result()
            if res:
                result[res[0]] = res[1]

    return result


load_wind_radar_data("00", radar_type="CR", num_workers=8)


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
