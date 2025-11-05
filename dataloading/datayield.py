import os
import json
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import logging
import time



logger = logging.getLogger(__name__)


# =====================================================
# ⚡ 并行加载 .npy 文件
# =====================================================
def _load_one(f):
    try:
        return np.load(f, mmap_mode='r', allow_pickle=False).astype(np.float32, copy=False)
    except Exception as e:
        logger.warning(f"⚠️ 文件加载失败: {f} ({e})")
        # 返回一个空矩阵防止维度错误
        return np.full((480, 480), np.nan, dtype=np.float32)

def load_npy_files(file_list, num_workers=8):
    """并行加载多个 npy 文件并生成 (data_array, time_array)"""
    if not file_list:
        return np.array([]), pd.to_datetime([])

    # 确保路径存在
    file_list = [f for f in file_list if os.path.exists(f)]
    if not file_list:
        logger.warning("⚠️ 所有文件路径均无效，跳过。")
        return np.array([]), pd.to_datetime([])

    # ---- 向量化解析时间 ----
    try:
        basenames = [os.path.basename(f) for f in file_list]
        times = ['2000' + b.split('-')[0][-4:] + b.split('-')[1][:4] for b in basenames]
        times = pd.to_datetime(times, format='%Y%m%d%H%M').round('6min')
    except Exception as e:
        logger.error(f"❌ 时间解析失败: {e}")
        return np.array([]), pd.to_datetime([])

    # ---- 并行读取 ----
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        data = list(ex.map(_load_one, file_list))

    data = np.stack(data, axis=0)

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        data = list(ex.map(_load_one, file_list))

    data = np.stack(data, axis=0)

    # ---- 缺失值修正 ----
    first_path = file_list[0]
    if "RADAR" in first_path:
        data[np.isin(data, (-32768, -32767, -1280))] = np.nan
        data /= 10.0
    elif "LABEL" in first_path:
        data[data == -9] = np.nan

    # ---- 去重时间 ----
    _, idx = np.unique(times, return_index=True)
    return data[idx], times[idx]


# =====================================================
# 🌀 从 JSON 索引懒加载风场 + 雷达，并对齐时间
# =====================================================
def yield_aligned_from_json(
    json_path,
    radar_type="CR",
    num_workers=8,
    max_samples=None
):
    """
    从 data_index_flat.json 懒加载并 yield 对齐后的 (key, wind_array, radar_array)

    ------------------------------------------------------------
    🧭 函数用途：
        从一个扁平化的 JSON 索引文件（例如 data_index_flat.json）中，
        按样本逐个读取“风场 (LABEL/WA)”与“雷达观测 (RADAR/<radar_type>)”文件，
        自动并行加载 .npy 数据并根据时间戳对齐，生成可以直接用于模型训练的成对数据。

    ------------------------------------------------------------
    📥 输入参数：
        json_path : str
            指向 JSON 索引文件的路径（例如 "data_index_flat.json"）。
            文件结构参考 build_flat_data_index() 的输出。

        radar_type : str, default = "CR"
            指定要加载的雷达通道类型。
            可选项包括: "CR", "R05", "RG1", "RG2", "V05", "V15", "VIL" 等。

        num_workers : int, default = 8
            并行加载 .npy 文件的进程数。
            对 CPU I/O 密集型任务有显著加速效果。

        max_samples : int or None, default = None
            限制最多 yield 的样本数。
            适合调试或小规模测试时使用。
            None 表示加载全部样本。

    ------------------------------------------------------------
    📤 输出格式：
        本函数为生成器（generator），每次 yield 一个样本三元组：

        (key, wind_array, radar_array)

        其中：
        - key : str
            样本唯一标识符，例如 "TSW_00_03010941_00095"
        - wind_array : np.ndarray(float32)
            风场标签数据，形状为 (T, H, W)
        - radar_array : np.ndarray(float32)
            对应的雷达观测数据，形状同样为 (T, H, W)

        两者时间维度完全对齐，间隔为 6 分钟；
        缺失值已统一为 np.nan；
        雷达数据已做异常值清理与单位标准化（除以 10）。

        ⚠️ 注意：
            - 风场和雷达文件数量不一致时会自动补 NaN；
            - 任何异常或缺失样本将被跳过；
            - 返回结果可直接用于模型训练、可视化或特征提取。

    ------------------------------------------------------------
    🧪 示例使用：
        >>> from your_module import yield_aligned_from_json
        >>> json_path = "/path/to/wind_radar_files.json"

        >>> for key, wind, radar in yield_aligned_from_json(
        ...     json_path=json_path,
        ...     radar_type="CR",
        ...     num_workers=8,
        ...     max_samples=2
        ... ):
        ...     print("🌀 Key:", key)
        ...     print("Wind shape:", wind.shape)
        ...     print("Radar shape:", radar.shape)
        ...     print("Wind dtype:", wind.dtype, "| Radar dtype:", radar.dtype)
        ...
        🌀 Key: TSW_00_03010941_00095
        Wind shape:  (15, 301, 301)
        Radar shape: (15, 301, 301)
        Wind dtype: float32 | Radar dtype: float32

    ------------------------------------------------------------
    📊 数据说明：
        时间分辨率：6 分钟,T大小不等
        空间维度：H×W， 为 301×301
        缺失值标志：np.nan
        数据类型：float32

        - wind_array：目标场（LABEL/WA），原始 -9 已替换为 NaN
        - radar_array：输入场（RADAR/<radar_type>），原始异常值已清除并除以 10.0

    ------------------------------------------------------------
    🔄 返回类型：
        generator of (str, np.ndarray, np.ndarray)

    ------------------------------------------------------------
    """
    if not os.path.exists(json_path):
        logger.error(f"❌ JSON 文件不存在: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        index_dict = json.load(f)

    wind_dict = index_dict.get("wind", {})
    radar_dict = index_dict.get(radar_type, {})

    total_keys = len(wind_dict)
    logger.info(f"加载索引: {total_keys} 个样本, 雷达类型: {radar_type}")

    # 外层 tqdm 显示样本进度
    count = 0
    for key in tqdm(wind_dict.keys(), desc=f"Aligning {radar_type}", dynamic_ncols=True):
        if max_samples and count >= max_samples:
            break

        if key not in radar_dict:
            continue  # 该时段没有对应雷达类型

        wind_files = wind_dict[key]
        radar_files = radar_dict[key]

        # 并行加载
        wind_data, wind_times = load_npy_files(wind_files, num_workers)
        radar_data, radar_times = load_npy_files(radar_files, num_workers)

        if len(wind_times) == 0 or len(radar_times) == 0:
            continue

        # 时间对齐（6分钟网格）
        start = min(wind_times.min(), radar_times.min()).floor("6min")
        end = max(wind_times.max(), radar_times.max()).ceil("6min")
        time_grid = pd.date_range(start, end, freq="6min")

        da_wind = xr.DataArray(wind_data, coords={"time": wind_times}, dims=["time", "y", "x"])
        da_radar = xr.DataArray(radar_data, coords={"time": radar_times}, dims=["time", "y", "x"])

        aligned_wind = da_wind.reindex(time=time_grid).values
        aligned_radar = da_radar.reindex(time=time_grid).values

        if aligned_wind.shape == aligned_radar.shape:
            yield key, aligned_wind, aligned_radar
            count += 1

    logger.info(f"✅ 已生成 {count} 个样本。")


# =====================================================
# 🚀 示例调用
# =====================================================
if __name__ == "__main__":
    json_path = "/home/dataset-assist-0/data/wind_radar_files.json"
    start_time = time.time()

    logger.info("开始加载与对齐数据 ...")
    for key, wind, radar in yield_aligned_from_json(
        json_path=json_path,
        radar_type="V05",
        num_workers=8,
        max_samples=3  # 示例只处理 3 个样本
    ):
        elapsed = time.time() - start_time
        logger.info(f"\n🌀 Key: {key}")
        logger.info(f"Wind shape:  {wind.shape}")
        logger.info(f"Radar shape: {radar.shape}")
        logger.info(f"Elapsed time: {elapsed:.2f} s\n")
        start_time = time.time()

    logger.info("🎯 全部处理完成。")
