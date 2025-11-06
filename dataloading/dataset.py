import json, os, numpy as np, pandas as pd, xarray as xr, torch
from functools import lru_cache

from torch.utils.data import Dataset
from dataloading.datayield  import yield_aligned_from_json
from dataloading.datayield import load_npy_files


class SimpleWindRadarDataset(Dataset):
    """
    基础版 Dataset：
    - 初始化时保存生成器对象；
    - 每次 __getitem__ 从生成器中取样并滑窗生成样本；
    - 不随机、不多进程；
    - 内存安全（不一次性加载所有数据）。
    """
    def __init__(self, json_path, radar_type="V05", input_len=12, pred_len=20, max_samples=None):
        self.gen = yield_aligned_from_json(json_path, radar_type=radar_type)
        self.input_len = input_len
        self.pred_len = pred_len
        self.buffer = []  # 暂存滑窗样本
        self.exhausted = False
        self.max_samples = max_samples

    def _refill_buffer(self):
        """从生成器中取一个 key 并生成所有滑窗"""
        try:
            key, wind, radar = next(self.gen)
        except StopIteration:
            self.exhausted = True
            return
        T = radar.shape[0]
        if T < self.input_len + self.pred_len:
            return
        for t in range(0, T - self.input_len - self.pred_len + 1):
            x = radar[t:t+self.input_len].astype(np.float32)
            y = wind[t+self.input_len:t+self.input_len+self.pred_len].astype(np.float32)
            self.buffer.append((x, y))

    def __getitem__(self, idx):
        # 当 buffer 为空时，从生成器补充数据
        while not self.buffer and not self.exhausted:
            self._refill_buffer()
        if not self.buffer:
            raise IndexError("Dataset exhausted")
        return self.buffer.pop(0)

    def __len__(self):
        return self.max_samples or 10000  # 虚值，用于 DataLoader 限制迭代次数

class FullSequenceWindRadarDataset(Dataset):
    """
    🚀 纯加载 Dataset
    - 训练时配合 collate_fn 或模型内部进行窗口化
    - 支持多进程并行 + 缓存
    """
    def __init__(self, json_path, radar_type="V05", cache_size=32):
        with open(json_path, "r") as f:
            idx = json.load(f)
        self.wind_dict = idx["wind"]
        self.radar_dict = idx[radar_type]
        self.keys = [k for k in self.wind_dict.keys() if k in self.radar_dict]

        self._load_key_data = lru_cache(maxsize=cache_size)(self._load_key_data_uncached)

    def __len__(self):
        return len(self.keys)

    def _load_key_data_uncached(self, key):
        wind_files = self.wind_dict[key]
        radar_files = self.radar_dict[key]
        wind_data, wind_times = load_npy_files(wind_files)
        radar_data, radar_times = load_npy_files(radar_files)

        start = min(wind_times.min(), radar_times.min()).floor("6min")
        end   = max(wind_times.max(), radar_times.max()).ceil("6min")
        grid  = pd.date_range(start, end, freq="6min")

        da_wind  = xr.DataArray(wind_data, coords={"time": wind_times}, dims=["time", "y", "x"])
        da_radar = xr.DataArray(radar_data, coords={"time": radar_times}, dims=["time", "y", "x"])

        wind_aligned  = da_wind.reindex(time=grid).values.astype(np.float32)
        radar_aligned = da_radar.reindex(time=grid).values.astype(np.float32)

        return torch.from_numpy(radar_aligned), torch.from_numpy(wind_aligned)

    def __getitem__(self, idx):
        key = self.keys[idx]
        radar, wind = self._load_key_data(key)
        return key, radar, wind

