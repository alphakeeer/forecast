import json, os, numpy as np, pandas as pd, xarray as xr, torch
from functools import lru_cache
import time

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
    - 支持多个 radar_type
    - 训练时配合 collate_fn 或模型内部进行窗口化
    - 支持多进程并行 + 缓存
    """
    def __init__(self, json_path, radar_type=["V05","CR"], cache_size=32):
        with open(json_path, "r") as f:
            idx = json.load(f)
        self.wind_dict = idx["wind"]

        # 检查雷达类型是否都存在
        self.radar_types = radar_type
        for rt in self.radar_types:
            if rt not in idx:
                raise ValueError(f"Radar type '{rt}' not found in dataset JSON keys: {list(idx.keys())}")

        # 每种雷达类型的索引
        self.radar_dicts = {rt: idx[rt] for rt in self.radar_types}

        # 仅保留同时存在于 wind 和所有 radar 类型的 key
        self.keys = [
            k for k in self.wind_dict.keys()
            if all(k in self.radar_dicts[rt] for rt in self.radar_types)
        ]

        # 缓存机制
        self._load_key_data = lru_cache(maxsize=cache_size)(self._load_key_data_uncached)

    def __len__(self):
        return len(self.keys)

    def _load_key_data_uncached(self, key):
        wind_files = self.wind_dict[key]
        wind_data, wind_times = load_npy_files(wind_files)

        radar_datas = []
        radar_times_all = []

        # 加载每种雷达数据
        for rt in self.radar_types:
            radar_files = self.radar_dicts[rt][key]
            radar_data, radar_times = load_npy_files(radar_files)
            radar_datas.append((radar_data, radar_times))
            radar_times_all.append(radar_times)

        # 时间对齐范围
        start = min(wind_times.min(), *(rt.min() for _, rt in radar_datas)).floor("6min")
        end   = max(wind_times.max(), *(rt.max() for _, rt in radar_datas)).ceil("6min")
        grid  = pd.date_range(start, end, freq="6min")

        # 对齐风场
        da_wind = xr.DataArray(wind_data, coords={"time": wind_times}, dims=["time", "y", "x"])
        wind_aligned = da_wind.reindex(time=grid).values.astype(np.float32)

        # 对齐多个雷达
        aligned_radars = []
        for radar_data, radar_times in radar_datas:
            da_radar = xr.DataArray(radar_data, coords={"time": radar_times}, dims=["time", "y", "x"])
            aligned = da_radar.reindex(time=grid).values.astype(np.float32)
            aligned_radars.append(aligned)

        # 堆叠多个雷达类型 => shape: [num_types, time, y, x]
        radar_stacked = np.stack(aligned_radars, axis=0)

        return torch.from_numpy(radar_stacked), torch.from_numpy(wind_aligned)

    def __getitem__(self, idx):
        key = self.keys[idx]
        radar, wind = self._load_key_data(key)
        return key, radar, wind
    
def test_dataset_shapes_and_times():
    json_path = "/home/dataset-assist-0/data/data_index_flat_train.json"
    radar_types = ['CR', 'R05', 'RG1', 'RG2', 'V05', 'V15', 'VIL']# 你要测试的雷达类型

    dataset = FullSequenceWindRadarDataset(json_path, radar_type=radar_types, cache_size=4)

    print(f"✅ Total samples: {len(dataset)}")
    if len(dataset) == 0:
        print("❌ Dataset is empty, please check your json or keys.")
        return

    # 随机抽几个样本进行测试
    for i in range(min(3, len(dataset))):
        start_time = time.time()
        key, radar, wind = dataset[i]
        print(f"\n🌀 Sample {i} — key: {key}")

        # 打印基本形状信息
        print(f"Radar shape: {radar.shape}")  # [num_types, time, y, x]
        print(f"Wind  shape: {wind.shape}")   # [time, y, x]

        e_time = time.time() - start_time
        print(f"⏱️  Load time: {e_time:.3f} seconds")

if __name__ == "__main__":
    test_dataset_shapes_and_times()