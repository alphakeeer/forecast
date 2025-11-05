import numpy as np
from torch.utils.data import Dataset
from dataloading.datayield  import yield_aligned_from_json

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
