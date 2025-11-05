import os
import glob
import json
import logging
from tqdm import tqdm
import random
from utils.logger_config import setup_logger
from tqdm.contrib.logging import logging_redirect_tqdm


def build_flat_data_index(
    base_dir="/home/dataset-assist-1/SevereWeather_AI_2025",
    save_path="data_index_flat.json",
    radar_types=("CR", "R05", "RG1", "RG2", "V05", "V15", "VIL")
):
    """
    构建一个扁平化的 JSON 索引:
    {
      "wind": { "TSW_00_03010941_00095": [path1, path2, ...], ... },
      "CR": { "TSW_00_03010941_00095": [path1, path2, ...], ... },
      ...
    }
    """
    logger = setup_logger()
    logger.info("开始构建扁平数据索引...")
    logger.info(f"基础目录: {base_dir}")
    logger.info(f"保存路径: {save_path}")
    logger.info(f"雷达类型: {radar_types}")

    train_root = os.path.join(base_dir, "TSW", "TrainSet")
    if not os.path.exists(train_root):
        logger.error(f"❌ 训练集目录不存在: {train_root}")
        return None

    index_dict = {"wind": {}}
    for r in radar_types:
        index_dict[r] = {}

    idx_list = sorted(os.listdir(train_root))
    logger.info(f"发现 {len(idx_list)} 个一级目录 (例如 00, 01, 02...)")

    # 外层进度条：一级目录
    for idx in tqdm(idx_list, desc="📂 Scanning main folders", position=0):
        idx_dir = os.path.join(train_root, idx)
        if not os.path.isdir(idx_dir):
            logger.debug(f"跳过非目录: {idx_dir}")
            continue

        subdirs = sorted(os.listdir(idx_dir))
        logger.info(f"[{idx}] 含 {len(subdirs)} 个样本目录")

        # 内层进度条：每个样本目录
        for sd in tqdm(subdirs, desc=f"   ↳ Processing {idx}", position=1, leave=False):
            sub_path = os.path.join(idx_dir, sd)
            if not os.path.isdir(sub_path):
                logger.debug(f"跳过非目录: {sub_path}")
                continue

            key = sd[-15:]  # e.g. TSW_00_03010941_00095
            logger.debug(f"处理样本: {key}")

            # 风场 (LABEL/WA)
            wind_pattern = os.path.join(sub_path, "LABEL", "WA", "*.npy")
            wind_files = sorted(glob.glob(wind_pattern))
            if wind_files:
                index_dict["wind"][key] = wind_files
                logger.debug(f"  🌬️ 风场文件数: {len(wind_files)}")

            # 各类雷达数据
            radar_root = os.path.join(sub_path, "RADAR")
            if os.path.exists(radar_root):
                for rt in radar_types:
                    pattern = os.path.join(radar_root, rt, "*.npy")
                    radar_files = sorted(glob.glob(pattern))
                    if radar_files:
                        index_dict[rt][key] = radar_files
                        logger.debug(f"  📡 {rt} 文件数: {len(radar_files)}")
            else:
                logger.warning(f"未找到 RADAR 目录: {radar_root}")

    # 保存 JSON 文件
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(index_dict, f, indent=2, ensure_ascii=False)

    wind_count = len(index_dict["wind"])
    total_samples = sum(len(v) for v in index_dict.values())

    logger.info(f"✅ 扁平索引已保存到: {save_path}")
    logger.info(f"包含 {wind_count} 个风场样本，总计 {total_samples} 个样本。")
    print(f"\n✅ 索引已完成并保存到: {save_path}")
    print(f"风场样本: {wind_count} | 总样本: {total_samples}")

    return index_dict

def split_data_index(
    json_path,
    out_dir=".",
    train_ratio=0.8,
    eval_ratio=0.1,
    test_ratio=0.1,
    seed=42
):
    """
    将 build_flat_data_index 生成的扁平 JSON 按比例拆分为 train/eval/test 三份
    """
    logger = setup_logger()
    logger.info("📂 开始拆分数据索引...")
    logger.info(f"输入文件: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        full_index = json.load(f)

    # 使用 "wind" 中的 key 作为主样本列表
    all_keys = sorted(full_index.get("wind", {}).keys())
    total = len(all_keys)
    if total == 0:
        logger.error("❌ 没有在索引中找到 'wind' 样本键。")
        return

    random.seed(seed)
    random.shuffle(all_keys)

    n_train = int(total * train_ratio)
    n_eval = int(total * eval_ratio)
    train_keys = set(all_keys[:n_train])
    eval_keys = set(all_keys[n_train:n_train + n_eval])
    test_keys = set(all_keys[n_train + n_eval:])

    logger.info(f"总样本: {total}")
    logger.info(f"训练集: {len(train_keys)} | 验证集: {len(eval_keys)} | 测试集: {len(test_keys)}")

    def filter_subset(keys):
        """根据指定样本键过滤数据"""
        subset = {}
        for radar_type, samples in full_index.items():
            subset[radar_type] = {k: v for k, v in samples.items() if k in keys}
        return subset

    subsets = {
        "train": filter_subset(train_keys),
        "eval": filter_subset(eval_keys),
        "test": filter_subset(test_keys),
    }

    os.makedirs(out_dir, exist_ok=True)

    for name, data in subsets.items():
        save_path = os.path.join(out_dir, f"data_index_flat_{name}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ 已保存 {name} 索引: {save_path} ({sum(len(v) for v in data.values())} 条记录)")

    logger.info("🎯 数据集拆分完成。")
    return subsets

if __name__ == "__main__":
    with logging_redirect_tqdm():
        # json_index = build_flat_data_index(
        #     base_dir="/home/dataset-assist-1/SevereWeather_AI_2025",
        #     save_path="/home/dataset-assist-0/data/wind_radar_files.json"
        # )
        split_data_index(
            json_path="/home/dataset-assist-0/data/wind_radar_files.json",
            out_dir="/home/dataset-assist-0/data",
            train_ratio=0.8,
            eval_ratio=0.1,
            test_ratio=0.1
        )