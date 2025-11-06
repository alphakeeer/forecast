import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from xgboost import XGBRegressor
import joblib
from dataloading.dataset import SimpleWindRadarDataset
import os
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
from utils.logger_config import setup_logger
from tqdm.contrib.logging import logging_redirect_tqdm

logger = setup_logger("output/log/train_xgboost2.log")

# 超参数
TRAIN_JSON_PATH = "/home/dataset-assist-0/data/data_index_flat_train.json"
EVAL_JSON_PATH = "/home/dataset-assist-0/data/data_index_flat_eval.json"
RADAR_TYPE = ["V05", "CR"]
INPUT_LEN = 12
PRED_LEN = 20
OUT_DIR = "output/models/xgb_baseline"


def train_xgb_baseline(json_path, radar_type="V05", input_len=12, pred_len=20, out_dir="xgb_baseline", eval_json_path=None):
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"🚀 启动 XGBoost Baseline 训练")
    logger.info(f"数据路径: {json_path}")
    logger.info(f"雷达类型: {radar_type} | 输入长度: {input_len} | 预测长度: {pred_len}")
    logger.info(f"模型输出目录: {out_dir}")

    dataset = SimpleWindRadarDataset(
        json_path, radar_type, input_len, pred_len)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0)

    models = []

    # 外层：预测步（horizon）循环
    for h in tqdm(range(pred_len), desc="🌈 Horizon Progress", position=0):
        logger.info(f"\n=== 训练 Horizon H{h+1}/{pred_len} ===")

        model = XGBRegressor(
            n_estimators=1000,
            max_depth=10,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            n_jobs=8,
        )

        X_list, y_list = [], []

        # 内层：样本加载循环
        for x_seq, y_seq in tqdm(dataloader, total=200, desc=f"H{h+1:02d} 数据加载", leave=False):
            x_np = x_seq.squeeze(0).numpy()    # (L,H,W)
            if x_np.ndim == 4:
                # 多通道: (L,C,H,W) → (L*C,H,W)
                L, C, H, W = x_np.shape
                x_np = x_np.reshape(L*C, H, W)
            else:
                # 单通道: (L,H,W)
                L, H, W = x_np.shape

            y_np = y_seq.squeeze(0)[h].numpy()  # (H,W)
            X_pix = x_np.reshape(-1, H * W).T   # (H*W, L*C)
            y_pix = y_np.reshape(-1)

            mask = (~np.isnan(X_pix).any(axis=1)) & (~np.isnan(y_pix))
            if mask.any():
                X_list.append(X_pix[mask])
                y_list.append(y_pix[mask])

            if len(X_list) >= 1000:  # 控制训练样本数
                break

        if not X_list:
            logger.warning(f"[跳过] H{h+1}: 无有效数据")
            continue

        X_all = np.concatenate(X_list)
        y_all = np.concatenate(y_list)

        logger.info(f"开始训练 H{h+1} 模型: 样本数={len(X_all):,}")
        model.fit(X_all, y_all)
        tag = "_".join([str(rt).lower() for rt in radar_type]) if isinstance(
            radar_type, (list, tuple)) else str(radar_type).lower()
        path = os.path.join(out_dir, f"xgb_{tag}_h{h+1:02d}.pkl")

        joblib.dump(model, path)

        models.append(model)
        logger.info(f"[✅ 保存完成] {path}")

    logger.info("🎯 Baseline 全流程训练完成！")

    if eval_json_path:
        logger.info("✅ Baseline training done. 开始评估模型性能...")
        evaluate_xgb_models(eval_json_path, models,
                            radar_type, input_len, pred_len)

    return models


def evaluate_xgb_models(json_path, models, radar_type="V05", input_len=12, pred_len=20, max_batches=100):
    """使用加载好的模型对数据进行评估"""
    logger.info("🔍 启动模型评估流程")
    logger.info(f"数据路径: {json_path}")
    logger.info(f"评估帧数: {pred_len}, 输入长度: {input_len}")

    dataset = SimpleWindRadarDataset(
        json_path, radar_type, input_len, pred_len)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0)

    mae_list, rmse_list, r2_list = [], [], []

    for h in tqdm(range(pred_len), desc="📊 Eval Horizons"):
        model = models[h] if h < len(models) else None
        if model is None:
            continue

        y_true_all, y_pred_all = [], []

        for x_seq, y_seq in tqdm(dataloader, total=max_batches, desc=f"H{h+1:02d} 预测中", leave=False):
            x_np = x_seq.squeeze(0).numpy()    # (L,C,H,W) 或 (L,H,W)

            # --- 支持多通道展开（与训练阶段一致） ---
            if x_np.ndim == 4:
                L, C, H, W = x_np.shape
                x_np = x_np.reshape(L * C, H, W)  # 合并通道到“序列维”
            else:
                L, H, W = x_np.shape

            y_np = y_seq.squeeze(0)[h].numpy()   # (H,W)

            # --- 拉平成像素点特征（与训练阶段一致） ---
            X_pix = x_np.reshape(-1, H * W).T    # (H*W, L*C)
            y_pix = y_np.reshape(-1)

            mask = (~np.isnan(X_pix).any(axis=1)) & (~np.isnan(y_pix))
            if not mask.any():
                continue

            X_valid = X_pix[mask]
            y_valid = y_pix[mask]
            y_pred = model.predict(X_valid)

            y_true_all.append(y_valid)
            y_pred_all.append(y_pred)

            if len(y_true_all) >= 5:  # 控制样本量，避免评估太慢
                break

        if not y_true_all:
            logger.warning(f"[跳过] H{h+1}: 无有效样本")
            continue

        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        mae_list.append(mae)
        rmse_list.append(rmse)
        r2_list.append(r2)

        logger.info(f"H{h+1:02d}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

    if mae_list:
        logger.info("🎯 平均评估结果：")
        logger.info(f"  MAE  = {np.mean(mae_list):.4f}")
        logger.info(f"  RMSE = {np.mean(rmse_list):.4f}")
        logger.info(f"  R²   = {np.mean(r2_list):.4f}")
    else:
        logger.warning("⚠️ 无评估结果，可能模型或数据为空。")

    logger.info("✅ 评估完成。")
    return {
        "mae_list": mae_list,
        "rmse_list": rmse_list,
        "r2_list": r2_list,
    }


def predict_future_wind(models, radar_window):
    """
    radar_window: (L, H, W) 或 (L, C, H, W)
    返回: (T_out, H, W)
    """
    arr = radar_window
    if arr.ndim == 3:
        L, H, W = arr.shape
        C = 1
        arr = arr.reshape(L, 1, H, W)
    else:
        L, C, H, W = arr.shape
    X = arr.reshape(L * C, H * W).T   # (H*W, L*C)
    outs = []
    for m in models:
        y_pred = m.predict(X)
        outs.append(y_pred.reshape(H, W))
    return np.stack(outs, axis=0)


if __name__ == "__main__":
    # 训练模型
    with logging_redirect_tqdm():
        trained_models = train_xgb_baseline(
            TRAIN_JSON_PATH,
            radar_type=RADAR_TYPE,
            input_len=INPUT_LEN,
            pred_len=PRED_LEN,
            out_dir=OUT_DIR,
            eval_json_path=EVAL_JSON_PATH
        )
