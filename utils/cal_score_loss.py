# 具体评分看：https://data.cma.cn/scwds/#/detail
import torch
import torch

# ======================================================
# 表 1：预报时效权重 (共 20 个)
# ======================================================
TIME_WEIGHTS = torch.tensor([
    0.0075, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
    0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.0075, 0.005
])

# ======================================================
# 表 2：风速等级权重 (对流性大风)
# ======================================================


def wind_level_weight(obs):
    """根据观测风速分级权重（表 2）"""
    weight = torch.zeros_like(obs)
    weight = torch.where(obs < 5.5, 0.1, weight)
    weight = torch.where((obs >= 5.5) & (obs < 10.8), 0.1, weight)
    weight = torch.where((obs >= 10.8) & (obs < 13.9), 0.1, weight)
    weight = torch.where((obs >= 13.9) & (obs < 17.2), 0.2, weight)
    weight = torch.where((obs >= 17.2) & (obs < 24.5), 0.2, weight)
    weight = torch.where(obs >= 24.5, 0.3, weight)
    return weight


# ======================================================
# 各项指标计算
# ======================================================
def compute_R(pred, obs, eps=1e-6):
    """计算相关系数 R_k"""
    B, K, H, W = pred.shape
    pred = pred.view(B, K, -1)
    obs = obs.view(B, K, -1)
    pred_mean = pred.mean(dim=-1, keepdim=True)
    obs_mean = obs.mean(dim=-1, keepdim=True)
    num = ((pred - pred_mean) * (obs - obs_mean)).sum(dim=-1)
    den = torch.sqrt(((pred - pred_mean)**2).sum(dim=-1) *
                     ((obs - obs_mean)**2).sum(dim=-1)) + eps
    return num / den


def compute_TS(pred, obs, threshold=10.8, eps=1e-6):
    """计算 Threat Score"""
    pred_event = (pred >= threshold).float()
    obs_event = (obs >= threshold).float()
    TP = (pred_event * obs_event).sum(dim=(-1, -2))
    FP = (pred_event * (1 - obs_event)).sum(dim=(-1, -2))
    FN = ((1 - pred_event) * obs_event).sum(dim=(-1, -2))
    return TP / (TP + FP + FN + eps)


# ======================================================
# 主函数：计算综合得分与分项指标
# ======================================================
def wind_forecast_score_total(pred, obs, eps=1e-6):
    """
    计算最终综合得分 Score_total，并返回分项指标
    pred, obs: (B, 20, 301, 301)
    返回:
      score_total: scalar
      metrics: dict 含 batch 平均的 TS, MAE, R
    """
    B, K, H, W = pred.shape
    assert K == 20, f"应包含 20 个预报时效，但得到 {K}"

    # --- 表 2 等级权重
    W_i = wind_level_weight(obs)  # (B, K, H, W)

    # --- 指标
    TS = compute_TS(pred, obs)                 # (B, K)
    MAE = (pred - obs).abs().mean(dim=(-1, -2))  # (B, K)
    R = compute_R(pred, obs)                  # (B, K)

    # --- 单时效得分 Score_k (公式 3)
    Score_k = torch.sqrt(torch.exp(R - 1)) * (
        W_i.mean(dim=(-1, -2)) * TS * torch.sqrt(torch.exp(-MAE))
    )

    # --- 表 1 加权平均 Score_p
    time_weights = TIME_WEIGHTS.to(pred.device)
    Score_p = (Score_k * time_weights).sum(dim=1)

    # --- 所有样本平均 Score_total
    score_total = Score_p.mean()

    # --- 输出监控指标
    metrics = {
        "TS_mean":  TS.mean().item(),
        "MAE_mean": MAE.mean().item(),
        "R_mean":   R.mean().item(),
        "Score_k_mean": Score_k.mean().item()
    }

    return score_total, metrics


def wind_forecast_loss(pred, obs):
    """训练用 loss (越小越好)，并返回分项指标"""
    score_total, metrics = wind_forecast_score_total(pred, obs)
    loss = 1 - score_total
    return loss, metrics
