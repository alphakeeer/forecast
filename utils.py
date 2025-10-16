import numpy as np, math

# 缺测处理
def mask_invalid(x, invalid_values):
    mask = np.ones_like(x, dtype=bool)
    for iv in invalid_values:
        mask &= (x != iv)
    mask &= ~np.isnan(x)
    return mask

def fill_invalid(x, fill_value=0.0, invalid_values=None):
    x = x.copy()
    if invalid_values is None:
        invalid_values = []
    m = mask_invalid(x, invalid_values)
    x[~m] = fill_value
    return x, m

# 指标
def mae(y_true, y_pred, mask=None):
    if mask is None:
        mask = np.ones_like(y_true, dtype=bool)
    diff = np.abs(y_true - y_pred)
    return diff[mask].mean() if mask.any() else np.nan

def rmse(y_true, y_pred, mask=None):
    if mask is None:
        mask = np.ones_like(y_true, dtype=bool)
    diff2 = (y_true - y_pred) ** 2
    return math.sqrt(diff2[mask].mean()) if mask.any() else np.nan

def event_scores(y_true, y_pred, thr=17.0, mask=None):
    if mask is None:
        mask = np.ones_like(y_true, dtype=bool)
    yt = y_true[mask] >= thr
    yp = y_pred[mask] >= thr
    TP = np.logical_and(yt, yp).sum()
    FP = np.logical_and(~yt, yp).sum()
    FN = np.logical_and(yt, ~yp).sum()
    precision = TP / (TP + FP + 1e-9)
    recall    = TP / (TP + FN + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    csi       = TP / (TP + FP + FN + 1e-9)
    return dict(precision=precision, recall=recall, f1=f1, csi=csi)