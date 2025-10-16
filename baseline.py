import os, glob, numpy as np, json
from utils import fill_invalid, mask_invalid, mae, rmse, event_scores

# ========== 配置 ==========
TRAIN_ROOT = "dataset/train"   # 训练数据根目录（有 label）
TESTA_ROOT = "dataset/testa"   # A榜目录（无 label）
VAL_RATIO  = 0.1               # 10% 过程做验证
WIND_THR   = 17.0              # 大风事件阈值（可改）

CR_NAME, VIL_NAME, LABEL_NAME = "CR.npy", "VIL.npy", "label.npy"
CR_IVALS, VIL_IVALS, LAB_IVALS = [-32768, -1280], [-32768, -1280], [-9]

# ========== 工具 ==========
def list_process_folders(root):
    procs = sorted(glob.glob(os.path.join(root, "*")))
    return [p for p in procs if os.path.isdir(p)]

def list_time_folders(proc_dir):
    times = sorted(glob.glob(os.path.join(proc_dir, "*")))
    return [t for t in times if os.path.isdir(t)]

def load_triplet(time_dir, need_label=True):
    cr   = np.load(os.path.join(time_dir, CR_NAME))
    vil  = np.load(os.path.join(time_dir, VIL_NAME))
    cr, mcr = fill_invalid(cr, 0.0, CR_IVALS)
    vil, mvl = fill_invalid(vil, 0.0, VIL_IVALS)
    if need_label:
        lab  = np.load(os.path.join(time_dir, LABEL_NAME))
        mlab = mask_invalid(lab, LAB_IVALS)
        mask = mcr & mvl & mlab
        return cr, vil, lab, mask
    else:
        mask = mcr & mvl
        return cr, vil, None, mask

def predict_linear(cr, vil, a, b, cr0, vil0, clip_min=0.0, clip_max=45.0):
    pred = a * np.maximum(cr - cr0, 0.0) + b * np.maximum(vil - vil0, 0.0)
    return np.clip(pred, clip_min, clip_max)

# ========== 训练：网格搜索 ==========
def grid_search(train_times, val_times):
    # 粗网格（够用，几分钟可跑完；想细再加点刻度）
    A  = [0.1, 0.2, 0.5, 1.0]
    B  = [0.1, 0.2, 0.5, 1.0]
    C0 = [20, 25, 30, 35]     # CR 阈值（dBZ）
    V0 = [2, 5, 8, 12]        # VIL 阈值（kg/m^2）

    best = None
    # 在训练集合上拟合超参（目标：最小 MAE）
    for a in A:
        for b in B:
            for cr0 in C0:
                for vil0 in V0:
                    maes=[]
                    for td in train_times:
                        cr, vil, lab, m = load_triplet(td, need_label=True)
                        pred = predict_linear(cr, vil, a, b, cr0, vil0)
                        maes.append(mae(lab, pred, m))
                    score = np.nanmean(maes)
                    record = dict(a=a,b=b,cr0=cr0,vil0=vil0,tr_mae=score)
                    if best is None or score < best['tr_mae']:
                        best = record

    # 在验证集报告
    v_mae, v_rmse, v_f1 = [], [], []
    for td in val_times:
        cr, vil, lab, m = load_triplet(td, need_label=True)
        pred = predict_linear(cr, vil, best['a'], best['b'], best['cr0'], best['vil0'])
        v_mae.append(mae(lab, pred, m))
        v_rmse.append(rmse(lab, pred, m))
        v_f1.append(event_scores(lab, pred, WIND_THR, m)['f1'])

    best['val_mae']  = float(np.nanmean(v_mae))
    best['val_rmse'] = float(np.nanmean(v_rmse))
    best['val_f1']   = float(np.nanmean(v_f1))
    return best

# ========== 推理：在 TestA 上生成预测 ==========
def infer_on_testa(params, out_tag="pred_baseline0"):
    procs = list_process_folders(TESTA_ROOT)
    for p in procs:
        times = list_time_folders(p)
        for td in times:
            cr, vil, _, m = load_triplet(td, need_label=False)
            pred = predict_linear(cr, vil, params['a'], params['b'], params['cr0'], params['vil0'])
            out_dir = os.path.join(td, out_tag)
            os.makedirs(out_dir, exist_ok=True)
            np.save(os.path.join(out_dir, "pred.npy"), pred.astype(np.float32))
    print(f"[Done] Wrote predictions to subfolders named '{out_tag}' under TestA time folders.")

def main():
    # —— 切分训练/验证（按“过程”切分，避免泄漏）
    procs = list_process_folders(TRAIN_ROOT)
    rng = np.random.default_rng(42)
    rng.shuffle(procs)
    cut = int(len(procs) * (1 - VAL_RATIO))
    tr_procs, va_procs = procs[:cut], procs[cut:]

    def flat_times(pl):
        tds=[]
        for p in pl:
            tds += list_time_folders(p)
        return tds

    tr_times = flat_times(tr_procs)
    va_times = flat_times(va_procs)

    # —— 网格搜索
    best = grid_search(tr_times, va_times)
    print("[Best Params]")
    print(json.dumps(best, ensure_ascii=False, indent=2))

    # ——（可选）在 TestA 生成预测
    # infer_on_testa(best, out_tag="pred_baseline0")

if __name__ == "__main__":
    main()