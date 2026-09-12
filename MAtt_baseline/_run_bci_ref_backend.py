"""一次性探针（跑完即删）:BCI 参考被试后端预训练
1) 计算黎曼几何中心被试（每被试 epoch 平均协方差 trace 归一 →
   Log-Euclidean 两两距离 → 到其他被试距离和最小者）
2) 对参考被试集（S3 + 中心被试）各做被试内训练，在最优 val_acc
   处保存后端（--save_backend 语义,E2R 之后全部张量含 Buffer）
"""
import sys
import numpy as np

base = "/mnt/d/EEG_Analysis/Riemannian_Trajectories/MAtt_baseline"
sys.path.insert(0, base)

import train_pair_with_recording as TP  # noqa: E402
from utils.GetPhysionetMI import load_preprocessed_dataset, split_train_valid_test  # noqa: E402
from utils.functions import build_matt_model  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from scipy.linalg import logm  # noqa: E402

subject_data, _ = load_preprocessed_dataset(TP.DATA_PATH)
N_SUB = len(subject_data)
T = subject_data[0][0].shape[-1]
M = (T - TP.WIN_SAMPLES) // TP.STEP_SAMPLES + 1
print(f"[数据] 被试数={N_SUB}, T={T}, M={M}")


# ---------------- 1) 黎曼几何中心被试 ----------------
def subject_cov(X):
    # X: (n,22,T) → 所有样本时间点外积均值,再 trace 归一
    Xm = X - X.mean(axis=2, keepdims=True)
    C = np.einsum("nct,ndt->ncd", Xm, Xm).sum(axis=0) / max(X.shape[0] * X.shape[2], 1)
    C = 0.5 * (C + C.T) + 1e-6 * np.eye(C.shape[0])  # 数值对称正定化
    return C / (np.trace(C) + 1e-12)


def log_euclid(C1, C2):
    # Log-Euclidean 距离 = ||logm(C1) - logm(C2)||_F（SPD → 对称矩阵线性空间再取 F 范数）
    return float(np.linalg.norm(logm(C1) - logm(C2), ord="fro"))


covs = [subject_cov(subject_data[s][0]) for s in range(N_SUB)]
dist = np.zeros((N_SUB, N_SUB))
for i in range(N_SUB):
    for j in range(i + 1, N_SUB):
        d = log_euclid(covs[i], covs[j])
        dist[i, j] = dist[j, i] = d
sumd = dist.sum(axis=1)
center = int(sumd.argmin())
print(f"[中心] 各被试到其他被试 Log-Euclidean 距离和: {list(np.round(sumd, 3))}")
print(f"[中心] 黎曼几何中心被试 = S{center + 1}")
for sid in (3, center + 1):
    if 1 <= sid <= N_SUB:
        print(f"[参考] 将训练: S{sid}")


# ---------------- 2) 后端预训练 ----------------
def train_ref(sid):
    X, y = subject_data[sid - 1]
    _, X, _, y = train_test_split(X, y, test_size=TP.SAMPLE_RATIO, stratify=y,
                                  random_state=42)
    X_tr, y_tr, X_va, y_va, X_te, y_te = split_train_valid_test(X, y)
    tr_ld, va_ld, te_ld = TP.make_loaders(X_tr, y_tr, X_va, y_va, X_te, y_te, bs=TP.BS)
    out_dir = TP.BASE / "results" / f"ref_backend_s{sid}"
    out_dir.mkdir(parents=True, exist_ok=True)
    TP.OUT_DIR = out_dir
    TP.TAG = f"ref_s{sid}"
    bk_path = str(TP.BASE / "results" / "checkpoint" / f"backend_ref_S{sid}.pt")
    TP.SAVE_BACKEND_PATH = bk_path
    net = build_matt_model(TP.WIN_SAMPLES, TP.STEP_SAMPLES, {
        'n_classes': 4, 'd_attn_in': 8, 'd_attn_out': 8,
        'local_radius': TP.LOCAL_RADIUS, 'anchor_stride': None, 'n_heads': 1,
        'n_windows': None, 'conv2_separable': False, 'conv2_multiscale': False,
        'conv2_ms_per_branch': None, 'conv2_ms_specs': None,
        'n_spatial_filters': 22, 'channel_gaussian': False, 'gaussian_sigma': 0.62,
        'attn_change': None, 'attn_prior': None, 'manifold_change': None,
        'dropout': TP.DROP_PROB, 'per_subject_conv1': 0, 'per_subject_conv2': 0})
    TP.train_with_record(f"S{sid} 参考被试后端", net, tr_ld, va_ld, te_ld, M,
                         f"ref_s{sid}", save_backend_on_best=True)
    print(f"[参考] S{sid} 后端已存 → {bk_path}")


train_ref(3) if N_SUB >= 3 else None
if center + 1 != 3:
    train_ref(center + 1)

print("[OK] 参考后端预训练完成")