# -*- coding: utf-8 -*-
"""
训练过程全记录：S03 / S09 被试内 + 9人混合跨被试
==================================================
重新训练三个模型（参数与 profile_10 完全一致），在训练过程中系统性采集：
  ① 训练曲线    : 每迭代 train/val loss、train/val acc、每迭代 test acc、每类 test acc
  ② 参数演化    : 各模块可学习参数范数 + SPDNorm 的 gamma/beta 随训练变化
  ③ 特征演化    : 每快照点验证集逐层特征（范数/SPD log-欧氏均值矩阵）
  ④ 注意力演化  : 注意力概率矩阵对角线集中度 + 均值矩阵
  ⑤ 可分性演化  : 切空间特征 silhouette（类间可分性）
  ⑥ 梯度诊断    : 每迭代各层梯度范数
  ⑦ 测试过程    : 每快照点 test acc / 每类 acc / 混淆矩阵

全部保存到 results/profile_11/pair_training_*，不覆盖原 checkpoint。
（profile_11 = 恢复全连接 conv2 + 保留 3 处等价优化 + 移除 log_trace 的新基线）
"""

import sys, json, time, pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from mAtt.optimizer import MixOptimizer
from mAtt.mAtt import (_PerSubjectConv2d, _PerSubjectConv2dTemporal,
                       BACKEND_PREFIXES)
from utils.functions import build_matt_model
from analyze_cross_pipeline import (
    FeatureRecorder, batch_to_vectors, LAYER_ORDER, LAYER_DISP,
    WIN_SAMPLES, STEP_SAMPLES, DEVICE,
    load_preprocessed_dataset, split_train_valid_test, le_mean,
)

print(f'Device: {DEVICE}')

# ============ 命令行参数：空间滤波器个数 / 输出目录 ============
import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument('--n_spatial_filters', type=int, default=22,
                     help='conv1 输出空间滤波器个数（22=基线，10/15=压缩版）')
_parser.add_argument('--out_dir', type=str, default='profile_11', help='输出目录名')
_parser.add_argument('--tag', type=str, default='trainpair', help='checkpoint 文件名前缀')
_parser.add_argument('--channel_gaussian', action='store_true',
                     help='通道加权改用固定 2D 高斯掩膜（C3/Cz/C4 中心，不参与训练）')
_parser.add_argument('--ortho_lambda', type=float, default=0.0,
                     help='Conv1 正交性正则系数 λ（0=关闭），L_reg=λ/2·mean(||WWᵀ-I||²)')
_parser.add_argument('--wd', type=float, default=0.02,
                     help='weight_decay（对 Adam 直接加进梯度；默认=0.02。0.5 已被证实为跨被试准确率塌缩主因）')
_parser.add_argument('--seed', type=int, default=None,
                     help='随机种子（固定 run 可复现；None=不固定，用于多种子统计）')
_parser.add_argument('--conv2_multiscale', action='store_true',
                     help='将 Conv2 替换为多尺度并联时间滤波器组(1×8/1×16/1×32/空洞1×16)')
_parser.add_argument('--conv2_ms_per_branch', type=int, default=None,
                     help='多尺度每分支输出通道数（None=均分到 d_attn_in；4=扩容后投影回）')
_parser.add_argument('--conv2_ms_specs', type=str, default=None,
                     help='多尺度核长组合 e.g. "8_1,16_1,24_1,24_2"（核长_空洞，逗号分隔）')
_parser.add_argument('--attn_change', type=str, default=None,
                     help='变化感知模式: None=原始匹配 | ref=显式静息参考(A) | diff=时间差分打分(B)')
_parser.add_argument('--attn_prior', type=str, default=None,
                     help='协议先验(逗号分隔,长度=窗口数): 在softmax前对key窗加偏置,'
                          '提升想象段注意力。e.g. "0,0,0.6,0.6,0.6,0.6,0.6,0.6" (win0静息低, win2-7想象高)')
_parser.add_argument('--manifold_change', type=str, default=None,
                     help='SPD 流形"相对变化"特征: None=恒等(绝对SPD) | le_win0=相对窗0静息 | '
                          'le_diff1=一阶时间差分 | le_diff2=二阶时间差分')
_parser.add_argument('--channel_prior_eps', type=float, default=0.0,
                     help='通道权重"先验锚定微调": >0 时每个优化步后把 scale 钳制到 '
                          '先验±eps·先验 的邻域内（仅非 channel_gaussian 模式生效），'
                          '验证"生理先验 + 受限学习"能否保住通道结构')
_parser.add_argument('--per_subject_conv1', type=int, default=0,
                     help='conv1 改为逐被试独立空间滤波器组(>1=被试数)。梯度隔离,'
                          '检验"空间滤波逐被试应对通道差异"能否解 conv1 塌缩;'
                          '仅作用于 9 人混合分支,被试内仍走共享路径')
_parser.add_argument('--per_subject_conv2', type=int, default=0,
                     help='conv2 改为逐被试独立时序滤波器组(>1=被试数)。与 '
                          '--per_subject_conv1 共用时检验"前端(空间+频率)全 per-subject '
                          '能否逼近被试内水平"（判定瓶颈在前端共享还是后端可分性）')
_parser.add_argument('--dataset', type=str, default='bci2a', choices=['bci2a', 'physionetmi'],
                     help='数据集: bci2a=BCI-IV-2a(9人,250Hz) | physionetmi=PhysionetMI'
                          '(103人,160Hz)。决定 pkl 路径与窗口采样数换算')
_parser.add_argument('--save_backend', type=str, default=None,
                     help='训练完成后把"后端统一模型"权重(=E2R之后 att2/ract2/tangent/flat/'
                          'dropout/linear/spdnorm/manifold_change 可学习层)单独保存到该路径,'
                          '供以后只改 conv 前端时复用;后端不依赖前端具体实现,只依赖'
                          'c2_out(d_attn_in)/d_attn_out/M 维度')
_parser.add_argument('--load_backend', type=str, default=None,
                     help='构建模型后只加载后端权重(前端 conv/IN/channel_weight 保持随机),'
                          '用于"换前端张量、复用后端"场景。'
                          '注意: 加载的后端必须与当前 d_attn_in/d_attn_out/M 维度一致')
_parser.add_argument('--freeze_backend', action='store_true',
                     help='与 --load_backend 并用:加载后端后冻结全部后端参数'
                          '(requires_grad=False),训练时只更新前端的 conv/IN/channel_weight,'
                          '实现"改进 conv 前端时后端完全不动、无需重训"。')
_parser.add_argument('--iterations', type=int, default=150,
                     help='训练迭代数(历史全部实验=150,已证伪方向复现勿改;只有做耗时权衡时降)')
_parser.add_argument('--batch_size', type=int, default=64,
                     help='batch size(历史=64。103人全量混合需 ≤10min 时用 128:步数减半)')
_parser.add_argument('--snap_every', type=int, default=10,
                     help='每 N 迭代拍一次诊断快照(特征/silhouette/注意力);频率越高曲线越密但越慢')
_parser.add_argument('--sil_max', type=int, default=2000,
                     help='silhouette 子采样上限样本数(N²d 复杂度,103人验证集>2000 时分层抽样到该值)')
_parser.add_argument('--grad_probe', action='store_true',
                     help='开启跨被试梯度冲突探针:每个快照点用固定探针批算各被试梯度'
                          '两两 cos 与对齐比 |mean g|/mean|g|,并额外算 subject-silhouette')
_parser.add_argument('--probe_per_subject', type=int, default=48,
                     help='梯度探针每被试抽取的试次数(探针批全程固定,不参与训练)')
_parser.add_argument('--skip_within', action='store_true',
                     help='跳过 S03/S09 被试内两个模型(纯 cross_mix 诊断用,省 2/3 时间)')
_parser.add_argument('--norm', type=str, default='in', choices=['in', 'bn'],
                     help='归一化层: in=InstanceNorm(历史默认,无 running stats) | '
                          'bn=BatchNorm(有 running stats,可用于目标批统计量重算)')
_parser.add_argument('--bn_recalib', action='store_true',
                     help='(需 --norm bn)逐被试重算 BN running stats(用目标被试的无标签'
                          '试次,不改任何权重),再在该被试上评估 —— 即"BN trick"的可部署版本')
_parser.add_argument('--max_subjects', type=int, default=0,
                     help='只用前 N 个被试(0=全部),用于 N-标度律实验')
_args = _parser.parse_args()
N_SPATIAL_FILTERS = _args.n_spatial_filters
TAG = _args.tag
CHANNEL_GAUSSIAN = _args.channel_gaussian
ORTHO_LAMBDA = _args.ortho_lambda
WD = _args.wd
CONV2_MULTISCALE = _args.conv2_multiscale
CONV2_MS_PER_BRANCH = _args.conv2_ms_per_branch
CONV2_MS_SPECS = (tuple((int(k), int(d)) for k, d in
                        (s.split('_') for s in _args.conv2_ms_specs.split(',')))
                  if _args.conv2_ms_specs else None)
ATTN_CHANGE = _args.attn_change
ATTN_PRIOR = (tuple(float(v) for v in _args.attn_prior.split(','))
              if _args.attn_prior else None)
if ATTN_PRIOR is not None and len(ATTN_PRIOR) < 8:
    print(f'[WARN] attn_prior 长度 {len(ATTN_PRIOR)} < 窗口数, 末窗可能缺值')
MANIFOLD_CHANGE = _args.manifold_change
CHANNEL_PRIOR_EPS = _args.channel_prior_eps
PER_SUBJECT_CONV1 = _args.per_subject_conv1
PER_SUBJECT_CONV2 = _args.per_subject_conv2
DATASET_NAME = _args.dataset
SAVE_BACKEND_PATH = _args.save_backend
LOAD_BACKEND_PATH = _args.load_backend
FREEZE_BACKEND = _args.freeze_backend
ITERATIONS = _args.iterations
BS = _args.batch_size
SNAP_EVERY = _args.snap_every
SIL_MAX = _args.sil_max
GRAD_PROBE = _args.grad_probe
PROBE_PER_SUBJECT = _args.probe_per_subject
SKIP_WITHIN = _args.skip_within
MAX_SUBJECTS = _args.max_subjects
NORM_TYPE = _args.norm
BN_RECALIB = _args.bn_recalib
if FREEZE_BACKEND and not LOAD_BACKEND_PATH:
    print('[WARN] --freeze_backend 未搭配 --load_backend,无后端可冻结,已忽略')

OUT_DIR = BASE / 'results' / _args.out_dir
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = _args.seed
if SEED is not None:
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
print(f'[配置] n_spatial_filters={N_SPATIAL_FILTERS} channel_gaussian={CHANNEL_GAUSSIAN} '
      f'ortho_lambda={ORTHO_LAMBDA} conv2_multiscale={CONV2_MULTISCALE} '
      f'per_branch={CONV2_MS_PER_BRANCH} specs={CONV2_MS_SPECS}, '
      f'输出目录: {OUT_DIR}')

# 数据集相关：pkl 路径 / 采样率 / 窗口采样数（bci2a=250Hz, physionetmi=160Hz，
# 窗长秒数与 bci2a 一致 1.5s/0.75s，只换算采样点）
if DATASET_NAME == 'physionetmi':
    DATA_PATH = str(BASE.parent / 'saved_data' / 'preprocessed_data_moabb_raw_ica.pkl')
    FS = 160
else:
    DATA_PATH = str(BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl')
    FS = 250
WIN_SAMPLES = int(round(1.5 * FS))    # 窗长 1.5s
STEP_SAMPLES = int(round(0.75 * FS))  # 步长 0.75s（50% 重叠）
print(f'[数据集] {DATASET_NAME}: fs={FS}Hz, win={WIN_SAMPLES}点({WIN_SAMPLES/FS:.2f}s), '
      f'step={STEP_SAMPLES}点({STEP_SAMPLES/FS:.2f}s)')
SAMPLE_RATIO = 0.5
N_CLASSES = 4
D_ATTN_IN, D_ATTN_OUT = 8, 8
PAIR_SUBJECTS = [3, 9]

ITERATIONS = _args.iterations
LR = 5e-4
GRAD_CLIP = 1.0
LOCAL_RADIUS = 6      # 全连接注意力（M=4 时覆盖全部窗口）；保留原注意力机制的完整交互
DROP_PROB = 0.3       # dropout（验证过的折中值：0.5 偏重，0.3 平衡效果与正则）
TEST_EVERY = 5

# 参数分组（用于参数演化）
PARAM_GROUPS = [
    ('channel_weight.scale', '通道加权scale'),
    ('conv1.weight', 'Conv1核'),
    ('In1.', 'InstNorm1'),
    ('conv2.weight', 'Conv2核'),
    ('conv2.bias', 'Conv2偏置'),
    ('In2.', 'InstNorm2'),
    ('spdnorm.gamma', 'SPDNorm.gamma'),
    ('spdnorm.beta', 'SPDNorm.beta'),
    ('att2.', '注意力模块'),
    ('linear.weight', '分类器W'),
    ('linear.bias', '分类器b'),
]


def load_data():
    subject_data, _ = load_preprocessed_dataset(DATA_PATH)
    if MAX_SUBJECTS > 0:
        subject_data = subject_data[:MAX_SUBJECTS]
    T = subject_data[0][0].shape[-1]
    M = (T - WIN_SAMPLES) // STEP_SAMPLES + 1
    from sklearn.model_selection import train_test_split
    splits = {}
    for sid in PAIR_SUBJECTS:
        if sid > len(subject_data):      # --max_subjects 截断后跳过不存在的被试
            continue
        X, y = subject_data[sid - 1]
        _, X, _, y = train_test_split(X, y, test_size=SAMPLE_RATIO, stratify=y, random_state=42)
        X_tr, y_tr, X_va, y_va, X_te, y_te = split_train_valid_test(X, y)
        splits[sid] = (X_tr, y_tr, X_va, y_va, X_te, y_te)
    X_all = np.concatenate([subject_data[s - 1][0] for s in range(1, len(subject_data) + 1)], axis=0)
    y_all = np.concatenate([subject_data[s - 1][1] for s in range(1, len(subject_data) + 1)], axis=0)
    sub_all = np.concatenate([np.full(len(subject_data[s - 1][1]), s - 1, dtype=np.int64)
                              for s in range(1, len(subject_data) + 1)], axis=0)
    # 混合训练/验证（全部被试拼接 0.6/0.2/0.2，train 用前两者）；
    # others=sub_all 让试次归属随 X/y 同步切分（per-subject conv1 路由用）
    cross = split_train_valid_test(X_all, y_all, others=[sub_all])
    return subject_data, T, M, splits, cross


def make_loaders(X_tr, y_tr, X_va, y_va, X_te, y_te, bs=BS,
                 s_tr=None, s_va=None, s_te=None):
    def tl(X, y, shuffle, s=None):
        if s is not None:
            ds = torch.utils.data.TensorDataset(
                torch.Tensor(X).unsqueeze(1), torch.Tensor(y).long(), torch.Tensor(s).long())
        else:
            ds = torch.utils.data.TensorDataset(
                torch.Tensor(X).unsqueeze(1), torch.Tensor(y).long())
        # pin_memory：GPU 训练时锁页缓冲，配合 non_blocking 异步传输提速；
        # 数据已驻留内存(TensorDataset)，num_workers 仅选 2 避免多余进程开销。
        return torch.utils.data.DataLoader(
            ds, batch_size=bs, shuffle=shuffle,
            pin_memory=True,
            num_workers=0 if DEVICE.type == 'cpu' else 2,
            persistent_workers=True if DEVICE.type == 'cuda' else False)
    return (tl(X_tr, y_tr, True, s_tr), tl(X_va, y_va, False, s_va),
            tl(X_te, y_te, False, s_te))


def _match_param(key, name):
    """参数分组匹配。普通前缀 startswith；conv2 需跨多尺度分支聚合
    （conv2.weight/conv2.bias 会漏掉 conv2.convs.N.weight/bias → 改成通配 *）。"""
    if key in ('conv2.weight', 'conv2.bias'):
        suffix = '.'.join(key.split('.')[1:])          # weight / bias
        return (name == key) or (name.startswith('conv2.') and name.endswith('.' + suffix))
    return name.startswith(key)


def param_snapshot(net):
    """记录各模块参数范数 + SPDNorm 仿射"""
    snap = {}
    for key, disp in PARAM_GROUPS:
        if key == 'channel_weight.scale':
            # scale 可能是 Parameter（可学习）或 buffer（高斯固定），统一直接读取
            s = net.channel_weight.scale.detach()
            snap[disp] = float(s.norm().item()) if s.numel() > 0 else 0.0
            continue
        tot = 0.0
        for name, p in net.named_parameters():
            if _match_param(key, name):
                tot += float(p.detach().norm().item() ** 2)
        snap[disp] = float(np.sqrt(tot)) if tot > 0 else 0.0
    spd = net.spdnorm
    snap['SPDNorm_gamma'] = float(spd.gamma.item())
    snap['SPDNorm_beta'] = float(spd.beta.item())
    snap['channel_scale'] = [float(v) for v in net.channel_weight.scale.detach().cpu().tolist()]
    return snap


def grad_snapshot(net):
    """记录各模块梯度范数"""
    snap = {}
    for key, disp in PARAM_GROUPS:
        tot = 0.0
        for name, p in net.named_parameters():
            if _match_param(key, name) and p.grad is not None:
                tot += float(p.grad.norm().item() ** 2)
        snap[disp] = float(np.sqrt(tot)) if tot > 0 else 0.0
    return snap


def _build_probe_batches(tr_ld, per_subject, n_subjects):
    """固定一批"梯度探针"试次：每个被试 per_subject 个，全程不变。

    只多走一遍训练 loader（不参与训练、不改梯度），使不同迭代的 cos 可比。
    loader 必须携带被试归属（第 3 列），否则返回 None。
    """
    buckets = {}
    for batch in tr_ld:
        if len(batch) <= 2:
            return None
        x, y, s = batch[0], batch[1].numpy(), batch[2].numpy()
        for i in range(len(y)):
            b = buckets.setdefault(int(s[i]), [[], []])
            if len(b[0]) < per_subject:
                b[0].append(x[i:i + 1])
                b[1].append(int(y[i]))
        if len(buckets) >= n_subjects and all(len(v[0]) >= per_subject
                                             for v in buckets.values()):
            break
    return {sid: (torch.cat(xs).to(DEVICE), torch.tensor(ys, device=DEVICE))
            for sid, (xs, ys) in buckets.items()}


# 梯度冲突探针的参数分组：前端 conv1/conv2 与"共享后端"分开看
_PROBE_GROUPS = {
    'conv1': lambda n: n.startswith('conv1'),
    'conv2': lambda n: n.startswith('conv2'),
    'backend': lambda n: n.startswith(BACKEND_PREFIXES),
    'all': lambda n: True,
}


def grad_conflict_probe(net, probe):
    """量化跨被试梯度冲突（"期望梯度是否被互斥方向抵消"的直接测量）。

    对被试 s 在固定探针批上算 g_s = ∇_θ L_s，定义
      非对角 cos 均值 μ_cos：≈0 = 方向近似正交(纯冲突)，=1 = 完全同向；
      对齐比 |mean_s g_s| / mean_s |g_s|：K 个正交梯度时 ≈1/√K，完全同向 =1。
    在 eval 模式取梯度（关 dropout），测期望梯度方向而非单步噪声；
    用 autograd.grad 而非 backward，不污染 .grad，训练不受影响。
    """
    was_training = net.training
    net.eval()
    named = [(n, p) for n, p in net.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    # 每个参数张量在展平梯度里的列区间：只做一次 backward 再按分组切列
    # （若每个分组各 backward 一次，计算图在第一次后即被释放会直接报错）
    sizes = [p.numel() for p in params]
    offs = np.concatenate([[0], np.cumsum(sizes)])

    vecs = []
    for sid in sorted(probe):
        xb, yb = probe[sid]
        sb = (torch.full((len(yb),), sid, dtype=torch.long, device=DEVICE)
              if (PER_SUBJECT_CONV1 > 0 or PER_SUBJECT_CONV2 > 0) else None)
        loss = nn.CrossEntropyLoss()(net(xb, sb) if sb is not None else net(xb), yb)
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        vecs.append(torch.cat([
            (g if g is not None else torch.zeros_like(p)).reshape(-1)
            for g, p in zip(grads, params)]).detach())
    G_all = torch.stack(vecs)                                  # (K, P_total)

    out = {}
    for gname, pred in _PROBE_GROUPS.items():
        sel = [i for i, (n, _) in enumerate(named) if pred(n)]
        if not sel:
            continue
        cols = torch.as_tensor(
            np.concatenate([np.arange(int(offs[i]), int(offs[i + 1])) for i in sel]),
            device=G_all.device)
        G = G_all.index_select(1, cols)
        Gn = G / (G.norm(dim=1, keepdim=True) + 1e-12)
        C = Gn @ Gn.T
        k = C.shape[0]
        C0 = C.clone()
        C0.fill_diagonal_(0.0)
        off_diag = C0[C0 != 0]
        row_mean = C0.sum(1) / max(k - 1, 1)
        g = {
            'subjects': sorted(probe),
            'mean_cos': float(C0.sum() / max(k * (k - 1), 1)),
            'std_cos': float(off_diag.std()) if k > 1 else 0.0,
            'min_cos': float(off_diag.min()) if k > 1 else 0.0,
            'max_cos': float(off_diag.max()) if k > 1 else 0.0,
            'align_ratio': float(G.mean(0).norm() / (G.norm(dim=1).mean() + 1e-12)),
            'grad_norm_mean': float(G.norm(dim=1).mean()),
            'cos_row_mean': row_mean.cpu().numpy().tolist(),
        }
        if k <= 24:            # 103 人时不落全矩阵,避免 JSON 爆炸
            g['cos_matrix'] = C.cpu().numpy().tolist()
        out[gname] = g
    net.zero_grad(set_to_none=True)
    if was_training:
        net.train()
    return out


def feature_snapshot(net, loader, M):
    """验证集逐层特征统计 + 注意力 + 切空间 silhouette"""
    recorder = FeatureRecorder(net)
    raw = {}
    att_probs = []
    feats_tan = []
    labels = []
    sub_ids = []
    net.eval()
    with torch.no_grad():
        for batch in loader:
            xb = batch[0].to(DEVICE)
            sb = batch[2].to(DEVICE) if len(batch) > 2 and (PER_SUBJECT_CONV1 > 0 or PER_SUBJECT_CONV2 > 0) else None
            recorder.clear()
            net(xb, sb) if sb is not None else net(xb)
            labels.append(batch[1].numpy())
            if len(batch) > 2:
                sub_ids.append(batch[2].numpy())
            for name, _, _ in LAYER_ORDER:
                if name in recorder.features:
                    raw.setdefault(name, []).append(recorder.features[name].float())
            spd_norm = recorder.features.get('spdnorm')
            if spd_norm is not None:
                att_probs.append(net.att2.attention_prob(spd_norm.to(DEVICE)).cpu().numpy())
            tan = recorder.features.get('tangent.vec')
            if tan is not None:
                b = tan.shape[0] // M
                feats_tan.append(tan.reshape(b, M * tan.shape[1]).numpy())
    labels = np.concatenate(labels)
    subs = np.concatenate(sub_ids) if sub_ids else None
    recorder.remove()
    att_mean = np.mean(np.concatenate(att_probs), axis=0) if att_probs else None

    stats = {}
    for name, _, _ in LAYER_ORDER:
        if name not in raw:
            continue
        t = torch.cat(raw[name], dim=0).float()
        vecs, mats = batch_to_vectors(name, t, M)
        stats[name] = {
            'norm_mean': float(np.linalg.norm(vecs, axis=1).mean()),
            'centroid_norm': float(np.linalg.norm(vecs.mean(0))),
        }
        if mats is not None:
            stats[name]['logeu_mean'] = le_mean(mats)

    sil = None
    sil_sub = None
    if feats_tan and len(np.unique(labels)) >= 2:
        from sklearn.metrics import silhouette_score
        try:
            # silhouette 是 O(N²·d):全量混合验证集 N≈9270、d≈1265 时单次 ~1.1e11 flop,
            # 是快照最大开销。>2000 样本时按类分层子采样到 2000,估算值误差 <1pp,
            # 不影响"分离度≈0 / 高于被试内"这类量级判断。
            feats_t = np.concatenate(feats_tan)
            idx = None
            if len(feats_t) > SIL_MAX:
                uniq, counts = np.unique(labels, return_counts=True)
                rng = np.random.RandomState(42)
                idx = np.concatenate([
                    rng.choice(np.where(labels == u)[0], size=min(c, SIL_MAX // len(uniq)),
                               replace=False) for u, c in zip(uniq, counts)
                ])
                feats_t, labels_s = feats_t[idx], labels[idx]
            else:
                labels_s = labels
            sil = float(silhouette_score(feats_t, labels_s))
            # 同一份特征换被试标签:s_sub 远高于 s_class 即说明特征主要编码
            # "身份"而非"类别"(身份记忆假说的直接量纲)
            if subs is not None and len(np.unique(subs)) >= 2:
                sil_sub = float(silhouette_score(
                    feats_t, subs if idx is None else subs[idx]))
        except Exception:
            sil = None
    return stats, att_mean, sil, sil_sub


def accuracy(net, loader):
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            xb, yb = batch[0].to(DEVICE), batch[1].to(DEVICE)
            sb = batch[2].to(DEVICE) if len(batch) > 2 and (PER_SUBJECT_CONV1 > 0 or PER_SUBJECT_CONV2 > 0) else None
            out = net(xb, sb) if sb is not None else net(xb)
            correct += (out.argmax(1) == yb).sum().item()
            total += len(yb)
    return correct / max(total, 1)


def accuracy_per_subject_recalib(net, loader):
    """BN trick 的逐被试版本(零标签适配)。

    对测试集里的每个被试:把 BN 的 running stats 重置回训练态,再用**该被试自己的
    试次**(不需要标签)在 train() 模式下前向一遍,让 BN 统计量适配到他身上,然后评估。
    全程不改任何权重;结束后恢复全局统计量。这正是文献里"在目标数据上重算归一化
    统计量"的可部署形式 —— 部署时拿到新被试的一批试次即可,无需重训。
    """
    if not any(isinstance(m, nn.BatchNorm2d) for m in net.modules()):
        return float('nan')
    groups = {}
    for batch in loader:
        xb, yb, sb = batch[0], batch[1], batch[2]
        for s in torch.unique(sb):
            m = (sb == s)
            groups.setdefault(int(s), []).append(
                (xb[m].to(DEVICE), yb[m].to(DEVICE)))
    bns = [m for m in net.modules() if isinstance(m, nn.BatchNorm2d)]
    saved = [(b.running_mean.clone(), b.running_var.clone(),
              b.num_batches_tracked.clone()) for b in bns]
    correct = total = 0
    try:
        for s, chunks in groups.items():
            for b, (rm, rv, nt) in zip(bns, saved):
                b.running_mean.copy_(rm); b.running_var.copy_(rv)
                b.num_batches_tracked.copy_(nt)
            net.train()                       # 让 BN 用该被试的数据更新 running stats
            with torch.no_grad():
                for xb, yb in chunks:
                    net(xb)
            net.eval()
            with torch.no_grad():
                for xb, yb in chunks:
                    correct += (net(xb).argmax(1) == yb).sum().item()
                    total += len(yb)
    finally:
        for b, (rm, rv, nt) in zip(bns, saved):
            b.running_mean.copy_(rm); b.running_var.copy_(rv)
            b.num_batches_tracked.copy_(nt)
        net.eval()
    return correct / max(total, 1)


def per_class_acc_conf(net, loader, n_classes=N_CLASSES):
    net.eval()
    conf = np.zeros((n_classes, n_classes), dtype=int)
    with torch.no_grad():
        for batch in loader:
            xb, yb = batch[0].to(DEVICE), batch[1].to(DEVICE)
            sb = batch[2].to(DEVICE) if len(batch) > 2 and (PER_SUBJECT_CONV1 > 0 or PER_SUBJECT_CONV2 > 0) else None
            pred = (net(xb, sb) if sb is not None else net(xb)).argmax(1).cpu().numpy()
            for i, j in zip(yb.cpu().numpy(), pred):
                conf[i, j] += 1
    acc = conf.diagonal() / conf.sum(axis=1).clip(1)
    return acc.tolist(), conf.tolist()


def conv1_ortho_penalty(net, lam):
    """Conv1 滤波器正交性正则：L_reg = λ/2 · ||W Wᵀ − I||_F²

    W 为 (n_sp, 22)（每个空间滤波器是一个 22 维通道向量），WWᵀ 是 n_sp×n_sp，
    强制滤波器两两正交且单位范数（满秩、不缩水）。注意：正交约束是 WWᵀ=I 而非
    WᵀW=I —— 后者是 22×22 且秩 ≤ n_sp < 22，数学上无法等于单位阵。
    用 Frobenius 范数平方（各元素平方和，与用户公式一致）；λ=0.02 时初始
    惩罚约 0.13（≈ CE 的 10%），足以抵抗跨被试的滤波器塌缩又不压制分类。
    """
    if lam <= 0:
        return 0.0
    if isinstance(net.conv1, _PerSubjectConv2d):
        # per-subject：对各组分别正则（正交约束定义在各组内部）
        w = net.conv1.weight   # (N, c1_out, 1, 22, 1)
        w = w.reshape(w.shape[0], w.shape[1], -1)
        g = w @ w.transpose(-1, -2)          # (N, c1_out, c1_out)
        eye = torch.eye(g.shape[-1], device=g.device)
        tot = 0.0
        for s in range(w.shape[0]):
            gs = g[s]
            tot = tot + ((gs - eye) ** 2).sum()
        return 0.5 * lam * tot
    w = net.conv1.weight
    w = w.view(w.shape[0], -1)               # (n_sp, 22)
    g = w @ w.t()                            # (n_sp, n_sp)
    eye = torch.eye(g.shape[0], device=g.device)
    return 0.5 * lam * ((g - eye) ** 2).sum()


def _infer_windows(net, tr_ld):
    """推断真实窗口数 m（卷积会改变时序长度，硬编码 M 可能与实际不符）

    取一个 mini-batch 前向到 att2，经 shape 返回获取实际窗口数。
    仅用于记录/切片语义（feature_snapshot），与 LazyLinear 的分类器输入维解耦。
    """
    net.eval()
    with torch.no_grad():
        for batch in tr_ld:
            probe_b = batch[0].to(DEVICE)
            break
        x = net.channel_weight(probe_b) if net.channel_weight is not None else probe_b
        # conv1: 共享或 per-subject（conv1 为 (22,1) 空间核，不改变时序长度，
        # 窗口数与 per-subject 路由无关；per-subject 用第 0 组即可探测）
        x = net.conv1(x, torch.zeros(x.shape[0], dtype=torch.long, device=DEVICE)) \
            if isinstance(net.conv1, _PerSubjectConv2d) else net.conv1(x)
        x = net.In1(x)
        # conv2: 同样探测；per-subject 用第 0 组（时序长度仅由核长/填充决定）
        x = net.conv2(x, torch.zeros(x.shape[0], dtype=torch.long, device=DEVICE)) \
            if isinstance(net.conv2, _PerSubjectConv2dTemporal) else net.conv2(x)
        x = net.In2(x)
        spd = net.ract1(x); spd = net.spdnorm(spd)
        _, shape = net.att2(spd)
        return int(shape[1])


# '后端统一模型'：E2R 之后所有可学习层（注意力+切空间+分类头+流形对齐）。
# 边界前缀集合由 mAtt.py 的 BACKEND_PREFIXES 统一定义（勿在此重复维护，
# 否则改一处会与模型层失联）。
# 前端(conv1/In1/conv2/In2/channel_weight)换成任何同输出通道数实现后，
# 此后端权重可直接复用 —— 后端输入是 SPD 协方差，不关心前端卷积如何产生。


def extract_backend_state(net):
    """从模型 state_dict 抽出后端权重（后端前缀下所有 tensor，含 Buffer——
    如 tangent 的索引掩码 spdnorm 的 running stats，这些 eval 时必须一致）。"""
    backend = {}
    for k, v in net.state_dict().items():
        if k.startswith(BACKEND_PREFIXES) and hasattr(v, 'dtype'):
            backend[k] = v
    return backend


def load_backend_state(net, path, freeze=False):
    """把后端权重加载进模型（strict=False 只匹配存在的键）；返回加载失败报告。

    freeze=True 时在加载后把全部后端参数 requires_grad=False
    （优化器已用 filter(requires_grad) 构建,冻结参数不进优化器,
    训练只更新前端 conv/IN/channel_weight → “换前端、后端不动”）。
    """
    try:
        ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    except Exception:
        # filters: 老式 pickle（含 Buffer/UninitializedParameter）回退明文加载
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    if not isinstance(ckpt, dict) or not any(k.startswith(BACKEND_PREFIXES) for k in ckpt):
        raise ValueError(f'{path} 不是后端权重文件（应含 att2.*/linear.* 等键）')
    missing, unexpected = net.load_state_dict(ckpt, strict=False)
    miss_bk = [k for k in missing if k.startswith(BACKEND_PREFIXES)]
    if miss_bk:
        print(f'[WARN] load_backend: 后端键缺失/维度不符 {len(miss_bk)} 个: {miss_bk[:5]}...')
    if unexpected:
        print(f'[WARN] load_backend: 多余键 {len(unexpected)} 个（忽略）')
    if freeze:
        n_frozen = 0
        for k, p in net.named_parameters():
            if k.startswith(BACKEND_PREFIXES):
                p.requires_grad = False
                n_frozen += 1
        print(f'[load_backend] 冻结后端参数 {n_frozen} 个（只训练前端 conv/IN/channel_weight）')
    print(f'[load_backend] {path} → 加载后端权重（前端保持随机）')
    return miss_bk, unexpected


def train_with_record(tag, net, tr_ld, va_ld, te_ld, M, ckpt_prefix,
                      ortho_lambda=ORTHO_LAMBDA, channel_prior=None,
                      channel_prior_eps=0.0, save_backend_on_best=False, probe=None):
    """训练一个模型并全程记录过程数据。save_backend_on_best=True 时在最优
    val_acc 处把后端统一模型权重存到 SAVE_BACKEND_PATH（仅主模型用）。"""
    if channel_prior_eps > 0 and channel_prior is not None:
        print(f'   [通道先验锚定] eps={channel_prior_eps}, '
              f'scale 每步钳制在 先验±{channel_prior_eps}·先验')
    print(f'\n===== 训练 {tag} =====')
    t0 = time.time()
    m_real = _infer_windows(net, tr_ld)
    if m_real and m_real != M:
        print(f'   [窗口数修正] M={M} → 实际 m={m_real}（卷积后时序长度变化导致）')
        M = m_real
    CE = nn.CrossEntropyLoss()
    base_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=LR, weight_decay=WD)
    opt = MixOptimizer(base_opt)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(base_opt, T_max=ITERATIONS)

    rec = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
           'test_acc': [], 'lr': [], 'grad': [], 'params': [], 'features': [],
           'attn_conc': [], 'silhouette': [], 'test_iter': [], 'snap_iter': [],
           'per_class_test': [], 'confusion': [],
           'silhouette_subject': [], 'grad_conflict': [], 'test_acc_recalib': []}
    best_loss, best_acc = float('inf'), 0.0

    for ite in range(ITERATIONS):
        net.train()
        loss_sum = acc_tr = n_tr = nb = 0
        for batch in tr_ld:
            xb, yb = batch[0].to(DEVICE), batch[1].to(DEVICE)
            sb = batch[2].to(DEVICE) if len(batch) > 2 and (PER_SUBJECT_CONV1 > 0 or PER_SUBJECT_CONV2 > 0) else None
            n_tr += len(yb)
            out = net(xb, sb) if sb is not None else net(xb)
            loss = CE(out, yb) + conv1_ortho_penalty(net, ortho_lambda)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            opt.step()
            if channel_prior_eps > 0 and channel_prior is not None:
                # 通道权重先验锚定：Δ=scale−prior 被钳制在 ±eps·prior 邻域，
                # 允许受限学习（而不被互斥投票推离生理先验/推向原点）
                with torch.no_grad():
                    s = net.channel_weight.scale
                    p = channel_prior.to(s.device)
                    s.copy_(p + torch.clamp(s - p, -channel_prior_eps * p,
                                            channel_prior_eps * p))
            loss_sum += loss.item()
            acc_tr += (out.argmax(1) == yb).sum().item()
            nb += 1
        rec['grad'].append(grad_snapshot(net))

        net.eval()
        vloss = vacc = vn = 0.0
        with torch.no_grad():
            for batch in va_ld:
                xb, yb = batch[0].to(DEVICE), batch[1].to(DEVICE)
                sb = batch[2].to(DEVICE) if len(batch) > 2 and (PER_SUBJECT_CONV1 > 0 or PER_SUBJECT_CONV2 > 0) else None
                vn += len(yb)
                out = net(xb, sb) if sb is not None else net(xb)
                vloss += CE(out, yb).item()
                vacc += (out.argmax(1) == yb).sum().item()
        rec['train_loss'].append(loss_sum / max(nb, 1))
        rec['val_loss'].append(vloss / max(vn, 1))
        rec['train_acc'].append(acc_tr / max(n_tr, 1))
        rec['val_acc'].append(vacc / max(vn, 1))
        rec['lr'].append(base_opt.param_groups[0]['lr'])
        sched.step()

        # 快照：特征/参数/注意力/可分性 + test 过程
        if ite % SNAP_EVERY == 0 or ite == ITERATIONS - 1:
            rec['snap_iter'].append(ite)
            stats, att_mean, sil, sil_sub = feature_snapshot(net, va_ld, M)
            rec['features'].append(stats)
            rec['params'].append(param_snapshot(net))
            if att_mean is not None:
                rec['attn_conc'].append(float(np.trace(att_mean) / att_mean.sum()))
                rec['attn_mat'] = rec.get('attn_mat', []) + [att_mean]
            else:
                rec['attn_conc'].append(float('nan'))
            rec['silhouette'].append(sil)
            rec['silhouette_subject'].append(sil_sub)
            if probe is not None:
                gc = grad_conflict_probe(net, probe)
                rec['grad_conflict'].append(gc)
                print('   [梯度冲突] ' + ' | '.join(
                    f'{k}: cos={v["mean_cos"]:+.3f} align={v["align_ratio"]:.3f}'
                    for k, v in gc.items() if k != 'all'))
        if ite % TEST_EVERY == 0 or ite == ITERATIONS - 1:
            rec['test_iter'].append(ite)
            rec['test_acc'].append(accuracy(net, te_ld))
            if BN_RECALIB:
                rec['test_acc_recalib'].append(
                    accuracy_per_subject_recalib(net, te_ld))
            pca, conf = per_class_acc_conf(net, te_ld)
            rec['per_class_test'].append(pca)
            rec['confusion'].append(conf)

        # 保存最优
        if rec['val_loss'][-1] < best_loss:
            best_loss = rec['val_loss'][-1]
            torch.save(net.state_dict(), OUT_DIR / f'{ckpt_prefix}_best_loss.pt')
        if rec['val_acc'][-1] > best_acc:
            best_acc = rec['val_acc'][-1]
            torch.save(net.state_dict(), OUT_DIR / f'{ckpt_prefix}_best_acc.pt')
            if SAVE_BACKEND_PATH and save_backend_on_best:
                bk_path = Path(SAVE_BACKEND_PATH)
                bk_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(extract_backend_state(net), bk_path)
                print(f'   [save_backend] 已保存后端统一模型 → {bk_path}')

        if ite % 10 == 0 or ite == ITERATIONS - 1:
            print(f'  ite {ite:02d}: trL={rec["train_loss"][-1]:.3f} vaL={rec["val_loss"][-1]:.3f} '
                  f'trA={rec["train_acc"][-1]*100:.1f}% vaA={rec["val_acc"][-1]*100:.1f}%')

    print(f'  [{tag}] 完成, 耗时 {time.time()-t0:.0f}s, best_val_loss={best_loss:.3f} best_val_acc={best_acc*100:.1f}%')
    return rec


def main():
    print('=' * 70)
    print(f'训练过程全记录（{DATASET_NAME}）：S03/S09 被试内 + 全量混合跨被试')
    print('=' * 70)
    subject_data, T, M, splits, cross_split = load_data()
    print(f'T={T}, n_windows M={M}, 混合被试数={len(subject_data)}')

    results = {'config': {'subjects': PAIR_SUBJECTS, 'iterations': ITERATIONS,
                          'lr': LR, 'wd': WD, 'n_windows': M}, 'models': {}}

    def _make_net(p1=PER_SUBJECT_CONV1, p2=PER_SUBJECT_CONV2):
        """统一经 build_matt_model 构建(参数齐全,dropout 走构造传入)

        p1/p2: 该模型 conv1/conv2 是否逐被试独立（被试内=0 共享，混合=对应开关）
        """
        return build_matt_model(WIN_SAMPLES, STEP_SAMPLES, {
            'n_classes': N_CLASSES,
            'd_attn_in': D_ATTN_IN,
            'd_attn_out': D_ATTN_OUT,
            'local_radius': LOCAL_RADIUS,
            'anchor_stride': None,
            'n_heads': 1,
            'n_windows': None,
            'conv2_separable': False,
            'conv2_multiscale': CONV2_MULTISCALE,
            'conv2_ms_per_branch': CONV2_MS_PER_BRANCH,
            'conv2_ms_specs': CONV2_MS_SPECS,
            'n_spatial_filters': N_SPATIAL_FILTERS,
            'channel_gaussian': CHANNEL_GAUSSIAN,
            'gaussian_sigma': 0.62,
            'attn_change': ATTN_CHANGE,
            'attn_prior': ATTN_PRIOR,
            'manifold_change': MANIFOLD_CHANGE,
            'dropout': DROP_PROB,
            'per_subject_conv1': p1,
            'per_subject_conv2': p2,
            'norm_type': NORM_TYPE,
        })

    # 被试内（--skip_within 时跳过，纯 cross_mix 诊断用）
    for sid in ([] if SKIP_WITHIN else PAIR_SUBJECTS):
        X_tr, y_tr, X_va, y_va, X_te, y_te = splits[sid]
        tr_ld, va_ld, te_ld = make_loaders(X_tr, y_tr, X_va, y_va, X_te, y_te)
        net = _make_net(0, 0)   # 被试内：共享 conv1/conv2，无塌缩且无需 per-subject 路由
        if LOAD_BACKEND_PATH:
            load_backend_state(net, LOAD_BACKEND_PATH, freeze=FREEZE_BACKEND)
        _prior = (net.channel_weight.scale.detach().clone() if
                  (CHANNEL_PRIOR_EPS > 0 and not CHANNEL_GAUSSIAN) else None)
        rec = train_with_record(f'S{sid:02d} 被试内', net, tr_ld, va_ld, te_ld, M,
                                f'{TAG}_within_S{sid}',
                                channel_prior=_prior, channel_prior_eps=CHANNEL_PRIOR_EPS)
        results['models'][f'within_S{sid}'] = {'tag': f'S{sid} 被试内', **rec}
        del net

    # 9人混合
    # cross_split 现在含 sub_id（6 元 + sub 的 train/valid/test 3 份）
    X_tr, y_tr, X_va, y_va, X_te, y_te, s_tr, s_va, s_te = cross_split
    # 混合分支一律携带被试归属：per-subject conv 用它路由，梯度探针与
    # subject-silhouette 也用它；共享 conv 时训练循环仍取 sb=None
    # （判据是 PER_SUBJECT_* 开关），数值与旧行为等价。
    tr_ld, va_ld, te_ld = make_loaders(X_tr, y_tr, X_va, y_va, X_te, y_te,
                                       s_tr=s_tr, s_va=s_va, s_te=s_te)
    net = _make_net()
    if LOAD_BACKEND_PATH:
        load_backend_state(net, LOAD_BACKEND_PATH, freeze=FREEZE_BACKEND)
    _prior = (net.channel_weight.scale.detach().clone() if
              (CHANNEL_PRIOR_EPS > 0 and not CHANNEL_GAUSSIAN) else None)
    _probe = None
    if GRAD_PROBE:
        _probe = _build_probe_batches(tr_ld, PROBE_PER_SUBJECT, len(subject_data))
        print(f'   [梯度探针] 固定 {len(_probe) if _probe else 0} 个被试的探针批, '
              f'每被试 {PROBE_PER_SUBJECT} 试次')
    rec = train_with_record('全量混合跨被试', net, tr_ld, va_ld, te_ld, M,
                            f'{TAG}_cross_mix',
                            channel_prior=_prior, channel_prior_eps=CHANNEL_PRIOR_EPS,
                            save_backend_on_best=bool(SAVE_BACKEND_PATH),
                            probe=_probe)

    # per-subject 身份捷径检验：把测试集中每个试次的路由 sub_id 整体错配
    # （+1 mod N），看 acc 是否断崖 —— 若断崖说明 W_s 只编码被试身份而非
    # 空间特征；若几乎不变说明学到的是共享可迁移的空间投影
    if (PER_SUBJECT_CONV1 > 0 or PER_SUBJECT_CONV2 > 0) and \
        (isinstance(net.conv1, _PerSubjectConv2d) or isinstance(net.conv2, _PerSubjectConv2dTemporal)):
        perm_correct = 0
        perm_wrong = 0
        n_te = 0
        net.eval()
        with torch.no_grad():
            for batch in te_ld:
                xb, yb = batch[0].to(DEVICE), batch[1].to(DEVICE)
                sb = batch[2].to(DEVICE)
                sb_wrong = (sb + 1) % PER_SUBJECT_CONV1
                out_c = net(xb, sb)
                out_w = net(xb, sb_wrong)
                perm_correct += (out_c.argmax(1) == yb).sum().item()
                perm_wrong += (out_w.argmax(1) == yb).sum().item()
                n_te += len(yb)
        print(f'   [身份置换] 正确路由 acc={(perm_correct/n_te)*100:.2f}% | '
              f'错配(+1 mod N) acc={(perm_wrong/n_te)*100:.2f}% | Δ={((perm_correct-perm_wrong)/n_te)*100:.2f}pp')
        rec['identity_perm_test'] = {
            'correct_route_acc': perm_correct / n_te,
            'wrong_route_acc': perm_wrong / n_te,
            'acc_delta': (perm_correct - perm_wrong) / n_te,
        }
    results['models']['cross_mix'] = {'tag': '全量混合跨被试', **rec}

    # 保存
    pkl_path = OUT_DIR / 'pair_training_process.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    json_path = OUT_DIR / 'pair_training.json'
    # 特征快照含 numpy 矩阵，转轻量版
    slim = json_slim(results)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(slim, f, indent=2, ensure_ascii=False, default=float)
    print(f'\n[OK] 完整训练过程数据: {pkl_path} ({pkl_path.stat().st_size/1e6:.1f} MB)')
    print(f'[OK] 结构化数据: {json_path}')
    make_plots(results)
    format_report(results)


def json_slim(results):
    slim = {'config': results['config'], 'models': {}}
    for k, r in results['models'].items():
        sr = {kk: vv for kk, vv in r.items() if kk not in ('features', 'attn_mat')}
        # 保留每层范数轨迹（去掉大矩阵）
        if 'features' in r:
            sr['features_norm_traj'] = [
                {name: st['norm_mean'] for name, st in fs.items()} for fs in r['features']]
        slim['models'][k] = sr
    return slim


def make_plots(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    model_keys = list(results['models'].keys())
    iters = list(range(ITERATIONS))

    # 图1: 训练曲线（loss + acc + test acc）
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    for k in model_keys:
        r = results['models'][k]
        axes[0].plot(iters, r['train_loss'], label=f'{r["tag"]} train')
        axes[0].plot(iters, r['val_loss'], '--', label=f'{r["tag"]} val')
    axes[0].set_title('训练/验证 loss 曲线'); axes[0].set_xlabel('iteration'); axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    for k in model_keys:
        r = results['models'][k]
        axes[1].plot(iters, np.array(r['train_acc']) * 100, label=f'{r["tag"]} train')
        axes[1].plot(iters, np.array(r['val_acc']) * 100, '--', label=f'{r["tag"]} val')
        if r.get('test_acc'):
            axes[1].plot(r['test_iter'], np.array(r['test_acc']) * 100, 'o-',
                         label=f'{r["tag"]} test', alpha=0.8)
    axes[1].set_title('准确率曲线'); axes[1].set_xlabel('iteration'); axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT_DIR / 'pair_training_curves.png', dpi=130)
    plt.close(fig)

    # 图2: 参数演化
    fig, axes = plt.subplots(len(model_keys), 1, figsize=(13, 3.2 * len(model_keys)), squeeze=False)
    disp_keys = [d for _, d in PARAM_GROUPS if 'gamma' not in d and 'beta' not in d]
    for ai, k in enumerate(model_keys):
        r = results['models'][k]
        snaps = r['params']
        xi = r['snap_iter']
        for dk in disp_keys:
            vals = [s[dk] for s in snaps]
            axes[ai][0].plot(xi, vals, label=dk, lw=1.2)
        axes[ai][0].set_title(f'{r["tag"]} · 模块参数范数演化'); axes[ai][0].set_xlabel('iteration')
        axes[ai][0].legend(fontsize=6); axes[ai][0].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT_DIR / 'pair_training_param_evo.png', dpi=130)
    plt.close(fig)

    # 图3: 特征演化（关键层 norm）
    key_layers = ['conv1', 'conv2', 'spdnorm', 'tangent.vec', 'linear']
    disp = {n: LAYER_DISP[n] for n, _, _ in LAYER_ORDER}
    fig, axes = plt.subplots(len(model_keys), 1, figsize=(13, 3.2 * len(model_keys)), squeeze=False)
    for ai, k in enumerate(model_keys):
        r = results['models'][k]
        xi = r['snap_iter']
        for name in key_layers:
            vals = [fs.get(name, {}).get('norm_mean', np.nan) for fs in r['features']]
            axes[ai][0].plot(xi, vals, label=disp.get(name, name), lw=1.2)
        axes[ai][0].set_title(f'{r["tag"]} · 验证集逐层特征范数演化'); axes[ai][0].set_xlabel('iteration')
        axes[ai][0].legend(fontsize=6); axes[ai][0].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT_DIR / 'pair_training_feature_evo.png', dpi=130)
    plt.close(fig)

    # 图4: 注意力集中度 + silhouette + 梯度
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for k in model_keys:
        r = results['models'][k]
        axes[0].plot(r['snap_iter'], r['attn_conc'], 'o-', label=r['tag'])
        axes[1].plot(r['snap_iter'], r['silhouette'], 's-', label=r['tag'])
        gavg = [np.mean(list(g.values())) if g else np.nan for g in r['grad']]
        axes[2].plot(iters, gavg, label=r['tag'])
    axes[0].set_title('注意力对角线集中度演化'); axes[0].set_xlabel('iteration'); axes[0].legend(fontsize=7)
    axes[1].set_title('切空间特征 silhouette 演化'); axes[1].set_xlabel('iteration'); axes[1].legend(fontsize=7)
    axes[2].set_title('平均梯度范数（对数）'); axes[2].set_xlabel('iteration'); axes[2].set_yscale('log')
    axes[2].legend(fontsize=7)
    for a in axes: a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT_DIR / 'pair_training_attn_sil_grad.png', dpi=130)
    plt.close(fig)
    print('[OK] 图已保存: pair_training_curves/param_evo/feature_evo/attn_sil_grad.png')


def format_report(results):
    L = []
    L.append('=' * 80)
    L.append(f'训练过程全记录报告（S03/S09 被试内 + 全量混合跨被试）')
    L.append('=' * 80)
    for k, r in results['models'].items():
        L.append(f'\n{"-"*80}\n[{k}] {r["tag"]}')
        L.append(f'初始/最终 train_loss: {r["train_loss"][0]:.3f} → {r["train_loss"][-1]:.3f}')
        L.append(f'初始/最终 val_loss  : {r["val_loss"][0]:.3f} → {r["val_loss"][-1]:.3f}')
        L.append(f'初始/最终 train_acc  : {r["train_acc"][0]*100:.1f}% → {r["train_acc"][-1]*100:.1f}%')
        L.append(f'初始/最终 val_acc    : {r["val_acc"][0]*100:.1f}% → {r["val_acc"][-1]*100:.1f}%')
        if r.get('test_acc'):
            L.append(f'test_acc 轨迹: ' + ', '.join(
                f'it{ti} {ta*100:.1f}%' for ti, ta in zip(r['test_iter'], r['test_acc'])))
        if r.get('silhouette'):
            L.append(f'silhouette 轨迹: ' + ', '.join(
                f'it{si} {s:.3f}' if s is not None else 'it{si} -'
                for si, s in zip(r['snap_iter'], r['silhouette'])))
        if r.get('attn_conc'):
            L.append(f'注意力对角线集中度轨迹: ' + ', '.join(
                f'it{si} {a:.3f}' for si, a in zip(r['snap_iter'], r['attn_conc'])))
        if r.get('params'):
            first = r['params'][0]; last = r['params'][-1]
            L.append('模块参数范数 (初始→最终):')
            for dk in first:
                if isinstance(first[dk], (int, float)):
                    L.append(f'  {dk:<22s} {first[dk]:.4f} → {last[dk]:.4f}')
    txt_path = OUT_DIR / 'pair_training_report.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    print(f'[OK] 文字报告: {txt_path}')
    print('\n' + '\n'.join(L))


if __name__ == '__main__':
    main()
