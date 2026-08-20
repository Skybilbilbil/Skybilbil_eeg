# -*- coding: utf-8 -*-
"""
训练过程全记录：S03 / S09 被试内 + 9人混合跨被试
==================================================
重新训练三个模型（参数与 profile_10 完全一致），在训练过程中系统性采集：
  ① 训练曲线    : 每迭代 train/val loss、train/val acc、每迭代 test acc、每类 test acc
  ② 参数演化    : 各模块可学习参数范数 + SPDNorm 的 gamma/beta 随训练变化
  ③ 特征演化    : 每快照点验证集逐层特征（范数/log-trace/SPD log-欧氏均值矩阵）
  ④ 注意力演化  : 注意力概率矩阵对角线集中度 + 均值矩阵
  ⑤ 可分性演化  : 切空间特征 silhouette（类间可分性）
  ⑥ 梯度诊断    : 每迭代各层梯度范数
  ⑦ 测试过程    : 每快照点 test acc / 每类 acc / 混淆矩阵

全部保存到 results/profile_10/pair_training_*，不覆盖原 checkpoint。
"""

import os, sys, json, time, pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from mAtt.mAtt import mAtt_bci
from mAtt.optimizer import MixOptimizer
from analyze_cross_pipeline import (
    FeatureRecorder, batch_to_vectors, LAYER_ORDER, LAYER_DISP,
    FS, WIN_SAMPLES, STEP_SAMPLES, DEVICE, OUT_DIR,
    load_preprocessed_dataset, split_train_valid_test, le_mean,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

DATA_PATH = str(BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl')
SAMPLE_RATIO = 0.5
N_CLASSES = 4
D_ATTN_IN, D_ATTN_OUT = 8, 8
PAIR_SUBJECTS = [3, 9]

ITERATIONS = 50
LR, WD, BS = 5e-4, 0.5, 64
GRAD_CLIP = 1.0
SNAP_EVERY = 5          # 每 5 迭代一次快照（特征/参数/注意力/可分性/test）
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
    T = subject_data[0][0].shape[-1]
    M = (T - WIN_SAMPLES) // STEP_SAMPLES + 1
    from sklearn.model_selection import train_test_split
    splits = {}
    for sid in PAIR_SUBJECTS:
        X, y = subject_data[sid - 1]
        _, X, _, y = train_test_split(X, y, test_size=SAMPLE_RATIO, stratify=y, random_state=42)
        X_tr, y_tr, X_va, y_va, X_te, y_te = split_train_valid_test(X, y)
        splits[sid] = (X_tr, y_tr, X_va, y_va, X_te, y_te)
    X_all = np.concatenate([subject_data[s - 1][0] for s in range(1, 10)], axis=0)
    y_all = np.concatenate([subject_data[s - 1][1] for s in range(1, 10)], axis=0)
    _, _, _, _, X9_te, y9_te = split_train_valid_test(X_all, y_all)
    # 混合训练/验证（9人全部数据 0.6/0.2/0.2，train 用前两者）
    X9_tr, y9_tr, X9_va, y9_va, _, _ = split_train_valid_test(X_all, y_all)
    return subject_data, T, M, splits, (X9_tr, y9_tr, X9_va, y9_va, X9_te, y9_te)


def make_loaders(X_tr, y_tr, X_va, y_va, X_te, y_te, bs=BS):
    def tl(X, y, shuffle):
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.Tensor(X).unsqueeze(1), torch.Tensor(y).long()),
            batch_size=bs, shuffle=shuffle)
    return tl(X_tr, y_tr, True), tl(X_va, y_va, False), tl(X_te, y_te, False)


def param_snapshot(net):
    """记录各模块参数范数 + SPDNorm 仿射"""
    snap = {}
    for key, disp in PARAM_GROUPS:
        tot = 0.0
        for name, p in net.named_parameters():
            if name.startswith(key):
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
            if name.startswith(key) and p.grad is not None:
                tot += float(p.grad.norm().item() ** 2)
        snap[disp] = float(np.sqrt(tot)) if tot > 0 else 0.0
    return snap


def feature_snapshot(net, loader, M):
    """验证集逐层特征统计 + 注意力 + 切空间 silhouette"""
    recorder = FeatureRecorder(net)
    raw = {}
    att_probs = []
    feats_tan = []
    labels = []
    net.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            recorder.clear()
            net(xb)
            labels.append(yb.numpy())
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
        if name == 'ract1.signal2spd[1]':
            stats[name]['log_trace_mean'] = float(t.numpy().mean())

    sil = None
    if feats_tan and len(np.unique(labels)) >= 2:
        from sklearn.metrics import silhouette_score
        try:
            sil = float(silhouette_score(np.concatenate(feats_tan), labels))
        except Exception:
            sil = None
    return stats, att_mean, sil


def accuracy(net, loader):
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = net(xb)
            correct += (out.argmax(1) == yb).sum().item()
            total += len(yb)
    return correct / max(total, 1)


def per_class_acc_conf(net, loader, n_classes=N_CLASSES):
    net.eval()
    conf = np.zeros((n_classes, n_classes), dtype=int)
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = net(xb).argmax(1).cpu().numpy()
            for i, j in zip(yb.cpu().numpy(), pred):
                conf[i, j] += 1
    acc = conf.diagonal() / conf.sum(axis=1).clip(1)
    return acc.tolist(), conf.tolist()


def train_with_record(tag, net, tr_ld, va_ld, te_ld, M, ckpt_prefix):
    """训练一个模型并全程记录过程数据"""
    print(f'\n===== 训练 {tag} =====')
    t0 = time.time()
    CE = nn.CrossEntropyLoss()
    base_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=LR, weight_decay=WD)
    opt = MixOptimizer(base_opt)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(base_opt, T_max=ITERATIONS)

    rec = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
           'test_acc': [], 'lr': [], 'grad': [], 'params': [], 'features': [],
           'attn_conc': [], 'silhouette': [], 'test_iter': [], 'snap_iter': [],
           'per_class_test': [], 'confusion': []}
    best_loss, best_acc = float('inf'), 0.0

    for ite in range(ITERATIONS):
        net.train()
        loss_sum = acc_tr = n_tr = nb = 0
        for xb, yb in tr_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            n_tr += len(yb)
            out = net(xb)
            loss = CE(out, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            opt.step()
            loss_sum += loss.item()
            acc_tr += (out.argmax(1) == yb).sum().item()
            nb += 1
        rec['grad'].append(grad_snapshot(net))

        net.eval()
        vloss = vacc = vn = 0.0
        with torch.no_grad():
            for xb, yb in va_ld:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                vn += len(yb)
                out = net(xb)
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
            stats, att_mean, sil = feature_snapshot(net, va_ld, M)
            rec['features'].append(stats)
            rec['params'].append(param_snapshot(net))
            if att_mean is not None:
                rec['attn_conc'].append(float(np.trace(att_mean) / att_mean.sum()))
                rec['attn_mat'] = rec.get('attn_mat', []) + [att_mean]
            else:
                rec['attn_conc'].append(float('nan'))
            rec['silhouette'].append(sil)
        if ite % TEST_EVERY == 0 or ite == ITERATIONS - 1:
            rec['test_iter'].append(ite)
            rec['test_acc'].append(accuracy(net, te_ld))
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

        if ite % 10 == 0 or ite == ITERATIONS - 1:
            print(f'  ite {ite:02d}: trL={rec["train_loss"][-1]:.3f} vaL={rec["val_loss"][-1]:.3f} '
                  f'trA={rec["train_acc"][-1]*100:.1f}% vaA={rec["val_acc"][-1]*100:.1f}%')

    print(f'  [{tag}] 完成, 耗时 {time.time()-t0:.0f}s, best_val_loss={best_loss:.3f} best_val_acc={best_acc*100:.1f}%')
    return rec


def main():
    print('=' * 70)
    print('训练过程全记录：S03/S09 被试内 + 9人混合跨被试')
    print('=' * 70)
    _, T, M, splits, cross_split = load_data()
    print(f'T={T}, n_windows M={M}')

    results = {'config': {'subjects': PAIR_SUBJECTS, 'iterations': ITERATIONS,
                          'lr': LR, 'wd': WD, 'n_windows': M}, 'models': {}}

    # 被试内
    for sid in PAIR_SUBJECTS:
        X_tr, y_tr, X_va, y_va, X_te, y_te = splits[sid]
        tr_ld, va_ld, te_ld = make_loaders(X_tr, y_tr, X_va, y_va, X_te, y_te)
        net = mAtt_bci(WIN_SAMPLES, STEP_SAMPLES, n_classes=N_CLASSES,
                       d_attn_in=D_ATTN_IN, d_attn_out=D_ATTN_OUT, n_windows=M).to(DEVICE)
        rec = train_with_record(f'S{sid:02d} 被试内', net, tr_ld, va_ld, te_ld, M,
                                f'trainpair_within_S{sid}')
        results['models'][f'within_S{sid}'] = {'tag': f'S{sid} 被试内', **rec}
        del net

    # 9人混合
    X_tr, y_tr, X_va, y_va, X_te, y_te = cross_split
    tr_ld, va_ld, te_ld = make_loaders(X_tr, y_tr, X_va, y_va, X_te, y_te)
    net = mAtt_bci(WIN_SAMPLES, STEP_SAMPLES, n_classes=N_CLASSES,
                   d_attn_in=D_ATTN_IN, d_attn_out=D_ATTN_OUT, n_windows=M).to(DEVICE)
    rec = train_with_record('9人混合跨被试', net, tr_ld, va_ld, te_ld, M,
                            'trainpair_cross_mix')
    results['models']['cross_mix'] = {'tag': '9人混合跨被试', **rec}

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
            sr['features_logtrace_traj'] = [
                {name: st['log_trace_mean'] for name, st in fs.items() if 'log_trace_mean' in st}
                for fs in r['features']]
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

    # 图3: 特征演化（关键层 norm + log-trace）
    key_layers = ['conv1', 'conv2', 'ract1.signal2spd[1]', 'spdnorm', 'tangent.vec', 'linear']
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
    L.append('训练过程全记录报告（S03/S09 被试内 + 9人混合跨被试）')
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
