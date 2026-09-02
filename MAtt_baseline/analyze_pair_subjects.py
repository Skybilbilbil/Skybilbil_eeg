# -*- coding: utf-8 -*-
"""
S03/S09 被试内 vs 9人混合跨被试 · 细化算法过程分析
====================================================
针对指定被试对（默认 S03、S09）:
  - 完整保存该被试在「被试内模型」与「9人混合跨被试模型」下的逐层算法过程数据
    （每层原始特征张量、逐窗口协方差能量/切空间范数、注意力概率矩阵、预测与标签）
  - 输出细化分析: 逐层分布对比 / 逐窗口轨迹 / 注意力结构 / 切空间PCA /
     逐类逐层偏移 / 混淆矩阵。
"""

import os, sys, json, time
import numpy as np
import torch
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from analyze_cross_pipeline import (
    FeatureRecorder, pair_metrics, le_mean, le_dist,
    LAYER_ORDER, LAYER_DISP, LAYER_KIND, CLASS_NAMES,
    FS, WIN_SAMPLES, STEP_SAMPLES, DEVICE, OUT_DIR,
    mAtt_bci, load_preprocessed_dataset, split_train_valid_test,
)

PAIR_SUBJECTS = [3, 9]
SAMPLE_RATIO = 0.5
BS = 16
N_CLASSES = 4
D_ATTN_IN, D_ATTN_OUT = 8, 8

CKPT_PATTERN_W = 'trainpair_within_S{sid}_best_loss.pt'
CKPT_CROSS = OUT_DIR / 'trainpair_cross_mix_best_loss.pt'
DATA_PATH = str(BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl')

# 画箱线图选用的关键层
BOX_LAYERS = ['channel_weight', 'conv1', 'conv2',
              'spdnorm', 'tangent.vec', 'dropout', 'linear']


def load_model(ckpt):
    net = mAtt_bci(WIN_SAMPLES, STEP_SAMPLES, n_classes=N_CLASSES,
                   d_attn_in=D_ATTN_IN, d_attn_out=D_ATTN_OUT, n_windows=M).to(DEVICE)
    try:
        net.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    except Exception:
        net.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    net.eval()
    return net


def extract_full(net, loader, device):
    """完整提取: 每层原始张量 / 注意力概率 / 标签 / 预测"""
    recorder = FeatureRecorder(net)
    raw = {}
    att_probs = []
    labels, preds = [], []
    net.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            recorder.clear()
            logits = net(xb)
            preds.append(logits.argmax(1).cpu().numpy())
            labels.append(yb.numpy())
            for name, t in recorder.features.items():
                raw.setdefault(name, []).append(t.float().numpy())
            spd_norm = recorder.features.get('spdnorm')
            if spd_norm is not None:
                att_probs.append(net.att2.attention_prob(spd_norm.to(device)).cpu().numpy())
    raw = {k: np.concatenate(v) for k, v in raw.items()}
    att_probs = np.concatenate(att_probs) if att_probs else None
    return raw, att_probs, np.concatenate(labels), np.concatenate(preds)


def main():
    t0 = time.time()
    global M
    print('=' * 70)
    print(f'加载 BCI2A 数据，分析被试 {PAIR_SUBJECTS}: 被试内 vs 9人混合跨被试')
    print('=' * 70)
    subject_data, _ = load_preprocessed_dataset(DATA_PATH)
    T = subject_data[0][0].shape[-1]
    M = (T - WIN_SAMPLES) // STEP_SAMPLES + 1
    print(f'T={T}, n_windows M={M}')

    from sklearn.model_selection import train_test_split
    subject_tests = {}
    for sid in PAIR_SUBJECTS:
        X, y = subject_data[sid - 1]
        _, X, _, y = train_test_split(X, y, test_size=SAMPLE_RATIO, stratify=y, random_state=42)
        _, _, _, _, X_te, y_te = split_train_valid_test(X, y)
        subject_tests[sid] = (X_te, y_te)
        print(f'  S{sid:02d} 测试集: {X_te.shape}, 类别 {np.bincount(y_te, minlength=4)}')

    net_cross = load_model(CKPT_CROSS)
    print('已加载 9人混合跨被试模型')

    process = {'config': {'subjects': PAIR_SUBJECTS, 'n_windows': M, 'fs': FS},
               'subjects': {}}
    for sid in PAIR_SUBJECTS:
        X_te, y_te = subject_tests[sid]
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.Tensor(X_te).unsqueeze(1),
                                           torch.Tensor(y_te).long()),
            batch_size=BS, shuffle=False)
        ckpt_w = OUT_DIR / CKPT_PATTERN_W.format(sid=sid)
        if not ckpt_w.exists():
            print(f'  [WARN] 缺少 {ckpt_w.name}, 跳过 S{sid:02d}')
            continue
        net_w = load_model(ckpt_w)
        rw = extract_full(net_w, loader, DEVICE)
        rc = extract_full(net_cross, loader, DEVICE)
        process['subjects'][str(sid)] = {
            'within': {'raw': rw[0], 'att_prob': rw[1], 'labels': rw[2], 'preds': rw[3]},
            'cross': {'raw': rc[0], 'att_prob': rc[1], 'labels': rc[2], 'preds': rc[3]},
        }
        acc_w = float((rw[2] == rw[3]).mean())
        acc_c = float((rc[2] == rc[3]).mean())
        print(f'  S{sid:02d}: 被试内 acc={acc_w*100:.1f}%  跨被试 acc={acc_c*100:.1f}%')
        del net_w

    # 保存完整过程数据
    pkl_path = OUT_DIR / 'pair_subjects_process.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(process, f)
    print(f'\n[OK] 完整过程数据已保存: {pkl_path}')
    print(f'     文件大小: {pkl_path.stat().st_size/1e6:.1f} MB')

    # ============ 细化分析 ============
    report = analyze(process)
    json_path = OUT_DIR / 'pair_subjects_detail.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=float)
    txt_path = OUT_DIR / 'pair_subjects_report.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(format_report(report))
    print(f'[OK] 细化分析数据已保存: {json_path}')
    print(f'[OK] 文字报告已保存: {txt_path}')
    print(f'总耗时: {time.time()-t0:.1f}s')


def analyze(process):
    report = {'per_subject': {}}
    for sid_str, d in process['subjects'].items():
        sid = int(sid_str)
        rw, rc = d['within'], d['cross']
        sub = {'accuracy': {'within': float((rw['labels'] == rw['preds']).mean()),
                            'cross': float((rc['labels'] == rc['preds']).mean())},
               'layers': {}, 'windows': {}, 'attention': {}, 'per_class': {}}

        # ---- 逐层: within vs cross 分布指标 ----
        for name, kind, disp in LAYER_ORDER:
            if name not in rw['raw'] or name not in rc['raw']:
                continue
            vecs_w, _ = to_vectors(name, rw['raw'][name], M)
            vecs_c, _ = to_vectors(name, rc['raw'][name], M)
            spdW = spd_mean_mat(name, rw['raw'][name], M)
            spdC = spd_mean_mat(name, rc['raw'][name], M)
            m = pair_metrics(vecs_w, vecs_c, kind, spdW, spdC)
            sub['layers'][name] = m

        # ---- 逐窗口: 协方差能量 / 切空间范数 ----
        cov_w = rw['raw']['ract1.signal2spd']  # (n, M, 8, 8)
        cov_c = rc['raw']['ract1.signal2spd']
        tan_w = rw['raw']['tangent.vec'].reshape(len(cov_w), M, -1)
        tan_c = rc['raw']['tangent.vec'].reshape(len(cov_c), M, -1)
        win = {'cov_ratio': [], 'tan_ratio': [],
               'cov_within': [], 'cov_cross': []}
        for w in range(M):
            mco = (np.linalg.norm(cov_w[:, w], axis=(1, 2)).mean(),
                   np.linalg.norm(cov_c[:, w], axis=(1, 2)).mean())
            mtn = (np.linalg.norm(tan_w[:, w], axis=1).mean(),
                   np.linalg.norm(tan_c[:, w], axis=1).mean())
            win['cov_ratio'].append(abs(mco[1] - mco[0]) / (mco[0] + 1e-8))
            win['tan_ratio'].append(abs(mtn[1] - mtn[0]) / (mtn[0] + 1e-8))
            win['cov_within'].append(float(mco[0]))
            win['cov_cross'].append(float(mco[1]))
        sub['windows'] = win

        # ---- 注意力结构 ----
        att_w = rw['att_prob'].mean(0)  # (M, M)
        att_c = rc['att_prob'].mean(0)
        diag_conc_w = np.trace(att_w) / att_w.sum()
        diag_conc_c = np.trace(att_c) / att_c.sum()
        sub['attention'] = {
            'within': att_w.tolist(), 'cross': att_c.tolist(),
            'diag_concentration_within': float(diag_conc_w),
            'diag_concentration_cross': float(diag_conc_c),
            'frobenius_dist': float(np.linalg.norm(att_w - att_c)),
        }

        # ---- 逐类逐层偏移（该被试内 within vs cross） ----
        for c in range(N_CLASSES):
            sub['per_class'][str(c)] = {}
            for name, kind, disp in LAYER_ORDER:
                if name not in rw['raw'] or name not in rc['raw']:
                    continue
                vecs_w, mats_w = to_vectors(name, rw['raw'][name], M)
                vecs_c, mats_c = to_vectors(name, rc['raw'][name], M)
                mw = vecs_w[rw['labels'] == c]
                mc = vecs_c[rc['labels'] == c]
                if len(mw) < 3 or len(mc) < 3:
                    continue
                spdW = le_mean(mats_w[rw['labels'] == c]) if (kind == 'spd' and mats_w is not None) else None
                spdC = le_mean(mats_c[rc['labels'] == c]) if (kind == 'spd' and mats_c is not None) else None
                m = pair_metrics(mw, mc, kind, spdW, spdC)
                if m:
                    sub['per_class'][str(c)][name] = m

        report['per_subject'][sid_str] = sub
    return report


def to_vectors(name, t, M):
    """原始张量 → 逐样本向量 + (SPD 层返回逐样本 log-均值矩阵)"""
    from analyze_cross_pipeline import batch_to_vectors
    return batch_to_vectors(name, torch.from_numpy(t), M)


def spd_mean_mat(name, t, M):
    if LAYER_KIND[name] != 'spd':
        return None
    _, mats = to_vectors(name, t, M)
    return le_mean(mats) if mats is not None else None


def format_report(report):
    lines = []
    lines.append('=' * 80)
    lines.append('S03 / S09 · 被试内 vs 9人混合跨被试 · 细化算法过程分析报告')
    lines.append('=' * 80)
    for sid, sub in report['per_subject'].items():
        acc = sub['accuracy']
        lines.append(f'\n{"-"*80}\nS{sid}  被试内 acc={acc["within"]*100:.1f}%  '
                     f'跨被试 acc={acc["cross"]*100:.1f}%  差距={acc["within"]-acc["cross"]:.3f}\n{"-"*80}')
        lines.append('\n[逐层 within vs cross 分布差异]')
        lines.append(f'{"层":<14s}{"范数比":<9s}{"质心偏移":<10s}{"MMD":<9s}{"方差比":<9s}{"测地":<8s}')
        for name, kind, disp in LAYER_ORDER:
            if name not in sub['layers']:
                continue
            m = sub['layers'][name]
            g = f'{m.get("geo_dist", float("nan")):.3f}' if 'geo_dist' in m else '-'
            lines.append(f'{disp:<14s}{m["norm_ratio"]:<9.3f}{m["centroid_shift"]:<10.3f}'
                         f'{m["mmd"]:<9.4f}{m["std_ratio"]:<9.3f}{g:<8s}')
        win = sub['windows']
        lines.append('\n[逐窗口差异] (窗口1=试次起始 0ms, 每窗口 500ms 步长)')
        lines.append(f'{"窗口":<6s}{"协方差 within":<14s}{"协方差 cross":<14s}{"协方差比值":<10s}{"切空间比值":<10s}')
        for w in range(len(win['cov_ratio'])):
            lines.append(f'  w{w+1}   {win["cov_within"][w]:.4f}          {win["cov_cross"][w]:.4f}          '
                         f'{win["cov_ratio"][w]:.3f}        {win["tan_ratio"][w]:.3f}')
        att = sub['attention']
        lines.append(f'\n[注意力结构] 对角线集中度: within={att["diag_concentration_within"]:.3f} '
                     f'cross={att["diag_concentration_cross"]:.3f}  两矩阵Frobenius距离={att["frobenius_dist"]:.3f}')
        lines.append('\n[逐类逐层 质心偏移]')
        lines.append(f'{"类":<5s}' + ''.join(f'{LAYER_DISP[n][:9]:<10s}' for n, _, _ in LAYER_ORDER))
        for c in range(N_CLASSES):
            row = ''
            for name, _, _ in LAYER_ORDER:
                m = sub['per_class'][str(c)].get(name)
                row += f'{m["centroid_shift"]:.3f}     ' if m else f'{"-":<10s}'
            lines.append(f'  {CLASS_NAMES[c]:<3s}{row}')
        lines.append('\n' + '=' * 80)
    return '\n'.join(lines)


if __name__ == '__main__':
    import pickle
    main()
