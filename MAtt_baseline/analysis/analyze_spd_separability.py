# -*- coding: utf-8 -*-
"""数据层 SPD 表示的可分性诊断(不看模型,只看特征本身)。

回答一个问题:**"SPD 特征可分性不足"是数据的真实上限,还是当前表示形式的缺陷?**

三条测量,全部在"原始 22 通道 → 逐试次协方差 → 切空间"这个最朴素的 SPD 表示上:
  ① 被试内 4 类可分性:每被试 5 折 CV 的 LDA 准确率(对照 25% 随机)
  ② 跨被试 9x9 迁移矩阵:被试 i 训 → 被试 j 测(对角线=被试内,非对角线=可迁移成分)
  ③ 不用模型、不用训练,所以任何"训练机制"的问题都被排除在外

若 ① 就远低于模型的被试内 82.8% → 表示形式是瓶颈(该换表示,不是换训练);
若 ① 高而 ② 的非对角线低 → 表示没问题,瓶颈在跨被试迁移(该换迁移方法)。
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold

BASE = Path(__file__).resolve().parent.parent   # = MAtt_baseline/（本脚本在 analysis/ 下）
sys.path.insert(0, str(BASE))
from utils.dataset import load_preprocessed_dataset

DATA = BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
FS = 250
WIN = int(1.5 * FS)      # 与模型一致:1.5s 窗
STEP = int(0.75 * FS)    # 0.75s 步长
MI_W0 = 375              # 只用 cue 后 0.5s 起(epoch 起点 -1s → +0s 在索引 250)
MI_W1 = 1126             # 到 epoch 末尾(+3.5s)
RIDGE = 1e-3


def trial_cov(X, w0, w1):
    """把 (N, C, T) 的 [w0:w1] 段做成逐试次协方差 (N, C, C)"""
    S = X[:, :, w0:w1]
    return np.einsum('nct,ndt->ncd', S, S) / (w1 - w0)


def tangent(C):
    """SPD → 切空间向量(用 eigh 手写 logm,比 scipy.logm 快且稳)"""
    C = C + RIDGE * np.trace(C, axis1=1, axis2=2)[:, None, None] / C.shape[1] * np.eye(C.shape[1])
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-12, None)
    L = (V * np.log(w)[:, None, :]) @ np.transpose(V, (0, 2, 1))
    iu = np.triu_indices(C.shape[1])
    return L[:, iu[0], iu[1]]


def feats(X):
    """MI 期协方差 → 切空间。

    空间基固定为 22 个原始通道(不学空间滤波),所以这条曲线是"学空间滤波之前"的
    下界 —— 模型能超过它,靠的是 conv1 学出来的空间投影。
    """
    return tangent(trial_cov(X, MI_W0, MI_W1))


def main():
    sd, _ = load_preprocessed_dataset(str(DATA))
    data = [(np.asarray(X, dtype=np.float64), np.asarray(y)) for X, y in sd]
    F = [feats(X) for X, _ in data]
    Y = [y for _, y in data]
    print(f'被试数={len(F)}  特征维={F[0].shape[1]}  每被试试次={F[0].shape[0]}')
    print('特征: 22ch 在 cue 后 [0.5,3.5]s 的逐试次协方差 → 切空间(253 维)\n')

    print('① 被试内 4 类可分性 (5 折 CV, LDA + Ledoit-Wolf 收缩):')
    accs = []
    for i, (f, y) in enumerate(zip(F, Y)):
        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        sc = []
        for tr, te in skf.split(f, y):
            clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            clf.fit(f[tr], y[tr])
            sc.append(clf.score(f[te], y[te]))
        accs.append(np.mean(sc))
        print(f'   S{i+1:02d}  {np.mean(sc)*100:5.1f}%')
    print(f'   平均 {np.mean(accs)*100:.1f}%   (随机 25%;模型被试内 S3 = 82.8%)\n')

    # 注意:对角线是"同一批数据既拟合又测试"(in-sample,乐观),不是可信的被试内数字;
    #      可信的被试内数字是 ① 的 5 折 CV。对角线只用于和它自己那一列比较。
    print('② 跨被试迁移矩阵 (被试 i 训 → 被试 j 测,行=训,列=测,单位 %):')
    L, C = len(F), len(Y)
    M = np.zeros((L, L))
    for i in range(L):
        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        clf.fit(F[i], Y[i])
        for j in range(L):
            M[i, j] = clf.score(F[j], Y[j]) * 100
    hdr = '      ' + ''.join(f'  S{j+1:02d} ' for j in range(C))
    print(hdr)
    for i in range(L):
        print(f'   S{i+1:02d} ' + ''.join(f'{M[i, j]:6.1f}' for j in range(C)))
    # 按"目标被试"拆开:该被试作为目标时,从其他 8 人迁移过来的平均准确率
    # 这直接回答"如果换一折做完全跨被试,大概会落在哪"
    colmean = (M.sum(0) - np.diag(M)) / (L - 1)
    print('\n   按目标被试的"接收性"(非对角列均值,即别人训它来测):')
    order = np.argsort(-colmean)
    for j in order:
        print(f'     S{j+1:02d}  接收 {colmean[j]:5.1f}%   (该被试内 CV {accs[j]*100:5.1f}%)')
    print(f'     列间: 均值 {colmean.mean():.1f}%  std {colmean.std():.1f}pp  '
          f'最高 S{colmean.argmax()+1:02d} {colmean.max():.1f}%  '
          f'最低 S{colmean.argmin()+1:02d} {colmean.min():.1f}%')
    print(f'     与"被试内 CV 准确率"的相关 r = {np.corrcoef(colmean, np.array(accs))[0, 1]:+.2f}')

    diag = np.mean(np.diag(M))
    off = (M.sum() - np.trace(M)) / (L * L - L)
    print(f'\n   对角(被试内) {diag:.1f}%  |  非对角(跨被试) {off:.1f}%  |  差 Δ={diag-off:.1f}pp')
    print(f'   → 这 Δ 就是"完全不用模型、不用训练"时,被试信息带来的全部增益')
    print(f'   → 供对照:模型共享 conv 54.6% / per-subject conv 62.6% / 被试内 82.8%')


if __name__ == '__main__':
    main()
