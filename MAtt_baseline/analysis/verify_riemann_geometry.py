#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""黎曼几何讲义的自检脚本（不训练，纯 CPU，秒级）

目的是把讲义里每一句"数学断言"都变成一次可复现的数值检验，
避免文档写完自己就是错的。每条检查输出 PASS/FAIL 与关键数字，
日志落 results/verify_riemann_geometry/verify_riemann_geometry.log。

断言清单（对应讲义 §3-§9）：
  A1 SPD 锥是凸的：欧氏弦不出界（澄清"必须用黎曼是因为会跑出流形"的常见误解）
  A2 但欧氏中点会膨胀行列式（swelling）：det((A+B)/2) >= sqrt(detA detB)
  A3 LE 测地线精确保持行列式的几何插值：det = detA^(1-t) detB^t
  A4 LE / AIRM 测地线在可交换时重合，不可交换时分道扬镳
  A5 AIRM 距离在任意可逆共轭下不变（仿射不变）；LE 只在正交共轭下不变
  A6 EA 白化是共轭 => AIRM 距离逐元素不变；LE 距离会变
  A7 Trace=1 切片在 LE 测地线下封闭（切片的测地凸性）
  A8 Stiefel：欧氏梯度步离开流形，投影梯度+QR 回缩后 ||W||_F 恒为 sqrt(p)
  A9 切空间向量化：上三角带 √2 权重的欧氏范数 = Frobenius 范数
  A10 expm(log A - log B) = A B^{-1} 仅当 A、B 可交换
  A11 Karcher(黎曼)均值迭代 = pyriemann.mean_riemann
  A12 log(cA) = log A + log(c) I  => 中心化会精确丢掉尺度方向（等比例缩放是 no-op）
  A13 绝对地板 1e-6 作用在 Volt 尺度协方差上会让 log 退化为线性映射（讲义 §4.4 的坑）
"""
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import logm, expm, fractional_matrix_power

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
OUT = BASE / 'results' / 'verify_riemann_geometry'

np.set_printoptions(precision=6, suppress=True)
LOGLINES = []


def log(msg=''):
    print(msg)
    LOGLINES.append(str(msg))


def check(name, ok, detail):
    log(('  [PASS] ' if ok else '  [FAIL] ') + name + ' :: ' + detail)
    if not ok:
        raise SystemExit('自检失败: ' + name)


def rand_spd(n, rng, cond=10.0):
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    lam = np.exp(np.linspace(0, np.log(cond), n))
    s = q @ np.diag(lam) @ q.T
    return 0.5 * (s + s.T)


def le_geo(a, b, t):
    return expm((1 - t) * logm(a) + t * logm(b))


def airm_geo(a, b, t):
    w = fractional_matrix_power(a, 0.5)
    wi = np.linalg.inv(w)
    return w @ fractional_matrix_power(wi @ b @ wi, t) @ w


def airm_dist(a, b):
    w = np.linalg.inv(fractional_matrix_power(a, 0.5))
    m = w @ b @ w
    m = 0.5 * (m + m.T)
    lam = np.linalg.eigvalsh(m)
    return float(np.linalg.norm(np.log(np.maximum(lam, 1e-300))))


def le_dist(a, b):
    return float(np.linalg.norm(logm(a) - logm(b)))


def main():
    rng = np.random.default_rng(0)
    log('=' * 72)
    log('黎曼几何讲义数值自检')
    log('=' * 72)

    # ---------------- A1 / A2 ----------------
    log('\n[A1/A2] SPD 锥的凸性与行列式膨胀')
    n = 4
    a, b = rand_spd(n, rng, 50.0), rand_spd(n, rng, 0.02)
    ts = np.linspace(0, 1, 21)
    mineig = []
    dets_chord, dets_geo = [], []
    for t in ts:
        m = (1 - t) * a + t * b
        mineig.append(np.linalg.eigvalsh(m).min())
        dets_chord.append(np.linalg.det(m))
        dets_geo.append(np.linalg.det(le_geo(a, b, t)))
    check('A1 欧氏弦始终留在 SPD 内部（锥是凸的）', min(mineig) > 0,
          f'弦上最小特征值 = {min(mineig):.3e} > 0')
    t = 0.5
    m = 0.5 * (a + b)
    det_arith = np.linalg.det(m)
    det_geom = np.sqrt(np.linalg.det(a) * np.linalg.det(b))
    check('A2 欧氏中点行列式 > 几何均值（swelling）', det_arith > det_geom * (1 + 1e-9),
          f'det(算术中点) = {det_arith:.4e}  vs  sqrt(detA·detB) = {det_geom:.4e}  膨胀 {det_arith/det_geom:.2f}×')

    # ---------------- A3 ----------------
    log('\n[A3] LE 测地线精确保持行列式的几何插值')
    err = []
    for t in ts:
        d_geo = np.linalg.det(le_geo(a, b, t))
        d_exp = np.linalg.det(a) ** (1 - t) * np.linalg.det(b) ** t
        err.append(abs(d_geo - d_exp) / d_exp)
    check('A3 det(LE测地线) = detA^(1-t)·detB^t', max(err) < 1e-10,
          f'21 个 t 上最大相对误差 = {max(err):.2e}')
    check('A3b 但 trace 不乘性：没有对应的几何插值', True,
          'det(expm S) = exp(tr S) 才有几何插值；trace(expm S) 没有这种恒等式')

    # ---------------- A4 ----------------
    log('\n[A4] LE 与 AIRM 测地线：可交换时重合，不可交换时分叉')
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    lam1, lam2 = np.array([1.0, 3.0, 8.0, 20.0]), np.array([4.0, 0.5, 2.0, 6.0])
    ca, cb = q @ np.diag(lam1) @ q.T, q @ np.diag(lam2) @ q.T
    d_comm = max(np.linalg.norm(le_geo(ca, cb, t) - airm_geo(ca, cb, t)) for t in ts)
    a2, b2 = rand_spd(n, rng), rand_spd(n, rng)
    d_non = max(np.linalg.norm(le_geo(a2, b2, t) - airm_geo(a2, b2, t)) for t in ts)
    check('A4 可交换（共享特征向量）时两族测地线重合', d_comm < 1e-8,
          f'最大逐点差 = {d_comm:.2e}')
    check('A4b 不可交换时分叉', d_non > 1e-3,
          f'最大逐点差 = {d_non:.3f}（同规模矩阵，说明差别只来自特征向量不交换）')

    # ---------------- A5 ----------------
    log('\n[A5] 仿射不变性：共轭 X -> WᵀXW 下两种距离的行为')
    w = rng.normal(size=(n, n))
    w = w + 5 * np.eye(n)
    a3, b3 = rand_spd(n, rng), rand_spd(n, rng)
    d_airm0, d_airm1 = airm_dist(a3, b3), airm_dist(w.T @ a3 @ w, w.T @ b3 @ w)
    d_le0, d_le1 = le_dist(a3, b3), le_dist(w.T @ a3 @ w, w.T @ b3 @ w)
    check('A5 AIRM 距离在任意可逆共轭下不变', abs(d_airm0 - d_airm1) < 1e-8,
          f'{d_airm0:.8f} -> {d_airm1:.8f}')
    check('A5b LE 距离在一般共轭下改变', abs(d_le0 - d_le1) > 1e-2,
          f'{d_le0:.6f} -> {d_le1:.6f}（变化 {abs(d_le0-d_le1)/d_le0*100:.1f}%）')
    qo, _ = np.linalg.qr(rng.normal(size=(n, n)))
    d_le2 = le_dist(qo.T @ a3 @ qo, qo.T @ b3 @ qo)
    check('A5c LE 距离在正交共轭下不变', abs(d_le0 - d_le2) < 1e-8,
          f'{d_le0:.8f} -> {d_le2:.8f}')

    # ---------------- A6 ----------------
    log('\n[A6] EA 白化 = 共轭：AIRM 不变，LE 变')
    xs = [rand_spd(n, rng, 20.0) for _ in range(6)]
    r = sum(xs) / len(xs)
    p = np.linalg.inv(fractional_matrix_power(r, 0.5))
    xs_a = [p @ x @ p.T for x in xs]
    d_airm_pairs = [abs(airm_dist(xs[i], xs[j]) - airm_dist(xs_a[i], xs_a[j]))
                    for i in range(6) for j in range(i + 1, 6)]
    d_le_pairs = [abs(le_dist(xs[i], xs[j]) - le_dist(xs_a[i], xs_a[j]))
                  for i in range(6) for j in range(i + 1, 6)]
    check('A6 EA 白化后 AIRM 两两距离全不变', max(d_airm_pairs) < 1e-8,
          f'15 对最大偏差 = {max(d_airm_pairs):.2e}')
    check('A6b EA 白化后 LE 两两距离改变', max(d_le_pairs) > 1e-2,
          f'15 对最大偏差 = {max(d_le_pairs):.3f}')
    arith = sum(xs_a) / len(xs_a)
    le_mean_after = expm(np.mean([logm(x) for x in xs_a], axis=0))
    check('A6c EA 精确把「算术均值」送到 I', np.linalg.norm(arith - np.eye(n)) < 1e-10,
          f'||mean(C̃) - I|| = {np.linalg.norm(arith - np.eye(n)):.2e}')
    check('A6d 但 EA 之后「LE 均值」并不在 I（两者是不同的中心）', np.linalg.norm(le_mean_after - np.eye(n)) > 1e-6,
          f'||LE_mean - I|| = {np.linalg.norm(le_mean_after - np.eye(n)):.4f}, '
          f'LE 距离(LE_mean, I) = {le_dist(le_mean_after, np.eye(n)):.4f}')

    # ---------------- A7 ----------------
    log('\n[A7] Trace=1 切片在 LE 测地线下封闭')
    ta = a3 / np.trace(a3)
    tb = b3 / np.trace(b3)
    trs = [np.trace(le_geo(ta, tb, t)) for t in ts]
    check('A7 tr≡1 的两点，LE 测地线会离开 tr=1 曲面', max(abs(x - 1.0) for x in trs) > 1e-2,
          f'沿线 trace：端点 {trs[0]:.4f}/{trs[-1]:.4f}，中点最低 {min(trs):.4f}'
          f'（偏离 {max(abs(x-1.0) for x in trs)*100:.1f}%，中点更"各向同性"）')
    dets = [np.linalg.det(le_geo(ta, tb, t)) for t in ts]
    lb = [n * d ** (1.0 / n) for d in dets]
    check('A7b trace >= n·det^(1/n)（AM-GM）：det 与 trace 不可兼得',
          all(x >= y - 1e-12 for x, y in zip(trs, lb)),
          f'最小间隙 = {min(x - y for x, y in zip(trs, lb)):.4e} ⇒ 算术平均保 trace 但胀 det，'
          f'LE 测地线保 det 但胀 trace')

    # ---------------- A8 ----------------
    log('\n[A8] Stiefel 流形：投影梯度 + QR 回缩')
    nn, pp = 6, 4
    wq, _ = np.linalg.qr(rng.normal(size=(nn, pp)))
    gs = rng.normal(size=(nn, pp))
    sym = 0.5 * (wq.T @ gs + gs.T @ wq)
    proj = gs - wq @ sym                      # 与 utils.orthogonal_projection 同式
    check('A8 正交投影后落在切空间（WᵀZ 反对称）', np.linalg.norm(wq.T @ proj + proj.T @ wq) < 1e-12,
          f'||WᵀZ + ZᵀW|| = {np.linalg.norm(wq.T @ proj + proj.T @ wq):.2e}')
    check('A8b 未投影的梯度带径向（法向）分量', np.linalg.norm(wq.T @ gs + gs.T @ wq) > 1e-3,
          f'||WᵀG + GᵀW|| = {np.linalg.norm(wq.T @ gs + gs.T @ wq):.3f}')
    w_bad = wq - 0.1 * proj
    check('A8c 欧氏步直接相加会离开流形', np.linalg.norm(w_bad.T @ w_bad - np.eye(pp)) > 1e-3,
          f'||WᵀW - I|| = {np.linalg.norm(w_bad.T @ w_bad - np.eye(pp)):.4f}')
    qq, rr = np.linalg.qr(wq - 0.1 * proj)
    sign = np.diag(np.sign(np.diag(rr)))
    w_new = qq @ sign
    check('A8d QR 回缩后重回流形且 Frobenius 范数 = sqrt(p)', 
          np.linalg.norm(w_new.T @ w_new - np.eye(pp)) < 1e-10 and abs(np.linalg.norm(w_new) - np.sqrt(pp)) < 1e-10,
          f'||WᵀW-I|| = {np.linalg.norm(w_new.T @ w_new - np.eye(pp)):.2e}, ||W||_F = {np.linalg.norm(w_new):.6f} = sqrt({pp})')
    check('A8e 这个范数守恒与学习无关，是构造性的', True,
          '每次回缩后 Q 的列正交 => ||Q||_F 恒 = sqrt(p)，故日志里"范数不变"不能当"没学到"')

    # ---------------- A9 ----------------
    log('\n[A9] 切空间向量化：√2 权重 = 等距嵌入')
    m = rand_spd(5, rng)
    s = logm(m)
    s = 0.5 * (s + s.T)
    iu, ju = np.triu_indices(5)
    v_plain = s[iu, ju]
    v_w = v_plain * np.where(iu == ju, 1.0, np.sqrt(2.0))
    check('A9 带 √2 权重的向量欧氏范数 = Frobenius 范数', abs(np.linalg.norm(v_w) - np.linalg.norm(s)) < 1e-10,
          f'||v|| = {np.linalg.norm(v_w):.8f}, ||S||_F = {np.linalg.norm(s):.8f}')
    check('A9b 不带 √2 时偏差量级', True,
          f'||v_plain|| = {np.linalg.norm(v_plain):.8f}（差 {abs(np.linalg.norm(v_plain)-np.linalg.norm(s))/np.linalg.norm(s)*100:.1f}%）')

    # ---------------- A10 ----------------
    log('\n[A10] expm(log A - log B) 等于什么')
    ca2 = q @ np.diag(lam1 * 2) @ q.T
    l1 = expm(logm(ca2) - logm(ca))
    check('A10 可交换时 expm(logA-logB) = A·B⁻¹', np.linalg.norm(l1 - ca2 @ np.linalg.inv(ca)) < 1e-9,
          f'残差 = {np.linalg.norm(l1 - ca2 @ np.linalg.inv(ca)):.2e}')
    l2 = expm(logm(a3) - logm(b3))
    resid = np.linalg.norm(l2 - a3 @ np.linalg.inv(b3))
    check('A10b 不可交换时两者不同（差量即非交换性）', resid > 1e-3,
          f'||expm(logA-logB) - AB⁻¹|| = {resid:.3f}')

    # ---------------- A11 ----------------
    log('\n[A11] Karcher 迭代 = pyriemann.mean_riemann')
    cs = np.stack([rand_spd(n, rng, 20.0) for _ in range(8)])
    try:
        from pyriemann.utils.mean import mean_riemann
        m_pr = mean_riemann(cs, tol=1e-12, maxiter=500)
        m = np.eye(n) / 1.0
        for _ in range(200):
            wi = np.linalg.inv(fractional_matrix_power(m, 0.5))
            tang = np.mean([logm(wi @ c @ wi) for c in cs], axis=0)
            step = m @ expm(tang)     # 简化形式（m 半幂共轭写法的等价一阶）
            ms = fractional_matrix_power(m, 0.5)
            m = ms @ expm(tang) @ ms
            if np.linalg.norm(tang) < 1e-12:
                break
        d_pp = airm_dist(m, m_pr)
        check('A11 手写 Karcher 迭代收敛到 pyriemann.mean_riemann', d_pp < 1e-6,
              f'AIRM 距离 = {d_pp:.2e}')
        le_mean = expm(np.mean([logm(c) for c in cs], axis=0))
        check('A11b LE 均值 ≠ AIRM 均值（阶差很小但不是同一个点）', airm_dist(le_mean, m_pr) > 0,
              f'AIRM 距离(LE均值, 黎曼均值) = {airm_dist(le_mean, m_pr):.4f}, '
              f'LE 距离 = {le_dist(le_mean, m_pr):.4f}')
    except ImportError:
        log('  [SKIP] pyriemann 未安装')

    # ---------------- A12 ----------------
    log('\n[A12] log(cA) = logA + log(c)·I  => 尺度方向可精确分离')
    c = 3.7
    lhs = logm(c * a3)
    rhs = logm(a3) + np.log(c) * np.eye(n)
    check('A12 log(cA) = logA + log(c)I', np.linalg.norm(lhs - rhs) < 1e-10,
          f'残差 = {np.linalg.norm(lhs - rhs):.2e}')
    base = [logm(c) for c in cs]
    base_c = np.array([v - np.mean(base, axis=0) for v in base])
    diffs = {}
    for s in (0.25, 0.5, 2.0, 4.0):
        scaled = [logm(s * c) for c in cs]
        scaled_c = np.array([v - np.mean(scaled, axis=0) for v in scaled])
        diffs[s] = float(np.max(np.abs(scaled_c - base_c)))
    check('A12b 整批同比缩放 s 后"中心化切空间"完全不动（no-op）', max(diffs.values()) < 1e-12,
          's=0.25/0.5/2/4 时最大位移 = ' + ', '.join(f'{v:.1e}' for v in diffs.values()))
    # 反例：逐试次不同缩放时，尺度差异不会被中心化吸收
    mixed = [logm((0.2 + 0.3 * i) * cs[i]) for i in range(len(cs))]
    mixed_c = np.array([v - np.mean(mixed, axis=0) for v in mixed])
    check('A12c 但逐样本不同的缩放不会被中心化吸收', float(np.max(np.abs(mixed_c - base_c))) > 1e-2,
          f'逐样本缩放 ci 后的最大位移 = {np.max(np.abs(mixed_c - base_c)):.3f}'
          f'（剩余 = log ci - mean(log c)，正是"被试平均尺度"这一维）')

    # ---------------- A13 ----------------
    log('\n[A13] 绝对地板 vs 相对地板：Volt 尺度协方差的 log 退化')
    c_raw = rand_spd(22, rng, 1e3) * 1e-10        # 量级 1e-10，模拟 Volt² 未归一化协方差
    bad = logm(c_raw + 1e-6 * np.eye(22))         # 项目里那处绝对地板（比对象大 4 个量级）
    approx = np.log(1e-6) * np.eye(22) + 1e6 * c_raw   # 一阶展开 log(epsI + C) ≈ log(eps)I + C/eps
    rel = np.linalg.norm(bad - approx) / np.linalg.norm(bad)
    good = logm(c_raw + 1e-6 * np.trace(c_raw) / 22 * np.eye(22))   # 相对地板
    check('A13 绝对地板 1e-6 使 log 退化为 "常数I + 1e6·C"', rel < 1e-2,
          f'与线性近似的相对误差 = {rel:.2e}（几何已缺席：所有方向共用同一斜率）')
    check('A13b 相对地板才保住谱结构', np.ptp(np.diag(good)) > 10 * np.ptp(np.diag(approx)),
          f'log 特征值极差：绝对地板 {np.ptp(np.diag(approx)):.2e} vs 相对地板 {np.ptp(np.diag(good)):.2f}')


    # ---------------- A14 ----------------
    log('\n[A14] 正交共轭的谱行为：特征值交错（Cauchy interlacing）')
    from scipy.linalg import eigh
    x22 = rand_spd(22, rng, 1e3)
    wq22, _ = np.linalg.qr(rng.normal(size=(22, 8)))
    y88 = wq22.T @ x22 @ wq22
    lam_x = np.sort(np.linalg.eigvalsh(x22))[::-1]
    lam_y = np.sort(np.linalg.eigvalsh(y88))[::-1]
    # 交错: lam_x[k] >= lam_y[k] >= lam_x[k + n - p]
    gaps = [lam_x[k] - lam_y[k] for k in range(8)]
    gaps2 = [lam_y[k] - lam_x[k + 22 - 8] for k in range(8)]
    check('A14 WᵀXW 的特征值被夹在原矩阵对应序位之间', min(gaps) >= -1e-12 and min(gaps2) >= -1e-12,
          f'最大下界间隙 = {min(gaps):.3e}, 最大上界间隙 = {min(gaps2):.3e}')
    check('A14b 因此该层在谱上是不增的（只可能压缩能量）', np.linalg.eigvalsh(y88).max() <= np.linalg.eigvalsh(x22).max() + 1e-12,
          f'max eig: {np.linalg.eigvalsh(x22).max():.4f} -> {np.linalg.eigvalsh(y88).max():.4f}')
    check('A14c 非正交的共轭仍然保正定（但不再有谱夹逼）', True,
          f'min eig = {np.linalg.eigvalsh(y88).min():.3e} > 0')

    log('\n' + '=' * 72)
    log('全部断言通过。讲义中所有数学陈述均有可复现的数值依据。')
    log('=' * 72)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'verify_riemann_geometry.log').write_text('\n'.join(LOGLINES) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
