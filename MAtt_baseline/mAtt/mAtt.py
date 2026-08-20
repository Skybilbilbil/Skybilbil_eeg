import torch
import torch.nn as nn
from mAtt.spd import SPDTransform, SPDTangentSpace, SPDRectified, SPDPower, SPDNorm


# ============================================================
# 解剖先验通道配置（A9）：10-20 系统下与手部运动区（C3/Cz/C4）的亲近度
# ============================================================
BCI2A_CHANNELS = [
    'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
    'P1', 'Pz', 'P2', 'POz',
]

# 运动想象（手 MI）相关：中央区核心 > 中央区邻近 > 额中央/中央顶辅助 > 远区删除
MOTOR_CHANNEL_PRIOR = {
    'C3': 1.0, 'Cz': 1.0, 'C4': 1.0,                  # 手区核心
    'C1': 0.8, 'C2': 0.8, 'C5': 0.8, 'C6': 0.8,       # 中央区邻近
    'FC3': 0.6, 'FC1': 0.6, 'FCz': 0.6, 'FC2': 0.6, 'FC4': 0.6,
    'CP3': 0.6, 'CP1': 0.6, 'CPz': 0.6, 'CP2': 0.6, 'CP4': 0.6,  # 辅助运动/体感区
    'Fz': 0.0, 'P1': 0.0, 'Pz': 0.0, 'P2': 0.0, 'POz': 0.0,      # 远区（等效删除）
}


class ChannelPriorWeighting(nn.Module):
    """解剖先验通道加权（A9）

    - 权重为 0 的通道等效删除（Fz/P/Pz/POz，共 5 个，离感觉运动区远）；
    - 其余通道以解剖先验初始化为可学习缩放因子，网络可自行调整关注强度；
    - 假设数据通道顺序与 BCI2A_CHANNELS 一致（GetPhysionetMI 已校验通道名集合）。
    """

    def __init__(self, channel_names=BCI2A_CHANNELS):
        super().__init__()
        prior = torch.tensor([MOTOR_CHANNEL_PRIOR.get(n, 0.0) for n in channel_names])
        keep_mask = prior > 0
        self.register_buffer('keep_mask', keep_mask)
        self.register_buffer('fixed_mask', prior.clone())  # 删除位恒 0
        self.n_channels = len(channel_names)
        self.n_kept = int(keep_mask.sum().item())
        self.scale = nn.Parameter(prior[keep_mask].clone())  # 仅保留通道可学习

    def forward(self, x):  # (B, 1, C, T)
        w = self.fixed_mask.clone()
        w[self.keep_mask] = self.scale
        return x * w.view(1, 1, -1, 1)


class signal2spd(nn.Module):
    """信号 → SPD 协方差矩阵（支持 Hann 加权，A3）

    返回 (归一化协方差, log(未归一化 trace))：
      - 协方差做 trace 归一化（通道间相对结构），同时把归一化前的 log(trace)
        作为一维信息传出（A1 折衷），由上层拼进切空间特征，模型自行决定用不用；
      - 加窗时（Hann）用加权均值与加权协方差，无偏分母为 Σw²。
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, window=None):
        if x.dim() == 4 and x.shape[-2] == 1:
            x = x.squeeze(-2)
        if x.dim() == 3:
            cov, log_tr = self._cov(x, window)
            return cov, log_tr
        elif x.dim() == 4:  # (B, M, C, T)
            b, m, c, t = x.shape
            cov, log_tr = self._cov(x.reshape(b * m, c, t),
                                    window.unsqueeze(0).expand(b * m, -1) if window is not None else None)
            return cov.view(b, m, c, c), log_tr.view(b, m)
        elif x.dim() == 2:
            return self._cov(x.unsqueeze(0), window)
        raise ValueError(f'signal2spd: unexpected input {x.dim()}D, shape {x.shape}')

    def _cov(self, x, w=None):
        # x: (N, C, T); w: (T,) 或 (N, T)
        if w is None:
            x = x - x.mean(dim=-1, keepdim=True)
            denom = x.shape[-1] - 1
        else:
            w = w if w.dim() == 2 else w.unsqueeze(0)  # (T,) → (1, T); (N, T) 保持不变
            w = w.unsqueeze(1)                        # (1, T) → (1, 1, T); (N, T) → (N, 1, T)
            sw = w.sum(dim=-1, keepdim=True)                       # (N, 1, 1)
            x = x - (x * w).sum(dim=-1, keepdim=True) / sw         # 加权均值
            x = x * w
            denom = (w * w).sum(dim=-1, keepdim=True)              # (N, 1, 1) 无偏分母 Σw²
        cov = x @ x.mT / denom
        trace0 = cov.diagonal(dim1=-2, dim2=-1).sum(-1)            # (N,) 未归一化 trace
        log_trace = trace0.clamp(min=1e-12).log()
        cov = cov / trace0.view(-1, 1, 1)
        eye = torch.eye(cov.shape[-1], device=cov.device).unsqueeze(0)
        return cov + 1e-5 * eye, log_trace


class E2R(nn.Module):
    """欧氏特征 → SPD 流形：Hann 加权滑动时间窗（A2/A3）

    「1s 窗 / 0.5s 步长」滑动窗 + Hann 窗：
      - 所有窗口等长，协方差估计样本数一致；
      - 50% 重叠 + Hann 加权（边缘样本权重低）抑制矩形窗边界截断引入的虚假高频；
      - 加权协方差用加权均值与 Σw² 无偏分母；
      - 尾部不足一个步长的采样点被丢弃（最多 step-1 点）。
    """

    def __init__(self, win_samples, step_samples, use_hann=True):
        super().__init__()
        if step_samples > win_samples:
            raise ValueError('step_samples 应 <= win_samples（否则窗口间存在未覆盖间隙）')
        if win_samples < 2:
            raise ValueError('win_samples 至少为 2，否则协方差不可计算')
        self.win = int(win_samples)
        self.step = int(step_samples)
        if use_hann:
            self.register_buffer('window', torch.hann_window(self.win))
        else:
            self.window = None
        self.signal2spd = signal2spd()

    @staticmethod
    def n_windows(T, win, step):
        """给定时间长度 T，返回滑动窗口数"""
        return (T - win) // step + 1

    def forward(self, x):
        # 输入 (B, C, T) 或 conv 输出 (B, C, 1, T)
        if x.dim() == 4:
            x = x.squeeze(-2)
        b, c, t = x.shape
        if t < self.win:
            raise ValueError(f'时间长度 {t} 小于窗长 {self.win}，无法滑窗')
        patches = x.unfold(dimension=-1, size=self.win, step=self.step)  # (B, C, M, win)
        patches = patches.permute(0, 2, 1, 3).contiguous()               # (B, M, C, win)
        return self.signal2spd(patches, window=self.window)              # (B,M,C,C), (B,M)


class AttentionManifold(nn.Module):
    """流形注意力：稀疏滑动窗口 + 锚点全局 + GQA + 共享正交基谱幂投影

    投影（A7+A13，共享基）:
      Q/K/V 共享同一组 Stiefel 正交变换 W（SPDTransform），各自通过 SPDPower 学
      一个逐特征值谱幂 w_q/w_k/w_v：Q = W^T X^{w_q} W 等。三个独立正交矩阵在
      小数据下易过拟合被试特异旋转；共享基后可学参数只剩三个 d 维谱幂向量。

    打分（A4/A5 修复）:
      α_ij = -d²(Q_i, K_j)，softmax(dim=-1) 按 query 行归一化，不做任何压缩变换；
      训练稳定性由外部梯度裁剪（grad_clip）保证。

    稀疏化: 局部带 |i-j|<=local_radius + 锚点双向全局（anchor_stride 控制），
      复杂度 O(m²) → O(m·(2r+1) + a·m)。FlashAttention 不适用（分数来自矩阵
      对数而非欧氏点积），省算力由稀疏对计算承担。

    GQA: n_heads>1 时共享 K/V 变换后按主子块切分给各头（n_kv_heads=1），
      头输出块对角拼回 d_out 维 SPD，下游维度不变。
    """

    def __init__(self, in_embed_size, out_embed_size,
                 local_radius=1, anchor_stride=None, n_heads=1):
        super().__init__()
        if out_embed_size % n_heads != 0:
            raise ValueError(f'out_embed_size={out_embed_size} 必须能被 n_heads={n_heads} 整除')
        self.d_in = in_embed_size
        self.d_out = out_embed_size
        self.local_radius = local_radius
        self.anchor_stride = anchor_stride
        self.n_heads = n_heads
        self.d_head = out_embed_size // n_heads

        # 共享正交基（A7）+ 各分支谱幂（A13 启用 SPDPower）
        self.shared_trans = SPDTransform(self.d_in, self.d_out)
        self.q_power = SPDPower(self.d_out)
        self.k_power = SPDPower(self.d_out)
        self.v_power = SPDPower(self.d_out)

        if n_heads > 1:
            mask = torch.zeros(n_heads, out_embed_size, self.d_head)
            for h in range(n_heads):
                mask[h, h * self.d_head:(h + 1) * self.d_head, :] = torch.eye(self.d_head)
            self.register_buffer('head_mask', mask)

        self._pair_cache = {}

    # ---------- 距离/均值（batched eigh） ----------

    def tensor_log(self, t):
        s, u = torch.linalg.eigh(t)
        s = s.abs().clamp(min=1e-6)
        return u @ torch.diag_embed(torch.log(s)) @ u.mT

    def tensor_exp(self, t):
        s, u = torch.linalg.eigh(t)
        return u @ torch.diag_embed(torch.exp(s)) @ u.mT

    def log_euclidean_distance(self, A, B):
        inner_term = self.tensor_log(A) - self.tensor_log(B)
        return (inner_term ** 2).sum(dim=(-2, -1))

    def LogEuclideanMean(self, weight, cov):
        # A6 说明: softmax 权重行和恒为 1，此时「在参考点 R 的切空间加权平均」
        #   expm(log R + Σ w (log V − log R)) = expm(Σ w log V)
        # 与直接 log 域线性组合数学恒等（log-Euclidean 度量的平移不变性）。
        # 真正的"先平移"收益由上游 SPDNorm 的 running 参考点（log 域批均值）实现，
        # 此处不再引入冗余参考点计算。
        bs, num_p, size = cov.shape[0], cov.shape[1], cov.shape[2]
        cov = self.tensor_log(cov).view(bs, num_p, -1)
        output = weight @ cov
        output = output.view(bs, -1, size, size)
        return self.tensor_exp(output)

    # ---------- 投影 / 头切分 ----------

    def _project(self, x):
        base = self.shared_trans(x)
        return self.q_power(base), self.k_power(base), self.v_power(base)

    def _split_heads(self, X):
        if self.n_heads == 1:
            return [X]
        return [self.head_mask[h].T @ (X @ self.head_mask[h]) for h in range(self.n_heads)]

    def _merge_heads(self, outs):
        if self.n_heads == 1:
            return outs[0]
        output = None
        for h, o in enumerate(outs):
            block = self.head_mask[h] @ o @ self.head_mask[h].T
            output = block if output is None else output + block
        return output

    # ---------- 稀疏注意力对 ----------

    def _anchors(self, m):
        if self.anchor_stride is None:
            return [0]
        return sorted(set(list(range(0, m, self.anchor_stride)) + [m - 1]))

    def _pairs(self, m):
        if m not in self._pair_cache:
            pairs = set()
            for i in range(m):
                for j in range(max(0, i - self.local_radius), min(m - 1, i + self.local_radius) + 1):
                    pairs.add((i, j))
            for a in self._anchors(m):
                for j in range(m):
                    pairs.add((a, j))
                    pairs.add((j, a))
            pairs = sorted(pairs)
            self._pair_cache[m] = (
                torch.tensor([p[0] for p in pairs], dtype=torch.long),
                torch.tensor([p[1] for p in pairs], dtype=torch.long),
            )
        return self._pair_cache[m]

    def _head_prob(self, Q, K):
        bs, m = Q.shape[0], Q.shape[1]
        qi, kj = self._pairs(m)
        qi, kj = qi.to(Q.device), kj.to(Q.device)
        energy = self.log_euclidean_distance(Q[:, qi], K[:, kj])
        scores = Q.new_full((bs, m, m), float('-inf'))
        scores[:, qi, kj] = -energy
        return torch.softmax(scores, dim=-1)

    # ---------- 前向 ----------

    def forward(self, x, shape=None):
        if len(x.shape) == 3 and shape is not None:
            x = x.view(shape[0], shape[1], self.d_in, self.d_in)
        x = x.to(torch.float)

        Qf, Kf, Vf = self._project(x)
        Qh, Kh, Vh = self._split_heads(Qf), self._split_heads(Kf), self._split_heads(Vf)

        outs = [self.LogEuclideanMean(self._head_prob(Qh[h], Kh[h]), Vh[h])
                for h in range(self.n_heads)]
        output = self._merge_heads(outs)

        shape = list(output.shape[:2])
        shape.append(-1)
        output = output.contiguous().view(-1, self.d_out, self.d_out)
        return output, shape

    @torch.no_grad()
    def attention_prob(self, x):
        """可视化用：返回 (bs, m, m) 注意力概率（多头平均，稀疏未计算处为 0）"""
        Qf, Kf, _ = self._project(x)
        Qh, Kh = self._split_heads(Qf), self._split_heads(Kf)
        probs = [self._head_prob(Qh[h], Kh[h]) for h in range(self.n_heads)]
        return torch.stack(probs).mean(0)


class mAtt_bci(nn.Module):
    """滑动窗稀疏流形注意力模型（PhysionetMI / BCI2A 22 通道）

    结构: 解剖通道加权(A9) → conv FE + InstanceNorm(A8) → Hann 滑动窗 E2R(A2/A3,
    返回协方差与 log-trace) → SPDNorm 流形仿射对齐(A12) → 稀疏流形注意力(A4/A5/A7/A13)
    → 平滑整流(A11) → 切空间 + log-trace 拼接(A1) → Linear。
    """

    def __init__(self, win_samples, step_samples, n_classes=4,
                 d_attn_in=20, d_attn_out=18,
                 local_radius=1, anchor_stride=None, n_heads=1,
                 n_windows=None, use_hann=True):
        super().__init__()
        # A9: 解剖先验通道加权/删除（数据 22 通道顺序 = BCI2A_CHANNELS）
        self.channel_weight = ChannelPriorWeighting(BCI2A_CHANNELS)

        # FE: (bs,1,22,T) → (bs,22,1,T) → (bs,d_attn_in,1,T)
        self.conv1 = nn.Conv2d(1, 22, (22, 1))
        self.In1 = nn.InstanceNorm2d(22, affine=True)   # A8: IN 替代 BN，消除跨被试统计漂移
        self.conv2 = nn.Conv2d(22, d_attn_in, (1, 12), padding=(0, 6))
        self.In2 = nn.InstanceNorm2d(d_attn_in, affine=True)

        # E2R: Hann 滑动窗
        self.ract1 = E2R(win_samples, step_samples, use_hann=use_hann)
        # A12: 流形仿射对齐（log 域平移 + 可学习谱缩放），先对齐再进注意力
        self.spdnorm = SPDNorm(d_attn_in)

        self.att2 = AttentionManifold(
            d_attn_in, d_attn_out,
            local_radius=local_radius, anchor_stride=anchor_stride, n_heads=n_heads,
        )
        self.ract2 = SPDRectified()  # A11: 平滑整流
        self.tangent = SPDTangentSpace(d_attn_out)
        self.flat = nn.Flatten()
        self.dropout = nn.Dropout(0.5)  # 对抗过拟合

        # 每窗口特征 = 切空间向量 d(d+1)/2 维 + 1 维 log-trace（A1 折衷拼接）
        self.feat_per_window = d_attn_out * (d_attn_out + 1) // 2 + 1
        if n_windows is None:
            self.linear = nn.LazyLinear(n_classes, bias=True)
        else:
            self.linear = nn.Linear(n_windows * self.feat_per_window, n_classes, bias=True)

    def forward(self, x):
        x = self.channel_weight(x)
        x = self.In1(self.conv1(x))
        x = self.In2(self.conv2(x))
        spd, log_trace = self.ract1(x)          # (B,M,C,C), (B,M)
        spd = self.spdnorm(spd)                 # A12
        x, shape = self.att2(spd)
        x = self.ract2(x)
        x = self.tangent(x)
        x = x.view(shape[0], shape[1], -1)
        x = self.flat(x)
        x = torch.cat([x, log_trace], dim=1)    # A1: 拼接 log-trace，模型自行决定用不用
        x = self.dropout(x)                     # 对抗过拟合
        return self.linear(x)

    def get_features(self, x):
        """分类头之前的特征（与 forward 共享主干，用于 silhouette 等）"""
        x = self.channel_weight(x)
        x = self.In1(self.conv1(x))
        x = self.In2(self.conv2(x))
        spd, log_trace = self.ract1(x)
        spd = self.spdnorm(spd)
        x, shape = self.att2(spd)
        x = self.ract2(x)
        x = self.tangent(x)
        return torch.cat([self.flat(x.view(shape[0], shape[1], -1)), log_trace], dim=1)


class mAtt_mamem(nn.Module):
    def __init__(self, win_samples, step_samples, n_classes=5,
                 d_attn_in=15, d_attn_out=12,
                 local_radius=1, anchor_stride=None, n_heads=1,
                 n_windows=None, use_hann=True):
        super().__init__()
        self.conv1 = nn.Conv2d(1, d_attn_in, (8, 1))
        self.In1 = nn.InstanceNorm2d(d_attn_in, affine=True)
        self.conv2 = nn.Conv2d(d_attn_in, d_attn_in, (1, 36), padding=(0, 18))
        self.In2 = nn.InstanceNorm2d(d_attn_in, affine=True)

        self.ract1 = E2R(win_samples, step_samples, use_hann=use_hann)
        self.spdnorm = SPDNorm(d_attn_in)
        self.att2 = AttentionManifold(
            d_attn_in, d_attn_out,
            local_radius=local_radius, anchor_stride=anchor_stride, n_heads=n_heads,
        )
        self.ract2 = SPDRectified()
        self.tangent = SPDTangentSpace(d_attn_out)
        self.flat = nn.Flatten()
        self.feat_per_window = d_attn_out * (d_attn_out + 1) // 2 + 1
        if n_windows is None:
            self.linear = nn.LazyLinear(n_classes, bias=True)
        else:
            self.linear = nn.Linear(n_windows * self.feat_per_window, n_classes, bias=True)

    def forward(self, x):
        x = self.In1(self.conv1(x))
        x = self.In2(self.conv2(x))
        spd, log_trace = self.ract1(x)
        spd = self.spdnorm(spd)
        x, shape = self.att2(spd)
        x = self.ract2(x)
        x = self.tangent(x)
        x = x.view(shape[0], shape[1], -1)
        x = self.flat(x)
        x = torch.cat([x, log_trace], dim=1)
        return self.linear(x)


class mAtt_cha(nn.Module):
    def __init__(self, win_samples, step_samples, n_classes=2,
                 d_attn_in=16, d_attn_out=8,
                 local_radius=1, anchor_stride=None, n_heads=1,
                 n_windows=None, use_hann=True):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 22, (56, 1))
        self.In1 = nn.InstanceNorm2d(22, affine=True)
        self.conv2 = nn.Conv2d(22, d_attn_in, (1, 64), padding=(0, 32))
        self.In2 = nn.InstanceNorm2d(d_attn_in, affine=True)

        self.ract1 = E2R(win_samples, step_samples, use_hann=use_hann)
        self.spdnorm = SPDNorm(d_attn_in)
        self.att2 = AttentionManifold(
            d_attn_in, d_attn_out,
            local_radius=local_radius, anchor_stride=anchor_stride, n_heads=n_heads,
        )
        self.ract2 = SPDRectified()
        self.tangent = SPDTangentSpace(d_attn_out)
        self.flat = nn.Flatten()
        self.feat_per_window = d_attn_out * (d_attn_out + 1) // 2 + 1
        if n_windows is None:
            self.linear = nn.LazyLinear(n_classes, bias=True)
        else:
            self.linear = nn.Linear(n_windows * self.feat_per_window, n_classes, bias=True)

    def forward(self, x):
        x = self.In1(self.conv1(x))
        x = self.In2(self.conv2(x))
        spd, log_trace = self.ract1(x)
        spd = self.spdnorm(spd)
        x, shape = self.att2(spd)
        x = self.ract2(x)
        x = self.tangent(x)
        x = x.view(shape[0], shape[1], -1)
        x = self.flat(x)
        x = torch.cat([x, log_trace], dim=1)
        return self.linear(x)
