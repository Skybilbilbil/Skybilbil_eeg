import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mAtt.spd import SPDTransform, SPDTangentSpace, SPDRectified, SPDNorm


class _PerSubjectConv2d(nn.Module):
    """逐被试独立空间滤波器组

    每个被试（sub_id 0..N-1）拥有一组独立的 (c1_out, 1, 22, 1) 空间卷积权重，
    forward 时按 batch 内每个样本的被试归属路由到对应滤波器组。
    数学动机：跨被试 conv1 塌缩源于"共享一个 W 无法同时拟合 9 个近乎正交的
    空间方向（子空间平均角 60.4°）→ 梯度互斥 → wd 指数压缩"。per-subject 把
    梯度隔离到各自滤波器（𝔼[∂L/∂W_s] 不再互斥），检验"空间滤波本身是否能
    以逐被试形式应对通道差异"。
    ⚠ 代价：identity shortcut 风险 —— 每个 W_s 可能退化成编码"被试身份"而
    非空间特征；用 wrong-sub 测试（路由给别的被试滤波器看 acc 是否断崖）检验。
    """

    def __init__(self, n_subjects, out_channels, kernel=22):
        super().__init__()
        self.n_subjects = n_subjects
        self.weight = nn.Parameter(torch.empty(n_subjects, out_channels, 1, kernel, 1))
        self.bias = nn.Parameter(torch.zeros(n_subjects, out_channels))
        # 每个滤波器组按 nn.Conv2d 默认方式独立初始化
        for s in range(n_subjects):
            nn.init.kaiming_uniform_(self.weight[s], a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[s])
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias[s], -bound, bound)

    def forward(self, x, sub_ids):
        # x: (B,1,22,T); sub_ids: (B,) int∈[0,n_subjects)
        # 空间核 (1,22,1) 沿通道维全连接,逐样本不同核 → 用 gather 取出每样本所属
        # 被试权重后以一次 einsum 完成全部样本,替代 for s in range(N) 的逐组
        # kernel 调用(103 组小 conv 的 kernel 启动开销是跨被试训练的主耗时)。
        # out[b,o,t] = Σ_c W[b][o,c]·x[b,c,t] + bias[b][o]  ← 对应线性层公式 W·x+b
        wb = self.weight[sub_ids]               # (B, Co,1,22,1) → squeeze 轴2/4
        wb = wb.squeeze(2).squeeze(3)           # (B, Co, 22)
        bb = self.bias[sub_ids]                 # (B, Co)
        xb = x.squeeze(1)                       # (B, 22, T)
        return torch.einsum('bct,boc->bot', xb, wb).unsqueeze(2) + bb[:, :, None, None]
        # 注:unsqueeze(2) 得 (B, Co, 1, T),与逐组 F.conv2d 输出布局一致(无 padding)


class _PerSubjectConv2dTemporal(nn.Module):
    """逐被试独立时序（频率）滤波器组

    与 _PerSubjectConv2d 同构，作用于 conv2：每个被试一组独立的
    (c_out, c_in, 1, k) 时序卷积核，forward 按 batch 内 sub_id 路由。
    数学动机：conv2 塌缩源于 μ 峰频率个体漂移（7.8-11.7Hz）导致共享
    卷积核的梯度互斥；per-subject 把梯度隔离到各自频率滤波器。
    """

    def __init__(self, n_subjects, in_channels, out_channels, kernel=12, padding=6):
        super().__init__()
        self.n_subjects = n_subjects
        self.padding = (0, padding)
        self.weight = nn.Parameter(torch.empty(n_subjects, out_channels, in_channels, 1, kernel))
        self.bias = nn.Parameter(torch.zeros(n_subjects, out_channels))
        for s in range(n_subjects):
            nn.init.kaiming_uniform_(self.weight[s], a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[s])
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias[s], -bound, bound)

    def forward(self, x, sub_ids):
        # x: (B, C_in, 1, T); sub_ids: (B,)
        # 一维时序核 (1,K),pad 每边 K//2=6 → 输出长 T+1。逐样本不同核,
        # 同样用 gather + 固定 K 次 einsum 完成,替代 for s in range(N) 逐组调用。
        # out[b,o,t] = Σ_{c,k} w[b,o,c,k]·x_pad[b,c,t+k]  ← F.conv2d 卷积定义
        K = self.weight.shape[-1]
        pad = int(self.padding[1])                # (0,6) → 时间维每边 6
        xb = x.squeeze(2)                         # (B, C_in, T)
        xp = F.pad(xb, (pad, pad))                # (B, C_in, T+2·pad)
        L = xb.shape[-1] + 1                      # 输出长 = T+1 (与 F.conv2d 一致)
        wb = self.weight[sub_ids].squeeze(3)      # (B, Co, C_in, K)
        bb = self.bias[sub_ids]                   # (B, Co)
        out = torch.einsum('bct,boc->bot', xp[:, :, :L], wb[:, :, :, 0])
        for k in range(1, K):
            out = out + torch.einsum('bct,boc->bot', xp[:, :, k:k + L], wb[:, :, :, k])
        return out.unsqueeze(2) + bb[:, :, None, None]


class _Conv2dTemporalSeparable(nn.Module):
    """时序深度可分离卷积: depthwise(1×k, groups=C) + pointwise(1×1)

    替代全连接时序 Conv2d(C_in, C_out, (1,k))：FLOPs 从 O(C_in·C_out·k·T)
    降为 O(C_in·k·T + C_out·C_in·T)（MobileNet 风格，22→8·12 下约 -5×），
    感受野不变。参数以叶子形式挂在本模块（不注册子模块），
    使 conv2 仍是叶子层，特征钩子可整体记录其输出。
    """

    def __init__(self, in_channels, out_channels, kernel_size=12, padding=6):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.dw_weight = nn.Parameter(torch.empty(in_channels, 1, 1, kernel_size))
        self.dw_bias = nn.Parameter(torch.zeros(in_channels))
        self.pw_weight = nn.Parameter(torch.empty(out_channels, in_channels, 1, 1))
        self.pw_bias = nn.Parameter(torch.zeros(out_channels))
        # 初始化与 nn.Conv2d 默认一致
        nn.init.kaiming_uniform_(self.dw_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.dw_weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.dw_bias, -bound, bound)
        nn.init.kaiming_uniform_(self.pw_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.pw_weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.pw_bias, -bound, bound)

    def forward(self, x):
        x = F.conv2d(x, self.dw_weight, self.dw_bias,
                     padding=(0, self.padding), groups=x.shape[1])
        return F.conv2d(x, self.pw_weight, self.pw_bias)


class _Conv2dTemporalMultiScale(nn.Module):
    """多尺度并联时间滤波器组（经路径A验证：跨被试有效性成立）

    针对单一 1×12 卷积核对特定频率（如 10.5Hz Mu 节律）强频域选择性、
    个体间峰值频率漂移（8.5–10.5Hz）与潜伏期漂移（200–500ms）导致的
    输出特征能量系统性偏移，用多个并联不同核长/空洞率的时序卷积捕捉
    不同尺度时序结构，扩大频带覆盖：

      分支 核长  空洞  捕捉目标
      a    1×8    1     快速时序变化/高频
      b    1×16   1     标准 Mu/Beta 节律
      c    1×32   1     低频慢波过渡
      d    1×16   2     大感受野，零增参扩宽频率响应

    参数化（路径A/B 共用同一模块）：
      - per_branch=每分支输出通道数；总输出 = per_branch·n_branch。
      - 当总输出 != out_channels 时自动补一个 1×1 卷积投影回 out_channels，
        使后端（E2R/SPD/注意力）接口不受分支通道数影响。
      - specs 可传入不同核长组合（B 用更贴近 EEG 频段的核长）。
    时间长度经左右非对称 padding 保持与输入一致。
    """

    def __init__(self, in_channels, out_channels,
                 specs=((8, 1), (16, 1), (32, 1), (16, 2)),
                 per_branch=None):
        super().__init__()
        self.specs = specs
        self.n_branch = len(specs)
        if per_branch is None:
            assert out_channels % self.n_branch == 0, \
                'out_channels 需能被分支数整除以便均分通道'
            per_branch = out_channels // self.n_branch
        self.per = per_branch
        # 各分支独立卷积，挂在叶子层保特征钩子整体记录
        self.convs = nn.ModuleList()
        for k, d in specs:
            c = nn.Conv2d(in_channels, per_branch, (1, k), dilation=d)
            self.convs.append(c)
        # 分支拼接总通道
        self.concat_channels = per_branch * self.n_branch
        # 需要压回 out_channels 时加 1×1 投影（后端接口不变）
        if self.concat_channels != out_channels:
            self.project = nn.Conv2d(self.concat_channels, out_channels, (1, 1))
        else:
            self.project = nn.Identity()

    def forward(self, x):  # (B, C_in, 1, T) → (B, C_out, 1, T)，T 不变
        outs = []
        for conv, (k, d) in zip(self.convs, self.specs):
            rf = (k - 1) * d + 1           # 感受野长度
            pad_l = (rf - 1) // 2          # 左右非对称 padding 使 T 保持
            pad_r = (rf - 1) - pad_l
            xi = F.pad(x, (pad_l, pad_r, 0, 0))
            outs.append(conv(xi))
        cat = torch.cat(outs, dim=1)
        return self.project(cat)


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

# 生理学先验通道权重（方案一）：
# 依据 10-20 系统相对 Cz=1.0 的运动皮层躯体拓扑（C3 右手 / C4 左手 / Cz 双足
# 运动带主峰）与 SMA/运动前区（FCz）的重要性（Wang, Hong & Gao 的 C3-FCz /
# C4-FCz 双极最优布局实证），C5/C6 属颞区、远离手部运动带故权重最低。
PHYSIO_CHANNEL_PRIOR = {
    'Cz': 1.00,                                    # 双足运动带主峰
    'C3': 0.95, 'C4': 0.95,                        # 手部运动带主峰（镜像对称）
    'C1': 0.88, 'C2': 0.88,                        # 手带内侧邻近
    'FCz': 0.82,                                   # 辅助运动区 SMA
    'CPz': 0.78,                                   # 中央后带中线
    'FC1': 0.74, 'FC2': 0.74, 'CP1': 0.74, 'CP2': 0.74,  # 中央前/后带
    'FC3': 0.60, 'FC4': 0.60, 'CP3': 0.60, 'CP4': 0.60,  # 手带外侧邻近
    'C5': 0.30, 'C6': 0.30,                        # 颞区，远离运动带，权重最低
    'Fz': 0.0, 'P1': 0.0, 'Pz': 0.0, 'P2': 0.0, 'POz': 0.0,  # 远区（等效删除）
}


class ChannelPriorWeighting(nn.Module):
    """解剖先验通道加权（A9）

    - 权重为 0 的通道等效删除（Fz/P/Pz/POz，共 5 个，离感觉运动区远）；
    - 其余通道默认以解剖先验（MOTOR_CHANNEL_PRIOR）初始化为可学习缩放因子，
      网络可自行调整关注强度；
    - gaussian=True 时改为固定生理学先验掩膜（PHYSIO_CHANNEL_PRIOR，不参与训练），
      以 C3/Cz/C4 运动带为主峰、SMA 次峰、颞区压低，强制网络聚焦中央区，
      避免跨被试时通过"关闭通道"（scale→0）逃避空间变异；
    - 假设数据通道顺序与 BCI2A_CHANNELS 一致（GetPhysionetMI 已校验通道名集合）。
    """

    def __init__(self, channel_names=BCI2A_CHANNELS, gaussian=False,
                 gaussian_sigma=0.62, gaussian_centers=('C3', 'Cz', 'C4')):
        super().__init__()
        prior = torch.tensor([MOTOR_CHANNEL_PRIOR.get(n, 0.0) for n in channel_names])
        keep_mask = prior > 0
        self.register_buffer('keep_mask', keep_mask)
        self.register_buffer('fixed_mask', prior.clone())  # 删除位恒 0
        self.n_channels = len(channel_names)
        self.n_kept = int(keep_mask.sum().item())
        if gaussian:
            # 固定生理学先验掩膜（相对范数，Cz=1.0），远区仍硬删除
            full = torch.tensor(
                [PHYSIO_CHANNEL_PRIOR.get(n, 0.0) for n in channel_names])
            self.register_buffer('fixed_mask', full.clone())
            self.register_buffer('scale', full[keep_mask].clone())  # 常数乘子（非 Parameter）
            # gaussian 模式下 scale 为固定 buffer：一次性合成最终权重 (1,C,1)，
            # forward 直接广播相乘，避免每步 clone+索引赋值。
            # persistent=False：不入 state_dict，旧 checkpoint 严格加载不受影响。
            _w = full.clone()
            self.register_buffer('final_weight', _w.view(1, 1, -1, 1), persistent=False)
        else:
            self.scale = nn.Parameter(prior[keep_mask].clone())  # 仅保留通道可学习

    def forward(self, x):  # (B, 1, C, T)
        if hasattr(self, 'final_weight'):
            return x * self.final_weight
        w = self.fixed_mask.clone()
        w[self.keep_mask] = self.scale
        return x * w.view(1, 1, -1, 1)


class signal2spd(nn.Module):
    """信号 → SPD 协方差矩阵（支持 Hann 加权，A3）

    返回 trace 归一化的协方差矩阵（只保留通道间相对结构，不带绝对能量信息）：
      - 加窗时（Hann）用加权均值与加权协方差，无偏分母为 Σw²。
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, window=None):
        if x.dim() == 4 and x.shape[-2] == 1:
            x = x.squeeze(-2)
        if x.dim() == 3:
            return self._cov(x, window)
        elif x.dim() == 4:  # (B, M, C, T)
            b, m, c, t = x.shape
            cov = self._cov(x.reshape(b * m, c, t),
                            window.unsqueeze(0).expand(b * m, -1) if window is not None else None)
            return cov.view(b, m, c, c)
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
        cov = cov / cov.diagonal(dim1=-2, dim2=-1).sum(-1).view(-1, 1, 1)
        eye = torch.eye(cov.shape[-1], device=cov.device).unsqueeze(0)
        return cov + 1e-5 * eye


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
        return self.signal2spd(patches, window=self.window)              # (B,M,C,C)


class ManifoldChange(nn.Module):
    """流形上"相对变化"特征（SPD 原域运算，结果仍为 SPD）

    插在 spdnorm 之后、注意力之前，把绝对 SPD 换成"相对静息/相邻窗的变化"，
    目标是去除被试绝对位主导（黎曼诊断：被试聚合比 1.33 ≫ 类别 1.11），
    让下游看到的是"从静息到想象的变化过程"而非个体绝对样子。

    mode：
      None       恒等（不影响原模型）
      le_win0    Y_t = exp(log X_t − log X_0)：每样本以自己窗0(静息)为参考的 LE 变化
      le_diff1   Y_t = exp(log X_{t+1} − log X_t)：一阶时间差分
      le_diff2   Y_t = exp(二阶时间差分)：加速度型变化，高阶噪声大
    全部用 Log-Euclidean：切空间做差、指数映射回流形，结果仍为正定 SPD。
    """

    def __init__(self, mode=None):
        super().__init__()
        if isinstance(mode, str) and mode.strip().lower() in ('none', ''):
            mode = None           # 命令行传入的字符串 'None' → 视为恒等
        if mode not in (None, 'le_win0', 'le_diff1', 'le_diff2'):
            raise ValueError(f"manifold_change 须为 None/'le_win0'/'le_diff1'/'le_diff2'，收到 {mode!r}")
        self.mode = mode

    @staticmethod
    def _logm(x):
        x = x.double()
        s, u = torch.linalg.eigh(x)
        s = s.abs().clamp(min=1e-6)
        return (u @ torch.diag_embed(s.log()) @ u.transpose(-1, -2)).float()

    @staticmethod
    def _expm(y):
        y = ((y + y.transpose(-1, -2)) / 2).double()
        s, u = torch.linalg.eigh(y)
        return (u @ torch.diag_embed(s.exp()) @ u.transpose(-1, -2)).float()

    def forward(self, spd):
        if self.mode is None:
            return spd
        L = self._logm(spd)                     # (B,M,d,d)
        if self.mode == 'le_win0':
            L0 = L[:, 0:1]                      # (B,1,d,d) 每样本窗0=自己的静息参考
            Y = L - L0
        elif self.mode == 'le_diff1':
            Y = L[:, 1:] - L[:, :-1]
            Y = torch.cat([Y, Y[:, -1:]], dim=1)  # 保持 M 窗（末窗复制）
        elif self.mode == 'le_diff2':
            Y = L[:, 2:] - 2 * L[:, 1:-1] + L[:, :-2]
            Y = torch.cat([Y[:, :1], Y, Y[:, -1:]], dim=1)  # 端窗复制补齐 M
        return self._expm(Y)


class AttentionManifold(nn.Module):
    """流形注意力：稀疏滑动窗口 + 锚点全局 + GQA + 独立正交投影

    投影（原始 MAtt 做法）:
      Q/K/V 各用一套独立的 Stiefel 正交变换（SPDTransform）：
      Q = W_q^T X W_q，K = W_k^T X W_k，V = W_v^T X W_v。
      三套独立基互不共享，不存在"q≡k"坍缩机制。

    打分（A4/A5 修复）:
      α_ij = -d²(Q_i, K_j)，softmax(dim=-1) 按 query 行归一化，不做任何压缩变换；
      训练稳定性由外部梯度裁剪（grad_clip）保证。

    稀疏化: 局部带 |i-j|<=local_radius；anchor_stride=None 时无锚点（纯局部，
      用户要求删除锚点），设整数 k 则每 k 窗一个锚点（双向全局）。

    GQA: n_heads>1 时共享 K/V 变换后按主子块切分给各头（n_kv_heads=1），
      头输出块对角拼回 d_out 维 SPD，下游维度不变。
    """

    def __init__(self, in_embed_size, out_embed_size,
                 local_radius=1, anchor_stride=None, n_heads=1,
                 attn_change=None, attn_prior=None, score_original=True):
        super().__init__()
        if out_embed_size % n_heads != 0:
            raise ValueError(f'out_embed_size={out_embed_size} 必须能被 n_heads={n_heads} 整除')
        self.d_in = in_embed_size
        self.d_out = out_embed_size
        self.local_radius = local_radius
        self.anchor_stride = anchor_stride
        self.n_heads = n_heads
        self.d_head = out_embed_size // n_heads
        # 打分映射: True=原文 1/(1+log(1+d²))+softmax(dim=-2)+permute; False=负距离 softmax(dim=-1)
        self.score_original = score_original
        # 变化感知模式：None=原始(相似匹配)；'ref'=显式静息参考(A)；'diff'=时间差分打分(B)
        if attn_change not in (None, 'ref', 'diff'):
            raise ValueError(f"attn_change 必须为 None/'ref'/'diff'，收到 {attn_change!r}")
        self.attn_change = attn_change
        # 可学习注意力先验种子（attn_prior 为可迭代, 长度=窗口数）：
        #   作为 nn.Parameter 参与训练，初始值=传入的每窗重要度（弱偏想象段），
        #   之后由每个样本的分类梯度驱动，把注意力从"更像想象的段"的起点
        #   逐步修到真正重要的窗口（不硬编码哪扇窗重要）。
        #   注入方式：在 softmax 前的 score 上对「被关注方(key)窗口 j」加偏置 score+=prior[j]。
        #   None=关闭（默认，不创建参数，旧模型严格加载不受影响）。
        if attn_prior is not None:
            p = torch.tensor(list(attn_prior), dtype=torch.float)
            self.prior_bias = nn.Parameter(p)   # 可学习，梯度驱动调整
        else:
            self.register_buffer('prior_bias', None)

        # 独立正交变换（原始 MAtt 做法）：Q/K/V 三套独立 Stiefel 基，
        # 不存在"q≡k"坍缩机制
        self.q_trans = SPDTransform(self.d_in, self.d_out)
        self.k_trans = SPDTransform(self.d_in, self.d_out)
        self.v_trans = SPDTransform(self.d_in, self.d_out)

        if n_heads > 1:
            mask = torch.zeros(n_heads, out_embed_size, self.d_head)
            for h in range(n_heads):
                mask[h, h * self.d_head:(h + 1) * self.d_head, :] = torch.eye(self.d_head)
            self.register_buffer('head_mask', mask)

        self._pair_cache = {}

    # ---------- 距离/均值（batched eigh） ----------

    def tensor_log(self, t):
        # 对称化并求矩阵对数。CUDA 的 syevd 对某些取值(即使严格正定)存在
        # kernel bug，会拒绝收敛(error code 1 / 反复失败)。已实测单元素 8×8
        # 矩阵 double + 扰动自愈在 GPU 仍失败 → 对 batch 数很小的输入改走
        # CPU LAPACK(torch 在 CPU 的 eigh 极稳且同为可微算子，自动处理反向)。
        t0 = (t + t.mT) / 2
        n_batch = int(t0.numel() / (t0.shape[-1] * t0.shape[-1]))
        do_cpu = t0.is_cuda and n_batch <= 4
        tt = t0.double()
        if do_cpu:
            tt = tt.cpu()
        try:
            s, u = torch.linalg.eigh(tt)
        except Exception:
            print('[DBG] tensor_log eigh failed. n_batch=%d shape=%s finite=%s min=%.3f max=%.3f'
                  % (n_batch, tuple(t.shape), bool(torch.isfinite(t).all().item()),
                     float(t.min().item()), float(t.max().item())), flush=True)
            s, u = torch.linalg.eigh(t0.double().cpu())
        s = s.abs().clamp(min=1e-6)
        out = (u @ torch.diag_embed(torch.log(s)) @ u.mT).float()
        return out.to(t.device)

    def tensor_exp(self, t):
        t0 = (t + t.mT) / 2
        n_batch = int(t0.numel() / (t0.shape[-1] * t0.shape[-1]))
        do_cpu = t0.is_cuda and n_batch <= 4
        tt = t0.double()
        if do_cpu:
            tt = tt.cpu()
        try:
            s, u = torch.linalg.eigh(tt)
        except Exception:
            s, u = torch.linalg.eigh(t0.double().cpu())
        out = (u.float() @ torch.diag_embed(s.exp().float()) @ u.mT.float()).float()
        return out.to(t.device)

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
        # 独立变换：Q/K/V 各自正交基投影，互不共享，无"q≡k"坍缩
        return self.q_trans(x), self.k_trans(x), self.v_trans(x)

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
        # 用户要求删除锚点：anchor_stride=None → 无锚点（纯局部带注意力）
        if self.anchor_stride is None:
            return []
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
        # 去重: 先对全量 Q/K 各做一次 log-Euclidean log（B·M 次 eigh），
        # 再按稀疏对索引取用（原按对 gather 重复 tensor_log，B·n_pairs 次，~3.5× 冗余）
        logQ = self.tensor_log(Q)               # (bs, m, d, d)
        logK = self.tensor_log(K)
        inner = logQ[:, qi] - logK[:, kj]       # (bs, n_pairs, d, d)
        energy = (inner ** 2).sum(dim=(-2, -1))  # (bs, n_pairs)
        scores = Q.new_full((bs, m, m), float('-inf'))
        if self.score_original:
            # 原文打分：相似度 = 1/(1+log(1+d²))（距离→相似度，压缩动态范围），
            # 然后沿 dim=-2（query 维）softmax，再 permute 使行=key、列=query
            sim = 1.0 / (1.0 + torch.log1p(energy))
            scores[:, qi, kj] = sim
            if self.prior_bias is not None:
                scores = scores + self.prior_bias.view(1, 1, -1).to(scores.device)
            # 与原文一致：softmax(dim=-2) → permute(0,2,1)
            p = torch.softmax(scores, dim=-2).permute(0, 2, 1)
            return p
        scores[:, qi, kj] = -energy
        if self.prior_bias is not None:  # 协议先验：提升想象段(key)窗的注意力
            scores = scores + self.prior_bias.view(1, 1, -1).to(scores.device)
        return torch.softmax(scores, dim=-1)

    # ---------- 黎曼差打分（方向B：Log-Euclidean 切空间参考差，以窗0静息为锚） ----------

    def _head_prob_ref(self, Q, K):
        """以窗0为参考点的切空间差值打分（LE 度量，梯度线性稳定）。

        把每个窗投影到恒等参考的切空间（log，即 LE 对数映射），再相对窗0静息
        基线取切向量差：Δ_i = log X_i − log X_0。Δ 是切空间元素（实对称矩阵，
        不是新 SPD），打分 = -‖Δ^Q_i - Δ^K_j‖² 衡量两窗「相对静息的改进方向」
        的距离。纯矩阵减 + 谱 log，无 P^{1/2} 白化，回避矩阵根反导的条件数爆炸。

        注：真仿射不变黎曼对数映射 Log_P(X)=P^{1/2}log(P^{-1/2}XP^{-1/2})P^{1/2}
        在手写 backward 经数值验证在本数据上因「双 eigh + 白化」复合条件数爆 nan，
        故此处采用 Log-Euclidean 切向量差——同为流形切向量，仅度量不同，数值稳定。
        """
        bs, m = Q.shape[0], Q.shape[1]
        logQ = self.tensor_log(Q)                  # (bs,m,d,d) 恒等参考切空间
        logK = self.tensor_log(K)
        dQ = logQ - logQ[:, 0:1]                   # 相对窗0静息基线
        dK = logK - logK[:, 0:1]
        qi, kj = self._pairs(m)
        qi, kj = qi.to(Q.device), kj.to(Q.device)
        dQf = dQ.reshape(bs, m, self.d_out, self.d_out)
        dKf = dK.reshape(bs, m, self.d_out, self.d_out)
        inner = dQf[:, qi] - dKf[:, kj]            # (bs,n_pairs,d,d)
        energy = (inner ** 2).sum(dim=(-2, -1))
        scores = Q.new_full((bs, m, m), float('-inf'))
        scores[:, qi, kj] = -energy
        if self.prior_bias is not None:  # 协议先验：提升想象段(key)窗的注意力
            scores = scores + self.prior_bias.view(1, 1, -1).to(scores.device)
        return torch.softmax(scores, dim=-1)

    # ---------- 前向 ----------

    def forward(self, x, shape=None):
        if len(x.shape) == 3 and shape is not None:
            x = x.view(shape[0], shape[1], self.d_in, self.d_in)
        x = x.to(torch.float)

        Qf, Kf, Vf = self._project(x)
        Qh, Kh, Vh = self._split_heads(Qf), self._split_heads(Kf), self._split_heads(Vf)

        outs = []
        for h in range(self.n_heads):
            Q, K, V = Qh[h], Kh[h], Vh[h]
            if self.attn_change is None:
                # 原始：相似匹配打分 + V 直接加权
                w = self._head_prob(Q, K)
                outs.append(self.LogEuclideanMean(w, V))
            elif self.attn_change == 'ref':
                # 方向A：显式静息参考（窗0作基线）。
                #   打分保持原相似匹配；仅把 value 聚合为「相对窗0静息基线的 log 域嵌值」：
                #   out_i = expm( Σ_j w[i,j]·(logV_j − logV_0) )，
                #   使后续窗的输出携带「偏离静息」的变化结构；窗0行强制保留自身基线。
                bs, m = Q.shape[0], Q.shape[1]
                w = self._head_prob(Q, K)
                logV = self.tensor_log(V).view(bs, m, -1)          # (B,M,d·d)
                base = logV[:, 0:1]                                # 静息窗0基线 (B,1,d·d)
                rel = logV - base                                  # (B,M,d·d) 相对静息的变化
                out_log = w @ rel                                  # Σ w·(logV_j − logV_0)
                out_log[:, 0, :] = base[:, 0, :]                   # 窗0保留静息基线
                outs.append(self.tensor_exp(out_log.view(bs, -1, self.d_out, self.d_out)))
            elif self.attn_change == 'diff':
                # 方向B：真黎曼差打分（仿射不变，参考窗0静息）。
                #   打分用相对窗0的黎曼对数映射切向量距离 score_ij=-‖ΔQ_i-ΔK_j‖²，
                #   凸出「偏离静息」的变化；value 仍走原 LogEuclideanMean(_,V)，
                #   即只换打分，满足"流形上的差分用黎曼方法"。
                w = self._head_prob_ref(Q, K)
                outs.append(self.LogEuclideanMean(w, V))
        output = self._merge_heads(outs)

        shape = list(output.shape[:2])
        shape.append(-1)
        output = output.contiguous().view(-1, self.d_out, self.d_out)
        return output, shape

    @torch.no_grad()
    def attention_prob(self, x):
        """可视化用：返回 (bs, m, m) 注意力概率（多头平均，稀疏未计算处为 0）。"""
        Qf, Kf, _ = self._project(x)
        Qh, Kh = self._split_heads(Qf), self._split_heads(Kf)
        if self.attn_change == 'diff':
            probs = [self._head_prob_ref(Qh[h], Kh[h]) for h in range(self.n_heads)]
        else:
            probs = [self._head_prob(Qh[h], Kh[h]) for h in range(self.n_heads)]
        return torch.stack(probs).mean(0)


# ============================================================
# 前端 / 后端边界（后端统一模型机制的核心约束）
#   前端 = 卷积特征提取:channel_weight/conv1/In1/conv2/In2（依赖信号形状、通道排布）
#   后端 = E2R 之后的流形注意力分类链:spdnorm/manifold_change/att2/ract2/tangent/
#          flat/dropout/linear（输入是 SPD 协方差，维度只依赖 d_attn_in/d_attn_out/M）
# 换任意前端实现（如 per-subject conv），只要保持 c2_out==d_attn_in、M 不变，
# 后端权重即可复用；train_pair 的 --save_backend/--load_backend/--freeze_backend
# 都以此边界提取/冻结，勿单独改前缀集合（会破坏已有 backend_*.pt 兼容）。
# ============================================================
BACKEND_PREFIXES = ('att2.', 'ract2.', 'tangent.', 'flat.', 'dropout.',
                    'linear.', 'spdnorm.', 'manifold_change.')


def _norm_layer(name):
    """归一化层选择。

    'in' = InstanceNorm2d(历史默认):逐样本归一化,**没有 running stats**,
           所以"在目标数据上重算统计量"这种零标签适配无从下手。
    'bn' = BatchNorm2d:有 running stats,部署时可用目标被试(无标签)的批次
           重算统计量 —— 跨被试最经典、最便宜的适配旋钮。
    """
    return nn.BatchNorm2d if str(name).lower() == 'bn' else nn.InstanceNorm2d


class _SparseAttentionBase(nn.Module):
    """三个模型类共享的稀疏流形注意力主干（构造顺序严格保持 mAtt_bci 上游语义，
    确保 RNG 消耗次序与 state_dict 键名兼容，数值行为逐位不变）。

    主干: [通道加权] → conv1 空间滤波+IN → conv2 时序滤波+IN → E2R 滑动窗协方差
    → SPDNorm 流形对齐 → ManifoldChange → 流形注意力 → 平滑整流 → 切空间 → 分类头。
    子类只把各自差异映射成基类参数；forward 由基类共享，get_features 仅在需要的子类暴露。
    """

    def __init__(self, win_samples, step_samples,
                 n_classes=4, d_attn_in=20, d_attn_out=18,
                 local_radius=1, anchor_stride=None, n_heads=1,
                 n_windows=None, use_hann=True,
                 use_channel_weight=False, n_spatial_filters=22,
                 channel_gaussian=False, gaussian_sigma=0.62,
                 conv1_kernel=22, conv1_out=None,
                 conv2_kernel=12, conv2_pad=6, conv2_out=None,
                 conv2_separable=False, conv2_multiscale=False,
                 conv2_ms_per_branch=None, conv2_ms_specs=None,
                 manifold_change=None, attn_change=None, attn_prior=None,
                 dropout=0.3, score_original=True, per_subject_conv1=0,
                 per_subject_conv2=0, norm_type='in'):
        super().__init__()
        c1_out = int(n_spatial_filters) if conv1_out is None else int(conv1_out)
        c2_out = int(d_attn_in) if conv2_out is None else int(conv2_out)
        # A9: 解剖先验通道加权/删除（数据 22 通道顺序 = BCI2A_CHANNELS）
        self.channel_weight = (ChannelPriorWeighting(
            BCI2A_CHANNELS, gaussian=channel_gaussian, gaussian_sigma=gaussian_sigma)
            if use_channel_weight else None)
        # FE: 空间滤波（conv1），将 22 通道压缩成 c1_out 个空间分量
        # per_subject_conv1>1 时改为逐被试独立滤波器组（梯度隔离，检验空间域自适应）
        self.n_spatial_filters = c1_out
        if int(per_subject_conv1) > 1:
            self.conv1 = _PerSubjectConv2d(int(per_subject_conv1), c1_out, kernel=conv1_kernel)
        else:
            self.conv1 = nn.Conv2d(1, c1_out, (conv1_kernel, 1))
        self.In1 = _norm_layer(norm_type)(c1_out, affine=True)   # A8: IN(默认) 或 BN
        # conv2 实现：全连接时序卷积 / 深度可分离 / 多尺度并联滤波器组
        # per_subject_conv2>1：每被试独立时序滤波（仅作用于全连接路径）
        if conv2_multiscale:
            self.conv2 = _Conv2dTemporalMultiScale(
                c1_out, c2_out,
                specs=(conv2_ms_specs if conv2_ms_specs is not None
                       else ((8, 1), (16, 1), (32, 1), (16, 2))),
                per_branch=conv2_ms_per_branch)
        elif conv2_separable:
            self.conv2 = _Conv2dTemporalSeparable(
                c1_out, c2_out, kernel_size=conv2_kernel, padding=conv2_pad)
        elif int(per_subject_conv2) > 1:
            self.conv2 = _PerSubjectConv2dTemporal(
                int(per_subject_conv2), c1_out, c2_out,
                kernel=conv2_kernel, padding=conv2_pad)
        else:
            self.conv2 = nn.Conv2d(c1_out, c2_out, (1, conv2_kernel),
                                   padding=(0, conv2_pad))
        self.In2 = _norm_layer(norm_type)(c2_out, affine=True)
        # E2R: Hann 滑动窗 → SPD 流形
        self.ract1 = E2R(win_samples, step_samples, use_hann=use_hann)
        # A12: 流形仿射对齐（log 域平移 + 可学习谱缩放），先对齐再进注意力
        self.spdnorm = SPDNorm(c2_out)
        # 流形变化特征（相对静息/差分），None=恒等
        self.manifold_change = ManifoldChange(manifold_change)
        self.att2 = AttentionManifold(
            c2_out, d_attn_out,
            local_radius=local_radius, anchor_stride=anchor_stride, n_heads=n_heads,
            attn_change=attn_change, attn_prior=attn_prior,
            score_original=score_original,
        )
        self.ract2 = SPDRectified()  # A11: 平滑整流
        self.tangent = SPDTangentSpace(d_attn_out)
        self.flat = nn.Flatten()
        self.dropout = nn.Dropout(dropout) if dropout is not None else None
        # 每窗口特征 = 切空间向量 d(d+1)/2 维
        self.feat_per_window = d_attn_out * (d_attn_out + 1) // 2
        if n_windows is None:
            self.linear = nn.LazyLinear(n_classes, bias=True)
        else:
            self.linear = nn.Linear(n_windows * self.feat_per_window, n_classes, bias=True)

    def _shared_backbone(self, x, sub_ids=None):
        if self.channel_weight is not None:
            x = self.channel_weight(x)
        # per-subject conv1：sub_ids 逐样本路由到对应滤波器组；共享 conv1 忽略
        x = self.conv1(x, sub_ids) if isinstance(self.conv1, _PerSubjectConv2d) else self.conv1(x)
        x = self.In1(x)
        # per-subject conv2：同样按 sub_ids 路由；共享 conv2 忽略
        x = self.conv2(x, sub_ids) if isinstance(self.conv2, _PerSubjectConv2dTemporal) else self.conv2(x)
        x = self.In2(x)
        spd = self.ract1(x)                     # (B,M,C,C)
        spd = self.spdnorm(spd)                 # A12
        spd = self.manifold_change(spd)         # 变化特征（None=恒等）
        return spd

    def _features(self, x, sub_ids=None):
        spd = self._shared_backbone(x, sub_ids)
        x, shape = self.att2(spd)
        x = self.ract2(x)
        x = self.tangent(x)
        return self.flat(x.view(shape[0], shape[1], -1))

    def forward(self, x, sub_ids=None):
        x = self._features(x, sub_ids)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.linear(x)


class mAtt_bci(_SparseAttentionBase):
    """滑动窗稀疏流形注意力模型（PhysionetMI / BCI2A 22 通道）

    结构: 解剖通道加权(A9) → conv FE + InstanceNorm(A8) → Hann 滑动窗 E2R(A2/A3,
    返回 trace 归一化协方差) → SPDNorm 流形仿射对齐(A12) → 稀疏流形注意力(A4/A5/A7/A13)
    → 平滑整流(A11) → 切空间 → Linear。
    """

    def __init__(self, win_samples, step_samples, n_classes=4,
                 d_attn_in=20, d_attn_out=18,
                 local_radius=1, anchor_stride=None, n_heads=1,
                 n_windows=None, use_hann=True, conv2_separable=False,
                 conv2_multiscale=False,
                 conv2_ms_per_branch=None, conv2_ms_specs=None,
                 n_spatial_filters=22, channel_gaussian=False,
                 gaussian_sigma=0.62, attn_change=None,
                 attn_prior=None, manifold_change=None, dropout=0.3,
                 per_subject_conv1=0, per_subject_conv2=0, norm_type='in'):
        super().__init__(
            win_samples, step_samples,
            n_classes=n_classes, d_attn_in=d_attn_in, d_attn_out=d_attn_out,
            local_radius=local_radius, anchor_stride=anchor_stride, n_heads=n_heads,
            n_windows=n_windows, use_hann=use_hann,
            use_channel_weight=True, n_spatial_filters=n_spatial_filters,
            channel_gaussian=channel_gaussian, gaussian_sigma=gaussian_sigma,
            conv1_kernel=22, conv1_out=n_spatial_filters,
            conv2_kernel=12, conv2_pad=6, conv2_out=d_attn_in,
            conv2_separable=conv2_separable, conv2_multiscale=conv2_multiscale,
            conv2_ms_per_branch=conv2_ms_per_branch, conv2_ms_specs=conv2_ms_specs,
            manifold_change=manifold_change, attn_change=attn_change,
            attn_prior=attn_prior, dropout=dropout,
            per_subject_conv1=per_subject_conv1,
            per_subject_conv2=per_subject_conv2,
            norm_type=norm_type,
        )

    def get_features(self, x, sub_ids=None):
        """分类头之前的特征（与 forward 共享主干，用于 silhouette 等）"""
        return self._features(x, sub_ids)
