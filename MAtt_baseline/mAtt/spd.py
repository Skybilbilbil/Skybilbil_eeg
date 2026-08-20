import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.autograd import Function
import numpy as np

from mAtt.utils import *
from mAtt import StiefelParameter

class SPDTransform(nn.Module):

    def __init__(self, input_size, output_size):
        super(SPDTransform, self).__init__()
        self.increase_dim = None
        if output_size > input_size:
            self.increase_dim = SPDIncreaseDim(input_size, output_size)
            input_size = output_size
        # 参数在 CPU 创建，由模型级 .cuda()/.to(device) 统一搬移，
        # 避免与输入设备不一致（原 self.device 硬编码在 CUDA 可用时与 CPU 输入冲突）
        self.weight = StiefelParameter(torch.empty(input_size, output_size), requires_grad=True)
        nn.init.orthogonal_(self.weight)

    def forward(self, input):
        output = input
        if self.increase_dim:
            output = self.increase_dim(output)
        # 兼容任意 batch 维度 (..., d_in, d_in)，flatten 后用 2D 权重广播 matmul
        orig_shape = output.shape
        d_out = self.weight.shape[-1]
        output = output.reshape(-1, orig_shape[-2], orig_shape[-1])
        output = self.weight.T @ output @ self.weight
        return output.view(*orig_shape[:-2], d_out, d_out)


class SPDIncreaseDim(nn.Module):

    def __init__(self, input_size, output_size):
        super(SPDIncreaseDim, self).__init__()
        # buffer 注册时不指定设备，register_buffer 会随模型 .to(device) 自动搬移
        self.register_buffer('eye', torch.eye(output_size, input_size))
        add = torch.as_tensor([0] * input_size + [1] * (output_size-input_size), dtype=torch.float32)
        self.register_buffer('add', torch.diag(add))

    def forward(self, input):
        # 兼容任意 batch 维度 (..., d_in, d_in)
        orig_shape = input.shape
        input = input.reshape(-1, orig_shape[-2], orig_shape[-1])
        eye = self.eye.unsqueeze(0).expand(input.size(0), -1, -1)
        add = self.add.unsqueeze(0).expand(input.size(0), -1, -1)
        output = torch.baddbmm(add, eye, torch.bmm(input, eye.transpose(1, 2)))
        return output.view(orig_shape)

class ParametricVectorize(nn.Module):

    def __init__(self, input_size, output_size):
        super(ParametricVectorize, self).__init__()
        self.weight = nn.Parameter(torch.ones(output_size, input_size), requires_grad=True)

    def forward(self, input):
        weight = self.weight.unsqueeze(0)
        weight = weight.expand(input.size(0), -1, -1)
        output = torch.bmm(weight, input)
        output = torch.bmm(output, weight.transpose(1,2))
        output = torch.mean(output, 2)
        return output


class SPDVectorize(nn.Module):

    def __init__(self, input_size):
        super(SPDVectorize, self).__init__()
        row_idx, col_idx = torch.triu_indices(input_size, input_size)
        self.register_buffer('row_idx', torch.LongTensor(row_idx))
        self.register_buffer('col_idx', torch.LongTensor(col_idx))

    def forward(self, input):
        output = input[:, self.row_idx, self.col_idx]
        return output

class SPDUnVectorizeFunction(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        n = int(-.5 + 0.5 * torch.sqrt(1 + 8 * input.size(1)))
        output = input.new(len(input), n, n)
        output.fill_(0)
        mask_upper = torch.triu_indices(n, n)
        mask_diag = torch.diag_indices(n)
        for k, x in enumerate(input):
            output[k][mask_upper] = x
            output[k] = output[k] + output[k].t()   
            output[k][mask_diag] /= 2
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables
        input = input[0]
        grad_input = None
        if ctx.needs_input_grad[0]:
            n = int(-.5 + 0.5 * torch.sqrt(1 + 8 * input.size(1)))
            grad_input = input.new(len(input), input.size(1))
            mask = torch.triu_indices(n, n)
            for k, g in enumerate(grad_output):
                grad_input[k] = g[mask]

        return grad_input


class SPDUnVectorize(nn.Module):

    def __init__(self):
        super(SPDUnVectorize, self).__init__()

    def forward(self, input):
        return SPDUnVectorizeFunction.apply(input)


class SPDTangentSpaceFunction(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        
        output = input.new(input.size(0), input.size(1), input.size(2))
        for k, x in enumerate(input):
            s, u = torch.linalg.eigh(x)
            s = s.abs()
            s.log_()
            output[k] = u @ (torch.diag(s) @ u.T)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        input = input[0]
        grad_input = None

        if ctx.needs_input_grad[0]:
            eye = input.new(input.size(1))
            eye.fill_(1)
            eye = torch.diag(eye)
            grad_input = input.new(input.size(0), input.size(1), input.size(1))
            for k, g in enumerate(grad_output):
                x = input[k]
                s, u = torch.linalg.eigh(x)
                s = s.abs()
                
                g = symmetric(g)
                
                s_log_diag = torch.diag(s.log())
                s_inv_diag = torch.diag(1 / s)
                
                dLdV = 2 * (g @ (u @ s_log_diag))
                dLdS = eye * (s_inv_diag @ (u.T @ (g @ u)))
                
                P = s.unsqueeze(1)
                P = P.expand(-1, P.size(0))
                P = P - P.t()
                mask_zero = torch.abs(P) == 0
                P = 1 / P
                P[mask_zero] = 0
                
                grad_input[k] = u @ (symmetric(P.T * (u.T @ dLdV)) + dLdS) @ u.T

        return grad_input


class SPDTangentSpace(nn.Module):

    def __init__(self, input_size, vectorize=True):
        super(SPDTangentSpace, self).__init__()
        self.vectorize = vectorize
        if vectorize:
            self.vec = SPDVectorize(input_size)

    def forward(self, input):
        output = SPDTangentSpaceFunction.apply(input)
        if self.vectorize:
            output = self.vec(output)

        return output


class SPDUnTangentSpaceFunction(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        
        output = input.new(input.size(0), input.size(1), input.size(2))
        for k, x in enumerate(input):
            s, u = torch.linalg.eigh(x)
            s = s.abs()
            s.exp_()
            output[k] = u @ (torch.diag(s) @ u.T)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        input = input[0]
        grad_input = None

        if ctx.needs_input_grad[0]:
            eye = input.new(input.size(1))
            eye.fill_(1)
            eye = torch.diag(eye)
            grad_input = input.new(input.size(0), input.size(1), input.size(1))
            for k, g in enumerate(grad_output):
                x = input[k]
                s, u = torch.linalg.eigh(x)
                s = s.abs()
                
                g = symmetric(g)
                
                s_exp_diag = torch.diag(s.exp())
                
                dLdV = 2 * (g @ (u @ s_exp_diag))
                dLdS = eye * (s_exp_diag @ (u.T @ (g @ u)))
                
                P = s.unsqueeze(1)
                P = P.expand(-1, P.size(0))
                P = P - P.t()
                mask_zero = torch.abs(P) == 0
                P = 1 / P
                P[mask_zero] = 0
                
                grad_input[k] = u @ (symmetric(P.T * (u.T @ dLdV)) + dLdS) @ u.T

        return grad_input


class SPDUnTangentSpace(nn.Module):

    def __init__(self, unvectorize=True):
        super(SPDUnTangentSpace, self).__init__()
        self.unvectorize = unvectorize
        if unvectorize:
            self.unvec = SPDUnVectorize()

    def forward(self, input):
        if self.unvectorize:
            input = self.unvec(input)
        output = SPDUnTangentSpaceFunction.apply(input)
        return output

# TODO: [算法确认] SPDTangentSpaceFunction / SPDUnTangentSpaceFunction 仍为
# Python 循环遍历 batch 维，可后续 batched 化（eigh 原生支持批量输入）。
# SPDRectifiedFunction / SPDPowerFunction 已 batched 重写。

def _safe_diff_matrix(s):
    """P_ij = s_i - s_j，P=0（特征值简并）处置 0 避免 1/0。batched: (B, n) → (B, n, n)"""
    P = s.unsqueeze(-1) - s.unsqueeze(-2)
    mask = P.abs() == 0
    P = torch.where(mask, torch.ones_like(P), P)
    P = 1.0 / P
    return torch.where(mask, torch.zeros_like(P), P)


class SPDRectifiedFunction(Function):
    """平滑整流（softplus 风格，A11）: s' = (s + sqrt(s² + 4ε)) / 2

    满足 s' = ε / s'（自对偶），处处可导、保序，替代原硬阈值 s[s<ε]=ε
    （硬阈值压平谱尾、破坏特征值排序，且固定 ε 不随被试噪声水平自适应）。
    forward/backward 均为 batched eigh，无 Python 循环。
    """

    @staticmethod
    def forward(ctx, input, epsilon):
        s, u = torch.linalg.eigh(input)
        s = s.abs().clamp(min=0.0)
        e = epsilon.reshape(())  # 标量化
        s_rect = 0.5 * (s + torch.sqrt(s * s + 4 * e))
        ctx.save_for_backward(s_rect, u, e)
        return u @ torch.diag_embed(s_rect) @ u.transpose(-1, -2)

    @staticmethod
    def backward(ctx, grad_output):
        s_rect, u, e = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            g = symmetric(grad_output)
            Gu = u.transpose(-1, -2) @ g @ u
            diagGu = torch.diagonal(Gu, dim1=-2, dim2=-1)          # (B, n)

            # s' 满足 s'² - s·s' - ε = 0，隐函数求导: ds'/ds = s'² / (s'² + ε)
            # （等价于 0.5·(1 + s/sqrt(s²+4ε))，但此式全部由 s' 表出，数值更稳）
            ds = s_rect * s_rect / (s_rect * s_rect + e)

            dLdS_vec = ds * diagGu                                 # (B, n)
            dLdV = 2.0 * (g @ (u * s_rect.unsqueeze(-2)))          # g @ (u @ diag(s'))
            s = s_rect - e / s_rect                                # 反解原始特征值，用于 P
            P = _safe_diff_matrix(s)
            grad_input = u @ (
                symmetric(P.transpose(-1, -2) * (u.transpose(-1, -2) @ dLdV))
                + torch.diag_embed(dLdS_vec)
            ) @ u.transpose(-1, -2)
        return grad_input, None


class SPDRectified(nn.Module):

    def __init__(self, epsilon=1e-4):
        super(SPDRectified, self).__init__()
        self.register_buffer('epsilon', torch.FloatTensor([epsilon]))

    def forward(self, input):
        output = SPDRectifiedFunction.apply(input, self.epsilon)
        return output


class SPDPowerFunction(Function):
    """可学习谱幂变换: X^w = U diag(exp(w ⊙ log(s))) U^T（batched，无循环）

    特征值取 abs 后 clamp(min=1e-6) 再取 log，避免近零特征值导致 -inf（原实现的隐患）。
    """

    @staticmethod
    def forward(ctx, input, weight):
        s, u = torch.linalg.eigh(input)
        s = s.abs().clamp(min=1e-6)
        log_s = s.log()
        s_power = torch.exp(weight * log_s)
        ctx.save_for_backward(s, u, weight)
        return u @ torch.diag_embed(s_power) @ u.transpose(-1, -2)

    @staticmethod
    def backward(ctx, grad_output):
        s, u, weight = ctx.saved_tensors
        grad_input = None
        grad_weight = None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            g = symmetric(grad_output)
            Gu = u.transpose(-1, -2) @ g @ u
            diagGu = torch.diagonal(Gu, dim1=-2, dim2=-1)      # (B, n)
            log_s = s.log()
            s_power = torch.exp(weight * log_s)                # (B, n)

            if ctx.needs_input_grad[1]:
                # ∂L/∂w_j = log(s_j)·s_j^{w_j}·[U^T G U]_jj（谱微扰标准结果），
                # 对 batch 求和（每个 batch 元素都对共享 w 有梯度贡献；
                # 原实现的 .mean(0) 有误且 U^T diag(s^w) U G 对角公式有误，此处一并修正）
                grad_weight = (log_s * s_power * diagGu).sum(0)

            if ctx.needs_input_grad[0]:
                pw1 = weight * torch.exp((weight - 1) * log_s)  # d s^w / d s = w·s^{w-1}
                dLdS_vec = pw1 * diagGu
                dLdV = 2.0 * (g @ (u * s_power.unsqueeze(-2)))  # g @ (u @ diag(s_power))
                P = _safe_diff_matrix(s)
                grad_input = u @ (
                    symmetric(P.transpose(-1, -2) * (u.transpose(-1, -2) @ dLdV))
                    + torch.diag_embed(dLdS_vec)
                ) @ u.transpose(-1, -2)

        return grad_input, grad_weight


class SPDPower(nn.Module):

    def __init__(self, input_dim):
        super(SPDPower, self).__init__()
        self.weight = nn.Parameter(torch.ones(input_dim), requires_grad=True)

    def forward(self, input):
        # 兼容任意 batch 维度 (..., n, n)
        orig_shape = input.shape
        x3 = input.reshape(-1, orig_shape[-2], orig_shape[-1])
        output = SPDPowerFunction.apply(x3, self.weight)
        return output.view(orig_shape)


class SPDNorm(nn.Module):
    """流形仿射对齐（A12，SPD 版 BatchNorm）：log 域先平移再缩放

    Y = expm( γ·(log X − log M) + β·I )
      - 平移: log M 为批内 log 域均值（训练时统计，EMA 维护 running mean，eval 使用）
        —— 把不同被试/批次协方差的"聚集中心"对齐到单位矩阵附近，补齐 SPDTransform
        只有旋转（WᵀXW）而无平移的缺陷；
      - 缩放: γ、β 为可学习标量（初始 γ=1、β=0 即恒等），在谱方向做仿射。
    旋转部分由下游 SPDTransform 的 Stiefel 正交矩阵承担（先平移后旋转的分工）。
    """

    def __init__(self, input_size, momentum=0.1):
        super(SPDNorm, self).__init__()
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(input_size, input_size))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, input):
        # 兼容任意 batch 维度 (..., n, n)
        orig_shape = input.shape
        x3 = input.reshape(-1, orig_shape[-2], orig_shape[-1])

        s, u = torch.linalg.eigh(x3)
        s = s.abs().clamp(min=1e-6)
        log_x = u @ torch.diag_embed(s.log()) @ u.transpose(-1, -2)

        if self.training and x3.shape[0] > 0:
            mean = log_x.mean(0)
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
                self.num_batches_tracked += 1
        else:
            mean = self.running_mean

        y = self.gamma * (log_x - mean) + self.beta
        # expm（对称阵谱指数化）
        sy, uy = torch.linalg.eigh(y)
        out = uy @ torch.diag_embed(sy.exp()) @ uy.transpose(-1, -2)
        return out.view(orig_shape)
