import torch
import numpy as np

def symmetric(A):
    return 0.5 * (A + A.transpose(-2, -1))

def orthogonal_projection(A, B):
    out = A - B @ symmetric(B.T @ A)
    return out

def retraction(A, ref):
    data = A + ref
    Q, R = torch.linalg.qr(data, mode='reduced')
    # To avoid (any possible) negative values in the output matrix, we multiply the negative values by -1
    sign = (torch.diag(R).sign() + 0.5).sign().diag()
    out = Q @ sign
    return out
