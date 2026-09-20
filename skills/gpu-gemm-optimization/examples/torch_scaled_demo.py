"""Agent-local benchmark plumbing demo, NOT an MXFP/HTI or optimized kernel.

Only the installed PyTorch package is required. Arithmetic is BF16 matmul,
followed by FP32 per-row/per-column scaling and a BF16 store. This is a
multi-operation A/A diagnostic, not a reproduction of historical performance.
"""
import torch


def run(a, b, scale_a, scale_b, *, out, workspace):
    torch.mm(a, b, out=workspace["matmul"])
    workspace["scaled"].copy_(workspace["matmul"])
    workspace["scaled"].mul_(scale_a[:, None])
    workspace["scaled"].mul_(scale_b[None, :])
    out.copy_(workspace["scaled"])
