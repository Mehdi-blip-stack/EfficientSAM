"""Hand-written PyTorch layers; autograd differentiates only basic tensor ops."""

from __future__ import annotations

import math

import torch
from torch import nn


class Linear(nn.Module):
    """Affine map y=xW+b with an explicitly stored (in,out) weight."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="linear")

    def forward(self, x):
        y = x @ self.weight
        return y + self.bias if self.bias is not None else y


class LayerNorm(nn.Module):
    """Last-axis population-variance normalization with learned scale and offset."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        centered = x - mean
        variance = (centered * centered).mean(dim=-1, keepdim=True)
        return centered / torch.sqrt(variance + self.eps) * self.weight + self.bias


class GELU(nn.Module):
    """Tanh-approximated GELU used identically by the NumPy model."""

    def forward(self, x):
        u = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)
        return 0.5 * x * (1.0 + torch.tanh(u))


class ReLU(nn.Module):
    def forward(self, x):
        return torch.maximum(x, torch.zeros_like(x))


def softmax(x, dim=-1):
    """Stable softmax written from max, exp, and sum tensor operations."""
    shifted = x - x.max(dim=dim, keepdim=True).values
    exponents = torch.exp(shifted)
    return exponents / exponents.sum(dim=dim, keepdim=True)


class MultiHeadAttention(nn.Module):
    """Multi-head scaled dot-product attention with explicit projections."""

    def __init__(self, dim, heads):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        self.q_proj = Linear(dim, dim)
        self.k_proj = Linear(dim, dim)
        self.v_proj = Linear(dim, dim)
        self.out_proj = Linear(dim, dim)

    def _split(self, x):
        b, n, _ = x.shape
        return x.reshape(b, n, self.heads, self.head_dim).permute(0, 2, 1, 3)

    def _merge(self, x):
        b, h, n, d = x.shape
        return x.permute(0, 2, 1, 3).reshape(b, n, h * d)

    def forward(self, q, k=None, v=None):
        k = q if k is None else k
        v = k if v is None else v
        qh, kh, vh = self._split(self.q_proj(q)), self._split(self.k_proj(k)), self._split(self.v_proj(v))
        scores = (qh @ kh.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = softmax(scores)
        return self.out_proj(self._merge(weights @ vh))


class MLP(nn.Module):
    """Two affine maps with a hand-written tanh-GELU or ReLU activation."""

    def __init__(self, in_dim, hidden_dim, out_dim, activation="gelu"):
        super().__init__()
        self.fc1 = Linear(in_dim, hidden_dim)
        self.activation = GELU() if activation == "gelu" else ReLU()
        self.fc2 = Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class MLP3(nn.Module):
    """Three affine layers with ReLU after the first two."""

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = Linear(in_dim, hidden_dim)
        self.activation1 = ReLU()
        self.fc2 = Linear(hidden_dim, hidden_dim)
        self.activation2 = ReLU()
        self.fc3 = Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.fc3(self.activation2(self.fc2(self.activation1(self.fc1(x)))))


class PatchEmbed(nn.Module):
    """Split non-overlapping patches and apply Linear(3P²,D), like stride-P conv."""

    def __init__(self, img_size, patch, out_dim):
        super().__init__()
        self.patch = patch
        self.proj = Linear(3 * patch * patch, out_dim)

    def forward(self, images):
        b, c, h, w = images.shape
        p = self.patch
        patches = images.reshape(b, c, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5)
        return self.proj(patches.reshape(b, (h // p) * (w // p), c * p * p))


class UpsampleConvT2x(nn.Module):
    """Kernel=stride=2 transposed convolution as per-pixel Linear + pixel shuffle."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.proj = Linear(in_channels, out_channels * 4)

    def forward(self, x):
        b, _c, h, w = x.shape
        y = self.proj(x.permute(0, 2, 3, 1)).reshape(b, h, w, self.out_channels, 2, 2)
        return y.permute(0, 3, 1, 4, 2, 5).reshape(b, self.out_channels, 2 * h, 2 * w)


class LayerNorm2d(nn.Module):
    """Layer normalization across channels for each spatial location."""

    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.norm = LayerNorm(channels, eps)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
