"""Elementary neural-network layers with explicit NumPy backward passes."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


class Module:
    """Base class with recursively named parameters and gradient buffers."""

    def __init__(self):
        self.params: dict[str, np.ndarray] = {}
        self.grads: dict[str, np.ndarray] = {}

    def add_parameter(self, name: str, value: np.ndarray) -> None:
        self.params[name] = value
        self.grads[name] = np.zeros_like(value)

    def named_parameters(self, prefix: str = ""):
        for name, value in self.params.items():
            yield f"{prefix}{name}", value
        for name, child in self.__dict__.items():
            if isinstance(child, Module):
                yield from child.named_parameters(f"{prefix}{name}.")
            elif isinstance(child, (list, tuple)):
                for index, module in enumerate(child):
                    if isinstance(module, Module):
                        yield from module.named_parameters(f"{prefix}{name}.{index}.")

    def zero_grad(self) -> None:
        for grad in self.grads.values():
            grad.fill(0)
        for child in self._children():
            child.zero_grad()

    def _children(self):
        for value in self.__dict__.values():
            if isinstance(value, Module):
                yield value
            elif isinstance(value, (list, tuple)):
                yield from (item for item in value if isinstance(item, Module))


def _init(shape, rng, dtype, scale=None):
    fan_in = shape[0] if shape else 1
    return (rng.normal(0, scale or math.sqrt(2.0 / fan_in), size=shape)).astype(dtype)


class Linear(Module):
    """Linear map y=xW+b; backward uses dx=dY Wᵀ, dW=xᵀdY."""

    def __init__(self, in_features: int, out_features: int, rng=None, dtype=np.float32,
                 bias=True):
        super().__init__()
        rng = rng or np.random.default_rng()
        self.add_parameter("weight", _init((in_features, out_features), rng, dtype))
        if bias:
            self.add_parameter("bias", np.zeros(out_features, dtype=dtype))
        self.cache = None

    def forward(self, x):
        self.cache = x
        y = x @ self.params["weight"]
        return y + self.params["bias"] if "bias" in self.params else y

    def backward(self, dout):
        """dX=dY Wᵀ; dW=Σ xᵀdY; db=ΣdY."""
        x = self.cache
        axes = tuple(range(dout.ndim - 1))
        self.grads["weight"] += (x.reshape(-1, x.shape[-1]).T @ dout.reshape(-1, dout.shape[-1]))
        if "bias" in self.grads:
            self.grads["bias"] += dout.sum(axis=axes) if axes else dout
        return dout @ self.params["weight"].T


class LayerNorm(Module):
    """Last-axis normalization, y=(x-μ)/sqrt(var+ε)·γ+β."""

    def __init__(self, dim, rng=None, dtype=np.float32, eps=1e-5):
        super().__init__()
        self.add_parameter("weight", np.ones(dim, dtype=dtype))
        self.add_parameter("bias", np.zeros(dim, dtype=dtype))
        self.eps = eps
        self.cache = None

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        centered = x - mean
        inv_std = 1.0 / np.sqrt(np.mean(centered * centered, axis=-1, keepdims=True) + self.eps)
        normalized = centered * inv_std
        self.cache = (normalized, inv_std, x.shape)
        return normalized * self.params["weight"] + self.params["bias"]

    def backward(self, dout):
        """dx=γ/σ·(dŷ−mean(dŷ)−x̂·mean(dŷ·x̂)); dγ=Σdŷx̂; dβ=Σdŷ."""
        normalized, inv_std, _ = self.cache
        axes = tuple(range(dout.ndim - 1))
        self.grads["weight"] += (dout * normalized).sum(axis=axes)
        self.grads["bias"] += dout.sum(axis=axes)
        dnormalized = dout * self.params["weight"]
        return inv_std * (dnormalized - dnormalized.mean(axis=-1, keepdims=True)
                          - normalized * (dnormalized * normalized).mean(axis=-1, keepdims=True))


class GELU(Module):
    """Tanh approximation: 0.5x[1+tanh(c(x+0.044715x³))]."""

    def __init__(self):
        super().__init__()
        self.cache = None

    def forward(self, x):
        u = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)
        t = np.tanh(u)
        self.cache = (x, t)
        return 0.5 * x * (1.0 + t)

    def backward(self, dout):
        """dGELU/dx=0.5(1+t)+0.5x(1−t²)c(1+3·0.044715x²)."""
        x, t = self.cache
        c = math.sqrt(2.0 / math.pi)
        derivative = 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * c * (1 + 3 * 0.044715 * x * x)
        return dout * derivative


class ReLU(Module):
    """Elementwise max(0,x)."""

    def __init__(self):
        super().__init__()
        self.cache = None

    def forward(self, x):
        self.cache = x > 0
        return np.maximum(x, 0)

    def backward(self, dout):
        """dx=dout·1[x>0]."""
        return dout * self.cache


def softmax(x, axis=-1):
    """Stable softmax, subtracting the per-row maximum."""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exponents = np.exp(shifted)
    return exponents / exponents.sum(axis=axis, keepdims=True)


def softmax_backward(probabilities, dout, axis=-1):
    """ds=p⊙(dp−Σ(dp⊙p)), the Jacobian-vector product."""
    return probabilities * (dout - np.sum(dout * probabilities, axis=axis, keepdims=True))


class MultiHeadAttention(Module):
    """Scaled dot-product attention with separate q/k/v/out projections."""

    def __init__(self, dim, heads, rng=None, dtype=np.float32):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        rng = rng or np.random.default_rng()
        self.q_proj = Linear(dim, dim, rng, dtype)
        self.k_proj = Linear(dim, dim, rng, dtype)
        self.v_proj = Linear(dim, dim, rng, dtype)
        self.out_proj = Linear(dim, dim, rng, dtype)
        self.cache = None

    def _split(self, x):
        b, n, _ = x.shape
        return x.reshape(b, n, self.heads, self.head_dim).transpose(0, 2, 1, 3)

    def _merge(self, x):
        b, h, n, d = x.shape
        return x.transpose(0, 2, 1, 3).reshape(b, n, h * d)

    def forward(self, q, k=None, v=None):
        k = q if k is None else k
        v = k if v is None else v
        qh, kh, vh = self._split(self.q_proj.forward(q)), self._split(self.k_proj.forward(k)), self._split(self.v_proj.forward(v))
        scores = qh @ kh.transpose(0, 1, 3, 2) / math.sqrt(self.head_dim)
        probs = softmax(scores)
        context = probs @ vh
        self.cache = (qh, kh, vh, probs, context.shape, q.shape, k.shape, v.shape)
        return self.out_proj.forward(self._merge(context))

    def backward(self, dout):
        """Backpropagate attention using dS=P⊙(dP−ΣdP·P) and dQ=dS K/√d."""
        qh, kh, vh, probs, _, qshape, kshape, vshape = self.cache
        dcontext = self._split(self.out_proj.backward(dout))
        dprobs = dcontext @ vh.transpose(0, 1, 3, 2)
        dvh = probs.transpose(0, 1, 3, 2) @ dcontext
        dscores = softmax_backward(probs, dprobs) / math.sqrt(self.head_dim)
        dqh = dscores @ kh
        dkh = dscores.transpose(0, 1, 3, 2) @ qh
        dq = self.q_proj.backward(self._merge(dqh)).reshape(qshape)
        dk = self.k_proj.backward(self._merge(dkh)).reshape(kshape)
        dv = self.v_proj.backward(self._merge(dvh)).reshape(vshape)
        return dq, dk, dv


class MLP(Module):
    """Two linear layers with a configurable activation between them."""

    def __init__(self, in_dim, hidden_dim, out_dim, rng=None, dtype=np.float32, activation="gelu"):
        super().__init__()
        rng = rng or np.random.default_rng()
        self.fc1 = Linear(in_dim, hidden_dim, rng, dtype)
        self.activation = GELU() if activation == "gelu" else ReLU()
        self.fc2 = Linear(hidden_dim, out_dim, rng, dtype)

    def forward(self, x):
        return self.fc2.forward(self.activation.forward(self.fc1.forward(x)))

    def backward(self, dout):
        """Chain rule: dX=W₁ᵀ·dAct(W₂ᵀ·dY)."""
        return self.fc1.backward(self.activation.backward(self.fc2.backward(dout)))


class MLP3(Module):
    """Three affine layers with ReLU after the first two."""

    def __init__(self, in_dim, hidden_dim, out_dim, rng=None, dtype=np.float32):
        super().__init__()
        rng = rng or np.random.default_rng()
        self.fc1 = Linear(in_dim, hidden_dim, rng, dtype)
        self.activation1 = ReLU()
        self.fc2 = Linear(hidden_dim, hidden_dim, rng, dtype)
        self.activation2 = ReLU()
        self.fc3 = Linear(hidden_dim, out_dim, rng, dtype)

    def forward(self, x):
        x = self.fc1.forward(x)
        x = self.activation1.forward(x)
        x = self.fc2.forward(x)
        x = self.activation2.forward(x)
        return self.fc3.forward(x)

    def backward(self, dout):
        """Apply the chain rule through fc3, ReLU, fc2, ReLU, then fc1."""
        return self.fc1.backward(self.activation1.backward(
            self.fc2.backward(self.activation2.backward(self.fc3.backward(dout)))))


class PatchEmbed(Module):
    """Non-overlapping patch split then Linear(3P²,D), equivalent to stride-P conv."""

    def __init__(self, img_size, patch, out_dim, rng=None, dtype=np.float32):
        super().__init__()
        self.size, self.patch = img_size, patch
        self.proj = Linear(3 * patch * patch, out_dim, rng, dtype)
        self.cache = None

    def forward(self, images):
        b, c, h, w = images.shape
        p = self.patch
        patches = images.reshape(b, c, h // p, p, w // p, p).transpose(0, 2, 4, 1, 3, 5)
        patches = patches.reshape(b, (h // p) * (w // p), c * p * p)
        self.cache = images.shape
        return self.proj.forward(patches)

    def backward(self, dout):
        """Invert patch flattening after applying the Linear transpose."""
        b, c, h, w = self.cache
        p = self.patch
        patches = self.proj.backward(dout).reshape(b, h // p, w // p, c, p, p)
        return patches.transpose(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)


class UpsampleConvT2x(Module):
    """Kernel=stride=2 transposed convolution as Linear + 2× pixel shuffle."""

    def __init__(self, in_channels, out_channels, rng=None, dtype=np.float32):
        super().__init__()
        self.proj = Linear(in_channels, out_channels * 4, rng, dtype)
        self.cache = None
        self.out_channels = out_channels

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.proj.forward(x.transpose(0, 2, 3, 1)).reshape(b, h, w, self.out_channels, 2, 2)
        self.cache = x.shape
        return y.transpose(0, 3, 1, 4, 2, 5).reshape(b, self.out_channels, 2 * h, 2 * w)

    def backward(self, dout):
        """Undo pixel shuffle, then apply the per-pixel Linear derivative."""
        b, c, h, w = self.cache
        chunks = dout.reshape(b, self.out_channels, h, 2, w, 2).transpose(0, 2, 4, 1, 3, 5)
        dx = self.proj.backward(chunks.reshape(b, h, w, self.out_channels * 4))
        return dx.transpose(0, 3, 1, 2)


class LayerNorm2d(Module):
    """LayerNorm across channels independently at each pixel."""

    def __init__(self, channels, rng=None, dtype=np.float32, eps=1e-5):
        super().__init__()
        self.norm = LayerNorm(channels, rng, dtype, eps)

    def forward(self, x):
        return self.norm.forward(x.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)

    def backward(self, dout):
        """Transpose to NHWC, use LayerNorm's last-axis derivative, transpose back."""
        return self.norm.backward(dout.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
