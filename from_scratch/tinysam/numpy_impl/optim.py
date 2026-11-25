"""A small, explicit AdamW implementation for NumPy parameter dictionaries."""

import numpy as np


class AdamW:
    """Adam with decoupled weight decay and the standard bias corrections."""

    def __init__(self, parameters, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
        self.parameters = parameters
        self.lr, self.weight_decay = lr, weight_decay
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.step_count = 0
        self.first = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.second = {name: np.zeros_like(value) for name, value in parameters.items()}

    def step(self, gradients, lr=None):
        """Update each parameter with m̂/(sqrt(v̂)+ε) plus decoupled decay."""
        self.step_count += 1
        rate = self.lr if lr is None else lr
        for name, parameter in self.parameters.items():
            gradient = gradients[name]
            self.first[name] = self.beta1 * self.first[name] + (1 - self.beta1) * gradient
            self.second[name] = self.beta2 * self.second[name] + (1 - self.beta2) * gradient ** 2
            m_hat = self.first[name] / (1 - self.beta1 ** self.step_count)
            v_hat = self.second[name] / (1 - self.beta2 ** self.step_count)
            parameter *= 1 - rate * self.weight_decay
            parameter -= rate * m_hat / (np.sqrt(v_hat) + self.eps)
