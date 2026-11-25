"""AdamW updates in ordinary PyTorch operations (no torch.optim)."""

import torch


class AdamW:
    """Minimal AdamW optimizer with per-parameter first/second moments."""

    def __init__(self, parameters, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
        self.parameters = list(parameters)
        self.lr, self.weight_decay = lr, weight_decay
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.step_count = 0
        self.first = [torch.zeros_like(p) for p in self.parameters]
        self.second = [torch.zeros_like(p) for p in self.parameters]

    def zero_grad(self):
        for parameter in self.parameters:
            if parameter.grad is not None:
                parameter.grad.zero_()

    @torch.no_grad()
    def step(self, lr=None):
        self.step_count += 1
        rate = self.lr if lr is None else lr
        for i, parameter in enumerate(self.parameters):
            if parameter.grad is None:
                continue
            gradient = parameter.grad
            self.first[i].mul_(self.beta1).add_(gradient, alpha=1 - self.beta1)
            self.second[i].mul_(self.beta2).addcmul_(gradient, gradient, value=1 - self.beta2)
            m_hat = self.first[i] / (1 - self.beta1 ** self.step_count)
            v_hat = self.second[i] / (1 - self.beta2 ** self.step_count)
            parameter.mul_(1 - rate * self.weight_decay)
            parameter.addcdiv_(m_hat, v_hat.sqrt().add(self.eps), value=-rate)
