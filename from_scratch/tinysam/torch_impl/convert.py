"""Convert named parameters between the NumPy and PyTorch implementations."""

from pathlib import Path

import numpy as np
import torch


def numpy_to_torch(numpy_model, torch_model):
    """Copy arrays by identical dotted parameter names into a torch model."""
    source = numpy_model.parameters()
    destination = torch_model.state_dict()
    copied = []
    with torch.no_grad():
        for name, value in source.items():
            if name not in destination:
                raise KeyError(f"torch model is missing parameter {name}")
            if tuple(destination[name].shape) != value.shape:
                raise ValueError(f"shape mismatch for {name}: {value.shape} != {tuple(destination[name].shape)}")
            destination[name].copy_(torch.as_tensor(value, dtype=destination[name].dtype))
            copied.append(name)
        gaussian_name = "prompt_encoder.gaussian"
        if gaussian_name in destination and hasattr(numpy_model.prompt_encoder, "gaussian"):
            destination[gaussian_name].copy_(
                torch.as_tensor(numpy_model.prompt_encoder.gaussian, dtype=destination[gaussian_name].dtype)
            )
    torch_model.load_state_dict(destination)
    return copied


def save_numpy_weights(model, path):
    """Store a model's named NumPy arrays as a compressed NPZ."""
    np.savez_compressed(path, **model.parameters())


def load_numpy_weights(model, path):
    """Load matching arrays into an existing NumPy model."""
    with np.load(path) as data:
        parameters = model.parameters()
        if set(data.files) != set(parameters):
            missing, extra = set(parameters) - set(data.files), set(data.files) - set(parameters)
            raise ValueError(f"weight keys differ; missing={sorted(missing)}, extra={sorted(extra)}")
        for name, parameter in parameters.items():
            if data[name].shape != parameter.shape:
                raise ValueError(f"shape mismatch for {name}")
            parameter[...] = data[name]


def save_torch_weights(model, path):
    """Save only the model state dictionary."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_torch_weights(model, path, map_location="cpu", strict=True):
    """Load a state dictionary (including the fixed Fourier buffer)."""
    state = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(state, strict=strict)
