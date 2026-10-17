"""The PyTorch autograd graph is checked against handwritten NumPy derivatives."""

import numpy as np
import torch

from tinysam.config import TinySAMConfig
from tinysam.numpy_impl.losses import tiny_sam_loss as numpy_loss
from tinysam.numpy_impl.model import TinySAM as NumpyTinySAM
from tinysam.torch_impl.convert import numpy_to_torch
from tinysam.torch_impl.losses import tiny_sam_loss as torch_loss
from tinysam.torch_impl.model import TinySAM as TorchTinySAM


def test_numpy_and_torch_forward_and_all_parameter_gradients_match():
    config = TinySAMConfig(img_size=8, patch=4, enc_dim=8, enc_depth=1, enc_heads=2,
                           emb_dim=8, dec_depth=1, dec_heads=2)
    numpy_model = NumpyTinySAM(config, seed=41, dtype=np.float64)
    torch_model = TorchTinySAM(config, seed=9).double()
    numpy_to_torch(numpy_model, torch_model)
    rng = np.random.default_rng(31)
    images = rng.normal(size=(1, 3, 8, 8)).astype(np.float64)
    points = np.asarray([[[2, 4], [0, 0], [0, 0]]], dtype=np.float64)
    labels = np.asarray([[1, -1, -1]], dtype=np.int64)
    target = rng.integers(0, 2, size=(1, 8, 8)).astype(np.float64)
    np_masks, np_quality = numpy_model(images, points, labels)
    torch_masks, torch_quality = torch_model(torch.from_numpy(images), torch.from_numpy(points),
                                             torch.from_numpy(labels))
    np.testing.assert_allclose(torch_masks.detach().numpy(), np_masks, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(torch_quality.detach().numpy(), np_quality, atol=1e-5, rtol=1e-5)
    low_target = target.reshape(1, np_masks.shape[-1], 8 // np_masks.shape[-1],
                                np_masks.shape[-1], 8 // np_masks.shape[-1]).mean(axis=(2, 4))
    low_target = (low_target >= 0.5).astype(np.float64)
    np_loss_value, dmask, diou = numpy_loss(np_masks, np_quality, low_target)
    torch_loss_value = torch_loss(torch_masks, torch_quality,
                                  torch.from_numpy(low_target))
    torch_loss_value.backward()
    np.testing.assert_allclose(torch_loss_value.item(), np_loss_value, atol=1e-8, rtol=1e-8)
    numpy_model.zero_grad()
    numpy_model.backward(dmask, diou)
    numpy_grads = {}
    for name, _ in numpy_model.named_parameters():
        owner = numpy_model
        parts = name.split(".")
        for part in parts[:-1]:
            owner = owner[int(part)] if part.isdigit() else getattr(owner, part)
        numpy_grads[name] = owner.grads[parts[-1]]
    for name, parameter in torch_model.named_parameters():
        np.testing.assert_allclose(parameter.grad.detach().numpy(), numpy_grads[name],
                                   atol=1e-5, rtol=1e-5, err_msg=name)
