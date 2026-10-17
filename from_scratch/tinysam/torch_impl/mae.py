"""A pixel-reconstruction MAE used as a small stand-in for SAMI pretraining."""

import torch
from torch import nn

from .layers import GELU, LayerNorm, Linear, MultiHeadAttention, PatchEmbed


class MAEBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, heads)
        self.norm2 = LayerNorm(dim)
        self.fc1 = Linear(dim, dim * mlp_ratio)
        self.activation = GELU()
        self.fc2 = Linear(dim * mlp_ratio, dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.fc2(self.activation(self.fc1(self.norm2(x))))


class MAEPretrainer(nn.Module):
    """Mask 75% of patches, encode visible tokens, and reconstruct masked pixels."""

    def __init__(self, config, decoder_dim=128, decoder_depth=2, mask_ratio=0.75):
        super().__init__()
        self.config = config
        self.mask_ratio = mask_ratio
        self.image_encoder = nn.Module()
        self.image_encoder.patch_embed = PatchEmbed(config.img_size, config.patch, config.enc_dim)
        self.image_encoder.pos_embed = nn.Parameter(torch.empty(1, config.Hp ** 2, config.enc_dim))
        nn.init.normal_(self.image_encoder.pos_embed, std=0.02)
        from .model import EncoderBlock
        self.image_encoder.blocks = nn.ModuleList([
            EncoderBlock(config.enc_dim, config.enc_heads, config.mlp_ratio)
            for _ in range(config.enc_depth)
        ])
        self.decoder_embed = Linear(config.enc_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(torch.empty(1, config.Hp ** 2, decoder_dim))
        nn.init.normal_(self.decoder_pos_embed, std=0.02)
        self.decoder_blocks = nn.ModuleList([MAEBlock(decoder_dim, 4) for _ in range(decoder_depth)])
        self.decoder_norm = LayerNorm(decoder_dim)
        self.decoder_pred = Linear(decoder_dim, 3 * config.patch * config.patch)

    def _patchify(self, images):
        b, c, h, w = images.shape
        p = self.config.patch
        return images.reshape(b, c, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5).reshape(
            b, (h // p) * (w // p), c * p * p
        )

    def forward(self, images):
        patches = self._patchify(images)
        batch, count, _ = patches.shape
        mask_count = max(1, int(count * self.mask_ratio))
        noise = torch.rand(batch, count, device=images.device)
        order = noise.argsort(dim=1)
        masked = order[:, :mask_count]
        visible = order[:, mask_count:]
        tokens = self.image_encoder.patch_embed(images) + self.image_encoder.pos_embed
        visible_tokens = tokens.gather(1, visible.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
        for block in self.image_encoder.blocks:
            visible_tokens = block(visible_tokens)
        decoded_visible = self.decoder_embed(visible_tokens)
        full = self.mask_token.expand(batch, count, -1).clone()
        full.scatter_(1, visible.unsqueeze(-1).expand(-1, -1, full.shape[-1]), decoded_visible)
        full = full + self.decoder_pos_embed
        for block in self.decoder_blocks:
            full = block(full)
        prediction = self.decoder_pred(self.decoder_norm(full))
        target = patches
        mean = target.mean(dim=-1, keepdim=True)
        variance = ((target - mean) ** 2).mean(dim=-1, keepdim=True)
        target = (target - mean) / torch.sqrt(variance + 1e-6)
        per_patch = ((prediction - target) ** 2).mean(dim=-1)
        mask = torch.zeros_like(per_patch)
        mask.scatter_(1, masked, 1.0)
        loss = (per_patch * mask).sum() / mask.sum().clamp_min(1)
        return loss


def mae_encoder_state_dict(model):
    """Return only the transferable encoder tensors on CPU."""
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if key.startswith("image_encoder.")
    }
