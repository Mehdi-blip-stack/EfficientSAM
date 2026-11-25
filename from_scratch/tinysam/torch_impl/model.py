"""PyTorch counterpart of the NumPy architecture, using only hand-built layers."""

import math

import torch
from torch import nn

from .layers import (
    GELU, LayerNorm, LayerNorm2d, Linear, MLP, MLP3, MultiHeadAttention,
    PatchEmbed, ReLU, UpsampleConvT2x,
)


class EncoderBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, heads)
        self.norm2 = LayerNorm(dim)
        self.mlp = MLP(dim, dim * mlp_ratio, dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ImageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed(config.img_size, config.patch, config.enc_dim)
        self.pos_embed = nn.Parameter(torch.empty(1, config.Hp ** 2, config.enc_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([
            EncoderBlock(config.enc_dim, config.enc_heads, config.mlp_ratio)
            for _ in range(config.enc_depth)
        ])
        self.neck1 = Linear(config.enc_dim, config.emb_dim)
        self.neck_norm1 = LayerNorm(config.emb_dim)
        self.neck2 = Linear(config.emb_dim, config.emb_dim)
        self.neck_norm2 = LayerNorm(config.emb_dim)

    def forward(self, images):
        x = self.patch_embed(images) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.neck_norm2(self.neck2(self.neck_norm1(self.neck1(x))))


class PromptEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        generator = torch.Generator().manual_seed(2024)
        self.register_buffer("gaussian", torch.randn(2, config.emb_dim // 2, generator=generator))
        self.label_embeddings = nn.Parameter(torch.empty(5, config.emb_dim))
        nn.init.normal_(self.label_embeddings, std=0.02)

    def _pe(self, coords):
        projected = (2.0 * coords - 1.0) @ self.gaussian
        projected = projected * (2.0 * math.pi)
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)

    def dense_pe(self, device=None, dtype=None):
        h = self.config.Hp
        axis = (torch.arange(h, device=device, dtype=dtype or self.gaussian.dtype) + 0.5) / h
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        return self._pe(torch.stack([xx, yy], dim=-1).reshape(1, h * h, 2))

    def forward(self, points, labels):
        pe = self._pe((points + 0.5) / self.config.img_size)
        indices = torch.where(labels >= 0, labels, torch.full_like(labels, 4))
        embeddings = self.label_embeddings[indices]
        return embeddings + pe * (labels >= 0).unsqueeze(-1)


class TwoWayLayer(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.self_attn = MultiHeadAttention(dim, heads)
        self.norm1 = LayerNorm(dim)
        self.token_to_image = MultiHeadAttention(dim, heads)
        self.norm2 = LayerNorm(dim)
        self.mlp = MLP(dim, dim * 2, dim, activation="relu")
        self.norm3 = LayerNorm(dim)
        self.image_to_token = MultiHeadAttention(dim, heads)
        self.norm4 = LayerNorm(dim)

    def forward(self, tokens, image, query_pe, dense_pe, skip_first_pe=False):
        q = tokens if skip_first_pe else tokens + query_pe
        tokens = self.norm1(tokens + self.self_attn(q, q, tokens))
        tokens = self.norm2(tokens + self.token_to_image(tokens + query_pe, image + dense_pe, image))
        tokens = self.norm3(tokens + self.mlp(tokens))
        image = self.norm4(image + self.image_to_token(image + dense_pe, tokens + query_pe, tokens))
        return tokens, image


class TwoWayTransformer(nn.Module):
    def __init__(self, dim, depth, heads):
        super().__init__()
        self.layers = nn.ModuleList([TwoWayLayer(dim, heads) for _ in range(depth)])
        self.final_attn = MultiHeadAttention(dim, heads)
        self.final_norm = LayerNorm(dim)

    def forward(self, tokens, image, query_pe, dense_pe):
        for index, layer in enumerate(self.layers):
            tokens, image = layer(tokens, image, query_pe, dense_pe, skip_first_pe=index == 0)
        return self.final_norm(tokens + self.final_attn(tokens + query_pe, image + dense_pe, image))


class MaskDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        e, k = config.emb_dim, config.num_masks
        self.config = config
        self.iou_token = nn.Parameter(torch.empty(1, e))
        self.mask_tokens = nn.Parameter(torch.empty(k, e))
        nn.init.normal_(self.iou_token, std=0.02)
        nn.init.normal_(self.mask_tokens, std=0.02)
        self.transformer = TwoWayTransformer(e, config.dec_depth, config.dec_heads)
        self.up1 = UpsampleConvT2x(e, e // 4)
        self.up_norm = LayerNorm2d(e // 4)
        self.up_gelu = GELU()
        self.up2 = UpsampleConvT2x(e // 4, e // 8)
        self.up_gelu2 = GELU()
        self.hypernets = nn.ModuleList([MLP3(e, e, e // 8) for _ in range(k)])
        self.iou_head = MLP3(e, e, k)

    def forward(self, image_embeddings, sparse_prompts, dense_pe):
        b, _length, e = image_embeddings.shape
        tokens = torch.cat([
            self.iou_token.expand(b, 1, e),
            self.mask_tokens.expand(b, self.config.num_masks, e),
            sparse_prompts,
        ], dim=1)
        decoded = self.transformer(tokens, image_embeddings, tokens, dense_pe)
        iou_pred = self.iou_head(decoded[:, 0])
        mask_tokens = decoded[:, 1:1 + self.config.num_masks]
        feature_map = image_embeddings.transpose(1, 2).reshape(
            b, e, self.config.Hp, self.config.Hp
        )
        feature_map = self.up_gelu(self.up_norm(self.up1(feature_map)))
        feature_map = self.up_gelu2(self.up2(feature_map))
        weights = torch.stack([head(mask_tokens[:, i]) for i, head in enumerate(self.hypernets)], dim=1)
        mask_logits = (weights[:, :, :, None, None] * feature_map[:, None, :, :, :]).sum(dim=2)
        return mask_logits, iou_pred


class TinySAM(nn.Module):
    """(images, pixel-coordinate points, integer labels) → mask logits and IoU scores."""

    def __init__(self, config, seed=0):
        super().__init__()
        self.config = config
        torch.manual_seed(seed)
        self.image_encoder = ImageEncoder(config)
        self.prompt_encoder = PromptEncoder(config)
        self.mask_decoder = MaskDecoder(config)

    def encode_image(self, images):
        return self.image_encoder(images)

    def decode(self, embeddings, points, labels):
        dense_pe = self.prompt_encoder.dense_pe(embeddings.device, embeddings.dtype)
        prompts = self.prompt_encoder(points, labels)
        return self.mask_decoder(embeddings, prompts, dense_pe)

    def forward(self, images, points, labels):
        return self.decode(self.encode_image(images), points, labels)
