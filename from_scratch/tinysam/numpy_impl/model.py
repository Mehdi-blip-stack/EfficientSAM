"""A compact ViT/prompt encoder/two-way decoder assembled from NumPy layers."""

from __future__ import annotations

import math

import numpy as np

from .layers import (
    GELU, LayerNorm, LayerNorm2d, Linear, MLP, MLP3, Module, MultiHeadAttention,
    PatchEmbed, ReLU, UpsampleConvT2x,
)


class EncoderBlock(Module):
    def __init__(self, dim, heads, mlp_ratio, rng, dtype):
        super().__init__()
        self.norm1 = LayerNorm(dim, dtype=dtype)
        self.attn = MultiHeadAttention(dim, heads, rng, dtype)
        self.norm2 = LayerNorm(dim, dtype=dtype)
        self.mlp = MLP(dim, dim * mlp_ratio, dim, rng, dtype)

    def forward(self, x):
        self.x = x
        self.n1 = self.norm1.forward(x)
        self.a = self.attn.forward(self.n1)
        self.r1 = x + self.a
        self.n2 = self.norm2.forward(self.r1)
        self.m = self.mlp.forward(self.n2)
        return self.r1 + self.m

    def backward(self, dout):
        dres1 = dout + self.norm2.backward(self.mlp.backward(dout))
        dq, dk, dv = self.attn.backward(dres1)
        return dres1 + self.norm1.backward(dq + dk + dv)


class ImageEncoder(Module):
    def __init__(self, config, rng, dtype):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed(config.img_size, config.patch, config.enc_dim, rng, dtype)
        self.add_parameter("pos_embed", rng.normal(0, 0.02, (1, config.Hp ** 2, config.enc_dim)).astype(dtype))
        self.blocks = [EncoderBlock(config.enc_dim, config.enc_heads, config.mlp_ratio, rng, dtype)
                       for _ in range(config.enc_depth)]
        self.neck1 = Linear(config.enc_dim, config.emb_dim, rng, dtype)
        self.neck_norm1 = LayerNorm(config.emb_dim, dtype=dtype)
        self.neck2 = Linear(config.emb_dim, config.emb_dim, rng, dtype)
        self.neck_norm2 = LayerNorm(config.emb_dim, dtype=dtype)

    def forward(self, images):
        x = self.patch_embed.forward(images) + self.params["pos_embed"]
        for block in self.blocks:
            x = block.forward(x)
        x = self.neck1.forward(x)
        x = self.neck_norm1.forward(x)
        x = self.neck2.forward(x)
        return self.neck_norm2.forward(x)

    def backward(self, dout):
        dout = self.neck_norm2.backward(dout)
        dout = self.neck2.backward(dout)
        dout = self.neck_norm1.backward(dout)
        dout = self.neck1.backward(dout)
        for block in reversed(self.blocks):
            dout = block.backward(dout)
        self.grads["pos_embed"] += dout.sum(axis=0, keepdims=True)
        return self.patch_embed.backward(dout)


class PromptEncoder(Module):
    """Fourier point PE and trainable point/box/padding label vectors."""

    def __init__(self, config, rng, dtype):
        super().__init__()
        self.config = config
        self.rng_pe = np.random.default_rng(2024)
        self.gaussian = self.rng_pe.normal(size=(2, config.emb_dim // 2)).astype(dtype)
        self.add_parameter("label_embeddings", rng.normal(0, 0.02, (5, config.emb_dim)).astype(dtype))

    def _pe(self, coords):
        scaled = 2.0 * coords - 1.0
        projected = (scaled @ self.gaussian) * (2.0 * math.pi)
        return np.concatenate([np.sin(projected), np.cos(projected)], axis=-1)

    def dense_pe(self):
        h = self.config.Hp
        axis = (np.arange(h, dtype=self.gaussian.dtype) + 0.5) / h
        xx, yy = np.meshgrid(axis, axis)
        return self._pe(np.stack([xx, yy], axis=-1).reshape(-1, 2))[None, :, :]

    def forward(self, points, labels):
        coords = (points + 0.5) / self.config.img_size
        pe = self._pe(coords)
        valid = labels >= 0
        embeddings = np.zeros_like(pe)
        embeddings[valid] = self.params["label_embeddings"][labels[valid]]
        embeddings += pe * valid[..., None]
        embeddings[~valid] = self.params["label_embeddings"][4]
        self.cache = (labels.copy(), valid, pe.shape)
        return embeddings

    def backward(self, dout):
        """Only learned label vectors receive gradients; Fourier PE is a fixed buffer."""
        labels, valid, _ = self.cache
        for label in range(4):
            mask = labels == label
            if mask.any():
                self.grads["label_embeddings"][label] += dout[mask].sum(axis=0)
        mask = labels < 0
        if mask.any():
            self.grads["label_embeddings"][4] += dout[mask].sum(axis=0)
        return np.zeros_like(dout)


class TwoWayLayer(Module):
    """SAM-style post-norm token self/cross attention and image-to-token update."""

    def __init__(self, dim, heads, rng, dtype, skip_first_pe=False):
        super().__init__()
        self.skip_first_pe = skip_first_pe
        self.self_attn = MultiHeadAttention(dim, heads, rng, dtype)
        self.norm1 = LayerNorm(dim, dtype=dtype)
        self.token_to_image = MultiHeadAttention(dim, heads, rng, dtype)
        self.norm2 = LayerNorm(dim, dtype=dtype)
        self.mlp = MLP(dim, dim * 2, dim, rng, dtype, activation="relu")
        self.norm3 = LayerNorm(dim, dtype=dtype)
        self.image_to_token = MultiHeadAttention(dim, heads, rng, dtype)
        self.norm4 = LayerNorm(dim, dtype=dtype)

    def forward(self, tokens, image, query_pe, dense_pe):
        self.tokens, self.image = tokens, image
        q = tokens if self.skip_first_pe else tokens + query_pe
        self.sa = self.self_attn.forward(q, q, tokens)
        t = self.norm1.forward(tokens + self.sa)
        self.t1 = t
        self.ta = self.token_to_image.forward(t + query_pe, image + dense_pe, image)
        t = self.norm2.forward(t + self.ta)
        self.t2 = t
        self.tm = self.mlp.forward(t)
        t = self.norm3.forward(t + self.tm)
        self.t3 = t
        self.ia = self.image_to_token.forward(image + dense_pe, t + query_pe, t)
        image_out = self.norm4.forward(image + self.ia)
        return t, image_out

    def backward(self, dt, di, query_pe, dense_pe):
        # image_out = LN(image + attention(image+PE, tokens+PE, tokens))
        dimage_res = self.norm4.backward(di)
        da_q, da_k, da_v = self.image_to_token.backward(dimage_res)
        dt3 = dt + da_k + da_v
        dquery_pe = da_k
        dimage = dimage_res + da_q
        # token MLP residual and post-norm.
        dt2 = self.norm3.backward(dt3)
        dt2 += self.mlp.backward(dt2)
        # token-to-image cross attention: q=t+PE, k=image+PE, v=image
        dt1 = self.norm2.backward(dt2)
        dq, dk, dv = self.token_to_image.backward(dt1)
        dt1 += dq
        dquery_pe += dq
        dimage += dk + dv
        # token self-attention, whose query/key PE is held constant.
        dt0 = self.norm1.backward(dt1)
        qgrad, kgrad, vgrad = self.self_attn.backward(dt0)
        dt0 += vgrad
        if self.skip_first_pe:
            dt0 += qgrad + kgrad
        else:
            dquery_pe += qgrad + kgrad
        return dt0, dimage, dquery_pe


class TwoWayTransformer(Module):
    def __init__(self, dim, depth, heads, rng, dtype):
        super().__init__()
        self.layers = [TwoWayLayer(dim, heads, rng, dtype, i == 0) for i in range(depth)]
        self.final_attn = MultiHeadAttention(dim, heads, rng, dtype)
        self.final_norm = LayerNorm(dim, dtype=dtype)

    def forward(self, tokens, image, query_pe, dense_pe):
        self.query_pe, self.dense_pe = query_pe, dense_pe
        for layer in self.layers:
            tokens, image = layer.forward(tokens, image, query_pe, dense_pe)
        self.tokens, self.image = tokens, image
        self.final = self.final_attn.forward(tokens + query_pe, image + dense_pe, image)
        return self.final_norm.forward(tokens + self.final)

    def backward(self, dout):
        dt = self.final_norm.backward(dout)
        dq, dk, dv = self.final_attn.backward(dt)
        dt += dq
        dimage = dk + dv
        dquery_pe = dq
        for layer in reversed(self.layers):
            dt, dimage, dpe = layer.backward(dt, dimage, self.query_pe, self.dense_pe)
            dquery_pe += dpe
        return dt + dquery_pe, dimage


class MaskDecoder(Module):
    def __init__(self, config, rng, dtype):
        super().__init__()
        e, k = config.emb_dim, config.num_masks
        self.config = config
        self.add_parameter("iou_token", rng.normal(0, 0.02, (1, e)).astype(dtype))
        self.add_parameter("mask_tokens", rng.normal(0, 0.02, (k, e)).astype(dtype))
        self.transformer = TwoWayTransformer(e, config.dec_depth, config.dec_heads, rng, dtype)
        self.up1 = UpsampleConvT2x(e, e // 4, rng, dtype)
        self.up_norm = LayerNorm2d(e // 4, dtype=dtype)
        self.up_gelu = GELU()
        self.up2 = UpsampleConvT2x(e // 4, e // 8, rng, dtype)
        self.up_gelu2 = GELU()
        self.hypernets = [MLP3(e, e, e // 8, rng, dtype) for _ in range(k)]
        self.iou_head = MLP3(e, e, k, rng, dtype)

    def forward(self, image_embeddings, sparse_prompts):
        b, length, e = image_embeddings.shape
        iou = np.broadcast_to(self.params["iou_token"], (b, 1, e))
        masks = np.broadcast_to(self.params["mask_tokens"], (b, self.config.num_masks, e))
        tokens = np.concatenate([iou, masks, sparse_prompts], axis=1)
        query_pe = tokens.copy()
        image = image_embeddings
        dense_pe = self._dense_pe
        decoded = self.transformer.forward(tokens, image, query_pe, dense_pe)
        self.num_prompt_tokens = sparse_prompts.shape[1]
        self.decoded = decoded
        iou_pred = self.iou_head.forward(decoded[:, 0])
        mask_tokens = decoded[:, 1:1 + self.config.num_masks]
        feat = image_embeddings.transpose(0, 2, 1).reshape(b, e, self.config.Hp, self.config.Hp)
        self.u1 = self.up1.forward(feat)
        self.un = self.up_norm.forward(self.u1)
        self.ug = self.up_gelu.forward(self.un)
        self.u2_pre = self.up2.forward(self.ug)
        self.u2 = self.up_gelu2.forward(self.u2_pre)
        weights = np.stack([mlp.forward(mask_tokens[:, i]) for i, mlp in enumerate(self.hypernets)], axis=1)
        self.weights = weights
        logits = np.einsum("bkc,bchw->bkhw", weights, self.u2)
        self._mask_tokens = mask_tokens
        return logits, iou_pred

    def backward(self, dlogits, diou):
        b, k, h, w = dlogits.shape
        dweights = np.einsum("bkhw,bchw->bkc", dlogits, self.u2)
        dfeat = np.einsum("bkhw,bkc->bchw", dlogits, self.weights)
        dmask_tokens = np.zeros_like(self._mask_tokens)
        for i, mlp in enumerate(self.hypernets):
            dmask_tokens[:, i] += mlp.backward(dweights[:, i])
        ddecoded = np.zeros_like(self.decoded)
        ddecoded[:, 0] = self.iou_head.backward(diou)
        ddecoded[:, 1:1 + k] += dmask_tokens
        dtokens, dimage = self.transformer.backward(ddecoded)
        self.grads["iou_token"] += dtokens[:, 0].sum(axis=0, keepdims=True)
        self.grads["mask_tokens"] += dtokens[:, 1:1 + k].sum(axis=0)
        dfeat = self.up2.backward(self.up_gelu2.backward(dfeat))
        dfeat = self.up1.backward(self.up_norm.backward(self.up_gelu.backward(dfeat)))
        dimage += dfeat.reshape(b, dfeat.shape[1], -1).transpose(0, 2, 1)
        return dimage, dtokens[:, 1 + k:]


class TinySAM(Module):
    """Promptable model returning (B,K,4Hp,4Hp) logits and (B,K) IoU estimates."""

    def __init__(self, config, seed=0, dtype=np.float32):
        super().__init__()
        self.config = config
        self.dtype = dtype
        rng = np.random.default_rng(seed)
        self.image_encoder = ImageEncoder(config, rng, dtype)
        self.prompt_encoder = PromptEncoder(config, rng, dtype)
        self.mask_decoder = MaskDecoder(config, rng, dtype)
        self.mask_decoder._dense_pe = self.prompt_encoder.dense_pe()

    def encode_image(self, images):
        return self.image_encoder.forward(images)

    def decode(self, embeddings, points, labels):
        prompts = self.prompt_encoder.forward(points, labels)
        return self.mask_decoder.forward(embeddings, prompts)

    def forward(self, images, points, labels):
        return self.decode(self.encode_image(images), points, labels)

    __call__ = forward

    def backward(self, dmask, diou):
        dembeddings, dprompts = self.mask_decoder.backward(dmask, diou)
        self.prompt_encoder.backward(dprompts)
        self.image_encoder.backward(dembeddings)

    def parameters(self):
        return dict(self.named_parameters())

    def zero_grad(self):
        super().zero_grad()
