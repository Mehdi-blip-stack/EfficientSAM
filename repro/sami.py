"""Steps B3/B4: SAMI pretraining model (written by our team, following the paper's Section 3.1 and Eq. 1).

Student encoder = the OFFICIAL EfficientSAM ImageEncoderViT (patch embedding, positional embedding, 12 ViT
blocks), but run only on the visible 25% of patches (MAE-style masking, implemented here).
Decoder        = cross-attention decoder implemented here: queries come from the MASKED tokens, keys/values from
                 the encoder output of the visible tokens concatenated with the masked tokens (paper Sec. 3.1).
Output         = decoder output at masked positions + encoder output at visible positions, put back in the
                 original order, then a linear projection to the target dimension.
Targets:
  'sami'  : SAM teacher features (256-d per token, standardised per channel); MSE over ALL tokens (paper Eq. 1).
  'mae'   : normalised pixels of each 16 x 16 patch (768-d); MSE over the MASKED tokens only (standard MAE).
Both variants share the encoder, masking, decoder and schedule, so the only difference is the target: this is
the paper's own SAMI-vs-MAE comparison.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .common import GRID, PATCH


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention (masked-token queries attend to all tokens) followed by an MLP."""

    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm_q, self.norm_kv = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, q, kv):
        kvn = self.norm_kv(kv)
        q = q + self.attn(self.norm_q(q), kvn, kvn, need_weights=False)[0]
        return q + self.mlp(self.norm2(q))


class SAMIPretrainer(nn.Module):
    def __init__(self, encoder, target="sami", target_dim=256, mask_ratio=0.75, dec_dim=256, dec_depth=4, dec_heads=8):
        super().__init__()
        assert target in ("sami", "mae")
        self.encoder, self.target, self.mask_ratio = encoder, target, mask_ratio
        enc_dim = encoder.patch_embed.proj.out_channels
        self.dec_in = nn.Linear(enc_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.dec_pos = nn.Parameter(torch.zeros(1, GRID * GRID, dec_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02); nn.init.trunc_normal_(self.dec_pos, std=0.02)
        self.blocks = nn.ModuleList([CrossAttentionBlock(dec_dim, dec_heads) for _ in range(dec_depth)])
        self.norm = nn.LayerNorm(dec_dim)
        out_dim = target_dim if target == "sami" else 3 * PATCH * PATCH
        self.proj = nn.Linear(dec_dim, out_dim)                     # paper: linear projection head
        self.register_buffer("pix_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("pix_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def encode_visible(self, x, keep_idx):
        """Official encoder components, run on the visible tokens only."""
        from efficient_sam.efficient_sam_encoder import get_abs_pos
        enc = self.encoder
        t = enc.patch_embed(x).permute(0, 2, 3, 1)                                    # B,H,W,C
        t = t + get_abs_pos(enc.pos_embed, enc.pretrain_use_cls_token, [t.shape[1], t.shape[2]])
        t = t.reshape(t.shape[0], -1, t.shape[-1])                                   # B,N,C
        t = torch.gather(t, 1, keep_idx[..., None].expand(-1, -1, t.shape[-1]))      # visible tokens only
        for blk in enc.blocks:
            t = blk(t)
        return t

    @staticmethod
    def patchify(img):
        b, c, h, w = img.shape
        return img.reshape(b, c, h // PATCH, PATCH, w // PATCH, PATCH).permute(0, 2, 4, 1, 3, 5).reshape(b, -1, c * PATCH * PATCH)

    def forward(self, images_uint8, target=None):
        """images_uint8: B x 3 x 256 x 256 (0-255). target: B x 256 x 16 x 16 standardised teacher features ('sami')."""
        x = (images_uint8.float() / 255 - self.pix_mean) / self.pix_std
        B, N = x.shape[0], GRID * GRID
        n_keep = int(round(N * (1 - self.mask_ratio)))
        order = torch.rand(B, N, device=x.device).argsort(1)
        keep, drop = order[:, :n_keep], order[:, n_keep:]

        vis = self.dec_in(self.encode_visible(x, keep))                               # B,keep,D
        pos = self.dec_pos.expand(B, -1, -1)
        q = self.mask_token.expand(B, N - n_keep, -1) + torch.gather(pos, 1, drop[..., None].expand(-1, -1, pos.shape[-1]))
        kv = torch.cat([vis + torch.gather(pos, 1, keep[..., None].expand(-1, -1, pos.shape[-1])), q], 1)
        for blk in self.blocks:
            q = blk(q, kv)
        # Under mixed precision (GPU) vis and q can come out in different dtypes: merge them in one common dtype.
        dt = torch.promote_types(vis.dtype, q.dtype)
        full = torch.zeros(B, N, vis.shape[-1], device=x.device, dtype=dt)
        full.scatter_(1, keep[..., None].expand(-1, -1, vis.shape[-1]), vis.to(dt))  # encoder output, visible
        full.scatter_(1, drop[..., None].expand(-1, -1, q.shape[-1]), q.to(dt))      # decoder output, masked
        pred = self.proj(self.norm(full))                                            # B,N,out_dim

        if self.target == "sami":
            tgt = target.flatten(2).transpose(1, 2).float()                          # B,N,256
            return F.mse_loss(pred.float(), tgt)                                     # Eq. 1: all tokens
        tgt = self.patchify(images_uint8.float() / 255)
        tgt = (tgt - tgt.mean(-1, keepdim=True)) / (tgt.var(-1, keepdim=True) + 1e-6).sqrt()   # per-patch norm (MAE)
        per_tok = ((pred.float() - tgt) ** 2).mean(-1)
        m = torch.zeros(B, N, device=x.device); m.scatter_(1, drop, 1.0)
        return (per_tok * m).sum() / m.sum()                                         # masked tokens only
