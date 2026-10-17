"""Small, explicit configurations for the teaching model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TinySAMConfig:
    img_size: int = 64
    patch: int = 8
    enc_dim: int = 64
    enc_depth: int = 4
    enc_heads: int = 4
    mlp_ratio: int = 4
    emb_dim: int = 64
    dec_depth: int = 2
    dec_heads: int = 4
    num_masks: int = 3

    @property
    def Hp(self) -> int:
        return self.img_size // self.patch

    def __post_init__(self) -> None:
        if self.img_size % self.patch:
            raise ValueError("img_size must be divisible by patch")
        if self.enc_dim % self.enc_heads or self.emb_dim % self.dec_heads:
            raise ValueError("embedding dimensions must be divisible by head counts")
        if self.emb_dim < 8 or self.emb_dim % 8:
            raise ValueError("emb_dim must be a positive multiple of 8 for mask upscaling")


NUMPY_TINY = TinySAMConfig()
TORCH_SMALL = TinySAMConfig(
    img_size=256, patch=16, enc_dim=192, enc_depth=6, enc_heads=3, emb_dim=128,
    dec_depth=2, dec_heads=4,
)
