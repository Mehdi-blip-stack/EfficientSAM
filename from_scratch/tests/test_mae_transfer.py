import torch

from tinysam.config import TinySAMConfig
from tinysam.torch_impl.mae import MAEPretrainer, mae_encoder_state_dict
from tinysam.torch_impl.model import TinySAM


def test_mae_encoder_state_transfers_exactly_to_tinysam(tmp_path):
    config = TinySAMConfig(img_size=16, patch=4, enc_dim=16, enc_depth=1, enc_heads=2,
                           emb_dim=16, dec_depth=1, dec_heads=2)
    pretrainer = MAEPretrainer(config, decoder_dim=16, decoder_depth=1)
    model = TinySAM(config)
    checkpoint = tmp_path / "mae_encoder.pt"
    torch.save(mae_encoder_state_dict(pretrainer), checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)

    model_state = model.state_dict()
    assert state
    assert all(key.startswith("image_encoder.") for key in state)
    assert all(key in model_state for key in state)
    result = model.load_state_dict(state, strict=False)
    assert result.unexpected_keys == []
    for key, value in state.items():
        assert torch.equal(model.state_dict()[key], value), key
