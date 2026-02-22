import torch
import pytest

from speechgr.models.rqvae import SlidingWindowDocumentRQVAE


def _make_model(aggregate: str) -> SlidingWindowDocumentRQVAE:
    return SlidingWindowDocumentRQVAE(
        input_dim=16,
        latent_dim=16,
        codebook_size=8,
        num_codebooks=2,
        window_size=4,
        window_stride=2,
        num_encoder_layers=1,
        num_decoder_layers=2,
        aggregate_for_retrieval=aggregate,
    )


@pytest.mark.parametrize("aggregate", ["mean", "vote", "first"])
def test_encode_single_code_shape(aggregate: str):
    model = _make_model(aggregate)
    x = torch.randn(2, 12, 16)
    mask = torch.ones(2, 12, dtype=torch.bool)

    codes = model.encode(x, mask)

    assert codes.shape == (2, 2)


def test_encode_all_keeps_window_axis():
    model = _make_model("all")
    x = torch.randn(2, 12, 16)
    mask = torch.ones(2, 12, dtype=torch.bool)

    codes = model.encode(x, mask)

    assert codes.ndim == 3
    assert codes.shape[0] == 2
    assert codes.shape[2] == 2
    assert codes.shape[1] >= 1


def test_vote_aggregation_uses_majority_per_codebook():
    model = _make_model("vote")
    x = torch.randn(2, 12, 16)
    mask = torch.ones(2, 12, dtype=torch.bool)

    fixed_codes = torch.tensor(
        [
            [[1, 3], [1, 4], [2, 4]],
            [[5, 6], [5, 7], [7, 7]],
        ],
        dtype=torch.long,
    )

    class DummyPooling(torch.nn.Module):
        def forward(self, features, attn_mask=None):
            return features.new_zeros((features.size(0), 3, features.size(-1)))

    class DummyRVQ(torch.nn.Module):
        def forward(self, windows):
            return windows, windows.new_tensor(0.0), fixed_codes.to(windows.device)

    model.pooling = DummyPooling()
    model.rvq = DummyRVQ()

    codes = model.encode(x, mask)

    expected = torch.tensor([[1, 4], [5, 7]], dtype=torch.long)
    assert torch.equal(codes.cpu(), expected)


def test_invalid_aggregation_raises():
    with pytest.raises(ValueError):
        _make_model("median")

