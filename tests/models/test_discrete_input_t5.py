import pytest
import torch
from transformers import T5Config

from speechgr.model import DiscreteInputT5


def _tiny_t5_config() -> T5Config:
    return T5Config(
        vocab_size=32,
        d_model=16,
        d_ff=32,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        d_kv=8,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    )


def test_discrete_input_t5_initializes_from_text_embedding_statistics():
    model = DiscreteInputT5(_tiny_t5_config(), discrete_vocab_size=9)
    with torch.no_grad():
        model.shared.weight.fill_(0.25)
    model.initialize_discrete_input_embeddings("random_text")

    discrete_mean = float(model.discrete_input_embeddings.weight.mean().item())
    assert abs(discrete_mean - 0.25) < 0.2


def test_discrete_input_t5_rejects_out_of_range_input_ids():
    model = DiscreteInputT5(_tiny_t5_config(), discrete_vocab_size=4)

    with pytest.raises(ValueError):
        model(input_ids=torch.tensor([[0, 1, 4]]), labels=torch.tensor([[1, 1, 1]]))
