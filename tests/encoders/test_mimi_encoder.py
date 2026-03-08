import numpy as np
import torch

from speechgr.encoders.mimi.encoder import (
    DEFAULT_MIMI_MODEL_NAME_OR_PATH,
    MimiEncoder,
)
from speechgr.encoders.registry import get_encoder_class, list_encoders


class DummyMimiTokenizer:
    def __call__(self, waveform: torch.Tensor, *, sampling_rate: int):
        assert sampling_rate == 24_000
        length = waveform.shape[-1]
        steps = max(1, length // 400)
        return {"codes": torch.arange(steps, dtype=torch.long).unsqueeze(0)}


def test_mimi_encoder_registration():
    assert get_encoder_class("mimi") is MimiEncoder
    assert list_encoders()["mimi"] is MimiEncoder


def test_mimi_encoder_defaults_to_hf_model_id(monkeypatch):
    monkeypatch.delenv("MIMI_MODEL_NAME_OR_PATH", raising=False)

    encoder = MimiEncoder(tokenizer=DummyMimiTokenizer())

    assert encoder.model_name_or_path == DEFAULT_MIMI_MODEL_NAME_OR_PATH


def test_mimi_encoder_allows_env_override(monkeypatch):
    monkeypatch.setenv("MIMI_MODEL_NAME_OR_PATH", "/tmp/local-mimi")

    encoder = MimiEncoder(tokenizer=DummyMimiTokenizer())

    assert encoder.model_name_or_path == "/tmp/local-mimi"


def test_mimi_encoder_encode_and_cache_roundtrip(tmp_path):
    encoder = MimiEncoder(
        tokenizer=DummyMimiTokenizer(),
        audio_field="question_audio",
        sample_id_field="question_id",
    )
    audio = np.linspace(-1.0, 1.0, 16_000, dtype=np.float32)

    codes = encoder.encode_audio(audio, sampling_rate=16_000)

    assert torch.equal(codes, torch.arange(60, dtype=torch.long))

    samples = [
        {
            "question_id": "q1",
            "question_audio": {"array": audio, "sampling_rate": 16_000},
        }
    ]
    encoder.precompute("train", str(tmp_path), samples)

    loaded = encoder.load_feature("train", str(tmp_path), "q1")
    assert torch.equal(loaded["codes"], codes)
