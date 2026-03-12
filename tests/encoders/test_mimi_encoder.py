import numpy as np
import torch

from speechgr.encoders.mimi.encoder import (
    DEFAULT_MIMI_MODEL_NAME_OR_PATH,
    MimiEncoder,
)
from speechgr.encoders.registry import get_encoder_class, list_encoders


class DummyMimiTokenizer:
    codebook_size = 2048
    num_semantic_quantizers = 1

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


def test_mimi_encoder_semantic_only_uses_first_quantizer():
    class MultiCodebookTokenizer:
        codebook_size = 8
        num_semantic_quantizers = 1

        def __call__(self, waveform: torch.Tensor, *, sampling_rate: int):
            del waveform, sampling_rate
            return {
                "codes": torch.tensor(
                    [[[1, 2, 3], [4, 5, 6]]],
                    dtype=torch.long,
                )
            }

    encoder = MimiEncoder(
        tokenizer=MultiCodebookTokenizer(),
        code_selection="semantic_only",
    )

    codes = encoder.encode_audio(np.zeros(24_000, dtype=np.float32), sampling_rate=24_000)
    assert torch.equal(codes, torch.tensor([1, 2, 3], dtype=torch.long))


def test_mimi_encoder_first_n_interleaves_with_offsets():
    class MultiCodebookTokenizer:
        codebook_size = 8
        num_semantic_quantizers = 1

        def __call__(self, waveform: torch.Tensor, *, sampling_rate: int):
            del waveform, sampling_rate
            return {
                "codes": torch.tensor(
                    [[[1, 2], [3, 4]]],
                    dtype=torch.long,
                )
            }

    encoder = MimiEncoder(
        tokenizer=MultiCodebookTokenizer(),
        code_selection="first_n",
        num_selected_quantizers=2,
    )

    codes = encoder.encode_audio(np.zeros(24_000, dtype=np.float32), sampling_rate=24_000)
    assert torch.equal(codes, torch.tensor([1, 11, 2, 12], dtype=torch.long))


def test_mimi_encoder_validates_output_vocab_size_for_interleaved_modes():
    class MultiCodebookTokenizer:
        codebook_size = 8
        num_semantic_quantizers = 1

        def __call__(self, waveform: torch.Tensor, *, sampling_rate: int):
            del waveform, sampling_rate
            return {
                "codes": torch.tensor(
                    [[[1, 2], [3, 4]]],
                    dtype=torch.long,
                )
            }

    encoder = MimiEncoder(
        tokenizer=MultiCodebookTokenizer(),
        code_selection="first_n",
        num_selected_quantizers=2,
        output_vocab_size=8,
    )

    try:
        encoder.encode_audio(np.zeros(24_000, dtype=np.float32), sampling_rate=24_000)
    except ValueError as exc:
        assert "output_vocab_size" in str(exc)
    else:  # pragma: no cover - keeps failure message explicit
        raise AssertionError("Expected output_vocab_size validation to raise ValueError")


def test_mimi_encoder_semantic_only_handles_single_frame_output():
    class SingleFrameTokenizer:
        codebook_size = 8
        num_semantic_quantizers = 1

        def __call__(self, waveform: torch.Tensor, *, sampling_rate: int):
            del waveform, sampling_rate
            return {
                "codes": torch.tensor(
                    [[[7], [3], [5]]],
                    dtype=torch.long,
                )
            }

    encoder = MimiEncoder(
        tokenizer=SingleFrameTokenizer(),
        code_selection="semantic_only",
    )

    codes = encoder.encode_audio(np.zeros(24_000, dtype=np.float32), sampling_rate=24_000)
    assert torch.equal(codes, torch.tensor([7], dtype=torch.long))


def test_mimi_encoder_precompute_batches_samples(tmp_path):
    class BatchTokenizer:
        codebook_size = 8
        num_semantic_quantizers = 1

        def __init__(self):
            self.batch_sizes = []

        def __call__(self, waveform, *, sampling_rate: int):
            assert sampling_rate == 24_000
            if isinstance(waveform, list):
                batch_size = len(waveform)
            else:
                batch_size = 1
            self.batch_sizes.append(batch_size)
            return {
                "codes": torch.tensor(
                    [[[1, 2, 3]]] * batch_size,
                    dtype=torch.long,
                )
            }

    tokenizer = BatchTokenizer()
    encoder = MimiEncoder(
        tokenizer=tokenizer,
        audio_field="question_audio",
        sample_id_field="question_id",
        batch_size=2,
    )

    audio = np.linspace(-1.0, 1.0, 16_000, dtype=np.float32)
    samples = [
        {"question_id": "q1", "question_audio": {"array": audio, "sampling_rate": 16_000}},
        {"question_id": "q2", "question_audio": {"array": audio, "sampling_rate": 16_000}},
        {"question_id": "q3", "question_audio": {"array": audio, "sampling_rate": 16_000}},
    ]

    encoder.precompute("train", str(tmp_path), samples)

    assert tokenizer.batch_sizes == [2, 1]
