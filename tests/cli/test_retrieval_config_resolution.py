from speechgr import DataConfig
from speechgr.cli.retrieval import (
    _resolve_discrete_code_num,
    _resolve_discrete_vocab_size,
    _resolve_special_token,
    _validate_discrete_settings,
)


def test_special_token_defaults_from_codebook_size():
    data_cfg = DataConfig(
        dataset_path="dummy",
        modality="discrete_precomputed",
        codebook_size=2048,
        discrete_code_num=None,
        special_token=None,
    )

    assert _resolve_discrete_code_num(data_cfg) == 2048
    assert _resolve_special_token(data_cfg) == 2048
    assert _resolve_discrete_vocab_size(data_cfg) == 2049


def test_special_token_defaults_from_discrete_code_num():
    data_cfg = DataConfig(
        dataset_path="dummy",
        modality="discrete_precomputed",
        codebook_size=None,
        discrete_code_num=512,
        special_token=None,
    )

    assert _resolve_special_token(data_cfg) == 512
    assert _resolve_discrete_vocab_size(data_cfg) == 513


def test_validate_discrete_settings_rejects_colliding_special_token():
    data_cfg = DataConfig(
        dataset_path="dummy",
        modality="discrete_precomputed",
        discrete_code_num=8,
        special_token=7,
    )

    try:
        _validate_discrete_settings(data_cfg)
    except ValueError as exc:
        assert "special_token" in str(exc)
    else:  # pragma: no cover - keeps failure message precise
        raise AssertionError("Expected special_token collision to raise ValueError")
