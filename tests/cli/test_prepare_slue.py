from types import SimpleNamespace
from pathlib import Path

from speechgr.cli import prepare_slue


class DummyDataset:
    def __init__(self):
        self.calls = []

    def cast_column(self, column_name, feature):
        self.calls.append((column_name, feature.decode))
        return self


def test_load_slue_dataset_uses_streaming_and_decode_flags(monkeypatch):
    captured = {}
    dataset = DummyDataset()

    def fake_load_dataset(name, config, streaming=False):
        captured["name"] = name
        captured["config"] = config
        captured["streaming"] = streaming
        return dataset

    monkeypatch.setattr(prepare_slue, "load_dataset", fake_load_dataset)

    cfg = SimpleNamespace(
        dataset_name="demo/slue",
        dataset_config="sqa5",
        streaming=True,
        decode_audio=False,
        get=lambda key, default=None: getattr(cfg, key, default),
    )

    loaded = prepare_slue._load_slue_dataset(cfg)

    assert loaded is dataset
    assert captured == {
        "name": "demo/slue",
        "config": "sqa5",
        "streaming": True,
    }
    assert dataset.calls == [
        ("question_audio", False),
        ("document_audio", False),
    ]


class DummyEncoder:
    def __init__(self):
        self.calls = []

    def cache_path(self, dataset_split: str, output_dir: str) -> Path:
        return Path(output_dir) / f"{dataset_split}_dummy.pt"

    def precompute(self, dataset_split: str, output_dir: str, dataset_split_data) -> None:
        self.calls.append((dataset_split, output_dir, dataset_split_data))


def test_precompute_split_skips_existing_cache(tmp_path):
    encoder = DummyEncoder()
    split_dir = tmp_path / "train"
    split_dir.mkdir(parents=True)
    (split_dir / "train_dummy.pt").write_text("cached")

    ran = prepare_slue._precompute_split(
        encoder,
        "train",
        [1, 2, 3],
        split_dir,
        skip_existing=True,
        force_recompute=False,
    )

    assert ran is False
    assert encoder.calls == []


def test_precompute_split_force_recompute_overrides_skip(tmp_path):
    encoder = DummyEncoder()
    split_dir = tmp_path / "train"
    split_dir.mkdir(parents=True)
    (split_dir / "train_dummy.pt").write_text("cached")

    ran = prepare_slue._precompute_split(
        encoder,
        "train",
        [1, 2, 3],
        split_dir,
        skip_existing=True,
        force_recompute=True,
    )

    assert ran is True
    assert len(encoder.calls) == 1
