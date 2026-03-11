from types import SimpleNamespace

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
