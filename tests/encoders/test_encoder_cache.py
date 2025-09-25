import torch

from speechgr.encoders.base import ModalityEncoder


class DummyEncoder(ModalityEncoder):
    def __init__(self):
        super().__init__(name="dummy")

    def supports_precompute(self) -> bool:
        return True

    def precompute(self, dataset_split, output_dir, samples):
        cache = {str(sample["id"]): torch.tensor(sample["value"]) for sample in samples}
        cache_path = self.cache_path(dataset_split, output_dir)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, cache_path)
        self._loaded_cache = cache
        self._loaded_cache_key = (dataset_split, output_dir)

    def _load_cache(self, path):
        return torch.load(path)


def test_encoder_cache_roundtrip(tmp_path):
    encoder = DummyEncoder()
    samples = [{"id": "item-1", "value": [1, 2, 3]}]

    encoder.precompute("train", str(tmp_path), samples)

    assert encoder.has_precomputed("train", str(tmp_path))

    loaded = encoder.load_feature("train", str(tmp_path), "item-1")
    assert torch.equal(loaded, torch.tensor([1, 2, 3]))

    # ensure memoised cache is reused without reloading from disk
    encoder.load_feature("train", str(tmp_path), "item-1")
