import torch

from speechgr.utils_legacy import RestrictDecodeVocab


class DummyTokenizer:
    eos_token_id = 1

    def __init__(self):
        self._lookup = {
            "doc0": [10],
            "doc1": [11, 12],
        }

    def encode(self, docid, add_special_tokens=False):
        assert add_special_tokens is False
        return self._lookup[docid]


def test_restrict_decode_vocab_allows_eos_at_terminal_node():
    restrict = RestrictDecodeVocab(["doc0", "doc1"], DummyTokenizer())

    assert restrict(0, torch.tensor([0])) == [10, 11]
    assert restrict(0, torch.tensor([0, 10])) == [1]
