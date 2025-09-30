import torch

from speechgr.data import SlueSQA5TextDataset
from speechgr.data.collators import TextIndexingCollator


class StubTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def pad(self, batch, padding="longest", return_tensors=None):
        max_len = max(len(item["input_ids"]) for item in batch)
        padded_ids, padded_mask = [], []
        for item in batch:
            ids = item["input_ids"] + [self.pad_token_id] * (max_len - len(item["input_ids"]))
            mask = item["attention_mask"] + [0] * (max_len - len(item["attention_mask"]))
            padded_ids.append(ids)
            padded_mask.append(mask)
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(padded_ids, dtype=torch.long),
                "attention_mask": torch.tensor(padded_mask, dtype=torch.long),
            }
        return {"input_ids": padded_ids, "attention_mask": padded_mask}

    def __call__(self, texts, padding="longest", return_tensors=None):
        encoded = [[len(text)] for text in texts]
        tensor = torch.tensor(encoded, dtype=torch.long)
        if return_tensors == "pt":
            return SimpleNamespace(input_ids=tensor)
        return {"input_ids": tensor}


class SimpleNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class StubTextEncoder:
    def __init__(self):
        self.tokenizer = StubTokenizer()

    def encode(self, text, max_length=None):
        tokens = [ord(ch) % 23 + 1 for ch in text][: max_length or len(text)]
        if not tokens:
            tokens = [1]
        attention = [1] * len(tokens)
        return {"input_ids": tokens, "attention_mask": attention}


def build_csv(path, rows):
    import csv

    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_text_dataset_and_collator(tmp_path):
    dataset_dir = tmp_path / "slue"
    dataset_dir.mkdir()

    train_rows = [
        {"question_id": "q1", "document_id": "d1", "post_query": "how are you"},
        {"question_id": "q2", "document_id": "d2", "post_query": "hello there"},
    ]
    pq_rows = [
        {"idx": 0, "question_id": "q1", "document_id": "d1", "post_query": "how are you"},
        {"idx": 1, "question_id": "q2", "document_id": "d2", "post_query": "hello there"},
    ]
    corpus_rows = [
        {"document_id": "d1", "normalized_document_text": "doc one"},
        {"document_id": "d2", "normalized_document_text": "doc two"},
    ]

    build_csv(dataset_dir / "train.csv", train_rows)
    build_csv(dataset_dir / "slue_sqa5_pq10_llama32_3b_clean.csv", pq_rows)
    build_csv(dataset_dir / "slue_sqa5_corpus.csv", corpus_rows)

    encoder = StubTextEncoder()

    dataset = SlueSQA5TextDataset(
        split="train",
        dataset_path=str(dataset_dir),
        text_encoder=encoder,
        query_text_field="post_query",
        corpus_text_field="normalized_document_text",
        include_corpus=True,
    )

    assert len(dataset) == len(train_rows) + len(corpus_rows)

    first_item = dataset[0]
    assert set(first_item[0].keys()) >= {"input_ids", "attention_mask"}
    assert isinstance(first_item[0]["input_ids"], list)

    collator = TextIndexingCollator(
        text_encoder=encoder,
        label_tokenizer=encoder.tokenizer,
        train_atomic=False,
    )

    batch = collator([dataset[0], dataset[1]])

    assert batch["input_ids"].shape[0] == 2
    assert batch["labels"].shape[0] == 2
    assert torch.all(batch["labels"][:, 0] > 0)
    assert "query_doc_id" in batch
