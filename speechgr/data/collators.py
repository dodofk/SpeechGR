"""Collator utilities for SpeechGR datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase

from speechgr.encoders import TextEncoder


@dataclass
class IndexingCollator(DataCollatorWithPadding):
    """Collate discrete unit inputs with tokenized document-id labels."""

    def __call__(self, features):
        input_ids = [{"input_ids": x[0]} for x in features]
        docids = [x[1] for x in features]
        inputs = super().__call__(input_ids)

        labels = self.tokenizer(
            docids, padding="longest", return_tensors="pt"
        ).input_ids

        labels[labels == self.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        inputs["query_doc_id"] = torch.tensor([x[2] for x in features])
        return inputs


@dataclass
class IndexingCollatorWithAtomic(DataCollatorWithPadding):
    """Collator variant that keeps atomic identifiers as numeric labels."""

    def __call__(self, features):
        input_ids = [{"input_ids": x[0]} for x in features]
        docids = [x[1] for x in features]
        inputs = super().__call__(input_ids)
        labels = torch.tensor([[label, self.tokenizer.eos_token_id] for label in docids])
        inputs["labels"] = labels
        inputs["query_doc_id"] = torch.tensor([x[2] for x in features])
        return inputs


@dataclass
class IndexingCollatorWithMetadata(DataCollatorWithPadding):
    """Collator that preserves metadata indices alongside padded docid labels."""

    def __call__(self, features):
        input_ids = [{"input_ids": x[0]} for x in features]
        docids = [x[1] for x in features]
        inputs = super().__call__(input_ids)

        labels = self.tokenizer(
            docids, padding="longest", return_tensors="pt"
        ).input_ids

        labels[labels == self.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        inputs["query_doc_id"] = torch.tensor([x[2] for x in features])

        return inputs


@dataclass
class ContinuousIndexingCollator(DataCollatorWithPadding):
    """Collate continuous speech features by leveraging HF padding helpers."""

    def __call__(self, features):
        feature_tensors = [x[0] for x in features]
        docids = [x[1] for x in features]

        batch_dict = [{"input_ids": feat.tolist()} for feat in feature_tensors]
        padded_inputs = super().__call__(batch_dict)
        padded_features = padded_inputs["input_ids"]

        if not isinstance(padded_features, torch.Tensor):
            padded_features = torch.tensor(padded_features)

        labels = self.tokenizer(
            docids, padding="longest", return_tensors="pt"
        ).input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_features": padded_features,
            "attention_mask": padded_inputs["attention_mask"],
            "labels": labels,
        }


@dataclass
class WhisperIndexingCollator(DataCollatorWithPadding):
    """Collator tailored for Whisper encoder features."""

    feature_dtype: torch.dtype = torch.float32

    def __call__(self, features):
        feature_tensors: List[torch.Tensor] = []
        docids: List[str] = []
        doc_indices: List[int] = []
        lengths: List[int] = []

        for sample in features:
            if len(sample) == 4:
                feat, docid, doc_idx, seq_len = sample
            else:
                feat, docid, doc_idx = sample
                seq_len = feat.shape[0]

            if feat.ndim != 2:
                raise ValueError(
                    f"Expected Whisper features with shape (frames, dim); got {tuple(feat.shape)}"
                )

            feature_tensors.append(feat.to(self.feature_dtype))
            docids.append(docid)
            doc_indices.append(int(doc_idx))
            lengths.append(int(seq_len))

        if not feature_tensors:
            raise ValueError("WhisperIndexingCollator received an empty batch")

        max_seq_len = max(lengths)
        feature_dim = feature_tensors[0].shape[1]

        batch_size = len(feature_tensors)
        batched_features = feature_tensors[0].new_zeros(batch_size, max_seq_len, feature_dim)
        batched_attention_masks = torch.zeros(batch_size, max_seq_len, dtype=torch.long)

        for idx, (feat, seq_len) in enumerate(zip(feature_tensors, lengths)):
            if feat.shape[1] != feature_dim:
                raise ValueError(
                    "All Whisper features in a batch must share the same dimensionality"
                )
            batched_features[idx, :seq_len] = feat[:seq_len]
            batched_attention_masks[idx, :seq_len] = 1

        labels = self.tokenizer(
            docids, padding="longest", return_tensors="pt"
        ).input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_features": batched_features,
            "attention_mask": batched_attention_masks,
            "labels": labels,
            "query_doc_id": torch.tensor(doc_indices, dtype=torch.long),
        }


@dataclass
class TextIndexingCollator:
    """Collate text-encoded inputs using a shared tokenizer."""

    text_encoder: TextEncoder
    label_tokenizer: PreTrainedTokenizerBase
    train_atomic: bool = False

    def __call__(self, features):
        token_features = [sample[0] for sample in features]
        doc_labels = [sample[1] for sample in features]
        doc_indices = [sample[2] for sample in features]

        padded_inputs = self.text_encoder.tokenizer.pad(
            token_features, padding="longest", return_tensors="pt"
        )

        if self.train_atomic:
            labels = torch.tensor(
                [
                    [label, self.label_tokenizer.eos_token_id]
                    for label in doc_labels
                ],
                dtype=torch.long,
            )
        else:
            labels = self.label_tokenizer(
                doc_labels, padding="longest", return_tensors="pt"
            ).input_ids
            labels[labels == self.label_tokenizer.pad_token_id] = -100

        batch = dict(padded_inputs)
        batch["labels"] = labels
        batch["query_doc_id"] = torch.tensor(doc_indices, dtype=torch.long)
        return batch


__all__ = [
    "IndexingCollator",
    "IndexingCollatorWithAtomic",
    "IndexingCollatorWithMetadata",
    "ContinuousIndexingCollator",
    "WhisperIndexingCollator",
    "TextIndexingCollator",
]
