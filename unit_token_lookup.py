"""Shared discrete-unit to T5-token lookup utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from transformers import AutoTokenizer, PreTrainedTokenizerBase


DEFAULT_TOKEN_LOOKUP_PATH = "ckpts/token_lookups/flan-t5-base-c500-l22-token-lookup.txt"
DEFAULT_LOOKUP_SOURCE_CSV = (
    "/home/ricky/dodofk/dataset/slue_sqa5/slue_sqa5_pq10_llama32_3b_clean.csv"
)
DEFAULT_LOOKUP_TEXT_COLUMN = "post_query"
CONFIG_LOOKUP_KEY = "speechgr_unit_token_lookup"


def sentinel_token_ids(
    sentinel_start_id: int = 32099,
    sentinel_direction: int = -1,
    count: int = 100,
) -> set[int]:
    return {sentinel_start_id + sentinel_direction * i for i in range(count)}


def _tokenizer_reserved_ids(tokenizer: PreTrainedTokenizerBase) -> set[int]:
    reserved = {int(tok) for tok in tokenizer.all_special_ids}
    for attr in ("pad_token_id", "eos_token_id", "unk_token_id", "bos_token_id"):
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
            reserved.add(int(token_id))
    reserved.update(sentinel_token_ids())
    return reserved


def load_token_lookup(
    path: str | Path,
    discrete_code_num: Optional[int] = None,
) -> np.ndarray:
    lookup = np.loadtxt(path, dtype=int)
    if lookup.ndim == 0:
        lookup = np.array([int(lookup)])
    lookup = lookup.astype(np.int64)
    if discrete_code_num is not None:
        if len(lookup) < discrete_code_num:
            raise ValueError(
                f"token lookup only provides {len(lookup)} ids, "
                f"but discrete_code_num={discrete_code_num}"
            )
        lookup = lookup[:discrete_code_num]
    return lookup


def save_token_lookup(path: str | Path, lookup: Sequence[int]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, np.asarray(lookup, dtype=np.int64), fmt="%d")
    return output_path


def build_unused_token_lookup(
    tokenizer: PreTrainedTokenizerBase,
    texts: Iterable[str],
    discrete_code_num: int,
    min_token_id: int = 20,
) -> np.ndarray:
    used_ids: set[int] = set()
    for text in texts:
        if text is None or pd.isna(text):
            continue
        used_ids.update(int(tok) for tok in tokenizer(str(text))["input_ids"])

    vocab_size = max(tokenizer.vocab_size, len(tokenizer))
    reserved_ids = _tokenizer_reserved_ids(tokenizer)
    candidates = [
        token_id
        for token_id in range(min_token_id, vocab_size)
        if token_id not in used_ids and token_id not in reserved_ids
    ]
    if len(candidates) < discrete_code_num:
        raise ValueError(
            "Not enough unused tokenizer ids for the discrete-unit lookup. "
            f"Need {discrete_code_num}, got {len(candidates)}."
        )
    return np.asarray(candidates[:discrete_code_num], dtype=np.int64)


def build_lookup_from_csv(
    csv_path: str | Path,
    tokenizer_name_or_path: str,
    discrete_code_num: int,
    text_column: str = DEFAULT_LOOKUP_TEXT_COLUMN,
) -> np.ndarray:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Lookup source CSV does not exist: {csv_path}. "
            "Provide --token_file or generate it from an available SLUE metadata CSV."
        )
    df = pd.read_csv(csv_path)
    if text_column not in df.columns:
        raise ValueError(
            f"Lookup source CSV {csv_path} does not contain column {text_column!r}."
        )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    return build_unused_token_lookup(
        tokenizer=tokenizer,
        texts=df[text_column].dropna().astype(str).tolist(),
        discrete_code_num=discrete_code_num,
    )


def lookup_from_model_config(
    config,
    discrete_code_num: Optional[int] = None,
) -> Optional[np.ndarray]:
    values = getattr(config, CONFIG_LOOKUP_KEY, None)
    if values is None:
        return None
    lookup = np.asarray(values, dtype=np.int64)
    if discrete_code_num is not None:
        lookup = lookup[:discrete_code_num]
    return lookup


def attach_lookup_to_config(config, lookup: Sequence[int], **metadata) -> None:
    setattr(config, CONFIG_LOOKUP_KEY, [int(value) for value in lookup])
    for key, value in metadata.items():
        setattr(config, f"speechgr_{key}", value)
