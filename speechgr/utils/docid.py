"""Shared helpers for hierarchical DocID text handling."""

from __future__ import annotations

import re
from typing import Iterable


DOCID_TOKEN_PATTERN = re.compile(r"<[^\s<>]+>")


def normalize_docid_text(docid: str) -> str:
    text = str(docid).strip()
    tokens = DOCID_TOKEN_PATTERN.findall(text)
    if tokens:
        return " ".join(tokens)
    return " ".join(text.split())


def collect_docid_tokens(valid_ids: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    for docid in valid_ids:
        for token in DOCID_TOKEN_PATTERN.findall(str(docid)):
            seen.add(token)
    return sorted(seen)


__all__ = [
    "DOCID_TOKEN_PATTERN",
    "collect_docid_tokens",
    "normalize_docid_text",
]
