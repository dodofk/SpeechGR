#!/usr/bin/env python
"""
Simple, one‑file‑at‑a‑time extractor that dumps each SLUE‑SQA‑5 audio sample's
SSL features to an individual **.npy** file.  No LMDB, no HDF5—just a folder
full of NumPy arrays you can load with `np.load(path)`.

* Each *question* sample →  `<output_dir>/<split>/q_<question_id>.npy`
* Each *document* (corpus) sample → `<output_dir>/corpus/doc_<document_id>.npy`

Change the default `--output_dir` to wherever you want the dump to live.  If you
rerun the script it will skip files that already exist, so you can resume after
interruptions.

Dependencies
------------
    pip install datasets soundfile torchaudio transformers numpy tqdm

Example
-------
    python save_ssl_feature_npz.py \
        --output_dir /path/to/feat_dump \
        --model microsoft/wavlm-large \
        --use_vad        # strip silence before encoding
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle  # only for optional meta save
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from datasets import Audio, DatasetDict, load_dataset
from tqdm.auto import tqdm
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    FeatureExtractionMixin,
    PreTrainedModel,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)8s | %(message)s"
)
logger = logging.getLogger("ssl‑npy")

################################################################################
# CLI
################################################################################


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Dump SSL features to .npy files")
    p.add_argument(
        "--output_dir",
        type=str,
        default="/home/ricky/dodofk/dataset/slue_sqa5_wavlm_large",
        help="Root directory for .npy dumps",
    )
    p.add_argument(
        "--model", type=str, default="microsoft/wavlm-large", help="HF model checkpoint"
    )
    p.add_argument(
        "--use_vad",
        action="store_true",
        default=False,
        help="Run Silero VAD before encoding",
    )
    p.add_argument(
        "--use_cuda", action="store_true", default=False, help="Use GPU if available"
    )
    p.add_argument(
        "--splits",
        type=str,
        nargs="*",
        default=["train", "validation", "test", "verified_test"],
        help="Dataset splits to process",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re‑compute even if .npy already exists",
    )
    return p.parse_args()


################################################################################
# VAD helpers (optional)
################################################################################


def load_vad_model(use_cuda: bool):
    logger.info("Loading Silero‑VAD …")
    model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
    (get_speech_timestamps, *_rest) = utils
    model.eval()
    if use_cuda and torch.cuda.is_available():
        model = model.to("cuda")
    return model, get_speech_timestamps


def strip_silence(
    wav: np.ndarray, vad_model, get_speech_timestamps, use_cuda: bool
) -> np.ndarray:
    """Return waveform with silent regions removed (mono 16 kHz expected)."""
    tensor = torch.tensor(wav, dtype=torch.float32).squeeze()
    if use_cuda and torch.cuda.is_available():
        tensor = tensor.to("cuda")
    ts = get_speech_timestamps(tensor, vad_model, threshold=0.5, sampling_rate=16000)
    if not ts:
        return wav
    chunks = [tensor[t["start"] : t["end"]] for t in ts]
    return torch.cat(chunks).cpu().numpy()


################################################################################
# Feature extraction (single example)
################################################################################


def extract_features(
    wav: np.ndarray,
    extractor: FeatureExtractionMixin,
    model: PreTrainedModel,
    use_cuda: bool,
) -> np.ndarray:
    """Return (T, hidden_size) float32 array."""
    inputs = extractor(wav, sampling_rate=16000, return_tensors="pt")
    if use_cuda and torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.inference_mode():
        hidden = model(**inputs).last_hidden_state.squeeze(0)  # T × 1024
    return hidden.cpu().numpy().astype(np.float32)


################################################################################
# Main processing routines
################################################################################


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def process_corpus(ds: DatasetDict, out_root: Path, extractor, model, args):
    corpus_dir = out_root / "corpus"
    ensure_dir(corpus_dir)
    logger.info("[Corpus] Saving to %s", corpus_dir)

    if args.use_vad:
        vad_model, get_speech_timestamps = load_vad_model(args.use_cuda)
    else:
        vad_model = get_speech_timestamps = None  # type: ignore

    processed = 0
    for split in args.splits:
        for ex in tqdm(ds[split], desc=f"corpus:{split}"):
            doc_id = ex["document_id"]
            wav = ex["document_audio"]["array"]
            out_path = corpus_dir / f"{doc_id}.npy"
            if out_path.exists() and not args.overwrite:
                continue
            if args.use_vad:
                wav = strip_silence(
                    wav, vad_model, get_speech_timestamps, args.use_cuda
                )
            feat = extract_features(wav, extractor, model, args.use_cuda)
            np.save(out_path, feat)
            processed += 1
    logger.info("[Corpus] %d files written", processed)


def process_questions(ds: DatasetDict, out_root: Path, extractor, model, args):
    if args.use_vad:
        vad_model, get_speech_timestamps = load_vad_model(args.use_cuda)
    else:
        vad_model = get_speech_timestamps = None  # type: ignore

    for split in args.splits:
        split_dir = out_root / split
        ensure_dir(split_dir)
        logger.info("[%s] Saving to %s", split, split_dir)
        processed = 0
        for ex in tqdm(ds[split], desc=split):
            qid = ex["question_id"]
            wav = ex["question_audio"]["array"]
            out_path = split_dir / f"{qid}.npy"
            if out_path.exists() and not args.overwrite:
                continue
            if args.use_vad:
                wav = strip_silence(
                    wav, vad_model, get_speech_timestamps, args.use_cuda
                )
            feat = extract_features(wav, extractor, model, args.use_cuda)
            np.save(out_path, feat)
            processed += 1
        logger.info("[%s] %d files written", split, processed)


################################################################################
# Entry point
################################################################################


def main():
    args = parse_args()
    out_root = Path(args.output_dir)
    ensure_dir(out_root)

    logger.info("Loading SLUE‑SQA‑5 … (this may take a moment)")
    ds: DatasetDict = load_dataset("asapp/slue-phase-2", "sqa5", streaming=True)
    # cast audio columns to ensure 16 kHz numpy arrays

    logger.info("Loading model %s", args.model)
    extractor = AutoFeatureExtractor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).eval()
    if args.use_cuda and torch.cuda.is_available():
        model = model.to("cuda")
        logger.info("CUDA enabled")

    # documents first (shared across splits)
    process_corpus(ds, out_root, extractor, model, args)
    # then questions per split
    process_questions(ds, out_root, extractor, model, args)

    logger.info("All done ✨  Features are under %s", out_root)


if __name__ == "__main__":
    main()
