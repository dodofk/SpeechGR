import argparse
import csv
import io
import json
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from tqdm import tqdm

from speechgr.models.rqvae import DocumentRQVAE, SlidingWindowDocumentRQVAE
from speechgr.models.ssl_wrapper import SSLModelWrapper


def _unwrap_state_dict(raw_checkpoint):
    if not isinstance(raw_checkpoint, dict):
        return raw_checkpoint
    for key in ("state_dict", "model_state_dict", "model"):
        if key in raw_checkpoint and isinstance(raw_checkpoint[key], dict):
            return raw_checkpoint[key]
    return raw_checkpoint


def _infer_pooling_type_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> str:
    if any(k.startswith("pooling.pool.") or k.startswith("decoder.self_attn_layers") for k in state_dict):
        return "sliding_window"
    return "global"


def _build_rqvae(args, input_dim: int, pooling_type: str):
    common_kwargs = {
        "input_dim": input_dim,
        "latent_dim": args.rqvae_latent_dim,
        "codebook_size": args.rqvae_codebook_size,
        "num_codebooks": args.rqvae_num_codebooks,
        "commitment_cost": args.rqvae_commitment_cost,
        "decay": args.rqvae_decay,
        "num_encoder_layers": args.rqvae_num_encoder_layers,
        "num_decoder_layers": args.rqvae_num_decoder_layers,
    }

    if pooling_type == "sliding_window":
        return SlidingWindowDocumentRQVAE(
            **common_kwargs,
            window_size=args.rqvae_window_size,
            window_stride=args.rqvae_window_stride,
            pooling_hidden_dim=args.rqvae_pooling_hidden_dim,
            aggregate_for_retrieval=args.rqvae_aggregate_for_retrieval,
        )

    return DocumentRQVAE(**common_kwargs)


def _compact_window_codes(codes: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "vote":
        return torch.mode(codes, dim=1).values
    if mode == "first":
        return codes[:, 0]
    raise ValueError(f"Unsupported compaction mode: {mode}")


def _load_audio_from_hf_field(audio_entry, target_sr: int) -> np.ndarray:
    path = audio_entry.get("path")
    raw_bytes = audio_entry.get("bytes")

    if path:
        wav, sr = sf.read(path)
    elif raw_bytes is not None:
        wav, sr = sf.read(io.BytesIO(raw_bytes))
    else:
        raise ValueError("Audio field missing both path and bytes")

    if wav.ndim > 1:
        wav = wav.mean(axis=1)

    if sr != target_sr:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=target_sr)

    return wav.astype(np.float32)


def _collect_unique_documents(dataset, splits: List[str]) -> List[Tuple[str, dict, str]]:
    seen = set()
    docs: List[Tuple[str, dict, str]] = []

    for split in splits:
        for row in dataset[split]:
            doc_id = str(row["document_id"])
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc_text = row.get("normalized_document_text") or row.get("raw_document_text") or ""
            docs.append((doc_id, row["document_audio"], str(doc_text)))

    return docs


def _write_pairs_csv(path: Path, rows: List[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["document_id", "document_text"])
        writer.writeheader()
        for doc_id, doc_text in rows:
            writer.writerow({"document_id": doc_id, "document_text": doc_text})


def main():
    parser = argparse.ArgumentParser(
        description="Build SLUE-SQA5 document id_map using a trained RQ-VAE checkpoint"
    )
    parser.add_argument("--output_id_map", type=str, required=True)
    parser.add_argument("--output_pairs_csv", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="asapp/slue-phase-2")
    parser.add_argument("--dataset_config", type=str, default="sqa5")
    parser.add_argument(
        "--splits",
        type=str,
        default="train,validation,test,verified_test",
        help="Comma-separated SLUE splits to scan for unique document IDs",
    )
    parser.add_argument("--max_docs", type=int, default=0)

    parser.add_argument("--ssl_model", type=str, default="microsoft/wavlm-large")
    parser.add_argument("--ssl_layer", type=int, default=22)
    parser.add_argument("--audio_sample_rate", type=int, default=16000)

    parser.add_argument("--rqvae_checkpoint", type=str, required=True)
    parser.add_argument(
        "--rqvae_pooling_type",
        type=str,
        default="auto",
        choices=["auto", "global", "sliding_window"],
    )
    parser.add_argument("--rqvae_latent_dim", type=int, default=1024)
    parser.add_argument("--rqvae_codebook_size", type=int, default=256)
    parser.add_argument("--rqvae_num_codebooks", type=int, default=4)
    parser.add_argument("--rqvae_num_encoder_layers", type=int, default=4)
    parser.add_argument("--rqvae_num_decoder_layers", type=int, default=6)
    parser.add_argument("--rqvae_commitment_cost", type=float, default=0.25)
    parser.add_argument("--rqvae_decay", type=float, default=0.99)
    parser.add_argument("--rqvae_window_size", type=int, default=25)
    parser.add_argument("--rqvae_window_stride", type=int, default=8)
    parser.add_argument("--rqvae_pooling_hidden_dim", type=int, default=128)
    parser.add_argument(
        "--rqvae_aggregate_for_retrieval",
        type=str,
        default="vote",
        choices=["vote", "mean", "first", "all"],
    )
    parser.add_argument(
        "--rqvae_compact_all_mode",
        type=str,
        default="vote",
        choices=["vote", "first"],
        help="Fallback compaction if encode() returns window-level codes",
    )

    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = requested_device

    print("Loading SLUE dataset metadata from Hugging Face...")
    dataset = load_dataset(args.dataset_name, args.dataset_config)
    dataset = dataset.cast_column("question_audio", Audio(decode=False))
    dataset = dataset.cast_column("document_audio", Audio(decode=False))

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    docs = _collect_unique_documents(dataset, splits=splits)
    if args.max_docs > 0:
        docs = docs[: args.max_docs]

    print(f"Collected {len(docs)} unique documents")

    print(f"Loading SSL model: {args.ssl_model} (layer={args.ssl_layer})")
    ssl_model = SSLModelWrapper(args.ssl_model, layer=args.ssl_layer, freeze=True).to(device)
    ssl_model.eval()

    print(f"Loading RQ-VAE checkpoint: {args.rqvae_checkpoint}")
    raw_checkpoint = torch.load(args.rqvae_checkpoint, map_location=device)
    state_dict = _unwrap_state_dict(raw_checkpoint)

    pooling_type = args.rqvae_pooling_type
    if pooling_type == "auto":
        pooling_type = _infer_pooling_type_from_state_dict(state_dict)
    print(f"Detected/selected RQ-VAE pooling type: {pooling_type}")

    rqvae = _build_rqvae(args, ssl_model.feature_dim, pooling_type).to(device)
    load_result = rqvae.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        print(f"Warning: missing keys when loading RQ-VAE: {len(load_result.missing_keys)}")
    if load_result.unexpected_keys:
        print(f"Warning: unexpected keys when loading RQ-VAE: {len(load_result.unexpected_keys)}")
    rqvae.eval()

    id_map: Dict[str, List[int]] = {}
    pair_rows: List[Tuple[str, str]] = []
    warned_window_codes = False

    with torch.no_grad():
        for doc_id, audio_entry, doc_text in tqdm(docs, desc="Encoding documents"):
            wav = _load_audio_from_hf_field(audio_entry, target_sr=args.audio_sample_rate)
            audio_tensor = torch.from_numpy(wav).unsqueeze(0).to(device)

            feats = ssl_model(audio_tensor)
            codes = rqvae.encode(feats)

            if codes.dim() == 3:
                if not warned_window_codes:
                    print(
                        "Warning: encode() returned window-level codes; "
                        f"compacting with mode={args.rqvae_compact_all_mode}"
                    )
                    warned_window_codes = True
                codes = _compact_window_codes(codes, args.rqvae_compact_all_mode)

            if codes.dim() != 2:
                raise ValueError(
                    "Expected code tensor shape [B, num_codebooks], "
                    f"got {tuple(codes.shape)}"
                )

            id_map[doc_id] = codes.squeeze(0).cpu().tolist()
            pair_rows.append((doc_id, doc_text))

    output_id_map = Path(args.output_id_map)
    output_id_map.parent.mkdir(parents=True, exist_ok=True)
    with output_id_map.open("w") as f:
        json.dump(id_map, f)
    print(f"Saved id_map to {output_id_map}")

    if args.output_pairs_csv:
        pairs_path = Path(args.output_pairs_csv)
        _write_pairs_csv(pairs_path, pair_rows)
        print(f"Saved document pairs CSV to {pairs_path}")


if __name__ == "__main__":
    main()

