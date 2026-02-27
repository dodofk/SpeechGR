import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def _load_checkpoint_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    raw = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(raw, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in raw and isinstance(raw[key], dict):
                return raw[key]
    if not isinstance(raw, dict):
        raise ValueError(
            f"Unsupported checkpoint format at {checkpoint_path}: {type(raw)}"
        )
    return raw


def _extract_rvq_embeddings(state_dict: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
    layer_to_table: Dict[int, torch.Tensor] = {}
    for key, value in state_dict.items():
        match = re.search(r"rvq\.layers\.(\d+)\.embedding$", key)
        if match is None:
            continue
        layer_idx = int(match.group(1))
        layer_to_table[layer_idx] = value.detach().cpu().float()

    if not layer_to_table:
        raise ValueError(
            "Could not find RVQ embedding tables in checkpoint. "
            "Expected keys ending with 'rvq.layers.<idx>.embedding'."
        )

    return [layer_to_table[i] for i in sorted(layer_to_table.keys())]


def _majority_vote(codes_2d: np.ndarray, num_embeddings: int) -> np.ndarray:
    if codes_2d.ndim != 2:
        raise ValueError(f"Expected 2D code array for voting, got shape {codes_2d.shape}")

    voted = []
    for col in range(codes_2d.shape[1]):
        counts = np.bincount(codes_2d[:, col], minlength=num_embeddings)
        voted.append(int(np.argmax(counts)))
    return np.array(voted, dtype=np.int64)


def _normalize_code_entry(
    entry,
    num_layers: int,
    codebook_size: int,
    compact_mode: str,
) -> np.ndarray:
    arr = np.asarray(entry, dtype=np.int64)

    if arr.ndim == 1:
        if arr.shape[0] != num_layers:
            raise ValueError(
                f"Code length mismatch: expected {num_layers}, got {arr.shape[0]}"
            )
        return arr

    if arr.ndim == 2:
        if arr.shape[1] != num_layers:
            raise ValueError(
                "Window-code layout mismatch: expected shape [num_windows, num_layers], "
                f"got {arr.shape}"
            )
        if compact_mode == "vote":
            return _majority_vote(arr, num_embeddings=codebook_size)
        if compact_mode == "first":
            return arr[0]
        raise ValueError(f"Unsupported compact mode: {compact_mode}")

    raise ValueError(f"Unsupported code entry rank: {arr.ndim}")


def _build_audio_vectors(
    id_map_path: str,
    rvq_embeddings: Sequence[torch.Tensor],
    compact_mode: str,
) -> Dict[str, np.ndarray]:
    with open(id_map_path, "r") as f:
        id_map = json.load(f)

    num_layers = len(rvq_embeddings)
    codebook_size = rvq_embeddings[0].shape[0]

    vectors: Dict[str, np.ndarray] = {}
    for audio_key, code_entry in id_map.items():
        codes = _normalize_code_entry(
            code_entry,
            num_layers=num_layers,
            codebook_size=codebook_size,
            compact_mode=compact_mode,
        )

        parts = []
        for layer_idx, code_id in enumerate(codes.tolist()):
            table = rvq_embeddings[layer_idx]
            if code_id < 0 or code_id >= table.shape[0]:
                raise ValueError(
                    f"Code {code_id} out of range for layer {layer_idx} "
                    f"(size {table.shape[0]})"
                )
            parts.append(table[code_id].numpy())

        vectors[audio_key] = np.concatenate(parts, axis=0)

    return vectors


def _read_pairs(path: str, audio_col: str, text_col: str) -> List[Tuple[str, str]]:
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".csv":
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            return [(row[audio_col], row[text_col]) for row in reader]

    if suffix == ".jsonl":
        rows = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rows.append((obj[audio_col], obj[text_col]))
        return rows

    if suffix == ".json":
        with open(path, "r") as f:
            obj = json.load(f)
        if not isinstance(obj, list):
            raise ValueError("JSON pairs file must be a list of objects")
        return [(row[audio_col], row[text_col]) for row in obj]

    raise ValueError(
        f"Unsupported pairs format: {suffix}. Use CSV, JSONL, or JSON list."
    )


def _normalize_key(key: str, match_mode: str) -> str:
    if match_mode == "exact":
        return key
    if match_mode == "basename":
        return Path(key).name
    if match_mode == "stem":
        return Path(key).stem
    raise ValueError(f"Unsupported match mode: {match_mode}")


def _encode_texts(
    texts: Sequence[str],
    model_name: str,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embeddings: List[torch.Tensor] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding text"):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            list(batch),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            token_emb = outputs.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (token_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = F.normalize(pooled, p=2, dim=1)
            all_embeddings.append(pooled.cpu())

    return torch.cat(all_embeddings, dim=0)


def _topk_neighbors(emb: torch.Tensor, k: int) -> torch.Tensor:
    emb = F.normalize(emb, p=2, dim=1)
    sim = emb @ emb.T
    sim.fill_diagonal_(-1e9)
    _, idx = torch.topk(sim, k=k, dim=1, largest=True, sorted=False)
    return idx


def _knn_agreement(a_idx: torch.Tensor, b_idx: torch.Tensor) -> float:
    if a_idx.shape != b_idx.shape:
        raise ValueError(
            f"kNN index shape mismatch: {tuple(a_idx.shape)} vs {tuple(b_idx.shape)}"
        )

    n, k = a_idx.shape
    total = 0.0
    for i in range(n):
        a_set = set(a_idx[i].tolist())
        b_set = set(b_idx[i].tolist())
        total += len(a_set.intersection(b_set)) / float(k)
    return total / float(n)


def main():
    parser = argparse.ArgumentParser(
        description="Cross-view kNN agreement between RQ-VAE codes and transcript embeddings"
    )
    parser.add_argument("--id_map", type=str, required=True)
    parser.add_argument("--rqvae_checkpoint", type=str, required=True)
    parser.add_argument("--pairs_file", type=str, required=True)
    parser.add_argument("--audio_col", type=str, default="audio_path")
    parser.add_argument("--text_col", type=str, default="transcript")
    parser.add_argument(
        "--key_match_mode",
        type=str,
        default="basename",
        choices=["exact", "basename", "stem"],
    )
    parser.add_argument(
        "--compact_window_codes",
        type=str,
        default="vote",
        choices=["vote", "first"],
        help="How to compact [num_windows, num_codebooks] entries in id_map",
    )
    parser.add_argument("--text_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--text_batch_size", type=int, default=64)
    parser.add_argument("--text_max_length", type=int, default=128)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_json", type=str, default="")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Loading RQ-VAE checkpoint embeddings...")
    state_dict = _load_checkpoint_state_dict(args.rqvae_checkpoint)
    rvq_embeddings = _extract_rvq_embeddings(state_dict)
    print(f"Found {len(rvq_embeddings)} RVQ layers")

    print("Building audio vectors from id_map...")
    audio_vec_map = _build_audio_vectors(
        id_map_path=args.id_map,
        rvq_embeddings=rvq_embeddings,
        compact_mode=args.compact_window_codes,
    )
    print(f"Loaded {len(audio_vec_map)} audio vectors")

    print("Loading paired transcript data...")
    pairs = _read_pairs(args.pairs_file, args.audio_col, args.text_col)

    normalized_audio_lookup = {}
    for key, vec in audio_vec_map.items():
        norm_key = _normalize_key(key, args.key_match_mode)
        if norm_key not in normalized_audio_lookup:
            normalized_audio_lookup[norm_key] = vec

    aligned_audio = []
    aligned_text = []
    for audio_key, text in pairs:
        norm_key = _normalize_key(audio_key, args.key_match_mode)
        if norm_key not in normalized_audio_lookup:
            continue
        if text is None or str(text).strip() == "":
            continue
        aligned_audio.append(normalized_audio_lookup[norm_key])
        aligned_text.append(str(text))

    if len(aligned_audio) < max(args.k + 1, 20):
        raise ValueError(
            f"Not enough aligned pairs after matching: {len(aligned_audio)}. "
            "Check key columns/match mode."
        )

    if args.max_samples > 0 and len(aligned_audio) > args.max_samples:
        indices = list(range(len(aligned_audio)))
        random.shuffle(indices)
        indices = indices[: args.max_samples]
        aligned_audio = [aligned_audio[i] for i in indices]
        aligned_text = [aligned_text[i] for i in indices]

    print(f"Aligned sample count: {len(aligned_audio)}")

    audio_emb = torch.from_numpy(np.stack(aligned_audio, axis=0)).float()
    audio_emb = F.normalize(audio_emb, p=2, dim=1)

    text_emb = _encode_texts(
        aligned_text,
        model_name=args.text_model,
        batch_size=args.text_batch_size,
        max_length=args.text_max_length,
        device=device,
    )

    k = min(args.k, len(aligned_audio) - 1)
    audio_knn = _topk_neighbors(audio_emb, k=k)
    text_knn = _topk_neighbors(text_emb, k=k)
    agreement = _knn_agreement(audio_knn, text_knn)

    shuffled = torch.randperm(text_emb.size(0))
    text_knn_shuffled = _topk_neighbors(text_emb[shuffled], k=k)
    agreement_shuffled = _knn_agreement(audio_knn, text_knn_shuffled)

    results = {
        "num_samples": len(aligned_audio),
        "k": k,
        "knn_agreement": agreement,
        "knn_agreement_shuffled_baseline": agreement_shuffled,
        "agreement_lift": agreement - agreement_shuffled,
        "text_model": args.text_model,
        "key_match_mode": args.key_match_mode,
        "compact_window_codes": args.compact_window_codes,
    }

    print(json.dumps(results, indent=2))

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved metrics to {args.output_json}")


if __name__ == "__main__":
    main()
