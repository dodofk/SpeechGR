import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


def _entropy_perplexity(values: List[int]) -> float:
    counts = Counter(values)
    total = len(values)
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log2(p)
    return 2 ** entropy


def _load_id_map(path: Path) -> Dict[str, List[int]]:
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("id_map must be a JSON object mapping doc_id -> code list")

    normalized: Dict[str, List[int]] = {}
    for doc_id, code in data.items():
        if not isinstance(code, list):
            raise ValueError(f"Code for {doc_id} is not a list")
        normalized[str(doc_id)] = [int(x) for x in code]
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick quality check for RQ-VAE id_map")
    parser.add_argument("--id_map", type=str, required=True)
    parser.add_argument("--top_collisions", type=int, default=10)
    parser.add_argument("--output_json", type=str, default="")
    args = parser.parse_args()

    id_map = _load_id_map(Path(args.id_map))
    if not id_map:
        raise ValueError("id_map is empty")

    codes = [tuple(v) for v in id_map.values()]
    n_docs = len(codes)
    n_unique = len(set(codes))
    collisions = n_docs - n_unique
    collision_rate = collisions / n_docs

    code_len_set = {len(c) for c in codes}
    if len(code_len_set) != 1:
        raise ValueError(f"Mixed code lengths found in id_map: {sorted(code_len_set)}")
    num_layers = next(iter(code_len_set))

    counts = Counter(codes)
    max_docs_per_id = max(counts.values())

    per_layer_ppl = {}
    for layer_idx in range(num_layers):
        layer_values = [code[layer_idx] for code in codes]
        per_layer_ppl[f"layer_{layer_idx}"] = _entropy_perplexity(layer_values)

    top_collisions: List[Tuple[str, int]] = []
    for code_tuple, c in counts.most_common(args.top_collisions):
        if c <= 1:
            break
        top_collisions.append(("-".join(str(x) for x in code_tuple), c))

    result = {
        "num_documents": n_docs,
        "num_unique_ids": n_unique,
        "collision_count": collisions,
        "collision_rate": collision_rate,
        "max_docs_per_id": max_docs_per_id,
        "num_codebooks": num_layers,
        "per_layer_effective_perplexity": per_layer_ppl,
        "top_collisions": top_collisions,
    }

    print(json.dumps(result, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved report to {output_path}")


if __name__ == "__main__":
    main()

