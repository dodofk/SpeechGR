"""Analyze hierarchical DocID distribution and cluster balance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from speechgr.docid import analyze_docid_map_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--docid-map",
        required=True,
        help="Path to docid_map.json",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to save the distribution report as JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze_docid_map_path(args.docid_map)
    payload = report.to_dict()

    print(json.dumps(payload, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))
        print(f"Saved distribution report to {output_path.resolve()}")


if __name__ == "__main__":
    main()
