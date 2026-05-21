"""CLI benchmark for the Phase 2 batch candidate kernel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.accelerate.fortran_batch import benchmark_to_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Fortran batch top-k candidate extraction")
    parser.add_argument("--anchor-label", default="french")
    parser.add_argument("--reference-label", default="latin")
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--build-dir", type=Path, default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or Path("data/validation/fortran_batch_benchmark_phase2.json")
    payload = benchmark_to_json(
        output,
        anchor_label=args.anchor_label,
        reference_label=args.reference_label,
        top_k=args.top_k,
        repeats=args.repeats,
        build_dir=args.build_dir,
        force_rebuild=args.force_rebuild,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
