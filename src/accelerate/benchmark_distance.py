"""CLI benchmark for the Session 1 Fortran distance scaffold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.accelerate.fortran_distance import (
    DEFAULT_CURRENT_MATRIX,
    DEFAULT_REFERENCE_MATRIX,
    DEFAULT_BUILD_DIR,
    benchmark_distance,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark NumPy vs Fortran distance kernel")
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT_MATRIX)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE_MATRIX)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    current = np.load(args.current)
    reference = np.load(args.reference)
    result = benchmark_distance(
        current,
        reference,
        repeats=args.repeats,
        build_dir=args.build_dir,
        force_rebuild=args.force_rebuild,
    )
    payload = {
        "current_matrix": str(args.current),
        "reference_matrix": str(args.reference),
        **result.to_dict(),
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

