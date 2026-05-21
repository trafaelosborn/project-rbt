"""
Phase 4 benchmark for accepted-mutation tensor refresh.

Compare:
    - incremental state.apply_sequences(...)
    - full state rebuild from the updated sequences

This is the first honest measure of whether the new tensor state scaffold is
worth carrying forward into the live engine.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.accelerate.incremental_tensor_state import (
    IncrementalFingerprintTensorState,
    NgramVectorBasis,
    TensorStateConfig,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "romance"


@dataclass(frozen=True)
class IncrementalBenchmarkResult:
    language: str
    sampled_sequences: int
    changed_sequences: int
    anchor_vocab_size: int
    update_mode: str
    incremental_seconds: float
    full_rebuild_seconds: float
    speedup_vs_rebuild: float

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "sampled_sequences": self.sampled_sequences,
            "changed_sequences": self.changed_sequences,
            "anchor_vocab_size": self.anchor_vocab_size,
            "update_mode": self.update_mode,
            "incremental_seconds": round(self.incremental_seconds, 6),
            "full_rebuild_seconds": round(self.full_rebuild_seconds, 6),
            "speedup_vs_rebuild": round(self.speedup_vs_rebuild, 6),
        }


def _load_sequences(language: str) -> list[list[str]]:
    path = PROCESSED_DIR / f"{language}_tokens.json"
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)["sequences"]


def _sample_sequences(sequences: list[list[str]], *, count: int, seed: int) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    n = min(count, len(sequences))
    idx = rng.choice(len(sequences), size=n, replace=False)
    return [list(sequences[int(i)]) for i in idx]


def _mutate_sequences_in_anchor(
    sequences: list[list[str]],
    *,
    change_count: int,
    seed: int,
) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    mutated = [list(seq) for seq in sequences]
    candidates = [i for i, seq in enumerate(mutated) if len(seq) >= 3]
    if not candidates:
        return mutated

    picked = rng.choice(candidates, size=min(change_count, len(candidates)), replace=False)
    for raw_idx in np.atleast_1d(picked):
        idx = int(raw_idx)
        seq = list(mutated[idx])
        if len(seq) >= 4 and rng.random() < 0.5:
            start = int(rng.integers(0, len(seq) - 2))
            seq[start : start + 3] = [seq[start + 1], seq[start + 2], seq[start]]
        else:
            pos = int(rng.integers(0, len(seq) - 1))
            seq[pos], seq[pos + 1] = seq[pos + 1], seq[pos]
        mutated[idx] = seq
    return mutated


def run_benchmark(
    *,
    language: str = "french",
    sample_count: int = 800,
    change_count: int = 12,
    seed: int = 42,
) -> IncrementalBenchmarkResult:
    sequences = _load_sequences(language)
    sampled = _sample_sequences(sequences, count=sample_count, seed=seed)
    updated = _mutate_sequences_in_anchor(sampled, change_count=change_count, seed=seed + 1)

    basis = NgramVectorBasis.from_reference_label("latin")
    config = TensorStateConfig()

    state = IncrementalFingerprintTensorState.from_sequences(
        sampled,
        config=config,
        ngram_basis=basis,
    )
    changed_sequences = sum(1 for before, after in zip(sampled, updated) if before != after)

    started = time.perf_counter()
    update_result = state.apply_sequences(updated)
    incremental_seconds = time.perf_counter() - started

    started = time.perf_counter()
    rebuilt = IncrementalFingerprintTensorState.from_sequences_with_anchor(
        updated,
        anchor_tokens=state.idx2token,
        config=config,
        ngram_basis=basis,
    )
    full_rebuild_seconds = time.perf_counter() - started

    if update_result.mode == "incremental":
        assert np.array_equal(state.cooccurrence_counts, rebuilt.cooccurrence_counts)
        assert np.allclose(state.tensor, rebuilt.tensor)

    speedup = (
        full_rebuild_seconds / incremental_seconds
        if incremental_seconds > 0.0
        else float("inf")
    )
    return IncrementalBenchmarkResult(
        language=language,
        sampled_sequences=len(sampled),
        changed_sequences=changed_sequences,
        anchor_vocab_size=len(state.idx2token),
        update_mode=update_result.mode,
        incremental_seconds=incremental_seconds,
        full_rebuild_seconds=full_rebuild_seconds,
        speedup_vs_rebuild=speedup,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark incremental tensor refresh vs full rebuild.")
    parser.add_argument("--language", default="french")
    parser.add_argument("--sample-count", type=int, default=800)
    parser.add_argument("--change-count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "validation" / "incremental_tensor_phase4_benchmark.json")
    args = parser.parse_args()

    result = run_benchmark(
        language=args.language,
        sample_count=args.sample_count,
        change_count=args.change_count,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, ensure_ascii=False, indent=2)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
