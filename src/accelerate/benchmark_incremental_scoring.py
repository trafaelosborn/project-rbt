"""
Benchmark: IncrementalScoringState vs _evaluate_sequences

Measures the per-proposal speedup from:
  Layer 1: score_token cache
  Layer 2: precomputed token/bigram counts
  Layer 3: incremental char-ngram evaluate()

Run:
    cd project_rbt
    python -m src.accelerate.benchmark_incremental_scoring
"""

from __future__ import annotations

import json
import time
import logging
from collections import Counter
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_corpus(n: int = 800) -> list[list[str]]:
    candidates = [
        PROJECT_ROOT / "data" / "retrodiction" / "french" / "v4_long_15000_local"
        / "blocks" / "block_0039" / "corpora" / "FR_v4_010_tokens.json",
        PROJECT_ROOT / "data" / "processed" / "romance" / "french_tokens.json",
    ]
    for path in candidates:
        if path.exists():
            with path.open(encoding="utf-8") as fh:
                seqs = json.load(fh)["sequences"]
            return [list(s) for s in seqs[:n]]
    raise FileNotFoundError("No French corpus found")


def _time(fn, n_reps: int) -> float:
    t0 = time.perf_counter()
    for _ in range(n_reps):
        fn()
    return (time.perf_counter() - t0) / n_reps


def main() -> None:
    from src.retrodiction.engine_reinforced import LatinReference
    from src.retrodiction.engine_reinforced_v2 import LatinFormReference
    from src.retrodiction.engine_reinforced_v4 import (
        ReinforcedV4Config,
        RelationalReinforcedRetrodictionEngineV4,
    )
    from src.retrodiction.similarity import ReferenceSet
    from src.accelerate.incremental_scoring_state import IncrementalScoringState

    N_SEQS = 800
    N_CANDIDATES = 8
    N_REPS = 5
    SEED = 42

    print(f"Loading corpus ({N_SEQS} sequences)...")
    sequences = _load_corpus(N_SEQS)
    total_tokens = sum(len(s) for s in sequences)
    print(f"  {len(sequences)} sequences, {total_tokens} tokens\n")

    print("Building references...")
    struct_ref = LatinReference()
    form_ref = LatinFormReference()
    refs = ReferenceSet()

    cfg_base = ReinforcedV4Config(
        num_sequences=N_SEQS,
        max_proposals=2,
        n_candidates=N_CANDIDATES,
        seed=SEED,
        use_incremental_scoring=False,
        acceleration_mode="python_only",
    )
    engine = RelationalReinforcedRetrodictionEngineV4(
        language="french",
        source_sequences=sequences,
        latin_structural_ref=struct_ref,
        latin_form_ref=form_ref,
        config=cfg_base,
        references=refs,
    )
    rng = np.random.default_rng(SEED)
    current_seqs = engine._sample_initial_corpus(rng)

    # --- Benchmark 1: score_token (cold vs warm cache) ---
    print("=== Layer 1: score_token cache ===")
    tc = Counter(tok for seq in current_seqs for tok in seq)
    tokens_sample = list(tc.keys())[:200]

    # Wipe cache
    form_ref._score_token_cache.clear()
    cold_ms = _time(lambda: [form_ref.score_token(t) for t in tokens_sample], N_REPS) * 1000
    warm_ms = _time(lambda: [form_ref.score_token(t) for t in tokens_sample], N_REPS) * 1000
    print(f"  200 score_token calls cold: {cold_ms:.1f}ms")
    print(f"  200 score_token calls warm: {warm_ms:.1f}ms")
    print(f"  speedup: {cold_ms / max(warm_ms, 0.001):.0f}x\n")

    # --- Benchmark 2: _evaluate_sequences vs incremental evaluate ---
    print("=== Layer 2+3: evaluate() vs _evaluate_sequences() ===")

    state = IncrementalScoringState.from_sequences(
        current_seqs, form_ref, struct_ref, refs
    )

    # Produce a set of mutated corpora to score
    rng2 = np.random.default_rng(SEED + 1)
    mutations: list[list[list[str]]] = []
    for _ in range(N_CANDIDATES):
        mutated, _, _, _ = engine._mutate_candidate(current_seqs, rng2)
        if mutated is not None:
            mutations.append(mutated)
    if not mutations:
        mutations = [current_seqs]  # fallback: noop
    print(f"  Generated {len(mutations)} candidate mutations to score")

    # Reference: _evaluate_sequences
    ref_ms = _time(
        lambda: [engine._evaluate_sequences(m, 0.3) for m in mutations],
        N_REPS,
    ) * 1000

    # Incremental: state.evaluate
    inc_ms = _time(
        lambda: [
            state.evaluate(m, 0.3, cfg_base.form_weight, cfg_base.coherence_weight, cfg_base.mutation_cost_weight)
            for m in mutations
        ],
        N_REPS,
    ) * 1000

    print(f"  _evaluate_sequences × {len(mutations)}: {ref_ms:.1f}ms")
    print(f"  state.evaluate()    × {len(mutations)}: {inc_ms:.1f}ms")
    print(f"  speedup: {ref_ms / max(inc_ms, 0.001):.1f}x\n")

    # --- Benchmark 3: _mutate_candidate (fresh counts vs precomputed) ---
    print("=== Layer 2 detail: precomputed token/bigram counts in _mutate_candidate ===")

    rng3 = np.random.default_rng(SEED + 2)

    def mutate_fresh():
        for _ in range(N_CANDIDATES):
            engine._mutate_candidate(current_seqs, rng3)

    precomp_tc = state.token_counts
    precomp_bg = state.word_bigram_counts

    def mutate_precomputed():
        for _ in range(N_CANDIDATES):
            engine._mutate_candidate(
                current_seqs, rng3,
                precomputed_token_counts=precomp_tc,
                precomputed_bigram_counts=precomp_bg,
            )

    fresh_ms = _time(mutate_fresh, N_REPS) * 1000
    precomp_ms = _time(mutate_precomputed, N_REPS) * 1000
    print(f"  {N_CANDIDATES}x _mutate_candidate fresh counts:      {fresh_ms:.1f}ms")
    print(f"  {N_CANDIDATES}x _mutate_candidate precomputed counts: {precomp_ms:.1f}ms")
    print(f"  speedup: {fresh_ms / max(precomp_ms, 0.001):.1f}x\n")

    # --- Benchmark 4: full proposal cycle ---
    print("=== Full proposal cycle: baseline vs incremental ===")

    rng4a = np.random.default_rng(SEED + 3)
    current_a = engine._evaluate_sequences(current_seqs, mutation_cost=0.0)

    def full_proposal_baseline():
        best = None
        for _ in range(N_CANDIDATES):
            mutated, op, details, cost = engine._mutate_candidate(current_seqs, rng4a)
            if mutated is None:
                continue
            cand = engine._evaluate_sequences(mutated, mutation_cost=cost)
            cand = engine._amplify_reward(current_a, cand)
            if best is None or cand.total_score > best.total_score:
                best = cand

    rng4b = np.random.default_rng(SEED + 3)
    current_b = engine._evaluate_sequences(current_seqs, mutation_cost=0.0)
    state_b = IncrementalScoringState.from_sequences(current_seqs, form_ref, struct_ref, refs)

    def full_proposal_incremental():
        best = None
        ptc = state_b.token_counts
        pbg = state_b.word_bigram_counts
        for _ in range(N_CANDIDATES):
            mutated, op, details, cost = engine._mutate_candidate(
                current_seqs, rng4b,
                precomputed_token_counts=ptc,
                precomputed_bigram_counts=pbg,
            )
            if mutated is None:
                continue
            scores = state_b.evaluate(
                mutated, cost,
                cfg_base.form_weight, cfg_base.coherence_weight, cfg_base.mutation_cost_weight,
            )
            cand = engine._candidate_state_from_scores(mutated, scores, cost)
            cand = engine._amplify_reward(current_b, cand)
            if best is None or cand.total_score > best.total_score:
                best = cand

    baseline_ms = _time(full_proposal_baseline, N_REPS) * 1000
    incremental_ms = _time(full_proposal_incremental, N_REPS) * 1000
    speedup = baseline_ms / max(incremental_ms, 0.001)

    print(f"  Baseline proposal ({N_CANDIDATES} candidates):    {baseline_ms:.0f}ms")
    print(f"  Incremental proposal ({N_CANDIDATES} candidates): {incremental_ms:.0f}ms")
    print(f"  Speedup: {speedup:.1f}x")

    est_baseline_per_hour = 3600_000 / baseline_ms
    est_incremental_per_hour = 3600_000 / incremental_ms
    print(f"\n  Estimated proposals/hour (baseline):    {est_baseline_per_hour:,.0f}")
    print(f"  Estimated proposals/hour (incremental): {est_incremental_per_hour:,.0f}")

    artifact = PROJECT_ROOT / "data" / "validation" / "incremental_scoring_benchmark.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    import json as _json
    with artifact.open("w", encoding="utf-8") as fh:
        _json.dump({
            "n_sequences": N_SEQS,
            "n_candidates": N_CANDIDATES,
            "total_tokens": total_tokens,
            "score_token_cold_ms": round(cold_ms, 3),
            "score_token_warm_ms": round(warm_ms, 3),
            "score_token_speedup": round(cold_ms / max(warm_ms, 0.001), 1),
            "evaluate_baseline_ms": round(ref_ms, 3),
            "evaluate_incremental_ms": round(inc_ms, 3),
            "evaluate_speedup": round(ref_ms / max(inc_ms, 0.001), 1),
            "mutate_fresh_ms": round(fresh_ms, 3),
            "mutate_precomputed_ms": round(precomp_ms, 3),
            "mutate_speedup": round(fresh_ms / max(precomp_ms, 0.001), 1),
            "full_proposal_baseline_ms": round(baseline_ms, 3),
            "full_proposal_incremental_ms": round(incremental_ms, 3),
            "full_proposal_speedup": round(speedup, 1),
            "est_proposals_per_hour_baseline": int(est_baseline_per_hour),
            "est_proposals_per_hour_incremental": int(est_incremental_per_hour),
        }, fh, indent=2)
    print(f"\nArtifact saved to {artifact}")


if __name__ == "__main__":
    main()
