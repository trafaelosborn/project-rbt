"""
Tests for IncrementalScoringState.

Correctness requirement: evaluate() must return the same scores as
_evaluate_sequences() to within floating-point tolerance.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def corpus_sequences():
    """Load the actual French v4 corpus used in the live run."""
    corpus_path = (
        PROJECT_ROOT
        / "data" / "retrodiction" / "french" / "v4_long_15000_local"
        / "blocks" / "block_0039" / "corpora" / "FR_v4_010_tokens.json"
    )
    if not corpus_path.exists():
        # Fall back to processed French corpus
        corpus_path = PROJECT_ROOT / "data" / "processed" / "romance" / "french_tokens.json"
    with corpus_path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    seqs = data["sequences"]
    return [list(s) for s in seqs[:800]]


@pytest.fixture(scope="module")
def latin_refs():
    from src.retrodiction.engine_reinforced import LatinReference
    from src.retrodiction.engine_reinforced_v2 import LatinFormReference
    from src.retrodiction.similarity import ReferenceSet
    return LatinReference(), LatinFormReference(), ReferenceSet()


@pytest.fixture(scope="module")
def scoring_state(corpus_sequences, latin_refs):
    from src.accelerate.incremental_scoring_state import IncrementalScoringState
    struct_ref, form_ref, refs = latin_refs
    return IncrementalScoringState.from_sequences(
        corpus_sequences, form_ref, struct_ref, refs
    )


# ---------------------------------------------------------------------------
# score_token cache
# ---------------------------------------------------------------------------

def test_score_token_cache_hit(latin_refs):
    """score_token returns the same value on repeated calls (cache correctness)."""
    _, form_ref, _ = latin_refs
    tok = "romani"
    first = form_ref.score_token(tok)
    second = form_ref.score_token(tok)
    assert first == second, "Cached score_token must be deterministic"


def test_score_token_cache_populated(latin_refs):
    """Cache is populated after a call."""
    _, form_ref, _ = latin_refs
    tok = "senatusque"
    _ = form_ref.score_token(tok)
    assert tok in form_ref._score_token_cache


def test_score_token_empty_string(latin_refs):
    _, form_ref, _ = latin_refs
    assert form_ref.score_token("") == 0.0


# ---------------------------------------------------------------------------
# IncrementalScoringState properties
# ---------------------------------------------------------------------------

def test_token_counts_match_direct(corpus_sequences, scoring_state):
    """token_counts property matches direct Counter of corpus."""
    expected = Counter(tok for seq in corpus_sequences for tok in seq)
    actual = scoring_state.token_counts
    assert actual == expected


def test_word_bigram_counts_match_direct(corpus_sequences, scoring_state):
    """word_bigram_counts property matches direct extraction."""
    from src.fingerprint.ngram import extract_ngrams
    expected = extract_ngrams(corpus_sequences, 2)
    actual = scoring_state.word_bigram_counts
    assert actual == expected


# ---------------------------------------------------------------------------
# evaluate() correctness against _evaluate_sequences()
# ---------------------------------------------------------------------------

def _make_engine(corpus_sequences, latin_refs):
    from src.retrodiction.engine_reinforced_v4 import (
        ReinforcedV4Config,
        RelationalReinforcedRetrodictionEngineV4,
    )
    struct_ref, form_ref, refs = latin_refs
    cfg = ReinforcedV4Config(
        num_sequences=len(corpus_sequences),
        max_proposals=1,
        use_incremental_scoring=False,
    )
    return RelationalReinforcedRetrodictionEngineV4(
        language="french",
        source_sequences=corpus_sequences,
        latin_structural_ref=struct_ref,
        latin_form_ref=form_ref,
        config=cfg,
        references=refs,
    )


def test_evaluate_noop_matches_current(corpus_sequences, scoring_state, latin_refs):
    """evaluate(same sequences) returns same scores as _evaluate_sequences."""
    engine = _make_engine(corpus_sequences, latin_refs)
    cfg = engine.config

    reference = engine._evaluate_sequences(corpus_sequences, mutation_cost=0.0)
    fast = scoring_state.evaluate(
        corpus_sequences, 0.0,
        cfg.form_weight, cfg.coherence_weight, cfg.mutation_cost_weight,
    )

    assert abs(fast.latin_form_score - reference.latin_form_score) < 1e-6, (
        f"form score mismatch: fast={fast.latin_form_score:.6f} ref={reference.latin_form_score:.6f}"
    )
    assert abs(fast.latin_structural_score - reference.latin_structural_score) < 1e-6, (
        f"structural score mismatch: fast={fast.latin_structural_score:.6f} ref={reference.latin_structural_score:.6f}"
    )
    assert abs(fast.type_token_ratio - reference.type_token_ratio) < 1e-6
    assert abs(fast.bigram_coverage - reference.bigram_coverage) < 1e-4
    assert abs(fast.trigram_coverage - reference.trigram_coverage) < 1e-4


def test_evaluate_token_rewrite_matches_reference(corpus_sequences, scoring_state, latin_refs):
    """evaluate() after a token rewrite matches _evaluate_sequences() on the mutated corpus."""
    engine = _make_engine(corpus_sequences, latin_refs)
    cfg = engine.config

    # Find a common token and rewrite it to a Latin-like form
    tc = scoring_state.token_counts
    common_tok = max(tc, key=tc.get)
    new_tok = common_tok + "us"

    mutated = [
        [new_tok if t == common_tok else t for t in seq]
        for seq in corpus_sequences
    ]

    reference = engine._evaluate_sequences(mutated, mutation_cost=0.3)
    fast = scoring_state.evaluate(
        mutated, 0.3,
        cfg.form_weight, cfg.coherence_weight, cfg.mutation_cost_weight,
    )

    assert abs(fast.latin_form_score - reference.latin_form_score) < 1e-6, (
        f"form score mismatch after rewrite: fast={fast.latin_form_score:.6f} ref={reference.latin_form_score:.6f}"
    )
    assert abs(fast.latin_structural_score - reference.latin_structural_score) < 1e-6
    assert abs(fast.type_token_ratio - reference.type_token_ratio) < 1e-6


def test_evaluate_does_not_mutate_state(corpus_sequences, scoring_state, latin_refs):
    """evaluate() must not change the committed state."""
    from src.accelerate.incremental_scoring_state import IncrementalScoringState
    struct_ref, form_ref, refs = latin_refs

    state = IncrementalScoringState.from_sequences(corpus_sequences, form_ref, struct_ref, refs)
    tc_before = Counter(state.token_counts)

    tc = state.token_counts
    common_tok = max(tc, key=tc.get)
    mutated = [
        [common_tok + "um" if t == common_tok else t for t in seq]
        for seq in corpus_sequences
    ]

    cfg_form, cfg_coh, cfg_mut = 0.75, 0.05, 0.005
    state.evaluate(mutated, 0.0, cfg_form, cfg_coh, cfg_mut)

    assert state.token_counts == tc_before, "evaluate() must not modify committed token_counts"


def test_evaluate_batch_matches_individual(corpus_sequences, latin_refs):
    """evaluate_batch() must agree with per-candidate evaluate()."""
    from src.accelerate.incremental_scoring_state import IncrementalScoringState

    struct_ref, form_ref, refs = latin_refs
    state = IncrementalScoringState.from_sequences(corpus_sequences, form_ref, struct_ref, refs)
    tc = state.token_counts
    common_tok = max(tc, key=tc.get)

    mutated_a = [
        [common_tok + "um" if t == common_tok else t for t in seq]
        for seq in corpus_sequences
    ]
    mutated_b = [
        [common_tok + "us" if t == common_tok else t for t in seq]
        for seq in corpus_sequences
    ]

    form_w, coh_w, mut_w = 0.75, 0.05, 0.005
    batch = state.evaluate_batch(
        [mutated_a, mutated_b],
        [0.2, 0.3],
        form_w,
        coh_w,
        mut_w,
    )
    single_a = state.evaluate(mutated_a, 0.2, form_w, coh_w, mut_w)
    single_b = state.evaluate(mutated_b, 0.3, form_w, coh_w, mut_w)

    assert abs(batch[0].latin_form_score - single_a.latin_form_score) < 1e-6
    assert abs(batch[0].latin_structural_score - single_a.latin_structural_score) < 1e-6
    assert abs(batch[0].total_score - single_a.total_score) < 1e-6
    assert abs(batch[1].latin_form_score - single_b.latin_form_score) < 1e-6
    assert abs(batch[1].latin_structural_score - single_b.latin_structural_score) < 1e-6
    assert abs(batch[1].total_score - single_b.total_score) < 1e-6


def test_evaluate_changed_sequences_matches_full_evaluate(corpus_sequences, latin_refs):
    """Sparse changed-row scoring must match full-corpus scoring."""
    from src.accelerate.incremental_scoring_state import IncrementalScoringState

    struct_ref, form_ref, refs = latin_refs
    state = IncrementalScoringState.from_sequences(corpus_sequences, form_ref, struct_ref, refs)
    tc = state.token_counts
    common_tok = max(tc, key=tc.get)
    new_tok = common_tok + "um"

    changed_sequences: dict[int, list[str]] = {}
    mutated: list[list[str]] = []
    for idx, seq in enumerate(corpus_sequences):
        new_seq = [new_tok if t == common_tok else t for t in seq]
        mutated.append(new_seq)
        if new_seq != seq:
            changed_sequences[idx] = new_seq

    form_w, coh_w, mut_w = 0.75, 0.05, 0.005
    sparse = state.evaluate_changed_sequences(
        changed_sequences,
        0.2,
        form_w,
        coh_w,
        mut_w,
    )
    full = state.evaluate(mutated, 0.2, form_w, coh_w, mut_w)

    assert abs(sparse.latin_form_score - full.latin_form_score) < 1e-6
    assert abs(sparse.latin_structural_score - full.latin_structural_score) < 1e-6
    assert abs(sparse.total_score - full.total_score) < 1e-6
    assert abs(sparse.type_token_ratio - full.type_token_ratio) < 1e-6


def test_evaluate_batch_changed_sequences_matches_full_batch(corpus_sequences, latin_refs):
    """Sparse batch scoring must match the materialized batch path."""
    from src.accelerate.incremental_scoring_state import IncrementalScoringState

    struct_ref, form_ref, refs = latin_refs
    state = IncrementalScoringState.from_sequences(corpus_sequences, form_ref, struct_ref, refs)
    tc = state.token_counts
    common_tok = max(tc, key=tc.get)

    changed_a: dict[int, list[str]] = {}
    mutated_a: list[list[str]] = []
    changed_b: dict[int, list[str]] = {}
    mutated_b: list[list[str]] = []
    for idx, seq in enumerate(corpus_sequences):
        new_a = [common_tok + "um" if t == common_tok else t for t in seq]
        new_b = [common_tok + "us" if t == common_tok else t for t in seq]
        mutated_a.append(new_a)
        mutated_b.append(new_b)
        if new_a != seq:
            changed_a[idx] = new_a
        if new_b != seq:
            changed_b[idx] = new_b

    form_w, coh_w, mut_w = 0.75, 0.05, 0.005
    sparse_batch = state.evaluate_batch_changed_sequences(
        [changed_a, changed_b],
        [0.2, 0.3],
        form_w,
        coh_w,
        mut_w,
    )
    full_batch = state.evaluate_batch(
        [mutated_a, mutated_b],
        [0.2, 0.3],
        form_w,
        coh_w,
        mut_w,
    )

    for sparse, full in zip(sparse_batch, full_batch):
        assert abs(sparse.latin_form_score - full.latin_form_score) < 1e-6
        assert abs(sparse.latin_structural_score - full.latin_structural_score) < 1e-6
        assert abs(sparse.total_score - full.total_score) < 1e-6


# ---------------------------------------------------------------------------
# commit() correctness
# ---------------------------------------------------------------------------

def test_commit_updates_token_counts(corpus_sequences, latin_refs):
    """After commit, token_counts reflects the new corpus."""
    from src.accelerate.incremental_scoring_state import IncrementalScoringState
    struct_ref, form_ref, refs = latin_refs

    state = IncrementalScoringState.from_sequences(corpus_sequences, form_ref, struct_ref, refs)
    tc = state.token_counts
    common_tok = max(tc, key=tc.get)
    new_tok = "novustestus"

    mutated = [
        [new_tok if t == common_tok else t for t in seq]
        for seq in corpus_sequences
    ]
    state.commit(mutated)

    expected = Counter(tok for seq in mutated for tok in seq)
    assert state.token_counts == expected
    assert common_tok not in state.token_counts or state.token_counts[common_tok] == 0


def test_commit_then_evaluate_consistent(corpus_sequences, latin_refs):
    """After commit, evaluate(same sequences) == evaluate on fresh state built from those sequences."""
    from src.accelerate.incremental_scoring_state import IncrementalScoringState
    struct_ref, form_ref, refs = latin_refs

    state = IncrementalScoringState.from_sequences(corpus_sequences, form_ref, struct_ref, refs)
    tc = state.token_counts
    common_tok = max(tc, key=tc.get)
    mutated = [
        [common_tok + "is" if t == common_tok else t for t in seq]
        for seq in corpus_sequences
    ]
    state.commit(mutated)

    # Fresh state built directly from mutated sequences
    fresh = IncrementalScoringState.from_sequences(mutated, form_ref, struct_ref, refs)

    form_w, coh_w, mut_w = 0.75, 0.05, 0.005
    scores_committed = state.evaluate(mutated, 0.0, form_w, coh_w, mut_w)
    scores_fresh = fresh.evaluate(mutated, 0.0, form_w, coh_w, mut_w)

    assert abs(scores_committed.latin_form_score - scores_fresh.latin_form_score) < 1e-6
    assert abs(scores_committed.latin_structural_score - scores_fresh.latin_structural_score) < 1e-6


# ---------------------------------------------------------------------------
# End-to-end: v4 engine with incremental scoring enabled
# ---------------------------------------------------------------------------

def test_v4_incremental_smoke(corpus_sequences, latin_refs):
    """v4 with use_incremental_scoring=True runs without error and reaches stable."""
    from src.retrodiction.engine_reinforced_v4 import (
        ReinforcedV4Config,
        RelationalReinforcedRetrodictionEngineV4,
    )
    import tempfile

    struct_ref, form_ref, refs = latin_refs
    cfg = ReinforcedV4Config(
        num_sequences=100,
        max_proposals=10,
        max_accepted_stages=5,
        n_candidates=3,
        seed=7,
        use_incremental_scoring=True,
        acceleration_mode="python_only",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        engine = RelationalReinforcedRetrodictionEngineV4(
            language="french",
            source_sequences=corpus_sequences,
            latin_structural_ref=struct_ref,
            latin_form_ref=form_ref,
            config=cfg,
            output_dir=Path(tmpdir),
            references=refs,
        )
        records = engine.run()

    assert len(records) >= 1
    # All records must have finite scores
    for r in records:
        assert math.isfinite(r.total_score)
        assert math.isfinite(r.latin_form_score)
        assert math.isfinite(r.latin_structural_score)
