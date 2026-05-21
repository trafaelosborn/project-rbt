"""
Benchmark Runner
================
Internal module for candidate-count throughput benchmarks.

Runs short trials (proposals_per_trial proposals each) across the
configured candidate counts and measures proposals/hour.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from src.control.run_config import BenchmarkConfig

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_candidate_count_benchmark(config: BenchmarkConfig) -> list[dict[str, Any]]:
    """
    Run a micro-benchmark across candidate counts.

    Returns a list of trial dicts, one per candidate count, each containing:
        candidate_count, proposals_attempted, wall_seconds, proposals_per_hour
    """
    import shutil
    import tempfile

    from src.retrodiction.engine_reinforced import LatinReference
    from src.retrodiction.engine_reinforced_v2 import LatinFormReference
    from src.retrodiction.engine_reinforced_v4 import ReinforcedV4Config, RelationalReinforcedRetrodictionEngineV4
    from src.retrodiction.similarity import ReferenceSet
    from src.validation.hungarian_alignment import extract_family_inventory

    log.info("Loading source corpus from %s", config.start_corpus)
    with Path(config.start_corpus).open(encoding="utf-8") as fh:
        corpus = json.load(fh)
    sequences = corpus["sequences"][: config.num_sequences]

    latin_structural_ref = LatinReference()
    latin_form_ref = LatinFormReference()
    references = ReferenceSet()

    # Build a minimal latin sample for alignment
    latin_path = PROJECT_ROOT / "data" / "sequestered" / "latin" / "latin_tokens.json"
    with latin_path.open(encoding="utf-8") as fh:
        latin_corpus = json.load(fh)
    latin_sequences = latin_corpus["sequences"][:72]

    trials: list[dict[str, Any]] = []

    for n_candidates in config.candidate_counts:
        trial_dir = Path(tempfile.mkdtemp(prefix=f"bench_c{n_candidates}_"))
        try:
            engine_cfg = ReinforcedV4Config(
                num_sequences=config.num_sequences,
                max_proposals=config.proposals_per_trial,
                max_accepted_stages=512,
                patience=config.proposals_per_trial,
                seed=config.seed,
                n_candidates=n_candidates,
                min_improvement=0.0001,
                save_dense_matrices=False,
                use_incremental_scoring=True,
            )
            from src.retrodiction.engine_reinforced_v4 import ReinforcedV4Config as _Cfg
            family_ref = extract_family_inventory(
                "latin", latin_sequences, engine_cfg.alignment_config
            )
            engine = RelationalReinforcedRetrodictionEngineV4(
                language=config.source_language,
                source_sequences=sequences,
                latin_structural_ref=latin_structural_ref,
                latin_form_ref=latin_form_ref,
                config=engine_cfg,
                output_dir=trial_dir,
                references=references,
                family_reference_inventory=family_ref,
            )

            t0 = time.perf_counter()
            engine.run()
            elapsed = time.perf_counter() - t0

            summary_path = trial_dir / "run_summary.json"
            with summary_path.open(encoding="utf-8") as fh:
                summary = json.load(fh)

            attempted = int(summary.get("proposals_attempted", config.proposals_per_trial))
            proposals_per_hour = attempted / elapsed * 3600 if elapsed > 0 else 0.0

            trial = {
                "candidate_count": n_candidates,
                "proposals_attempted": attempted,
                "wall_seconds": round(elapsed, 2),
                "proposals_per_hour": round(proposals_per_hour, 1),
                "accepted_mutation_stages": summary.get("accepted_mutation_stages"),
                "final_struct": summary.get("final_latin_structural_score"),
                "final_form": summary.get("final_latin_form_score"),
            }
            log.info(
                "candidate_count=%d: %d proposals in %.1fs = %.0f proposals/hour",
                n_candidates, attempted, elapsed, proposals_per_hour,
            )
            trials.append(trial)

        finally:
            shutil.rmtree(trial_dir, ignore_errors=True)

    return trials
