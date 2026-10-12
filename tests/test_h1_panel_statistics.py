import itertools
import json
from datetime import datetime, timezone

import numpy as np
import pytest

from src.validation.h1_convergence import (
    ROMANCE, CONTROLS, SEEDS, panel_summary, whole_language_reference,
    leave_one_language_out, analyze, sample_deciles, pair_auc, render_report,
)
from src.validation.h1_protocol import (
    protocol_identity, write_registration, validate_registration,
    validate_registered_manifest, SIDECAR,
)
from src.control.h1_matrix import build_jobs
from src.retrodiction.h1_objective import stable_hash


def pairs(focal_value=0.2, other_value=1.0):
    return [{"seed": seed, "left": a, "right": b,
             "auc": focal_value if a in ROMANCE and b in ROMANCE else other_value}
            for seed in SEEDS for a, b in itertools.combinations(ROMANCE + CONTROLS, 2)]


def test_known_contrast_and_exact_reference():
    rows = pairs()
    result = whole_language_reference(rows)
    assert result["observed_effect"] == pytest.approx(0.8)
    assert result["upper_tail_fraction"] == pytest.approx(1 / 126)
    assert result["extreme_allocations"] == 1
    assert result["unconditional_significance_claim_allowed"] is False
    assert result["exchangeability_established"] is False
    assert len({tuple(a["focal_languages"]) for a in result["allocations"]}) == 126
    assert sum(set(a["focal_languages"]) == set(ROMANCE) for a in result["allocations"]) == 1


def test_ties_and_opposite_effect_do_not_manufacture_significance():
    tied = whole_language_reference(pairs(1.0, 1.0))
    assert tied["upper_tail_fraction"] == 1
    assert tied["observed_effect"] == 0
    assert panel_summary(pairs(1.0, 1.0))["direction_replicates_all_seeds"] is False
    reverse = whole_language_reference(pairs(1.0, 0.2))
    assert reverse["upper_tail_fraction"] == 1
    assert reverse["observed_effect"] < 0


def test_duplicated_seed_evidence_does_not_shrink_reference_fraction():
    rows = pairs()
    replicated = rows + [{**p, "seed": p["seed"] + 10000000} for p in rows]
    twice = whole_language_reference(replicated, seeds=SEEDS + tuple(s + 10000000 for s in SEEDS))
    once = whole_language_reference(rows)
    assert twice["upper_tail_fraction"] == once["upper_tail_fraction"]
    assert twice["observed_effect"] == pytest.approx(once["observed_effect"])
    assert twice["allocation_count"] == once["allocation_count"]


def test_language_order_pair_orientation_and_row_order_are_irrelevant():
    rows = [{**p, "left": p["right"], "right": p["left"]} for p in reversed(pairs())]
    result = whole_language_reference(rows, languages=tuple(reversed(ROMANCE + CONTROLS)))
    assert result["upper_tail_fraction"] == pytest.approx(1 / 126)
    assert result["observed_effect"] == pytest.approx(0.8)


def test_positive_pooled_effect_does_not_hide_a_reversed_seed():
    rows = [{**p, "auc": 1.8 if p["seed"] == SEEDS[-1] and p["left"] in ROMANCE and p["right"] in ROMANCE else p["auc"]}
            for p in pairs()]
    summary = panel_summary(rows)
    assert summary["effect"] > 0
    assert summary["direction_replicates_all_seeds"] is False
    assert summary["per_seed"][-1]["other_minus_romance_auc"] < 0


@pytest.mark.parametrize("kind", ["missing", "duplicate", "unknown", "nan", "negative", "self"])
def test_invalid_pair_panels_fail_closed(kind):
    rows = pairs()
    if kind == "missing":
        rows.pop()
    elif kind == "duplicate":
        rows.append(dict(rows[0]))
    elif kind == "unknown":
        rows[0]["seed"] = -1
    elif kind == "self":
        rows[0]["right"] = rows[0]["left"]
    else:
        rows[0]["auc"] = float("nan") if kind == "nan" else -1
    with pytest.raises(ValueError):
        whole_language_reference(rows)


def test_leave_one_out_reports_all_languages_without_p_values():
    result = leave_one_language_out(pairs())
    assert len(result["analyses"]) == 9
    assert result["effect_range"] == pytest.approx([0.8, 0.8])
    assert all("p_value" not in r for r in result["analyses"])


def test_full_allocation_null_calibration_with_dependent_edges():
    rng = np.random.default_rng(20261001)
    languages = ROMANCE + CONTROLS
    for _ in range(10):
        latent = dict(zip(languages, rng.normal(size=9)))
        rows = [{"seed": s, "left": a, "right": b,
                 "auc": float(np.exp(0.2 * (latent[a] + latent[b]) + rng.normal(0, 0.01)))}
                for s in SEEDS for a, b in itertools.combinations(languages, 2)]
        ref = whole_language_reference(rows)
        stats = np.array([a["effect"] for a in ref["allocations"]])
        # All focal allocations are equally likely in this constructed null.
        tails = np.array([np.mean(stats >= t - 1e-12 * max(1, abs(t))) for t in stats])
        assert np.mean(tails < 0.05) <= 6 / 126 + 1e-15


def test_pinned_protocol_and_registration_integrity(tmp_path):
    assert protocol_identity()["protocol"]["unconditional_significance_gate"] is False
    root = tmp_path / "study"
    written = write_registration(root, build_jobs(root))
    assert validate_registration(root) == written
    assert len(written["jobs"]) == 33
    with pytest.raises(ValueError, match="new output root"):
        write_registration(root, build_jobs(root))
    path = root / SIDECAR
    changed = json.loads(path.read_text())
    changed["analysis_files_sha256_lf"]["src/validation/h1_convergence.py"] = "tampered"
    changed["registration_sha256"] = stable_hash({k: v for k, v in changed.items() if k != "registration_sha256"})
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="implementation changed"):
        validate_registration(root)


def test_missing_and_foreign_registration_refused(tmp_path):
    with pytest.raises(ValueError, match="Missing pre-launch"):
        validate_registration(tmp_path)
    root = tmp_path / "study"
    registered = write_registration(root, build_jobs(root))
    job = registered["jobs"][0]
    manifest = {"config": dict(job["config"]), "created_utc": datetime.now(timezone.utc).isoformat(),
                "h1_execution": {"source_sha256_lf": job["source_sha256_lf"],
                                 "objective": {"identity_sha256": registered["objective_identity"]}}}
    validate_registered_manifest(registered, manifest)
    manifest["created_utc"] = "2000-01-01T00:00:00+00:00"
    with pytest.raises(ValueError, match="predates"):
        validate_registered_manifest(registered, manifest)
    manifest["config"]["n_candidates"] = 8
    with pytest.raises(ValueError, match="job contract"):
        validate_registered_manifest(registered, manifest)


def test_changed_retained_reward_reference_is_detected(tmp_path, monkeypatch):
    import src.validation.h1_protocol as protocol
    root = tmp_path / "study"
    registered = write_registration(root, build_jobs(root))
    assert "data/sequestered/latin/latin_tokens.json" in registered["reference_inputs_sha256_lf"]
    changed = dict(registered["reference_inputs_sha256_lf"])
    changed["data/sequestered/latin/latin_tokens.json"] = "changed"
    monkeypatch.setattr(protocol, "reference_input_hashes", lambda: changed)
    with pytest.raises(ValueError, match="reference inputs changed"):
        validate_registration(root)


def test_complete_but_unregistered_matrix_cannot_be_adopted(tmp_path):
    for language in ROMANCE + CONTROLS:
        for seed in SEEDS:
            path = tmp_path / language / f"seed_{seed}" / "manifest.json"
            path.parent.mkdir(parents=True)
            path.write_text("{}")
    with pytest.raises(ValueError, match="Missing pre-launch"):
        analyze(tmp_path)


def test_mismatched_trajectory_shapes_are_not_broadcast_silently():
    with pytest.raises(ValueError, match="finite measured"):
        sample_deciles([-2, -1], [[0]])
    with pytest.raises(ValueError, match="matching 11-decile"):
        pair_auc(np.zeros((11, 6)), np.ones((1, 6)))


def test_failed_readout_is_never_written_as_a_negative_scientific_answer():
    text = render_report({"status": "not_evaluable", "error": "Missing Finnish seed 42"})
    assert "Missing Finnish seed 42" in text
    assert "not a negative scientific answer" in text
