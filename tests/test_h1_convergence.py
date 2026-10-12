import json
import numpy as np
import pytest
from src.validation.h1_convergence import exact_mann_whitney_less, sample_deciles, pair_auc, load_trajectory, analyze
from src.control.h1_matrix import build_jobs


def test_exact_rank_probability_and_ties():
    assert exact_mann_whitney_less([1, 2], [3, 4])["p_less"] == pytest.approx(1 / 6)
    assert exact_mann_whitney_less([3, 4], [1, 2])["p_less"] == 1
    assert exact_mann_whitney_less([1, 1], [1, 1])["p_less"] == 1
    assert exact_mann_whitney_less([1, 1], [2, 2])["p_less"] == pytest.approx(1 / 6)


def test_deciles_and_auc_have_known_geometry():
    left = sample_deciles([-10, -5, 0], [[1, 0], [0.5, 0], [0, 0]])
    right = sample_deciles([-10, 0], [[-1, 0], [0, 0]])
    area, curve = pair_auc(left, right)
    assert len(curve) == 11
    assert area == pytest.approx(0.5)
    assert curve[0] == 1 and curve[-1] == 0


def test_progress_uses_first_crossing_and_actual_endpoint():
    deciles = sample_deciles([-10, 0, -5], [[0], [10], [100]])
    assert deciles[1, 0] == pytest.approx(0.5)
    assert deciles[-1, 0] == 100


def test_degenerate_trajectories_are_not_silently_dropped():
    with pytest.raises(ValueError, match="positive"):
        sample_deciles([-1, -1], [[0], [0]])
    with pytest.raises(ValueError, match="zero"):
        pair_auc(np.zeros((11, 6)), np.zeros((11, 6)))


def test_partial_matrix_cannot_produce_verdict(tmp_path):
    with pytest.raises(ValueError, match="Missing required"):
        analyze(tmp_path)


def test_legacy_and_bounded_runs_cannot_produce_verdict(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"config": {}, "status": "plateau_hit"}))
    with pytest.raises(ValueError, match="legacy"):
        load_trajectory(path)
    path.write_text(json.dumps({"h1_execution": {"active": True}, "config": {"total_target_proposals": 4}, "status": "complete"}))
    with pytest.raises(ValueError, match="Bounded pilot"):
        load_trajectory(path)


def test_matrix_contains_primary_controls_and_separate_stress_jobs(tmp_path):
    jobs = build_jobs(tmp_path)
    assert len(jobs) == 33
    assert {j.language for j in jobs} == {"french", "spanish", "italian", "portuguese", "romanian",
                                         "finnish", "turkish", "vietnamese", "sumerian", "markov_null", "klingon"}
    assert all(j.start_corpus.exists() and j.corpus_budget == 20000 for j in jobs)
    assert not list(tmp_path.iterdir()), "Planning does not launch jobs or create run outputs"


def test_complete_synthetic_matrix_gate_and_mixed_contract_guard(tmp_path, monkeypatch):
    import src.validation.h1_convergence as h1
    for language in h1.ROMANCE + h1.CONTROLS:
        for seed in h1.SEEDS:
            path = tmp_path / language / f"seed_{seed}" / "manifest.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
    def trajectory(path, *, require_progress=True):
        language = path.parent.parent.name
        seed = int(path.parent.name.split("_")[1])
        index = (h1.ROMANCE + h1.CONTROLS + ("markov_null", "klingon")).index(language)
        if language in ("markov_null", "klingon"):
            assert require_progress is False, "Stress endpoints need no positive structural progress"
        vector = np.zeros((11, 6))
        if language in h1.ROMANCE:
            vector[:, 0] = (index + 1) * np.linspace(1, 0, 11)
        else:
            vector[:, 0] = 100 * (index + 1)
        return {"config": {"language": language, "seed": seed},
                "h1_execution": {"objective": {"identity_sha256": "fixture"},
                                 "launch_contract": {"candidate_count": 16}}}, vector
    monkeypatch.setattr(h1, "load_trajectory", trajectory)
    monkeypatch.setattr(h1, "validate_registration", lambda root: {"registration_sha256": "fixture"})
    monkeypatch.setattr(h1, "validate_registered_manifest", lambda registration, manifest: None)
    monkeypatch.setattr(h1, "trajectory_diagnostics", lambda manifest: {"fixture": True})
    result = h1.analyze(tmp_path)
    assert result["gate"] == "DESCRIPTIVE_ONLY"
    assert result["observed_pattern"] == "CONSISTENT_ROMANCE_DIRECTION"
    assert result["h1_inference_status"] == "not_established_by_this_design"
    assert len(result["pairs"]) == 108
    assert result["language_label_reference"]["upper_tail_fraction"] < 0.05
    assert result["language_label_reference"]["allocation_count"] == 126
    assert len(result["leave_one_language_out"]["analyses"]) == 9
    assert result["stress_complete"] is False
    assert all(s["status"] == "missing" for s in result["stress_diagnostics"])
    for language in ("markov_null", "klingon"):
        for seed in h1.SEEDS:
            path = tmp_path / language / f"seed_{seed}" / "manifest.json"
            path.parent.mkdir(parents=True)
            path.write_text("{}")
    with_stress = h1.analyze(tmp_path)
    assert with_stress["stress_complete"] is True
    assert len(with_stress["inputs"]) == 33
    assert len(with_stress["pairs"]) == 108, "Stress jobs do not enlarge the primary pair pool"
    assert with_stress["primary"] == result["primary"]
    text = h1.render_report(with_stress)
    assert "not an unconditional H1 significance verdict" in text
    assert "reference tail" in text
    def mixed(path):
        manifest, vector = trajectory(path)
        if manifest["config"]["language"] == "french":
            manifest["h1_execution"]["launch_contract"]["candidate_count"] = 32
        return manifest, vector
    monkeypatch.setattr(h1, "load_trajectory", mixed)
    with pytest.raises(ValueError, match="launch conditions"):
        h1.analyze(tmp_path)
