"""Behavioral checks for Decision 044, including the real production route."""
import itertools
import json
from pathlib import Path
import numpy as np
import pytest
from src.retrodiction.h1_objective import (
    BudgetError, FEATURES, FEATURE_SHA256, FREEZE_COMMIT, H1Objective,
    PROJECT_ROOT, measurement_prefix,
)
from src.retrodiction.h1_execution import prepare_execution, audit_block
from src.retrodiction.long_run_v4 import LongRunConfig, run_long_continuation
from src.retrodiction.long_run_v5 import _default_v5_config
from src.retrodiction.engine_reinforced_v4 import MutationPayload
from src.retrodiction.engine_reinforced_v5 import ReinforcedV5Config, RelationalReinforcedRetrodictionEngineV5
from src.validation.static_feature_matrix import compute_feature_vector
from src.validation.hungarian_alignment import extract_family_inventory
from tests.test_engine_reinforced_v3 import _FakeLatinFormRef, _FakeReferenceSet


def corpus():
    return [["abacus", "domina", "regnum", "servus", "portae", "templo"]] * 20


def collapsed():
    # Bijective word relabeling preserves the entire legacy reward vector.
    return [["aaaazz", "bbbbzz", "cccczz", "ddddzz", "eeeezz", "ffffzz"]] * 20


@pytest.fixture
def objective():
    target = compute_feature_vector(corpus())
    rows = [{"corpus_id": str(i), "language": "latin" if i == 1 else str(i),
             "role": "test", **{c: target[c] + offset for c in FEATURES}}
            for i, offset in enumerate((-1, 0, 1))]
    return H1Objective(rows, FEATURES, 120, {"test_fixture": True})


class ForbiddenLegacyReward:
    def score(self, vector):
        raise AssertionError("H1 invoked the legacy reward")


def engine_for(tmp_path, monkeypatch, objective, **options):
    monkeypatch.setattr(H1Objective, "from_config", classmethod(lambda cls, cfg: objective))
    cfg = ReinforcedV5Config(num_sequences=1, corpus_budget=objective.budget, max_proposals=1, n_candidates=2,
                             form_weight=0, coherence_weight=0, mutation_cost_weight=0,
                             save_dense_matrices=False, **options)
    family = extract_family_inventory("latin", corpus(), cfg.alignment_config)
    return RelationalReinforcedRetrodictionEngineV5(
        "french", collapsed(), ForbiddenLegacyReward(), _FakeLatinFormRef(), cfg,
        tmp_path, _FakeReferenceSet(), family,
    )


def test_frozen_calibration_matches_explicit_distance():
    obj = H1Objective.load(FREEZE_COMMIT, FEATURE_SHA256, 20000)
    result = obj.evaluate(corpus() * 167)
    terms = []
    latin = next(r for r in obj.rows if r["language"] == "latin")
    for feature in FEATURES:
        values = [float(r[feature]) for r in obj.rows]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        terms.append((result["features"][feature] - float(latin[feature])) ** 2 / variance)
    assert result["structural_score"] == pytest.approx(-5 * sum(terms) ** 0.5)
    assert result["measured_token_count"] == 20000


@pytest.mark.parametrize("index", range(6))
def test_each_registered_coordinate_changes_reward(objective, monkeypatch, index):
    import src.validation.static_feature_matrix as sfm
    features = compute_feature_vector(corpus())
    features[FEATURES[index]] += 0.5
    monkeypatch.setattr(sfm, "compute_feature_vector", lambda seqs: features)
    score = objective.evaluate(corpus())["structural_score"]
    assert score == pytest.approx(-5 * 0.5 / np.sqrt(2 / 3))


@pytest.mark.parametrize("incremental,batch,cosine", list(itertools.product([False, True], repeat=3)))
def test_all_requested_backends_accept_h1_improvement(tmp_path, monkeypatch, objective, incremental, batch, cosine):
    before, after = compute_feature_vector(collapsed()), compute_feature_vector(corpus())
    for key in ("f01_ttr", "f02_top100_bigram_coverage", "f03_top100_trigram_coverage"):
        assert before[key] == after[key]
    engine = engine_for(tmp_path, monkeypatch, objective,
                        use_incremental_scoring=incremental, use_fortran_batch=batch, use_fortran_cosine=cosine)
    def mutate(sequences, rng, **kwargs):
        return MutationPayload(sequences=corpus()), "test_relabel", "legacy-vector tie", 0.0
    monkeypatch.setattr(engine, "_mutate_candidate", mutate)
    records = engine.run()
    assert len(records) == 2, "Improvement invisible to legacy scoring must be accepted"
    assert records[0].latin_structural_score < records[1].latin_structural_score
    assert records[1].latin_structural_score == pytest.approx(0)
    assert records[1].diagnostics["candidate_scoring_backend"] == "h1_python_reference"
    path = tmp_path / "run_summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    audit_block(objective, engine.config, summary, path, engine)
    assert summary["h1_audit"]["stages_replayed"] == 2


def test_saved_score_tampering_is_detected(tmp_path, monkeypatch, objective):
    engine = engine_for(tmp_path, monkeypatch, objective, use_fortran_batch=False)
    engine.config.max_proposals = 0
    engine.run()
    path = tmp_path / "run_summary.json"
    summary = json.loads(path.read_text())
    summary["stages"][0]["latin_structural_score"] += 0.01
    with pytest.raises(ValueError, match="independent replay"):
        audit_block(objective, engine.config, summary, path, engine)


def test_sparse_payload_uses_materialized_candidate(tmp_path, monkeypatch, objective):
    engine = engine_for(tmp_path, monkeypatch, objective, use_fortran_batch=True)
    current = engine._evaluate_sequences(collapsed(), 0)
    payload = MutationPayload(changed_sequences={i: seq for i, seq in enumerate(corpus())})
    result = engine._score_mutation_pool(current=current, mutation_pool=[(payload, "relabel", "", 0)], proposal_guidance=None)
    assert result[0][0].latin_structural_score == pytest.approx(0)
    assert result[0][0].sequences == corpus()


def test_undersized_candidates_rejected_and_logged(tmp_path, monkeypatch, objective):
    engine = engine_for(tmp_path, monkeypatch, objective, use_fortran_batch=True)
    monkeypatch.setattr(engine, "_mutate_candidate", lambda *a, **k:
                        (MutationPayload(sequences=corpus()[:-1]), "merge", "under budget", 0))
    records = engine.run()
    assert len(records) == 1
    assert engine._h1_budget_rejections == 2
    assert "invalid_token_budget" in (tmp_path / "live_events.jsonl").read_text()


def test_budget_prefix_never_pads_and_retains_sentence_boundaries():
    assert measurement_prefix([["a", "b"], [], ["c", "d", "e"]], 4) == [["a", "b"], ["c", "d"]]
    with pytest.raises(BudgetError):
        measurement_prefix([["a"]], 2)


def test_initial_sampling_and_checkpoint_preservation(tmp_path, monkeypatch, objective):
    engine = engine_for(tmp_path, monkeypatch, objective)
    engine.config.corpus_budget = 120
    engine.source_sequences = corpus() + [["tail"]]
    initial = engine._sample_initial_corpus(np.random.default_rng(42))
    assert sum(map(len, initial)) == 120
    engine.config.preserve_initial_corpus = True
    assert engine._sample_initial_corpus(np.random.default_rng(42)) == engine.source_sequences


def config_for(tmp_path):
    return LongRunConfig("french", PROJECT_ROOT / "data/processed/romance/french_tokens.json",
                         tmp_path / "run", features_freeze_commit=FREEZE_COMMIT,
                         features_freeze_sha256=FEATURE_SHA256)


@pytest.mark.parametrize("change", [{"features_freeze_commit": "bad"}, {"features_freeze_sha256": ""}, {"corpus_budget": 9000}])
def test_invalid_registration_fails_before_artifacts(tmp_path, change):
    cfg = config_for(tmp_path)
    for key, value in change.items():
        setattr(cfg, key, value)
    with pytest.raises(ValueError):
        run_long_continuation(cfg)
    assert not cfg.output_dir.exists()


@pytest.mark.parametrize("status", ["running", "complete", "plateau_hit"])
def test_old_manifests_cannot_resume_even_when_completed(tmp_path, status):
    cfg = config_for(tmp_path)
    cfg.output_dir.mkdir()
    path = cfg.output_dir / "manifest.json"
    path.write_text(json.dumps({"status": status, "config": cfg.to_dict()}))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="fresh output directory"):
        run_long_continuation(cfg)
    assert path.read_bytes() == before


def test_resume_contract_detects_configuration_and_checkpoint_changes(tmp_path):
    cfg = config_for(tmp_path)
    obj, execution = prepare_execution(cfg, None)
    manifest = {"h1_execution": execution, "blocks": [], "config": cfg.to_dict()}
    assert prepare_execution(cfg, manifest)[1] == execution
    cfg.seed += 1
    with pytest.raises(ValueError, match="resume contract changed"):
        prepare_execution(cfg, manifest)
    cfg.seed -= 1
    manifest["blocks"] = [{"h1_audit": {"checkpoint_sha256_lf": "wrong"}}]
    manifest["current_corpus"] = str(cfg.start_corpus)
    with pytest.raises(ValueError, match="checkpoint"):
        prepare_execution(cfg, manifest)


def test_engine_builder_receives_active_registration(tmp_path):
    cfg = config_for(tmp_path)
    assert H1Objective.from_config(_default_v5_config(cfg, 2)).features == FEATURES


def test_missing_freeze_does_not_downgrade_h1_to_legacy(tmp_path):
    cfg = config_for(tmp_path)
    manifest = {"config": cfg.to_dict()}
    cfg.features_freeze_commit = cfg.features_freeze_sha256 = ""
    with pytest.raises(ValueError, match="legacy scorer"):
        prepare_execution(cfg, manifest)


def test_real_engine_blocks_stop_and_resume_without_resampling(tmp_path, monkeypatch, objective):
    import src.retrodiction.long_run_v4 as driver
    import src.retrodiction.engine_reinforced_v4 as v4
    monkeypatch.setattr(H1Objective, "from_config", classmethod(lambda cls, cfg: objective))
    monkeypatch.setattr(driver, "LatinReference", ForbiddenLegacyReward)
    monkeypatch.setattr(driver, "LatinFormReference", _FakeLatinFormRef)
    monkeypatch.setattr(driver, "ReferenceSet", _FakeReferenceSet)
    monkeypatch.setattr(v4, "load_latin_family_reference", lambda cfg, **kw: extract_family_inventory("latin", corpus(), cfg))
    class StopAfterFirstBlock(RelationalReinforcedRetrodictionEngineV5):
        def _mutate_candidate(self, *args, **kwargs):
            return MutationPayload(sequences=corpus()), "test_relabel", "", 0

        def run(self):
            result = super().run()
            if self.output_dir.name == "block_0001":
                (self.output_dir.parent.parent / "stop_requested").touch()
            return result
    cfg = config_for(tmp_path)
    cfg.corpus_budget = objective.budget
    cfg.total_target_proposals = 2
    cfg.block_proposals = 1
    cfg.n_candidates = 1
    first = run_long_continuation(cfg, engine_class=StopAfterFirstBlock, config_builder=_default_v5_config)
    assert first["status"] == "stopped_manual"
    assert len(first["blocks"]) == 1
    checkpoint = json.loads(Path(first["current_corpus"]).read_text())["sequences"]
    (cfg.output_dir / "stop_requested").unlink()
    second = run_long_continuation(cfg, engine_class=StopAfterFirstBlock, config_builder=_default_v5_config)
    assert second["status"] == "complete"
    assert len(second["blocks"]) == 2
    summary = json.loads((cfg.output_dir / "blocks/block_0002/run_summary.json").read_text())
    baseline = json.loads(Path(summary["stages"][0]["artifacts"]["corpus_json"]).read_text())["sequences"]
    assert baseline == checkpoint
    assert summary["config"]["preserve_initial_corpus"] is True
