"""Tests for src.control.run_config."""

import json
import tempfile
from pathlib import Path

import pytest

from src.control.run_config import (
    BLOCK_PROPOSALS_PRESETS,
    CANDIDATE_PRESETS,
    BenchmarkConfig,
    ConfigLockedError,
    ConfigValidationError,
    HIGH_DEPTH_V5_CANDIDATE_COUNT,
    LIVE_EVENT_MODES,
    RECOMMENDED_V5_CANDIDATE_COUNT,
    RECOMMENDED_V5_USE_FORTRAN_BATCH,
    RunConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_config(tmp_path: Path) -> RunConfig:
    """A RunConfig that passes validate_strict() — corpus file exists."""
    corpus = tmp_path / "corpus_tokens.json"
    corpus.write_text('{"sequences": []}', encoding="utf-8")
    return RunConfig(
        source_language="french",
        target_language="latin",
        start_corpus=corpus,
        until_mode="latin_hit",
        candidate_count=8,
        block_proposals=1000,
    )


# ---------------------------------------------------------------------------
# Preset enforcement
# ---------------------------------------------------------------------------

class TestCandidateCountPresets:
    def test_recommended_defaults_are_pinned(self):
        cfg = RunConfig()
        assert cfg.candidate_count == RECOMMENDED_V5_CANDIDATE_COUNT
        assert cfg.use_fortran_batch is RECOMMENDED_V5_USE_FORTRAN_BATCH

    def test_high_depth_mode_is_a_valid_preset(self):
        assert HIGH_DEPTH_V5_CANDIDATE_COUNT in CANDIDATE_PRESETS

    def test_valid_preset_accepted(self, tmp_path):
        for c in CANDIDATE_PRESETS:
            cfg = _valid_config(tmp_path)
            cfg.candidate_count = c
            assert cfg.validate() == []

    def test_invalid_count_rejected(self, tmp_path):
        cfg = _valid_config(tmp_path)
        cfg.candidate_count = 7
        errors = cfg.validate()
        assert any("candidate_count" in e for e in errors)

    def test_zero_count_rejected(self, tmp_path):
        cfg = _valid_config(tmp_path)
        cfg.candidate_count = 0
        errors = cfg.validate()
        assert any("candidate_count" in e for e in errors)


class TestBlockProposalsPresets:
    def test_valid_preset_accepted(self, tmp_path):
        for b in BLOCK_PROPOSALS_PRESETS:
            cfg = _valid_config(tmp_path)
            cfg.block_proposals = b
            assert cfg.validate() == []

    def test_invalid_block_size_rejected(self, tmp_path):
        cfg = _valid_config(tmp_path)
        cfg.block_proposals = 500
        errors = cfg.validate()
        assert any("block_proposals" in e for e in errors)


# ---------------------------------------------------------------------------
# Until-mode validation
# ---------------------------------------------------------------------------

class TestUntilMode:
    def test_budget_mode_requires_proposals(self, tmp_path):
        cfg = _valid_config(tmp_path)
        cfg.until_mode = "budget"
        cfg.total_target_proposals = 0
        errors = cfg.validate()
        assert any("total_target_proposals" in e for e in errors)

    def test_budget_mode_valid_with_proposals(self, tmp_path):
        cfg = _valid_config(tmp_path)
        cfg.until_mode = "budget"
        cfg.total_target_proposals = 1000
        assert cfg.validate() == []

    def test_invalid_until_mode_rejected(self, tmp_path):
        cfg = _valid_config(tmp_path)
        cfg.until_mode = "never"
        errors = cfg.validate()
        assert any("until_mode" in e for e in errors)


# ---------------------------------------------------------------------------
# Semantic transparency validation
# ---------------------------------------------------------------------------

class TestSemanticTransparency:
    def test_transparency_enabled_requires_weight(self, tmp_path):
        cfg = _valid_config(tmp_path)
        cfg.use_semantic_transparency = True
        cfg.transparency_weight = 0.0
        errors = cfg.validate()
        assert any("transparency_weight" in e for e in errors)

    def test_transparency_enabled_with_weight_valid(self, tmp_path):
        cfg = _valid_config(tmp_path)
        cfg.use_semantic_transparency = True
        cfg.transparency_weight = 0.1
        assert cfg.validate() == []

    def test_transparency_disabled_zero_weight_valid(self, tmp_path):
        cfg = _valid_config(tmp_path)
        cfg.use_semantic_transparency = False
        cfg.transparency_weight = 0.0
        assert cfg.validate() == []


class TestLiveEventControls:
    def test_valid_live_event_modes_are_accepted(self, tmp_path):
        for mode in LIVE_EVENT_MODES:
            cfg = _valid_config(tmp_path)
            cfg.live_event_mode = mode
            assert cfg.validate() == []

    def test_invalid_live_event_mode_rejected(self, tmp_path):
        cfg = _valid_config(tmp_path)
        cfg.live_event_mode = "maximum_drama"
        errors = cfg.validate()
        assert any("live_event_mode" in e for e in errors)

    def test_live_event_buffer_size_must_be_positive(self, tmp_path):
        cfg = _valid_config(tmp_path)
        cfg.live_event_buffer_size = 0
        errors = cfg.validate()
        assert any("live_event_buffer_size" in e for e in errors)


class TestValidatorSnapshots:
    def test_validator_snapshot_every_blocks_must_be_nonnegative(self, tmp_path):
        cfg = _valid_config(tmp_path)
        cfg.validator_snapshot_every_blocks = -1
        errors = cfg.validate()
        assert any("validator_snapshot_every_blocks" in e for e in errors)

    def test_validator_snapshot_warning_mentions_scope(self, tmp_path):
        cfg = _valid_config(tmp_path)
        cfg.validator_snapshot_every_blocks = 1
        cfg.validator_set = ["old_french", "middle_french"]
        warns = cfg.warnings()
        assert any("validator-bank snapshots" in w for w in warns)


# ---------------------------------------------------------------------------
# Corpus path validation
# ---------------------------------------------------------------------------

class TestCorpusPath:
    def test_missing_corpus_produces_error(self, tmp_path):
        cfg = _valid_config(tmp_path)
        cfg.start_corpus = tmp_path / "nonexistent.json"
        errors = cfg.validate()
        assert any("start_corpus" in e for e in errors)

    def test_existing_corpus_accepted(self, tmp_path):
        cfg = _valid_config(tmp_path)
        assert cfg.validate() == []


# ---------------------------------------------------------------------------
# Config lock
# ---------------------------------------------------------------------------

class TestConfigLock:
    def test_unlocked_config_is_mutable(self, tmp_path):
        cfg = _valid_config(tmp_path)
        cfg.seed = 99
        assert cfg.seed == 99

    def test_locked_config_raises_on_mutation(self, tmp_path):
        cfg = _valid_config(tmp_path)
        cfg.lock()
        with pytest.raises(ConfigLockedError):
            cfg.seed = 99

    def test_lock_is_irreversible(self, tmp_path):
        cfg = _valid_config(tmp_path)
        cfg.lock()
        assert cfg.is_locked
        # calling lock() again is idempotent
        cfg.lock()
        assert cfg.is_locked

    def test_locked_config_still_readable(self, tmp_path):
        cfg = _valid_config(tmp_path)
        cfg.lock()
        assert cfg.seed == 42
        assert cfg.candidate_count == 8

    def test_validate_strict_raises_on_errors(self, tmp_path):
        cfg = _valid_config(tmp_path)
        cfg.candidate_count = 7
        with pytest.raises(ConfigValidationError):
            cfg.validate_strict()


# ---------------------------------------------------------------------------
# Warnings (non-fatal)
# ---------------------------------------------------------------------------

class TestWarnings:
    def test_fortran_batch_high_count_warns(self, tmp_path):
        cfg = _valid_config(tmp_path)
        cfg.use_fortran_batch = True
        cfg.candidate_count = 32
        warns = cfg.warnings()
        assert any("Fortran-backed batch candidate scoring path" in w for w in warns)

    def test_fortran_batch_count_8_still_warns(self, tmp_path):
        cfg = _valid_config(tmp_path)
        cfg.use_fortran_batch = True
        cfg.candidate_count = 8
        warns = cfg.warnings()
        assert any("Fortran-backed batch candidate scoring path" in w for w in warns)

    def test_transparency_always_warns(self, tmp_path):
        cfg = _valid_config(tmp_path)
        cfg.use_semantic_transparency = True
        cfg.transparency_weight = 0.1
        warns = cfg.warnings()
        assert any("experimental condition" in w for w in warns)


# ---------------------------------------------------------------------------
# Serialisation round-trip
# ---------------------------------------------------------------------------

class TestSerialisation:
    def test_to_dict_from_dict_roundtrip(self, tmp_path):
        cfg = _valid_config(tmp_path)
        cfg.seed = 77
        cfg.candidate_count = 16
        restored = RunConfig.from_dict(cfg.to_dict())
        assert restored.seed == 77
        assert restored.candidate_count == 16
        assert restored.source_language == "french"

    def test_save_and_load_from_json(self, tmp_path):
        cfg = _valid_config(tmp_path)
        cfg.block_proposals = 5000
        save_path = tmp_path / "run_config.json"
        cfg.save(save_path)
        loaded = RunConfig.from_json(save_path)
        assert loaded.block_proposals == 5000

    def test_locked_config_serialises(self, tmp_path):
        cfg = _valid_config(tmp_path)
        cfg.lock()
        d = cfg.to_dict()
        assert d["seed"] == 42
        # _locked must not appear in the serialised dict
        assert "_locked" not in d

    def test_new_runtime_fields_roundtrip(self, tmp_path):
        cfg = _valid_config(tmp_path)
        cfg.validator_set = ["old_french"]
        cfg.validator_snapshot_every_blocks = 2
        cfg.live_event_mode = "selected"
        cfg.live_event_buffer_size = 128
        restored = RunConfig.from_dict(cfg.to_dict())
        assert restored.validator_set == ["old_french"]
        assert restored.validator_snapshot_every_blocks == 2
        assert restored.live_event_mode == "selected"
        assert restored.live_event_buffer_size == 128
