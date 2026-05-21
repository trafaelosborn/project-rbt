"""
Sequestration firewall verification tests
==========================================
These tests are the automated enforcement of the Island B firewall.

Critical invariants verified here:
    1. load_sequestered("latin") raises SequestrationViolation by default.
    2. load_sequestered("portuguese") raises SequestrationViolation by default.
    3. After unlock_sequestration(reason), load_sequestered() no longer raises
       (assuming the corpus file exists — we mock the file access here).
    4. unlock_sequestration with an empty/short reason raises ValueError.
    5. lock_sequestration() re-engages the firewall.
    6. load_sequestered() with an unknown corpus name raises ValueError.
    7. No reconstruction module (src/fingerprint/, src/nullmodel/) references
       unlock_sequestration — enforced by source code inspection.

These tests must pass at every commit. A failing sequestration test is a
methodological violation, not a code quality issue.
"""

import sys
import importlib
from pathlib import Path
from unittest.mock import mock_open, patch
import json
import pytest

# ---------------------------------------------------------------------------
# Ensure the src package is importable from the project root
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reload_guard():
    """
    Reload the guard module to reset firewall state between tests.
    The _SEQUESTRATION_LOCKED global must start True for each test.
    """
    import src.sequester.guard as guard
    importlib.reload(guard)
    return guard


# ---------------------------------------------------------------------------
# Default firewall state
# ---------------------------------------------------------------------------

class TestFirewallDefault:
    def test_latin_blocked_by_default(self):
        guard = _reload_guard()
        with pytest.raises(guard.SequestrationViolation):
            guard.load_sequestered("latin")

    def test_portuguese_blocked_by_default(self):
        guard = _reload_guard()
        with pytest.raises(guard.SequestrationViolation):
            guard.load_sequestered("portuguese")

    def test_is_locked_returns_true_by_default(self):
        guard = _reload_guard()
        assert guard.is_locked() is True

    def test_unknown_corpus_raises_value_error(self):
        guard = _reload_guard()
        with pytest.raises(ValueError):
            guard.load_sequestered("italian")

    def test_unknown_corpus_raises_even_when_unlocked(self):
        guard = _reload_guard()
        guard.unlock_sequestration("Phase 5 validation: testing unknown corpus")
        with pytest.raises(ValueError):
            guard.load_sequestered("german")


# ---------------------------------------------------------------------------
# Unlock / lock cycle
# ---------------------------------------------------------------------------

class TestUnlockLock:
    def test_unlock_requires_substantive_reason(self):
        guard = _reload_guard()
        with pytest.raises(ValueError):
            guard.unlock_sequestration("")

    def test_unlock_requires_reason_at_least_20_chars(self):
        guard = _reload_guard()
        with pytest.raises(ValueError):
            guard.unlock_sequestration("too short")

    def test_unlock_with_valid_reason_succeeds(self):
        guard = _reload_guard()
        guard.unlock_sequestration("Phase 5 validation: reconstruction complete")
        assert guard.is_locked() is False

    def test_lock_restores_firewall(self):
        guard = _reload_guard()
        guard.unlock_sequestration("Phase 5 validation: reconstruction complete")
        guard.lock_sequestration()
        assert guard.is_locked() is True

    def test_latin_accessible_after_unlock_with_mock_file(self):
        guard = _reload_guard()
        guard.unlock_sequestration("Phase 5 validation: reconstruction complete")

        fake_corpus = {"language": "latin", "sequences": [["arma", "virumque", "cano"]]}
        fake_json = json.dumps(fake_corpus)

        with patch("builtins.open", mock_open(read_data=fake_json)), \
             patch.object(Path, "exists", return_value=True), \
             patch.object(Path, "open", mock_open(read_data=fake_json)):
            result = guard.load_sequestered("latin")
        assert result["language"] == "latin"

    def test_firewall_re_engaged_after_lock(self):
        guard = _reload_guard()
        guard.unlock_sequestration("Phase 5 validation: reconstruction complete")
        guard.lock_sequestration()
        with pytest.raises(guard.SequestrationViolation):
            guard.load_sequestered("latin")


# ---------------------------------------------------------------------------
# Source code inspection: no reconstruction module references unlock
# ---------------------------------------------------------------------------

class TestSourceCodeFirewall:
    """
    Verify that reconstruction-adjacent modules do not bypass the sequestration firewall.

    Rules:
    - fingerprint, nullmodel, ingest must never call unlock_sequestration or load_sequestered.
    - retrodiction, accelerate, control must never call load_sequestered directly.
      (They access Latin only via LatinFormReference / load_latin_family_reference,
      which handle unlock/lock internally.)
    - retrodiction, accelerate, control must never call sequestered_path().
      (That function bypasses the firewall check and is only for ingest writes.)
    """

    INGEST_ADJACENT = [
        PROJECT_ROOT / "src" / "fingerprint",
        PROJECT_ROOT / "src" / "nullmodel",
        PROJECT_ROOT / "src" / "ingest",
    ]

    RECONSTRUCTION_MODULES = [
        PROJECT_ROOT / "src" / "retrodiction",
        PROJECT_ROOT / "src" / "accelerate",
        PROJECT_ROOT / "src" / "control",
    ]

    def _scan_for_pattern(self, directory: Path, pattern: str) -> list[Path]:
        """Return list of .py files under directory that contain pattern."""
        hits = []
        if not directory.exists():
            return hits
        for py_file in directory.rglob("*.py"):
            if pattern in py_file.read_text(encoding="utf-8"):
                hits.append(py_file)
        return hits

    def test_fingerprint_modules_do_not_call_unlock(self):
        hits = self._scan_for_pattern(
            PROJECT_ROOT / "src" / "fingerprint",
            "unlock_sequestration",
        )
        assert hits == [], (
            f"Found unlock_sequestration in fingerprint modules: {hits}. "
            "Fingerprint modules must never bypass the sequestration firewall."
        )

    def test_nullmodel_modules_do_not_call_unlock(self):
        hits = self._scan_for_pattern(
            PROJECT_ROOT / "src" / "nullmodel",
            "unlock_sequestration",
        )
        assert hits == [], (
            f"Found unlock_sequestration in nullmodel modules: {hits}."
        )

    def test_ingest_modules_do_not_call_unlock(self):
        hits = self._scan_for_pattern(
            PROJECT_ROOT / "src" / "ingest",
            "unlock_sequestration",
        )
        assert hits == [], (
            f"Found unlock_sequestration in ingest modules: {hits}. "
            "Ingest modules may WRITE to sequestered paths but must not unlock the firewall."
        )

    def test_fingerprint_modules_do_not_call_load_sequestered(self):
        hits = self._scan_for_pattern(
            PROJECT_ROOT / "src" / "fingerprint",
            "load_sequestered",
        )
        assert hits == [], (
            f"Found load_sequestered in fingerprint modules: {hits}."
        )

    def test_nullmodel_modules_do_not_call_load_sequestered(self):
        hits = self._scan_for_pattern(
            PROJECT_ROOT / "src" / "nullmodel",
            "load_sequestered",
        )
        assert hits == [], (
            f"Found load_sequestered in nullmodel modules: {hits}."
        )

    def test_reconstruction_modules_do_not_call_load_sequestered_directly(self):
        """
        Retrodiction, accelerate, and control modules must not call load_sequestered()
        directly. Latin access must go through LatinFormReference or
        load_latin_family_reference, which manage unlock/lock internally.
        Direct load_sequestered calls bypass the abstraction boundary.
        """
        for directory in self.RECONSTRUCTION_MODULES:
            hits = self._scan_for_pattern(directory, "load_sequestered")
            assert hits == [], (
                f"Found load_sequestered in {directory.name} modules: {hits}. "
                "Reconstruction modules must access Latin only via LatinFormReference "
                "or load_latin_family_reference — never directly via load_sequestered."
            )

    def test_reconstruction_modules_do_not_call_sequestered_path(self):
        """
        sequestered_path() bypasses the firewall check and is only for ingest writes.
        Reconstruction modules must never call it — doing so would allow raw corpus
        files to be read without triggering SequestrationViolation.
        """
        for directory in self.RECONSTRUCTION_MODULES:
            hits = self._scan_for_pattern(directory, "sequestered_path")
            assert hits == [], (
                f"Found sequestered_path in {directory.name} modules: {hits}. "
                "sequestered_path() is for ingest writes only and must not be used "
                "in reconstruction modules."
            )
