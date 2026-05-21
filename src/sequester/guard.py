"""
Sequestration Guard
===================
Purpose:
    Enforce the Island B firewall. Sequestered corpora (Latin ground truth,
    Portuguese positive control) must never be accessed by reconstruction modules.
    This module provides the only sanctioned path to sequestered data, and raises
    SequestrationViolation if that path is called while the firewall is active.

Design:
    The firewall is a module-level boolean, locked by default. All access to
    sequestered corpus files must go through load_sequestered(). Reconstruction
    modules should never call load_sequestered() — they have no reason to.

    The sequestration verification test (tests/test_sequestration.py) confirms:
        1. load_sequestered("latin") raises SequestrationViolation by default.
        2. load_sequestered("portuguese") raises SequestrationViolation by default.
        3. After unlock_sequestration(reason), load_sequestered() no longer raises.
        4. unlock_sequestration("") raises ValueError (empty reason not accepted).
        5. Re-locking with lock_sequestration() restores the firewall.

    Reconstruction modules are forbidden from importing or calling this module's
    unlock function. The test suite enforces this by grepping reconstruction
    source files for any reference to unlock_sequestration.

Sequestered corpora:
    - "latin":      Island B. Classical Latin ground truth. Opened only in Phase 5.
    - "portuguese": Positive control. Opened only after reconstruction is complete.

Usage (Phase 5 validation only):
    from src.sequester.guard import unlock_sequestration, load_sequestered

    unlock_sequestration("Phase 5 validation: reconstruction complete, lifting firewall")
    tokens = load_sequestered("latin")
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEQUESTERED_DIR = PROJECT_ROOT / "data" / "sequestered"

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# All corpora in this set require firewall bypass to access.
SEQUESTERED_CORPORA: frozenset[str] = frozenset(["latin", "portuguese"])

# ---------------------------------------------------------------------------
# Firewall state
# ---------------------------------------------------------------------------

# Locked by default. Never set this directly — use unlock_sequestration().
_SEQUESTRATION_LOCKED: bool = True


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class SequestrationViolation(Exception):
    """
    Raised when a sequestered corpus is accessed while the firewall is active.

    If you are seeing this exception, you have attempted to access Island B
    (Latin ground truth) or the Portuguese positive control before reconstruction
    is complete. This is a methodological violation.

    To lift the firewall intentionally (Phase 5 validation only), call:
        unlock_sequestration(reason="Phase 5 validation: ...")
    """


# ---------------------------------------------------------------------------
# Firewall controls
# ---------------------------------------------------------------------------

def unlock_sequestration(reason: str) -> None:
    """
    Lift the sequestration firewall.

    This function must only be called from validation scripts (Phase 5) after
    all reconstruction is complete. It must not be called from any module in
    src/retrodiction/, src/ingest/, or src/fingerprint/.

    Args:
        reason: A substantive string documenting why the firewall is being lifted.
                Must be at least 20 characters. This is recorded in the log.

    Raises:
        ValueError: If reason is empty or too short.
    """
    global _SEQUESTRATION_LOCKED
    if not reason or len(reason.strip()) < 20:
        raise ValueError(
            "unlock_sequestration requires a reason string of at least 20 characters. "
            "Document why the firewall is being lifted."
        )
    _SEQUESTRATION_LOCKED = False
    log.warning(
        "SEQUESTRATION FIREWALL LIFTED. Reason: %s",
        reason,
    )


def lock_sequestration() -> None:
    """Re-engage the sequestration firewall. Used in tests and after validation."""
    global _SEQUESTRATION_LOCKED
    _SEQUESTRATION_LOCKED = True
    log.info("Sequestration firewall re-engaged.")


def is_locked() -> bool:
    """Return True if the sequestration firewall is currently active."""
    return _SEQUESTRATION_LOCKED


# ---------------------------------------------------------------------------
# Sequestered data access
# ---------------------------------------------------------------------------

def load_sequestered(corpus_name: str) -> dict:
    """
    Load a sequestered corpus's token file.

    This is the ONLY sanctioned path to sequestered data. Raises
    SequestrationViolation if the firewall is active.

    Args:
        corpus_name: One of "latin" or "portuguese".

    Returns:
        The parsed JSON corpus dict (same format as processed romance corpora).

    Raises:
        SequestrationViolation: If the firewall is active.
        ValueError: If corpus_name is not a known sequestered corpus.
        FileNotFoundError: If the corpus file has not yet been ingested.
    """
    if corpus_name not in SEQUESTERED_CORPORA:
        raise ValueError(
            f"'{corpus_name}' is not a sequestered corpus. "
            f"Sequestered corpora: {sorted(SEQUESTERED_CORPORA)}"
        )

    if _SEQUESTRATION_LOCKED:
        raise SequestrationViolation(
            f"Access to sequestered corpus '{corpus_name}' was blocked by the "
            f"sequestration firewall. "
            f"Call unlock_sequestration(reason=...) in the validation phase only."
        )

    tokens_path = SEQUESTERED_DIR / corpus_name / f"{corpus_name}_tokens.json"
    if not tokens_path.exists():
        raise FileNotFoundError(
            f"Sequestered corpus file not found: {tokens_path}. "
            f"Run the ingestion script first."
        )

    log.info("Loading sequestered corpus '%s' from %s", corpus_name, tokens_path)
    with tokens_path.open(encoding="utf-8") as fh:
        return json.load(fh)


def sequestered_path(corpus_name: str) -> Path:
    """
    Return the filesystem path to a sequestered corpus directory.
    Does NOT check the firewall — use this only for writing during ingest,
    not for reading during reconstruction.

    Args:
        corpus_name: One of "latin" or "portuguese".
    """
    if corpus_name not in SEQUESTERED_CORPORA:
        raise ValueError(f"'{corpus_name}' is not a sequestered corpus.")
    return SEQUESTERED_DIR / corpus_name
