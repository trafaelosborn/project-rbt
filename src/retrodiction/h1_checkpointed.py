"""Continuous, atomic-checkpoint H1 search. See Decision 047.

This controller deliberately has a distinct schema from the historical block
driver. Scientific termination is a fixed proposal budget; session limits pause.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import time
import uuid

import numpy as np

from src.retrodiction.engine_reinforced_v2 import CandidateState
from src.retrodiction.h1_objective import BudgetError, stable_hash

SCHEMA = "h1_continuous_checkpoint_v1"
OUTCOMES = ("accepted", "rejected", "no_op", "invalid_token_budget")
FORM_TERMS = ("form_weight", "reward_form_gain_weight", "reward_suffix_gain_weight",
              "reward_trigram_gain_weight", "reward_joint_bonus", "reward_penalty_relief")


@dataclass(frozen=True)
class SearchConfig:
    language: str
    corpus_seed: int = 42
    search_seed: int = 42
    form_reward_mode: str = "full"
    proposal_budget: int = 1000
    n_candidates: int = 16
    proposal_checkpoints: tuple[int, ...] = (0, 100, 250, 500, 1000)

    def validate(self):
        if self.form_reward_mode not in {"full", "disabled"}:
            raise ValueError("form_reward_mode must be full or disabled")
        if any(type(v) is not int or v < 0 for v in (self.corpus_seed, self.search_seed)):
            raise ValueError("Seeds must be nonnegative integers")
        if any(type(v) is not int or v < 1 for v in (self.proposal_budget, self.n_candidates)):
            raise ValueError("Budgets and candidate counts must be positive integers")
        points = self.proposal_checkpoints
        if (any(type(p) is not int for p in points) or not points or points[0] != 0
                or points[-1] != self.proposal_budget or tuple(sorted(set(points))) != points):
            raise ValueError("Checkpoints must be unique increasing integers from zero to the budget")

    def to_dict(self):
        self.validate()
        return {**asdict(self), "proposal_checkpoints": list(self.proposal_checkpoints),
                "rng_policy": "continuous_PCG64_checkpoint_v1",
                "diagnostics_policy": "initial_and_every_accepted_state_with_every_proposal_counters",
                "stopping_rule": "fixed_proposal_budget_no_form_or_alignment_early_stop"}

    @classmethod
    def from_dict(cls, value):
        names = cls.__dataclass_fields__
        result = cls(**{k: tuple(v) if k == "proposal_checkpoints" else v
                        for k, v in value.items() if k in names})
        if result.to_dict() != value:
            raise ValueError("Search configuration schema or policy mismatch")
        return result


def engine_config(config, source, output):
    from src.retrodiction.long_run_v4 import LongRunConfig
    from src.retrodiction.long_run_v5 import _default_v5_config
    from src.retrodiction.h1_objective import FREEZE_COMMIT, FEATURE_SHA256
    config.validate()
    cfg = _default_v5_config(LongRunConfig(
        config.language, source, output, seed=config.search_seed,
        n_candidates=config.n_candidates, max_accepted_stages=config.proposal_budget + 1,
        features_freeze_commit=FREEZE_COMMIT, features_freeze_sha256=FEATURE_SHA256,
        use_incremental_scoring=False, use_fortran_cosine=False, use_fortran_batch=False,
    ), config.proposal_budget)
    if config.form_reward_mode == "disabled":
        for name in FORM_TERMS:
            setattr(cfg, name, 0.0)
    return cfg


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    """Publish one complete JSON file; interruption preserves the older pointer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def immutable_json(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Conflicting immutable evidence: {path}")
    else:
        atomic_json(path, value)


def seal(value):
    return {**value, "sha256": stable_hash(value)}


def verify(value):
    if value.get("sha256") != stable_hash({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError("Checkpoint/evidence integrity failure")
    return value


def candidate_dict(candidate):
    result = asdict(candidate)
    result["structural_vector"] = candidate.structural_vector.tolist()
    return result


def candidate_from_dict(value):
    value = dict(value)
    value["structural_vector"] = np.array(value["structural_vector"], dtype=float)
    return CandidateState(**value)


def _entropy(counts):
    total = sum(counts.values())
    return -sum((n / total) * math.log2(n / total) for n in counts.values()) if total else 0.0


def diversity(sequences):
    words = Counter(token for sequence in sequences for token in sequence)
    bigrams = Counter(tuple(sequence[i:i + 2]) for sequence in sequences
                      for i in range(len(sequence) - 1))
    total = sum(words.values())
    return {"measured_token_count": total, "vocabulary_size": len(words),
            "maximum_token_frequency": max(words.values(), default=0) / max(1, total),
            "word_entropy_bits": _entropy(words), "bigram_entropy_bits": _entropy(bigrams)}


def reward_components(engine, before, after):
    cfg = engine.config
    form_gain = after.latin_form_score - before.latin_form_score
    struct_gain = after.latin_structural_score - before.latin_structural_score
    coherent = (after.diagnostics["language_likeness_margin"]
                - before.diagnostics["language_likeness_margin"] >= -cfg.reward_max_coherence_drop)
    joint = struct_gain > 0 and form_gain > 0 and coherent
    result = dict(after.diagnostics["h1"]["score_components"])
    result.update({
        "struct_gain_reward": cfg.reward_struct_gain_weight * max(0.0, struct_gain),
        "form_gain_reward": cfg.reward_form_gain_weight * max(0.0, form_gain),
        "suffix_gain_reward": cfg.reward_suffix_gain_weight * max(0.0,
            after.form_details.get("latin_suffix_cosine", 0) - before.form_details.get("latin_suffix_cosine", 0)),
        "trigram_gain_reward": cfg.reward_trigram_gain_weight * max(0.0,
            after.form_details.get("latin_char_trigram_cosine", 0) - before.form_details.get("latin_char_trigram_cosine", 0)),
        "joint_reward": cfg.reward_joint_bonus if joint else 0.0,
        "cost_relief": cfg.mutation_cost_weight * after.mutation_cost * cfg.reward_penalty_relief if joint else 0.0,
    })
    if not math.isclose(sum(result.values()), after.total_score, rel_tol=0, abs_tol=1e-10):
        raise ValueError("Reward contribution audit failed")
    return result


def save_state(root, engine, current, accepted, proposal, contributions):
    state = seal({"candidate": candidate_dict(current), "accepted_states": accepted,
                  "accepted_at_proposal": proposal, "contributions": contributions,
                  "operator_weights": engine._current_operator_weights.tolist(),
                  "diversity": diversity(engine._h1_objective.measure(current.sequences))})
    immutable_json(root / "states" / (state["sha256"] + ".json"), state)
    return state["sha256"]


def load_state(root, digest):
    # Digests are filenames, never arbitrary paths from a checkpoint.
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Invalid state digest")
    state = verify(read_json(root / "states" / (digest + ".json")))
    if state["sha256"] != digest:
        raise ValueError("State filename/digest mismatch")
    return state


def load_checkpoint(root, contract_hash):
    checkpoint = verify(read_json(root / "checkpoint.json"))
    if checkpoint.get("schema") != SCHEMA or checkpoint.get("contract_sha256") != contract_hash:
        raise ValueError("Resume contract changed; use a new registered run")
    previous = None
    previous_state = None
    counters = {}
    for index in range(checkpoint["proposal_index"] + 1):
        event = verify(read_json(root / "proposals" / f"{index:09d}.json"))
        if (event["proposal_index"] != index or event["previous_sha256"] != previous
                or event["contract_sha256"] != contract_hash
                or (index and event["before_state"] != previous_state)):
            raise ValueError("Proposal chain is discontinuous")
        if index:
            if [a["slot"] for a in event["attempts"]] != list(range(len(event["attempts"]))):
                raise ValueError("Candidate slot sequence is invalid")
            for attempt in event["attempts"]:
                outcome = attempt["outcome"]
                if outcome not in OUTCOMES:
                    raise ValueError("Unknown candidate outcome")
                if attempt["evaluated"] != (outcome in {"accepted", "rejected"}):
                    raise ValueError("Candidate evaluation flag/outcome mismatch")
                counts = counters.setdefault(attempt["operator"], dict.fromkeys(OUTCOMES, 0))
                counts[outcome] += 1
        if event["counters"] != counters:
            raise ValueError("Candidate counter replay failed")
        previous, previous_state = event["sha256"], event["after_state"]
    if previous != checkpoint["event_sha256"] or previous_state != checkpoint["state_sha256"]:
        raise ValueError("Checkpoint pointer differs from proposal chain")
    if counters != checkpoint["counters"]:
        raise ValueError("Checkpoint counters differ from proposal chain")
    return checkpoint


def audit_search(engine, config, root, contract):
    """Replay every committed accepted state through scoring and reward logic."""
    checkpoint = load_checkpoint(root, stable_hash(contract))
    if not 0 <= checkpoint["proposal_index"] <= config.proposal_budget:
        raise ValueError("Checkpoint outside the registered proposal budget")
    previous = None
    last_digest = None
    accepted = 0
    evaluated = 0
    stagnation = 0
    selected_checkpoints = {}
    for index in range(checkpoint["proposal_index"] + 1):
        event = read_json(root / "proposals" / f"{index:09d}.json")
        if index and len(event["attempts"]) != config.n_candidates:
            raise ValueError("Proposal candidate count differs from the contract")
        accepted_attempts = [a for a in event["attempts"] if a["outcome"] == "accepted"]
        evaluated += sum(a["evaluated"] for a in event["attempts"])
        changed = last_digest != event["after_state"]
        if index and (len(accepted_attempts) != int(changed)):
            raise ValueError("Accepted candidate/state transition mismatch")
        if index:
            scored = [a for a in event["attempts"] if a["evaluated"]]
            for attempt in scored:
                if (not math.isclose(sum(attempt["contributions"].values()), attempt["total_score"], abs_tol=1e-10, rel_tol=0)
                        or not math.isclose(attempt["score_delta"], attempt["total_score"] - previous.total_score,
                                            abs_tol=1e-10, rel_tol=0)):
                    raise ValueError("Candidate contribution/score-delta audit failed")
            if scored:
                best = max(scored, key=lambda a: a["total_score"])
                should_accept = best["score_delta"] > engine.config.min_improvement
                if should_accept != changed or (should_accept and best["outcome"] != "accepted"):
                    raise ValueError("Best-candidate selection audit failed")
            stagnation = 0 if changed else stagnation + 1
        if changed:
            saved = load_state(root, event["after_state"])
            stored = candidate_from_dict(saved["candidate"])
            if index == 0 and contract.get("sampled_corpus_sha256") not in {None, stable_hash(stored.sequences)}:
                raise ValueError("Initial state differs from the registered sampled source")
            replay = engine._evaluate_sequences(stored.sequences, stored.mutation_cost)
            if index:
                replay = engine._amplify_reward(previous, replay)
                replay.operator, replay.details = stored.operator, stored.details
                parts = reward_components(engine, previous, replay)
                attempt = accepted_attempts[0]
                if (attempt["corpus_sha256"] != stable_hash(stored.sequences)
                        or attempt["contributions"] != parts
                        or attempt["total_score"] != replay.total_score
                        or replay.total_score - previous.total_score <= engine.config.min_improvement):
                    raise ValueError("Accepted candidate failed independent score replay")
                accepted += 1
            else:
                parts = dict(replay.diagnostics["h1"]["score_components"])
            engine._annotate_alignment(replay)
            if (candidate_dict(replay) != saved["candidate"] or parts != saved["contributions"]
                    or engine._current_operator_weights.tolist() != saved["operator_weights"]
                    or saved["diversity"] != diversity(engine._h1_objective.measure(stored.sequences))
                    or saved["accepted_states"] != accepted or saved["accepted_at_proposal"] != index):
                raise ValueError("Saved state failed independent feature/reward/schedule replay")
            previous = replay
            last_digest = event["after_state"]
        if index in config.proposal_checkpoints:
            selected_checkpoints[str(index)] = {
                "state_sha256": event["after_state"],
                "measurement": previous.diagnostics["h1"],
                "diversity": saved["diversity"],
                "contributions": saved["contributions"],
                "counters": event["counters"], "evaluated_candidates": evaluated,
            }
    if accepted != checkpoint["accepted_states"] or stagnation != checkpoint["stagnation"]:
        raise ValueError("Accepted-state counter mismatch")
    return {"proposals_replayed": checkpoint["proposal_index"], "states_replayed": accepted + 1,
            "evaluated_candidates": evaluated, "checkpoints": selected_checkpoints,
            "checkpoint_sha256": checkpoint["sha256"],
            "complete": checkpoint["proposal_index"] == config.proposal_budget}


def run_search(engine, config, root, contract, initial_sequences, *,
               max_session_proposals=None, max_session_seconds=None):
    """Run/resume at proposal boundaries; a killed partial proposal is replayed.

    The public registered launcher supplies the engine, frozen inputs, and a
    sealed contract. No pickle or untrusted executable checkpoint is used.
    """
    config.validate()
    if max_session_proposals is not None and (type(max_session_proposals) is not int or max_session_proposals < 1):
        raise ValueError("Session proposal limit must be a positive integer")
    if max_session_seconds is not None and (not math.isfinite(max_session_seconds) or max_session_seconds <= 0):
        raise ValueError("Session seconds must be positive and finite")
    if engine._h1_objective is None or engine.config.acceleration_mode != "python_only":
        raise ValueError("Continuous H1 requires the reference scorer and Python proposal policy")
    if engine.config.enable_culture_bombs or engine.config.use_semantic_transparency:
        raise ValueError("Continuous H1 does not support shocks or semantic transparency")
    root = Path(root)
    contract_hash = stable_hash(contract)
    if stable_hash(contract["search_config"]) != stable_hash(config.to_dict()):
        raise ValueError("Search configuration differs from execution contract")
    if contract["engine_config"] != engine.config.to_dict():
        raise ValueError("Engine configuration differs from execution contract")
    root.mkdir(parents=True, exist_ok=True)
    # OS-held lock survives exceptions safely and is released if the process dies.
    from src.retrodiction.h1_run_lock import run_lock
    with run_lock(root / "run.lock"):
        if (root / "checkpoint.json").exists():
            audit_search(engine, config, root, contract)
        return _run_locked(engine, config, root, contract_hash, initial_sequences,
                           max_session_proposals, max_session_seconds)


def _run_locked(engine, config, root, contract_hash, initial_sequences, proposal_limit, seconds_limit):
    started = time.monotonic()
    rng = np.random.Generator(np.random.PCG64(config.search_seed))
    if (root / "checkpoint.json").exists():
        checkpoint = load_checkpoint(root, contract_hash)
        state = load_state(root, checkpoint["state_sha256"])
        current = candidate_from_dict(state["candidate"])
        engine._current_operator_weights = np.array(state["operator_weights"], dtype=float)
        rng.bit_generator.state = checkpoint["rng_state"]
    else:
        current = engine._evaluate_sequences(initial_sequences, 0.0)
        engine._annotate_alignment(current)
        state_hash = save_state(root, engine, current, 0, 0,
                                dict(current.diagnostics["h1"]["score_components"]))
        event = seal({"proposal_index": 0, "previous_sha256": None,
                      "contract_sha256": contract_hash, "before_state": None,
                      "after_state": state_hash, "attempts": [], "counters": {}})
        immutable_json(root / "proposals" / "000000000.json", event)
        checkpoint = seal({"schema": SCHEMA, "contract_sha256": contract_hash,
                           "proposal_index": 0, "accepted_states": 0, "stagnation": 0,
                           "state_sha256": state_hash, "event_sha256": event["sha256"],
                           "rng_state": rng.bit_generator.state, "counters": {}})
        atomic_json(root / "checkpoint.json", checkpoint)
    initial_index = checkpoint["proposal_index"]
    if initial_index > config.proposal_budget:
        raise ValueError("Checkpoint exceeds the registered budget")
    reason = "fixed_budget_complete"
    while checkpoint["proposal_index"] < config.proposal_budget:
        if (root / "PAUSE").exists():
            reason = "pause_file"
            break
        if proposal_limit is not None and checkpoint["proposal_index"] - initial_index >= proposal_limit:
            reason = "session_proposal_limit"
            break
        if seconds_limit is not None and time.monotonic() - started >= seconds_limit:
            reason = "session_time_limit"
            break
        index = checkpoint["proposal_index"] + 1
        attempts, candidates = [], []
        # Mutate the whole pool before scoring, preserving the original policy.
        pool = [engine._mutate_candidate(current.sequences, rng) for _ in range(config.n_candidates)]
        for slot, (payload, operator, details, cost) in enumerate(pool):
            attempt = {"slot": slot, "operator": operator, "details": details,
                       "mutation_cost": cost, "outcome": "no_op", "evaluated": False}
            attempts.append(attempt)
            if payload is None:
                continue
            sequences = payload.materialize(current.sequences)
            attempt["corpus_sha256"] = stable_hash(sequences)
            # A returned unchanged payload was still scored in the historical
            # policy. Preserve that behavior; mark it separately for diagnostics.
            attempt["unchanged_corpus"] = sequences == current.sequences
            try:
                candidate = engine._evaluate_sequences(sequences, cost)
            except BudgetError:
                attempt["outcome"] = "invalid_token_budget"
                continue
            candidate = engine._amplify_reward(current, candidate)
            candidate.operator, candidate.details = operator, details
            parts = reward_components(engine, current, candidate)
            attempt.update(outcome="rejected", evaluated=True, total_score=candidate.total_score,
                           score_delta=candidate.total_score - current.total_score, contributions=parts)
            candidates.append((slot, candidate, parts))
        accepted = False
        state_hash = checkpoint["state_sha256"]
        if candidates:
            slot, best, parts = max(candidates, key=lambda item: item[1].total_score)
            if best.total_score - current.total_score > engine.config.min_improvement:
                accepted = True
                current = best
                engine._annotate_alignment(current)
                attempts[slot]["outcome"] = "accepted"
                state_hash = save_state(root, engine, current, checkpoint["accepted_states"] + 1, index, parts)
        counters = json.loads(json.dumps(checkpoint["counters"]))
        for attempt in attempts:
            counts = counters.setdefault(attempt["operator"], dict.fromkeys(OUTCOMES, 0))
            counts[attempt["outcome"]] += 1
        event = seal({"proposal_index": index, "previous_sha256": checkpoint["event_sha256"],
                      "contract_sha256": contract_hash, "before_state": checkpoint["state_sha256"],
                      "after_state": state_hash, "attempts": attempts, "counters": counters})
        immutable_json(root / "proposals" / f"{index:09d}.json", event)
        checkpoint = seal({"schema": SCHEMA, "contract_sha256": contract_hash,
                           "proposal_index": index, "accepted_states": checkpoint["accepted_states"] + int(accepted),
                           "stagnation": 0 if accepted else checkpoint["stagnation"] + 1,
                           "state_sha256": state_hash, "event_sha256": event["sha256"],
                           "rng_state": rng.bit_generator.state, "counters": counters})
        atomic_json(root / "checkpoint.json", checkpoint)
    complete = checkpoint["proposal_index"] == config.proposal_budget
    result = {"status": "complete" if complete else "paused", "reason": reason,
              "proposal_index": checkpoint["proposal_index"], "accepted_states": checkpoint["accepted_states"],
              "checkpoint_sha256": checkpoint["sha256"], "contract_sha256": contract_hash,
              "scientific_termination": "fixed_budget" if complete else None,
              "session_start_proposal": initial_index,
              "session_wall_seconds": time.monotonic() - started}
    atomic_json(root / "sessions" / (uuid.uuid4().hex + ".json"), result)
    return result
