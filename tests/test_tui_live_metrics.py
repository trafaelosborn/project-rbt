"""Tests for Logan's Run live comparison helpers in src.control.tui."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.control import tui


def test_relative_lift_handles_zero_baseline():
    assert tui._relative_lift(1.0, 0.0) is None


def test_relative_lift_positive_when_current_is_better():
    lift = tui._relative_lift(-0.12, -0.24)
    assert lift is not None
    assert lift > 0.0


def test_load_v4_baseline_reference_reads_report_and_events(tmp_path, monkeypatch):
    report_path = tmp_path / "headtohead_report.json"
    events_path = tmp_path / "live_events.jsonl"

    report_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "label": "v4_baseline",
                        "proposals_per_hour": 3301.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    events_path.write_text(
        "\n".join(
            [
                json.dumps({"operator": "macro_bundle_rewrite", "score_delta": -0.20}),
                json.dumps({"operator": "macro_bundle_rewrite", "score_delta": -0.10}),
                json.dumps({"operator": "split_token", "score_delta": -0.30}),
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(tui, "V4_BASELINE_REPORT", report_path)
    monkeypatch.setattr(tui, "V4_BASELINE_EVENTS", events_path)

    baseline = tui._load_v4_baseline_reference()
    assert baseline is not None
    assert baseline.proposals_per_hour == 3301.0
    assert baseline.aggregate_mean_score_delta == pytest.approx((-0.20 - 0.10 - 0.30) / 3)
    assert baseline.operator_mean_score_delta["macro_bundle_rewrite"] == pytest.approx((-0.20 - 0.10) / 2)


def test_live_event_accumulator_ingests_incrementally(tmp_path):
    run_dir = tmp_path / "run"
    block_dir = run_dir / "blocks" / "block_0001"
    block_dir.mkdir(parents=True)
    events_path = block_dir / "live_events.jsonl"

    first_batch = [
        {
            "proposal_index": 1,
            "candidate_index": 1,
            "operator": "macro_bundle_rewrite",
            "score_delta": -0.20,
            "details": "a",
        },
        {
            "proposal_index": 1,
            "candidate_index": 2,
            "operator": "split_token",
            "score_delta": -0.30,
            "details": "b",
        },
    ]
    events_path.write_text(
        "\n".join(json.dumps(event) for event in first_batch) + "\n",
        encoding="utf-8",
    )

    acc = tui._LiveEventAccumulator(max_recent=10)
    recent = acc.ingest_run_dir(run_dir)
    assert len(recent) == 2
    assert acc.aggregate_mean_score_delta() == pytest.approx((-0.20 - 0.30) / 2)

    second_batch = first_batch + [
        {
            "proposal_index": 2,
            "candidate_index": 1,
            "operator": "macro_bundle_rewrite",
            "score_delta": -0.10,
            "details": "c",
        }
    ]
    events_path.write_text(
        "\n".join(json.dumps(event) for event in second_batch) + "\n",
        encoding="utf-8",
    )

    recent = acc.ingest_run_dir(run_dir)
    assert len(recent) == 3
    assert acc.aggregate_mean_score_delta() == pytest.approx((-0.20 - 0.30 - 0.10) / 3)
    top = acc.top_operator_means(limit=2)
    assert top[0][0] == "macro_bundle_rewrite"
    assert top[0][2] == 2


def test_find_latest_validator_summary_uses_run_specific_artifacts(tmp_path, monkeypatch):
    run_dir = tmp_path / "my_run"
    run_dir.mkdir()
    validation_dir = tmp_path / "validation"
    validation_dir.mkdir()

    older = validation_dir / "other_run_vs_validator_bank_chronology.json"
    older.write_text("{}", encoding="utf-8")

    target = validation_dir / "my_run_vs_validator_bank_chronology.json"
    target.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(tui, "VALIDATION_DIR", validation_dir)

    found = tui._find_latest_validator_summary(run_dir)
    assert found == target


def test_validator_bank_lines_render_summary():
    summary = {
        "validator_count": 6,
        "block_count": 614,
        "structural_path": ["middle_french", "anglo_norman", "old_occitan"],
        "form_path": ["middle_french", "langue_d_oil", "middle_french"],
        "nearest_structural_by_block": [
            {
                "validator_corpus": "old_occitan",
                "validator_period": "Old Occitan",
                "validator_date_start": 1100,
                "validator_date_end": 1400,
                "validator_structural_distance": 1.9953,
            }
        ],
        "nearest_form_by_block": [
            {
                "validator_corpus": "middle_french",
                "validator_period": "Middle French",
                "validator_date_start": 1450,
                "validator_date_end": 1489,
                "validator_form_score": 0.4162,
            }
        ],
    }

    lines = tui._validator_bank_lines(
        summary,
        summary_path=Path("paper_run_vs_validator_bank_chronology.json"),
        status_text="updated 12:34:56",
    )

    rendered = "\n".join(lines)
    assert "updated 12:34:56" in rendered
    assert "6 validators across 614 blocks" in rendered
    assert "old_occitan" in rendered
    assert "middle_french" in rendered
    assert "middle_french -> anglo_norman -> old_occitan" in rendered


@pytest.mark.skipif(not tui._TEXTUAL_AVAILABLE, reason="textual not installed")
def test_run_status_panel_live_progress_uses_in_block_proposals():
    panel = tui.RunStatusPanel()
    status = tui.RunStatus(
        output_dir=Path("."),
        status="running",
        cumulative_proposals=0,
        blocks_completed=0,
        last_struct=None,
        last_form=None,
        last_align=None,
        last_updated_utc=None,
        current_corpus=None,
    )

    panel.update_status(status)
    panel.update_live_progress(
        status,
        current_block_proposals=44,
        current_block_name="block_0001",
        live_struct=-1.23,
        live_form=0.71,
        live_align=0.44,
    )

    assert panel.proposals == 44
    assert panel.blocks == 0
    assert panel.struct == pytest.approx(-1.23)
    assert panel.form == pytest.approx(0.71)
    assert panel.align == pytest.approx(0.44)
    assert panel.live_note == "block_0001 in progress"


@pytest.mark.skipif(not tui._TEXTUAL_AVAILABLE, reason="textual not installed")
def test_throughput_panel_renders_metrics():
    panel = tui.ThroughputPanel()
    panel.update_metrics(
        elapsed_seconds=125.0,
        proposals_per_hour=6487.0,
        total_proposals=2212,
        blocks_completed=2,
        current_block_name="block_0003",
    )

    rendered = panel.render()
    assert "Throughput" in rendered
    assert "06,487 p/h" not in rendered
    assert "6,487 p/h" in rendered
    assert "2,212 proposals across 2 completed blocks" in rendered
    assert "block_0003 in progress" in rendered


@pytest.mark.skipif(not tui._TEXTUAL_AVAILABLE, reason="textual not installed")
def test_logans_run_panel_renders_candidate_events():
    panel = tui.LogansRunPanel()
    panel.update_from_events(
        [
            {
                "proposal_index": 12,
                "candidate_index": 3,
                "outcome": "accepted",
                "operator": "macro_bundle_rewrite",
                "score_delta": -0.1535,
                "backend": "fortran",
                "details": "est->eta across 34 occurrences",
                "timestamp_utc": "2026-04-18T17:20:27+00:00",
            }
        ]
    )

    rendered = panel.render()
    assert "Logan's Run" in rendered
    assert "p000012 c03 accepted" in rendered
    assert "macro_bundle_rewrite" in rendered
    assert "est->eta across 34 occurrences" in rendered


@pytest.mark.skipif(not tui._TEXTUAL_AVAILABLE, reason="textual not installed")
def test_comparison_panel_renders_lines():
    panel = tui.LiveMovePanel()
    panel.update_lines(
        [
            "Wall clock : 00:20:27",
            "Pace      : 6,487 p/h vs v4 3,301 (+96.5%)",
            "Aggregate : dscore -0.1535 vs v4 -0.2531 (+39.3%)",
        ]
    )

    rendered = panel.render()
    assert "v4 Comparison" in rendered
    assert "6,487 p/h vs v4 3,301" in rendered
    assert "Aggregate : dscore -0.1535 vs v4 -0.2531" in rendered
