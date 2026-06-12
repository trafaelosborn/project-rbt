"""Tests for src.ingest.v2_corpus_manifest."""

import json
from pathlib import Path

from src.ingest import v2_corpus_manifest


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_manifest_marks_missing_required_corpora_as_blockers(tmp_path, monkeypatch):
    specs = (
        v2_corpus_manifest.LocalCorpusSpec(
            corpus_id="latin",
            language="latin",
            role="target",
            family="italic",
            tokens_path="data/latin_tokens.json",
            manifest_path="data/latin_manifest.json",
        ),
    )
    monkeypatch.setattr(v2_corpus_manifest, "LOCAL_CORPORA", specs)
    monkeypatch.setattr(v2_corpus_manifest, "REQUIRED_LEIPZIG_IDS", set())
    monkeypatch.setattr(v2_corpus_manifest, "LEIPZIG_PRESETS", {})
    monkeypatch.setattr(v2_corpus_manifest, "TATOEBA_PRESETS", {})
    monkeypatch.setattr(v2_corpus_manifest, "OPTIONAL_PLACEHOLDER_CORPORA", ())

    manifest = v2_corpus_manifest.build_manifest(project_root=tmp_path, corpora_dir=tmp_path / "corpora")

    assert manifest["status"] == "h1_blocked"
    assert manifest["h1_blockers"] == ["latin: missing"]


def test_build_manifest_reads_local_and_leipzig_metadata(tmp_path, monkeypatch):
    _write_json(
        tmp_path / "data" / "latin_manifest.json",
        {"language": "latin", "sequence_count": 10, "total_tokens": 100},
    )
    _write_json(tmp_path / "data" / "latin_tokens.json", {"sequences": [["arma"]]})
    _write_json(
        tmp_path / "corpora" / "turkish_news_2024_100k_manifest.json",
        {
            "corpus_id": "turkish_news_2024_100k",
            "language": "turkish",
            "sequence_count": 100,
            "total_tokens": 1000,
        },
    )
    _write_json(
        tmp_path / "corpora" / "turkish_news_2024_100k_tokens.json",
        {"sequences": [["ev", "var"]]},
    )
    specs = (
        v2_corpus_manifest.LocalCorpusSpec(
            corpus_id="latin",
            language="latin",
            role="target",
            family="italic",
            tokens_path="data/latin_tokens.json",
            manifest_path="data/latin_manifest.json",
        ),
    )
    monkeypatch.setattr(v2_corpus_manifest, "LOCAL_CORPORA", specs)
    monkeypatch.setattr(v2_corpus_manifest, "REQUIRED_LEIPZIG_IDS", {"turkish_news_2024_100k"})
    monkeypatch.setattr(
        v2_corpus_manifest,
        "LEIPZIG_PRESETS",
        {
            "turkish_news_2024_100k": v2_corpus_manifest.LEIPZIG_PRESETS["turkish_news_2024_100k"],
            "japanese_news_2023_100k": v2_corpus_manifest.LEIPZIG_PRESETS["japanese_news_2023_100k"],
        },
    )
    monkeypatch.setattr(v2_corpus_manifest, "OPTIONAL_PLACEHOLDER_CORPORA", ())

    manifest = v2_corpus_manifest.build_manifest(project_root=tmp_path, corpora_dir=tmp_path / "corpora")

    assert manifest["status"] == "h1_ready"
    assert manifest["h1_blockers"] == []
    latin = next(entry for entry in manifest["corpora"] if entry["corpus_id"] == "latin")
    turkish = next(entry for entry in manifest["corpora"] if entry["corpus_id"] == "turkish_news_2024_100k")
    japanese = next(entry for entry in manifest["corpora"] if entry["corpus_id"] == "japanese_news_2023_100k")
    assert latin["status"] == "ready"
    assert latin["stats"]["total_tokens"] == 100
    assert turkish["status"] == "ready"
    assert turkish["stats"]["sequence_count"] == 100
    assert japanese["status"] == "needs_tokenization_policy"
    assert japanese["required_for_h1"] is False


def test_occitan_is_auxiliary_positive_diagnostic_not_h1_required():
    occitan = next(spec for spec in v2_corpus_manifest.LOCAL_CORPORA if spec.corpus_id == "occitan")

    assert occitan.role == "auxiliary_modern_romance_positive_diagnostic"
    assert occitan.required_for_h1 is False
