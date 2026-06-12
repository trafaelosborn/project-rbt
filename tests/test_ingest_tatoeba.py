"""Tests for src.ingest.tatoeba."""

import bz2
import json
from pathlib import Path

import pytest

from src.ingest import tatoeba


def _write_tatoeba_archive(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = (
        "1\ttlh\ttlhIngan Hol Dajatlh'a'?\n"
        "2\ttlh\tbortaS bIr jablu'DI' reH QaQqu' nay'.\n"
        "3\ttlh\t!!!\n"
    )
    path.write_bytes(bz2.compress(rows.encode("utf-8")))


def test_tokenize_klingon_latin_preserves_case_and_apostrophe():
    assert tatoeba.tokenize_klingon_latin("tlhIngan Hol Dajatlh'a'?") == [
        "tlhIngan",
        "Hol",
        "Dajatlh'a'",
    ]
    assert tatoeba.tokenize_klingon_latin("bIjatlh 'e' yImev!") == [
        "bIjatlh",
        "'e'",
        "yImev",
    ]


def test_extract_token_sequences_reads_tatoeba_bz2(tmp_path):
    archive_path = tmp_path / "tlh_sentences.tsv.bz2"
    _write_tatoeba_archive(archive_path)

    sequences, meta = tatoeba.extract_token_sequences(
        archive_path,
        expected_lang="tlh",
        tokenization_mode="klingon_latin_case_sensitive",
    )

    assert sequences[0] == ["tlhIngan", "Hol", "Dajatlh'a'"]
    assert sequences[1] == ["bortaS", "bIr", "jablu'DI'", "reH", "QaQqu'", "nay'"]
    assert meta["pre_cleaning_sentence_count"] == 3
    assert meta["skipped_empty_after_tokenization"] == 1


def test_ingest_archive_writes_tokens_and_manifest_from_cached_archive(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "corpora"
    archive_path = raw_dir / "tlh_sentences.tsv.bz2"
    _write_tatoeba_archive(archive_path)

    monkeypatch.setattr(tatoeba, "download_archive", lambda *args, **kwargs: archive_path)

    tokens_path = tatoeba.ingest_archive(
        corpus_id="klingon_tatoeba_tlh",
        language="klingon",
        family="constructed",
        role="typological_control_optional",
        source_url="https://example.test/tlh_sentences.tsv.bz2",
        iso_code="tlh",
        tokenization_mode="klingon_latin_case_sensitive",
        output_dir=output_dir,
        raw_dir=raw_dir,
    )

    manifest_path = output_dir / "klingon_tatoeba_tlh_manifest.json"
    assert tokens_path.exists()
    assert manifest_path.exists()

    corpus = json.loads(tokens_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert corpus["sequence_count"] == 2
    assert corpus["total_tokens"] == 9
    assert corpus["tokenization"]["mode"] == "klingon_latin_case_sensitive"
    assert "sequences" in corpus
    assert "sequences" not in manifest


def test_sentence_from_tatoeba_line_rejects_wrong_language():
    with pytest.raises(ValueError, match="Expected Tatoeba language"):
        tatoeba.sentence_from_tatoeba_line("1\teng\thello", "tlh")
