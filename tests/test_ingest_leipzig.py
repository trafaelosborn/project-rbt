"""Tests for src.ingest.leipzig."""

import json
import tarfile
from pathlib import Path

import pytest

from src.ingest import leipzig


def _write_leipzig_archive(path: Path, corpus_name: str = "toy_news_2024_10K") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sentence_file = path.parent / f"{corpus_name}-sentences.txt"
    sentence_file.write_text(
        "1\tDas Haus ist alt.\n"
        "2\tLa maison est grande!\n"
        "3\t42 ???\n",
        encoding="utf-8",
    )
    try:
        with tarfile.open(path, "w:gz") as tar:
            tar.add(sentence_file, arcname=f"{corpus_name}/{sentence_file.name}")
    finally:
        sentence_file.unlink(missing_ok=True)


def test_sentence_from_leipzig_line_strips_numeric_id():
    assert leipzig.sentence_from_leipzig_line("12\tLa casa canta.\n") == "La casa canta."
    assert leipzig.sentence_from_leipzig_line("No tab here") == "No tab here"


def test_extract_token_sequences_reads_sentences_file(tmp_path):
    archive_path = tmp_path / "toy_news_2024_10K.tar.gz"
    _write_leipzig_archive(archive_path)

    sequences, meta = leipzig.extract_token_sequences(archive_path)

    assert sequences == [["das", "haus", "ist", "alt"], ["la", "maison", "est", "grande"]]
    assert meta["pre_cleaning_sentence_count"] == 3
    assert meta["skipped_empty_after_tokenization"] == 1
    assert meta["sentence_file"].endswith("-sentences.txt")


def test_ingest_archive_writes_tokens_and_manifest_from_cached_archive(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "corpora"
    archive_path = raw_dir / "toy_news_2024_10K.tar.gz"
    _write_leipzig_archive(archive_path)

    monkeypatch.setattr(leipzig, "download_archive", lambda *args, **kwargs: archive_path)

    tokens_path = leipzig.ingest_archive(
        corpus_id="toy_news_2024_10k",
        language="toy",
        family="test",
        role="typological_control",
        source_url="https://example.test/toy_news_2024_10K.tar.gz",
        iso_code="toy",
        source_type="news",
        source_year="2024",
        sentence_count_label="10K",
        output_dir=output_dir,
        raw_dir=raw_dir,
    )

    manifest_path = output_dir / "toy_news_2024_10k_manifest.json"
    assert tokens_path.exists()
    assert manifest_path.exists()

    corpus = json.loads(tokens_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert corpus["sequence_count"] == 2
    assert corpus["total_tokens"] == 8
    assert "sequences" in corpus
    assert "sequences" not in manifest
    assert manifest["archive_sha256"] == leipzig.sha256_file(archive_path)


def test_ingest_archive_refuses_to_overwrite_without_force(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "corpora"
    archive_path = raw_dir / "toy_news_2024_10K.tar.gz"
    _write_leipzig_archive(archive_path)
    output_dir.mkdir()
    (output_dir / "toy_tokens.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(leipzig, "download_archive", lambda *args, **kwargs: archive_path)

    with pytest.raises(FileExistsError):
        leipzig.ingest_archive(
            corpus_id="toy",
            language="toy",
            family="test",
            role="typological_control",
            source_url="https://example.test/toy_news_2024_10K.tar.gz",
            output_dir=output_dir,
            raw_dir=raw_dir,
        )
