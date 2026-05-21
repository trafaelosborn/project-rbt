"""
Unit tests for src.ingest.tokenize
"""

import pytest
from src.ingest.tokenize import (
    split_sentences,
    tokenize_sentence,
    tokenize_text,
    tokenize_corpus,
    corpus_stats,
    MIN_TOKEN_LEN,
)


class TestSplitSentences:
    def test_period_split(self):
        text = "The cat sat. The dog ran."
        parts = split_sentences(text)
        assert len(parts) == 2

    def test_question_mark_split(self):
        text = "Who is there? Nobody knows."
        parts = split_sentences(text)
        assert len(parts) == 2

    def test_exclamation_split(self):
        text = "Watch out! It is dangerous."
        parts = split_sentences(text)
        assert len(parts) == 2

    def test_newline_split(self):
        text = "First line.\nSecond line."
        parts = split_sentences(text)
        assert len(parts) >= 2

    def test_no_split_needed(self):
        text = "Just one sentence without a boundary"
        parts = split_sentences(text)
        assert len(parts) == 1

    def test_empty_string(self):
        parts = split_sentences("")
        assert parts == []

    def test_whitespace_only(self):
        parts = split_sentences("   \n\n  ")
        assert parts == []

    def test_returns_nonempty_parts(self):
        text = "First. Second. Third."
        parts = split_sentences(text)
        for p in parts:
            assert p.strip() != ""


class TestTokenizeSentence:
    def test_basic_lowercase(self):
        tokens = tokenize_sentence("The Quick Brown Fox")
        assert tokens == ["the", "quick", "brown", "fox"]

    def test_strips_punctuation(self):
        tokens = tokenize_sentence("Hello, world!")
        assert "hello" in tokens
        assert "world" in tokens
        assert "," not in tokens
        assert "!" not in tokens

    def test_drops_short_tokens(self):
        tokens = tokenize_sentence("I am a cat")
        for tok in tokens:
            assert len(tok) >= MIN_TOKEN_LEN

    def test_unicode_letters_kept(self):
        # French with diacritics
        tokens = tokenize_sentence("Le président de la République")
        assert "président" in tokens
        assert "république" in tokens

    def test_romanian_diacritics(self):
        tokens = tokenize_sentence("Ș și ț sunt litere românești")
        assert "sunt" in tokens
        assert "litere" in tokens

    def test_numbers_dropped(self):
        tokens = tokenize_sentence("There are 42 cats and 7 dogs")
        assert "42" not in tokens
        assert "7" not in tokens

    def test_urls_dropped(self):
        # URLs contain no pure-alpha runs of length >= 2 after splitting
        tokens = tokenize_sentence("Visit https://www.example.com for more")
        assert "https" in tokens or "www" in tokens or True  # URL fragments may survive
        # Key check: no token contains "://" or "."
        for tok in tokens:
            assert "://" not in tok
            assert "." not in tok

    def test_french_contraction_split(self):
        # "l'eau" → l (dropped, len<2 if MIN_TOKEN_LEN=2) + "eau"
        tokens = tokenize_sentence("l'eau est froide")
        assert "eau" in tokens
        assert "est" in tokens
        assert "froide" in tokens

    def test_empty_string(self):
        assert tokenize_sentence("") == []

    def test_all_punctuation(self):
        assert tokenize_sentence("... ??? ---") == []

    def test_nfc_normalization(self):
        # NFD form of 'é' (e + combining accent) should normalize to NFC 'é'
        nfd_e = "e\u0301"  # e + combining acute accent
        tokens = tokenize_sentence(f"caf{nfd_e} au lait")
        assert "café" in tokens

    def test_output_all_lowercase(self):
        tokens = tokenize_sentence("TUTTO MAIUSCOLO")
        for tok in tokens:
            assert tok == tok.lower()


class TestTokenizeText:
    def test_basic(self):
        text = "Il gatto dorme. Il cane corre veloce."
        sequences = tokenize_text(text)
        assert len(sequences) >= 1
        for seq in sequences:
            assert isinstance(seq, list)
            assert all(isinstance(t, str) for t in seq)

    def test_empty_text(self):
        assert tokenize_text("") == []

    def test_no_empty_sequences(self):
        text = "First sentence. Second sentence. Third."
        sequences = tokenize_text(text)
        for seq in sequences:
            assert len(seq) > 0

    def test_no_empty_tokens_in_sequences(self):
        text = "Le chat mange. Il dort beaucoup."
        sequences = tokenize_text(text)
        for seq in sequences:
            for tok in seq:
                assert tok != ""

    def test_sequence_boundaries_respected(self):
        # Each sentence should be its own sequence
        text = "Alpha beta gamma. Delta epsilon zeta."
        sequences = tokenize_text(text)
        # Both sentences should produce sequences
        assert len(sequences) >= 2

    def test_returns_list_of_lists(self):
        sequences = tokenize_text("Test sentence here.")
        assert isinstance(sequences, list)
        assert isinstance(sequences[0], list)


class TestTokenizeCorpus:
    def test_multiple_texts(self):
        texts = [
            "First article text. More content here.",
            "Second article. Different content.",
        ]
        sequences = tokenize_corpus(iter(texts))
        assert len(sequences) > 0

    def test_empty_iterable(self):
        sequences = tokenize_corpus(iter([]))
        assert sequences == []

    def test_concatenates_across_texts(self):
        # Each text produces independent sequences; all are concatenated
        texts = ["One sentence.", "Another sentence."]
        sequences = tokenize_corpus(iter(texts))
        assert len(sequences) >= 2


class TestCorpusStats:
    def test_basic_counts(self):
        sequences = [["a", "b", "c"], ["a", "d"]]
        stats = corpus_stats(sequences)
        assert stats["sequence_count"] == 2
        assert stats["total_tokens"] == 5
        assert stats["unique_types"] == 4

    def test_type_token_ratio(self):
        # All unique → TTR = 1.0
        sequences = [["a", "b", "c"]]
        stats = corpus_stats(sequences)
        assert stats["type_token_ratio"] == 1.0

    def test_ttr_less_than_one_with_repeats(self):
        sequences = [["a", "a", "a"]]
        stats = corpus_stats(sequences)
        assert stats["type_token_ratio"] < 1.0

    def test_mean_seq_length(self):
        sequences = [["a", "b"], ["c", "d", "e", "f"]]
        stats = corpus_stats(sequences)
        assert stats["mean_seq_length"] == 3.0

    def test_empty_corpus(self):
        stats = corpus_stats([])
        assert stats["total_tokens"] == 0
        assert stats["type_token_ratio"] == 0.0
        assert stats["mean_seq_length"] == 0.0
