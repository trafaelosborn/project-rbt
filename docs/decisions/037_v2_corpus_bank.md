# Decision 037: v2 Corpus Bank, Controls, and Tokenization Exceptions

Date: 2026-06-12

Status: frozen for H1 as part of the v2 preregistration freeze.

## Context

The v2 brief requires a corpus bank broad enough to support static
collinearity calibration, Romance-family geometry, typological controls, donor
comparators, null controls, and optional constructed-language stress tests.

The previous v1/v5 corpus set was enough for French-to-Latin production runs,
but not enough for the v2 H1 preregistration gate. In particular, the v2 draft
requires:

- Latin target.
- Modern Romance sources.
- A withheld Romance relative.
- Medieval Romance validators.
- Typological controls.
- Donor comparators for contact diagnostics.
- Markov noise.
- Klingon if cleanly available.

## Decision

The v2 corpus bank is tracked by:

```text
data/v2/corpora_manifest.json
```

Each corpus row has one of three artifact levels:

1. A lightweight manifest JSON tracked in git.
2. A bulky token JSON ignored by git but reproducible from the ingester.
3. A raw downloaded archive ignored by git but checksummed in the manifest.

The primary ingesters are:

```text
python -m src.ingest.leipzig
python -m src.ingest.tatoeba
python -m src.ingest.v2_corpus_manifest
```

Regeneration commands:

```text
python -m src.ingest.leipzig ingest-all
python -m src.ingest.tatoeba ingest --corpus-id klingon_tatoeba_tlh
python -m src.ingest.v2_corpus_manifest
```

## Leipzig Controls

Leipzig/Wortschatz corpora are used for most modern controls because the
download portal provides comparable plain-text corpora split into sentence
rows, with fixed size buckets.

Default Leipzig rows:

```text
turkish_news_2024_100k
finnish_news_2022_100k
vietnamese_news_2022_100k
catalan_news_2022_100k
arabic_news_2022_100k
bulgarian_news_2022_100k
german_news_2025_100k
```

The Leipzig tokenization policy is the normal Project RBT Unicode word-token
policy:

```text
Leipzig sentence row -> Unicode word tokens -> lowercase -> drop tokens shorter than 2
```

Known caveats:

- The Arabic row is used as the Modern Standard Arabic proxy, but Leipzig labels
  the row simply as Arabic. This must be noted when reporting contact tests.
- German is an imperfect Germanic donor proxy for French contact diagnostics.
- Bulgarian is the default Slavic donor proxy. Serbian remains an alternate.
- Catalan is the default withheld Romance relative because Leipzig has a 100K
  Catalan corpus. Sardinian remains an alternate but Leipzig currently only
  provided 10K Wikipedia archives in the local audit.

## Klingon

Klingon is included as an optional constructed-language typological control:

```text
klingon_tatoeba_tlh
```

Source:

```text
https://downloads.tatoeba.org/exports/per_language/tlh/tlh_sentences.tsv.bz2
```

Reason for Tatoeba:

- Tatoeba provides a per-language Klingon export.
- The export is sentence-level TSV with `Sentence id`, `Lang`, and `Text`.
- Tatoeba documents the export files as CC BY 2.0 FR.
- The local ingested corpus is large enough for the v2 static budgets below
  50K tokens.

Klingon tokenization is intentionally not the normal Project RBT tokenizer.
Standard Latin-script Klingon uses uppercase letters and apostrophe as
meaningful orthographic/phonemic signal. Lowercasing or splitting apostrophes
would destroy signal before the control test begins.

The Klingon tokenizer therefore uses:

```text
case-sensitive Latin-script tokens, preserving apostrophe inside tokens
```

Current local ingest stats:

```text
sequence_count: 42262
total_tokens: 141318
unique_types: 25804
mean_seq_length: 3.343854999763381
archive_sha256: fe2f6d4cd7d5d799b66f48854223fb21ad91b645a8dcef61f8e1666edd39cda6
```

Interpretation caveat:

Klingon is a constructed-language stress test, not a natural-language
genealogical comparator. It should not be used as evidence about language
families. Its job is to test whether the feature space behaves sensibly when a
deliberately designed, non-Romance, non-natural row is included.

## Auxiliary Modern Romance Positive Diagnostic

Modern Occitan is retained as a non-required auxiliary positive-control
diagnostic:

```text
occitan
```

Current local ingest stats:

```text
sequence_count: 10857
total_tokens: 105768
unique_types: 20530
```

Reason:

Occitan is a modern Romance corpus already available locally at sufficient size
for the 20K static/reporting budget. It gives H1 an additional Romance
near-neighbor sanity check without reopening the required corpus bank.

Policy:

```text
required_for_h1: false
role: auxiliary_modern_romance_positive_diagnostic
```

It is visible in `data/v2/corpora_manifest.json` and may be reported as an
auxiliary positive diagnostic, but it is not required for freeze and is not used
to select the static in-loop tier.

## Japanese and Mandarin Deferred Rows

Japanese remains in the manifest as a post-H1 optional typological control, but
it is no longer required for the H1 freeze:

```text
japanese_news_2023_100k: needs_tokenization_policy
```

Reason:

The current Unicode word tokenizer is not an acceptable segmentation policy for
Japanese. It can extract long script runs from Leipzig sentence rows, but that
is not equivalent to word tokenization and would make word-level features
misleading.

Mandarin is retained only as an alternate for the Mandarin-or-Vietnamese row and
is also blocked by the same segmentation issue. Vietnamese is the default row
for that control slot because whitespace-delimited word-like tokenization is
usable under the current tokenizer.

Decision:

```text
Japanese and Mandarin remain visible as non-H1 rows with
needs_tokenization_policy status. They must not block H1 unless the corpus
requirements are amended again before freeze.
```

## Historical and Null Rows

The existing historical validator bank remains part of the v2 manifest:

```text
old_french
middle_french
langue_d_oil
anglo_norman
old_occitan
old_spanish
```

Decision 038 records the first v2 deepening pass for Middle French and Old
Occitan validators.

Sumerian is retained as a structured non-Indo-European ceiling, but its manifest
already records the source limitation: the current fallback is lexical text, not
running prose.

Markov noise remains the generated null control.

## Reporting Rules

Every corpus-related report should include:

- Corpus id.
- Source provider and URL.
- Access or ingest date.
- License note.
- Archive SHA-256 if downloaded.
- Tokenization mode.
- Pre-cleaning sentence and whitespace-token counts where available.
- Final sequence, token, type, TTR, and mean sequence length.
- Known limitation notes.

Any row with source, tokenization, or status ambiguity must remain visible in
`data/v2/corpora_manifest.json`; it must not be silently omitted.

## Consequences

This decision makes the corpus bank more reproducible but also stricter:

- H1 may proceed without Japanese because the required control list now uses
  Finnish, Turkish, Vietnamese, Sumerian, Klingon, and donor comparators for the
  non-Romance baseline spread.
- Modern Occitan is available as an auxiliary positive diagnostic without
  changing the required H1 corpus set or static in-loop tier.
- Klingon is available and documented, but remains optional.
- Corpus artifacts can be regenerated, while bulky token and raw files stay out
  of git.
