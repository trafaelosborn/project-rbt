# Decision: 500 articles per language as default corpus size

Date: 2026-04-07
Phase: P2

## What was decided

The default target for Wikipedia ingestion is 500 articles per language. This is
configurable via `--articles` on the ingester command line.

## Why

**Statistical stability floor:** A co-occurrence matrix built from 500 Wikipedia
articles (approximately 100,000–250,000 tokens per language) will have seen the
majority of common grammatical function words enough times to produce stable positional
and co-occurrence statistics. Pilot testing with the Minos fingerprinting infrastructure
suggests that vocabulary sizes of ~20,000–50,000 types at this token count produce
non-degenerate matrices.

**Thin-corpus languages:** Occitan and Genoese Wikipedias have significantly fewer
articles than 500 quality articles on varied topics. The ingester will process as many
as are available and record the shortfall in the manifest. These languages are included
despite thin corpora because they represent important branches (Gallo-Romance for
Occitan; Ligurian for Genoese) that constrain the retrodiction geometry.

**Practical runtime:** 500 articles with batches of 20 requires 25 API requests per
language plus the random ID requests. With rate-limiting, this takes approximately
3–5 minutes per language (6 Romance languages + Portuguese = ~35 minutes total).
This is acceptable for a one-time ingestion.

**Register stability:** Above ~200 articles, the function word frequency distribution
stabilizes. 500 provides margin against the variance introduced by topic clustering
in random samples.

## Impact

Corpus sizes will vary between languages. The fingerprint scoring must account for
corpus size differences when comparing languages. The type/token ratio and frequency
normalization in the fingerprint components are designed to be size-robust.

## Revision history

- 2026-04-07: Initial decision.
