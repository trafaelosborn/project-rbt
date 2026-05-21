# Decision: Simple heuristic sentence splitting, no language-specific models

Date: 2026-04-07
Phase: P2

## What was decided

Sentences are split using a simple regex heuristic: terminal punctuation (`.`, `!`, `?`)
followed by whitespace and an uppercase letter, or a newline character. No
language-specific sentence boundary detection models (e.g. NLTK Punkt) are used.

## Why

The core methodology of RBT is that all languages are processed identically with no
linguistic assumptions. Using a language-specific sentence splitter (even a statistical
one trained on modern language data) would introduce implicit knowledge about the
language being processed. This contradicts the methodology.

The heuristic has known failure modes: abbreviations ("Dr. Smith"), decimal numbers
("3.14"), and some ellipses will cause incorrect splits. In a large corpus (500 Wikipedia
articles per language), these are a small fraction of total sentence boundaries. The
false-split sequences are short and contribute noise, not signal.

The heuristic also misses sentence boundaries that don't end in terminal punctuation
(particularly in headers, which may survive Wikipedia API text extraction). These are
accepted as long sequences that get split into shorter sub-sequences by the positional
statistics.

NLTK's Punkt tokenizer was evaluated and rejected because:
1. It requires language-specific trained models (separate models for each Romance language).
2. It encodes lexical knowledge of abbreviations and sentence starters.
3. It is not readily applicable to Latin, where Punkt has no pre-trained model.
4. Consistency across all languages (including Latin and Sumerian) requires a single
   uniform approach.

## Impact

Some sequences will be longer than a natural sentence (missed boundary) or shorter
(false boundary from abbreviation). This noise is documented here and accepted.

## Revision history

- 2026-04-07: Initial decision.
