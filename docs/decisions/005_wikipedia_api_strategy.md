# Decision: Wikipedia random article sampling via MediaWiki API

Date: 2026-04-07
Phase: P2

## What was decided

Modern Romance language corpora are sourced via the MediaWiki API using random article
sampling (`action=query&list=random&rnnamespace=0`), then fetching plain-text extracts
(`action=query&prop=extracts&explaintext=true`). Articles are processed in batches of
20. Raw article text is tokenized on receipt and discarded.

## Why

**Why Wikipedia:** Register consistency is the primary reason. Classical Latin as
preserved is almost entirely formal register — Cicero, Livy, Virgil. Wikipedia prose
in all Romance languages is comparable: formal, encyclopedic, non-conversational.
If the input corpora were drawn from social media, news, or literary fiction, the
register mismatch with the Latin ground truth would be a significant confound.

**Why random sampling:** Topic diversity reduces the risk of fingerprints that reflect
domain-specific vocabulary rather than grammatical structure. A corpus sampled only
from articles about medieval history would have different function word patterns than
one sampled randomly across all topics.

**Why not download full dumps:** Full Wikipedia XML dumps are multi-gigabyte files.
The brief specifies "no local storage of raw corpora." Streaming via API satisfies
this requirement while remaining fully reproducible by API query date.

**Why 20 articles per batch:** The MediaWiki API's `list=random` endpoint returns up
to 500 results per request, but `prop=extracts` is most reliable with smaller batches.
Batch size of 20 balances throughput against per-request reliability.

## Acknowledged limitations

Random sampling does not guarantee balanced topic or genre coverage. Wikipedia article
distributions are skewed toward certain domains (geography, history, biographies).
This is accepted given the register-consistency rationale above.

The Occitan (oc) and Genoese (lij) Wikipedias are thin (<50,000 articles combined).
Actual article counts for these languages may fall short of the 500-article target.
The manifest records actual counts; downstream weighting should account for corpus size.

## Revision history

- 2026-04-07: Initial decision.
