# French V4 Post-Plateau 50% Probe

Date: 2026-04-08

## Purpose

Test whether the apparent `v4` plateau at `FR_v4_006` was a real exhaustion of the
current search basin or just a consequence of the original stopping budget.

The probe continues from the previous `v4` endpoint with the same engine and the
same objective, but grants an extra proposal budget equal to roughly 50% of the
original run length:

- original `v4` proposals attempted: `51`
- extra probe proposals: `26`

## Setup

Seed corpus:

- `data/retrodiction/french/v4_from_v3_endpoint/corpora/FR_v4_006_tokens.json`

Output:

- `data/retrodiction/french/v4_post_plateau_50pct_probe/`

Configuration differences from the original `v4` run:

1. source corpus = prior endpoint `FR_v4_006`
2. `max_proposals = 26`
3. `patience = 26`

Everything else remained on the same `v4` rule stack.

## Result

Run summary:

- total stages: `5`
- accepted mutation stages: `4`
- proposals attempted: `26`
- halt reason: `max_proposals`
- best / final stage: `FR_v4_004`
- final coherence: `coherent`

Accepted operator counts:

- `token_char_edit = 1`
- `paradigm_family_rewrite = 2`
- `split_token = 1`

## Raw-axis comparison to the prior endpoint

Because `v4` total score includes run-local gain bonuses, the clean cross-run
comparison is on the raw Latin axes rather than on `total_score`.

From old endpoint `FR_v4_006` to new endpoint `FR_v4_004`:

- Latin structural score: `-1.295360 -> -1.288622`
- Latin form score: `0.806686 -> 0.808937`
- family alignment score: `0.536294 -> 0.520696`
- coherence label: `coherent -> coherent`

So the continuation found additional Latin-directed improvement, but it paid for it
with lower Hungarian family alignment.

## Interpretation

This probe falsifies the strongest form of the earlier plateau claim.

The previous endpoint did not remain frozen when given a modest extra post-plateau
budget. Under the same `v4` rules, the engine found four more accepted moves and
did not halt as `stable`; it halted only because the added proposal budget ran out.

That means:

1. the prior `v4` plateau was not absolute
2. the current search basin still contains more Latin-directed movement
3. the extra movement is not free, because family alignment drops as the raw Latin
   axes improve

The cleanest read is that the engine can still climb, but the additional climb is
currently biased more toward direct Latin reward than toward the independent family
alignment diagnostic.

## Practical takeaway

The main lesson is not "run forever." It is:

1. post-plateau continuation is a real axis of experimentation
2. patience / budget choices matter more than the earlier `stable` halt suggested
3. any future continuation should be read on raw Latin axes plus validator signals,
   not only on the run-local `total_score`

## Next move

Two reasonable next moves now exist:

1. extend the continuation again and see where the new basin actually stabilizes
2. compare the new endpoint against Old French before pushing farther, so we can see
   whether the extra Latin movement is still historically legible
