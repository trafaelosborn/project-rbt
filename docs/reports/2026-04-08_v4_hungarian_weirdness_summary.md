# V4 Direction Summary: Hungarian-Guided Weirdness Scheduling

Date: 2026-04-08

## Purpose

Capture the proposed `v4` direction before implementation:

1. use a Hungarian assignment over mutable language families rather than blunt
   scalar ranking alone
2. derive the next mutation "weirdness" from that global mismatch
3. let proposal weirdness run wide while keeping adoption gated by coherence and
   score improvement

This is a design summary, not yet an implemented method.

## Why this exists

The project has already learned two important things:

1. simply increasing perturbation window size did not break the French `v2` plateau
2. stranger mutations plus louder Latin reward in `v3` *did* reopen movement above
   `FR_v2_061`

So the next likely bottleneck is not just "more iterations" or "bigger windows." The
search still lacks a strong global notion of *how misaligned the bridge is*, and it
still uses mostly fixed weirdness settings.

## Core idea

Instead of asking only:

- "Is this candidate better than the current baseline?"

`v4` would also ask:

- "How globally misaligned is the current bridge relative to Latin across mutable
  feature families?"

That mismatch would then control how weird the *next* mutation is allowed to be.

## Why Hungarian

The Hungarian algorithm is good when we have:

1. a set of candidate units
2. a set of target units
3. a cost matrix between them
4. a need for a globally consistent one-to-one assignment

That is useful here because current RBT scoring can give too much credit to a few
locally good features. Hungarian matching would force the system to explain *multiple*
families coherently instead of letting one Latin-like cluster stand in for everything.

Important caveat:

Whole languages are not naturally one-to-one, so Hungarian should **not** be applied
to raw vocabulary as the main objective. It makes much more sense over aggregated,
mutable families.

## Recommended matching units

The safest first-pass units are:

1. suffix families
2. function-word families
3. top character trigram bundles
4. prefix-linked paradigm groups
5. optionally, high-frequency local order templates

These are better than raw token matching because historical change is often:

- many-to-one
- one-to-many
- family-level rather than token-level

## Proposed control loop

### Step 1: build family inventories

For the current bridge state and the Latin reference, extract the top `N` mutable
families:

1. family label
2. frequency / mass
3. suffix profile
4. char-trigram profile
5. function-word flag or class
6. optional local-order signature

### Step 2: compute family-to-family cost matrix

For bridge family `i` and Latin family `j`, define a weighted cost:

`cost(i, j) =`

1. suffix mismatch
2. char-trigram mismatch
3. mass mismatch
4. optional function-word class mismatch
5. optional order-template mismatch

Lower is better.

### Step 3: run Hungarian assignment

Use Hungarian on the family cost matrix to obtain the best one-to-one alignment.

Outputs:

1. matched pairs
2. normalized total assignment cost
3. per-family residuals

### Step 4: convert mismatch into alignment score

Let:

- `c` = normalized Hungarian cost in `[0, 1]`
- `a = 1 - c` = alignment score in `[0, 1]`

Higher `a` means the bridge is globally more aligned with Latin family structure.

### Step 5: compute weirdness from inverse-log schedule

Instead of fixed weirdness, define:

`weirdness(a) = w_min + (w_max - w_min) * (1 - log(1 + beta * a) / log(1 + beta))`

Interpretation:

1. when alignment is poor (`a` low), weirdness stays high
2. as alignment improves, weirdness cools down gradually rather than collapsing
3. the schedule is aggressive early and conservative late

This is the "inverse logarithmic" part: better alignment lowers weirdness, but only
slowly.

## What weirdness should control

Weirdness should not just be a decorative scalar. It should drive real search knobs:

1. operator weights
   more mismatch -> more `macro_bundle_rewrite`, `function_word_burst`,
   `paradigm_family_rewrite`
2. mutation span
   more mismatch -> more families / more tokens touched
3. bundle depth
   more mismatch -> larger macro bundles
4. candidate exploration
   more mismatch -> more proposals sampled before choice
5. mutation-cost forgiveness
   more mismatch -> larger tolerated cost *if* coherence survives

## Very important safety rule

Do **not** fully uncap accepted weirdness.

Uncap **proposal generation**. Keep **adoption** gated.

That means:

1. let the generator propose very weird moves when mismatch is high
2. accept only candidates that:
   - remain `coherent`
   - improve the active objective
   - do not catastrophically increase assignment residuals in important families

This keeps the search adventurous without turning the chain into reward-hacked junk.

## How this would fit the current stack

### Current `v2`

- fixed operator weights
- local-to-meso mutations
- scalar total score

### Current `v3`

- stranger operators
- amplified Latin reward
- still mostly fixed weirdness settings

### Proposed `v4`

- family-level Hungarian mismatch diagnostic
- inverse-log weirdness scheduler
- coherence-gated adoption
- scalar score remains, but is no longer the only steering signal

So `v4` would extend the current design rather than replacing it.

## Why this may help

The project already knows that:

1. local and meso moves alone are not enough
2. louder Latin reward helps
3. the bridge can still improve above `FR_v2_061`

The remaining gap is a better global controller. Hungarian assignment gives the model
an interpretable notion of *where it is still wrong as a system*, not just whether a
single candidate happened to score slightly better.

## Phased implementation plan

### Phase 1: diagnostic only

Add family extraction plus Hungarian scoring, but do **not** change mutation behavior.

Goal:

- verify that the alignment measure is stable and interpretable

### Phase 2: schedule operator weights

Use alignment-derived weirdness only to shift operator probabilities.

Goal:

- test whether endogenous weirdness improves exploration without destabilizing the run

### Phase 3: schedule bundle size and penalty relief

Let weirdness also control:

1. macro-bundle size
2. burst width
3. mutation-cost forgiveness

Goal:

- see whether the system can open deeper basins safely

### Phase 4: evaluate alternatives

If Hungarian proves too rigid even at family level, consider:

1. soft assignment
2. entropic optimal transport
3. many-to-one family matching

## Proposed success criteria

`v4` would count as a real improvement if it does at least one of these:

1. improves raw Latin structural and form scores beyond `FR_v3_008`
2. preserves coherence while accepting larger family-level mutations
3. produces bridges that are more legible to attested intermediate validators
4. reveals which family mismatches remain stubborn late in the search

## Short version

The proposed `v4` idea is:

- measure global bridge-vs-Latin mismatch with Hungarian matching over mutable
  language families
- convert that mismatch into an inverse-log weirdness schedule
- let proposals get weird when mismatch is high
- keep acceptance conservative through coherence and score gates

That gives RBT something it does not yet have: an endogenous mutation temperature
driven by global alignment rather than fixed hyperparameters alone.
