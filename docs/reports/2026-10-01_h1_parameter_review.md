# H1 parameter review, 2026-10-01

Later implementation: the author approved these priorities. See
[Decision 047](../decisions/047_h1_continuous_search_and_form_reward_ablation.md)
and the [extension audit](2026-10-01_h1_extension_audit.md) for the implemented
continuous controller, diagnostic ledger, resource pauses, and form-reward
ablation. The text below preserves the recommendation-stage record. Independent
corpus/control expansion remains future work.

Status: recommendations following the author's question about additional
parameters. These names describe proposed interfaces, not implemented options.
No execution code, registered protocol, frozen input, or production setting was
changed by this review. Production remains unlaunched; the fixed-panel versus
expanded-design scope question remains open.

## Recommendation

Prioritize reproducible randomness and informative measurements. Add experimental
arms only when each answers a specified question. Preserve the current candidate
protocol and evidence; adopt any execution or analysis changes through a new,
versioned amendment before its production runs.

| Priority | Proposed setting | Purpose and recommended policy |
|---|---|---|
| First | `corpus_seed`, `search_seed`, `rng_policy` | Separate selection of the initial 20,000 tokens from the mutation search. Record both seeds and the generator/version. Prefer checkpointed continuation for a newly amended protocol. |
| First | `diagnostics_policy` | Record diversity and individual score components at the initial state and every accepted state, with cumulative proposal position and aggregate rejection counts. Use diagnostics to explain behavior, without changing acceptance. |
| First | `proposal_checkpoints`, resource limits | Permit a prespecified secondary comparison at equal proposal budgets, and safe operational pauses. Keep budget exhaustion distinct from natural termination. Choose checkpoint values from feasibility evidence before production. |
| First experimental extension | `form_reward_mode = full / disabled` | Compare the existing composite objective with an arm that disables all direct form-linked reward terms, while holding the proposal mechanism fixed. This isolates the contribution of those reward terms. |
| Expanded design | `source_control_mode`, `control_seed`, `corpus_replica_id` | Add source-matched controls or independent corpus samples with explicit generation/sampling recipes. Each condition must identify what information it preserves and removes. |

## Code findings and implementation requirements

### Randomness and resume

`engine_reinforced_v2.py::_sample_initial_corpus` uses the configured seed for
H1 corpus sampling. The long-run builders pass that same configured seed to each
new block engine. The engine initializes a fresh generator in `run`; it does
not currently continue a saved generator state across blocks. This was already
disclosed in Decision 044. It is a defined historical execution policy, not
evidence that the biological hypothesis has failed.

Separating seeds allows us to hold the sampled corpus fixed while measuring
search variability, or hold a search seed fixed while varying corpus samples.
The present three diagonal seed combinations do not separately estimate both
sources of variation. Do not silently relabel them as independent replicates.

Persisting RNG state alone is insufficient for exact continuation: persist every
search state that affects the next proposal or its acceptance, together with the
corpus, counters, and effective configuration. Verify uninterrupted versus
interrupted/resumed execution produces identical proposals and accepted states.
Changing block sizes must not silently change the stopping horizon: the current
plateau window is expressed in blocks. Any claimed block-boundary invariance
needs an explicit proposal-based stopping rule and a corresponding test.

### Collapse measurements

The engine and readout already retain entropy/features and endpoint diagnostics.
Expose a consistent trajectory report of word/bigram entropy, vocabulary size,
maximum token frequency, and all six frozen features, together with form,
coherence, cost, and reward contributions. Record accepted, rejected, no-op, and
token-budget-invalid candidates by operator with clearly defined denominators.
Some underlying values are already logged; avoid redundant recomputation.

Recommend continuous measurements first. Any alert threshold must be specified
prospectively with a rationale. An alert must not automatically remove a run,
change the reward, or make an otherwise valid result inadmissible. A Klingon
nearest-neighbor result is an outcome to explain, not an exclusion criterion.

### Computation budget and checkpoints

The existing protocol measures normalized structural-score progress and requires
natural termination. That addresses a different question from improvement after
the same number of proposals. Add the latter only as an explicitly specified
secondary readout, with its own treatment of stalled trajectories. Keep the
candidate count fixed and also report actual candidate evaluations and invalid
proposals, since proposal counts alone need not represent equal computational
work. A finite or interrupted trajectory must not enter the existing primary
readout as if naturally complete.

The 32-proposal benchmarks measured approximately 6.13 and 7.51 seconds per
proposal for French and Finnish at 16 candidates, including startup/audit costs.
The longer French benchmark timed out at its 30-minute cap. These observations
support durable checkpoints and resource controls; they do not establish a
scientifically sufficient budget or forecast full-run termination. Operational
limits should save state and pause; they must not declare convergence or reject
the linguistic hypothesis.

### A precise ablation

Setting `form_weight=0` alone is inadequate for a form-reward ablation. The current
v3 reward path, inherited by the later engines, also adds form, suffix, and
trigram gains, a joint structural/form bonus, and form-conditioned mutation-cost
relief. The disabled arm must remove all of those terms. Keep coherence, mutation
cost, structural reward, and the proposal mechanism fixed and documented.

Latin-informed character/suffix proposals and local candidate selection would
remain in this comparison. Therefore call it a *form-reward ablation*, not a
Latin-free or purely structural search. Removing target-informed proposal
guidance is a separate intervention requiring its own matched mutation policy.
Also audit form/alignment-dependent stopping conditions: choose common stopping
criteria for the comparison, or make the ablation a separately registered
equal-budget experiment. Do not interpret differing stop rules as a pure reward
effect. Changing objective terms requires coordinated scorer, audit, provenance,
and readout changes; an inert configuration label is insufficient.

### Controls and corpus replication

A possible inexpensive-to-construct control is to shuffle token order within
each sequence of the exact sampled source corpus, using a separate recorded
control seed. This preserves the token multiset, token forms, and sequence
lengths, while perturbing order and bigram statistics. It does not erase every
linguistic feature, guarantee a changed corpus, or supply a general null of
unrelated ancestry. Verify the claimed invariants and record the parent corpus
hash, generated corpus hash, algorithm version, and seed. Existing Markov and
Klingon sources answer different stress-test questions.

For wider claims, independent documents/registers and a defensible language
sampling design are more useful than treating additional optimizer seeds as
new languages. Record document overlap and sampling rules. Resamples of one
corpus remain dependent. An end-to-end synthetic positive control with known
transformation history could test recovery under that generator; the existing
3,000 synthetic pair-statistic simulations validate analysis behavior, not
recovery from generated linguistic corpora.

## Avoid outcome-directed tuning

Do not add an entropy reward, increase Latin attraction, change feature weights,
or penalize Klingon proximity merely to remove the observed anomalies. Such
interventions could be separate, explicitly motivated algorithm experiments;
they do not repair the interpretation of earlier runs. Keep the 20,000-token
budget, feature definition, candidate count, and main weights fixed for the
current candidate comparison unless a new amendment explains a change.

The current `1e-12` direction tolerance is numerical, not a minimum scientifically
meaningful effect. A substantive effect threshold would need an independent
rationale or calibration; do not invent one to obtain a favorable classification.

## Evidence and documentation

Reviewed `long_run_v5.py`, `long_run_v4.py`, the v2-v5 engines, Decision 046,
`h1_e1_panel_v2.json`, and the design repair audit. This is a documentation-only
review; no execution tests were warranted or run, and no new result is claimed.
The earlier source archive and completion inventory remain historical evidence
and do not include this later review.

Methodological context: Morris, White, and Crowther (2019),
[Using simulation studies to evaluate statistical methods](https://doi.org/10.1002/sim.8086),
recommends specifying aims, data-generating mechanisms, estimands, methods, and
performance measures when planning simulation studies. The parameter choices
above are recommendations from this repository review, not prescriptions from
that paper.

Related records: [design amendment](../decisions/046_h1_e1_dependence_and_claim_scope.md),
[design repair audit](2026-10-01_h1_design_repair_audit.md), and
[execution repair](../decisions/044_h1_objective_execution_repair.md).
