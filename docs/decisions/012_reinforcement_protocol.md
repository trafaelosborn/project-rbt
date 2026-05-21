# Decision 012: Latin-Conditioned Reinforcement Protocol

Date: 2026-04-07
Phase: P3R

## What was decided

Phase 3 is reframed as a target-conditioned bridge-generation experiment rather than a
fully blind retrodiction exercise.

Latin is placed explicitly inside the optimization loop as the reinforcer. Modern
Romance corpora are the starting state. The destination is not the scientific result;
the bridge trajectory is.

Two algorithms are retained:

1. Stochastic search: random perturbations of the current bigram model are generated,
   scored against Latin, and the best candidate is kept.
2. Directed gradient: the current bigram transition matrix is mixed one step toward
   the Latin transition matrix.

Both algorithms emit full bridge-stage records and synthetic corpora at every stage.

## Why this change was made

The interesting claim is not that French can be pushed into Latin. Any sufficiently
plastic learner can be pushed toward a target with enough reinforcement.

The interesting claim is that the induced bridge may reveal something about the
geometry of historical change:

- whether the path passes near attested intermediate languages
- whether multiple algorithms converge on the same bridge region
- whether coherent but unattested "ghost" bridge languages exist
- whether the whole procedure collapses into noise

Those are all meaningful outcomes. The path is the experiment.

## What counts as a valid result

Three classes of outcomes are treated as methodologically meaningful:

1. Attested match  
   A generated bridge aligns statistically with attested historical intermediates.

2. Coherent alternate bridge  
   The generated bridge is language-like and internally coherent but does not align
   with attested intermediates. This implies additional degrees of freedom in the
   historical path.

3. Incoherent collapse  
   The generated bridge behaves like noise or reward-hacked junk. This falsifies
   either the implementation or the ontology.

## What remains sequestered

Portuguese remains outside the optimization loop as a withheld positive control.
Latin unlocks are allowed only inside the reinforced engines and must be logged with
a substantive reason string. Latin text is not copied into generated bridge corpora;
the source vocabulary remains fixed.

## Immediate scope

The primary initial reinforced run is French -> Latin using both algorithms:

- `stochastic`
- `gradient`

French is the first case study because it is the clearest motivating example in the
project framing and provides a tractable reference run before scaling to additional
Romance languages.
