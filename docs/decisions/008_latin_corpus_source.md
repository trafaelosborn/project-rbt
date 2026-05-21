# Decision: Latin corpus sourced from Perseus canonical-latinLit GitHub repository

Date: 2026-04-07
Phase: P2

## What was decided

The Latin Island B corpus is sourced from the Perseus Digital Library's canonical-latinLit
repository on GitHub (https://github.com/PerseusDL/canonical-latinLit), which contains
428 Latin XML files in TEI P5 format. The repository is the canonical digital Latin
corpus and is used by Perseus, CLARIN, and many other digital humanities projects.

Access method: GitHub API (1 request for file tree listing) + raw.githubusercontent.com
(one request per file). Raw file fetches are not subject to GitHub API rate limits.

## Why

The brief specifies "Perseus Digital Library + PHI Latin Texts" as the Island B source.
The canonical-latinLit repository is the digital form of the Perseus corpus. It covers:

- Caesar: De Bello Gallico, De Bello Civili
- Cicero: speeches, philosophical works, letters
- Virgil: Aeneid, Eclogues, Georgics
- Livy: Ab Urbe Condita
- Sallust: Bellum Catilinae, Bellum Iugurthinum
- Tacitus: Annales, Germania, Historiae
- Ovid: Metamorphoses, Amores, Ars Amatoria
- Plus approximately 400 other works

This corpus provides register consistency with the Wikipedia Romance language corpora:
classical Latin prose is formal and formal-register Wikipedia text is the closest modern
equivalent. Poetry (Virgil, Ovid, Horace) is included; its shorter sequences provide
variance in sequence length distribution.

**Why not PHI separately:** The PHI Latin Texts (http://latin.packhum.org/) requires
browser-based access and does not provide a programmatic bulk download API. Perseus
canonical-latinLit overlaps significantly with PHI and is fully accessible
programmatically. For reproducibility, a single well-documented source is preferable.

**Why not a local Latin corpus:** The brief's methodology requires that Island B data
never be stored outside of the sequestered directory. Fetching from a remote source
during ingestion and writing directly to sequestered storage satisfies this requirement.

## Corpus register note

Classical Latin as preserved is almost entirely formal register. The corpus does not
include:
- Conversational Latin (Plautus, Terence — comedies with colloquial register)
- Vulgar Latin (unattested in writing)
- Medieval or Church Latin

This is intentional. The register of Wikipedia Romance language corpora (formal,
encyclopedic) is most comparable to classical formal Latin. Including Vulgar Latin
would require either attested fragments (too small) or reconstructed forms (circular).

The absence of Vulgar Latin from Island B is documented as a potential limitation:
if the reconstruction's bridge stages correspond more closely to Vulgar Latin than
to Classical Latin, the vs_latin_ground_truth score at validation may understate
the accuracy of the reconstruction. This is flagged in METHODOLOGY.md.

## Revision history

- 2026-04-07: Initial decision.
