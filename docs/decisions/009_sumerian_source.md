# Decision: Sumerian corpus — CDLI primary, ORACC DCCLT fallback

Date: 2026-04-07
Phase: P2

## What was decided

The Sumerian null model corpus uses:

**Primary source (per project brief):** CDLI bulk ATF download
- URL: https://cdli.mpsa.cnrs.fr/dl/bulk_data/ATF/cdli_atf_20150104.zip
- Format: ASCII Transliteration Format (ATF)
- Content: ~330,000 cuneiform texts; approximately 100,000+ are Sumerian
- Genre: Mixed — administrative, literary, lexical, royal inscriptions
- Period: Mostly Ur III (ca. 2100–2000 BCE) administrative tablets

**Fallback source (if CDLI unreachable):** ORACC DCCLT JSON
- URL: https://oracc.museum.upenn.edu/json/dcclt.zip
- Format: ORACC CDL JSON
- Content: ~10,200 texts total; ~3,257 catalogued as Sumerian
- Genre: Lexical texts only (vocabulary lists, sign lists)
- Period: Various

## Why CDLI is preferred

Administrative Ur III tablets from CDLI contain running Sumerian text that reflects
genuine grammatical structure: verbal chains, case markers, personal names, quantities
and commodities in consistent syntactic patterns. These texts have been used in
computational Sumerian linguistics research and are the standard reference corpus.

The ORACC DCCLT corpus contains only lexical texts — structured vocabulary lists
organized by semantic field ("wooden objects," "birds," "fish," etc.). These texts
have much shorter sequences (typically 3–8 tokens per entry) and the co-occurrence
structure reflects semantic proximity in a lexical system, not grammatical structure.

## Why ORACC DCCLT is the fallback

The CDLI server (cdli.mpsa.cnrs.fr) was inaccessible from the build environment
during Phase 2 ingestion. ORACC (oracc.museum.upenn.edu) was accessible and provided
a usable Sumerian corpus. ORACC is operated by the Penn Museum and is a well-maintained,
citable scholarly resource.

## Limitation of the DCCLT fallback

When ORACC DCCLT is used instead of CDLI:
- Sequences are shorter (mean ~5 tokens vs. ~15+ for CDLI administrative)
- Positional statistics are less grammatically rich
- Co-occurrence statistics reflect lexical organization, not syntactic structure

The null model is still valid under these conditions: Sumerian lexical structure is
genuinely non-IE and statistically real. The ceiling effect (reconstruction must
score closer to Latin than to Sumerian) is preserved. The gap between the ceiling
and the reconstruction may be larger than it would be with CDLI administrative
texts, which would make the null model a weaker (more easily cleared) ceiling.

This limitation is noted in the manifest when ORACC is used:
`"limitation": "DCCLT contains lexical texts..."`.

## Action required

Before final validation (Phase 5), CDLI ingestion should be re-attempted from a
network environment with CDLI access and the Sumerian corpus should be updated to
use CDLI administrative texts. The METHODOLOGY.md section on the null model should
be updated accordingly.

## Revision history

- 2026-04-07: Initial decision. CDLI unreachable; ORACC DCCLT used as fallback.
