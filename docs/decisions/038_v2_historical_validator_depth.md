# Decision 038: v2 Historical Validator Deepening

Date: 2026-06-12

Status: frozen for H1 as part of the v2 preregistration freeze.

## Context

The v2 design depends on intermediate validation corpora more heavily than the
v1 French-to-Latin paper run. The question is not only whether endpoints move
toward Latin, but whether Romance-source trajectories pass through plausible
historical and family-relative regions.

The existing validator bank was useful but uneven:

```text
middle_french: 10 files, 6217 sequences, 31818 tokens
old_occitan: 1 file, 9899 sequences, 70826 tokens
```

Middle French was too Villon-centered. Old Occitan had decent token mass but
was dominated by one Crusade narrative witness.

## Decision

Deepen the v2 historical validators before H1 freeze by adding targeted,
attested Wikisource witnesses:

1. Add a late medieval dramatic Middle French witness:
   `La Farce de Maistre Pathelin`.
2. Add a Bernart de Ventadorn lyric packet to Old Occitan, preserving the
   existing Crusade narrative witness.

This is a validator-bank deepening, not an analysis result. These corpora may
be used for static calibration and checkpoint comparison only after the H1
feature/corpus freeze.

## Added Middle French Witness

```text
corpus_id: middle_french
file: data/raw/historical/middle_french/farce_maistre_pathelin_ancien_francais.txt
source: https://fr.wikisource.org/wiki/La_Farce_de_Ma%C3%AEtre_Pathelin,_traduction_Fournier,_1872/Version_en_ancien_fran%C3%A7ais
manifest title: La Farce de Maistre Pathelin
role: late medieval dramatic witness
```

Reason:

The Middle French packet needed a non-Villon witness and a different register.
Pathelin adds dramatic dialogue and late medieval urban/comic language without
mixing the row into early modern prose.

## Added Old Occitan Witnesses

```text
corpus_id: old_occitan
source author page: https://fr.wikisource.org/wiki/Auteur:Bernard_de_Ventadour
role: troubadour lyric packet
```

Fetched Bernart de Ventadorn witnesses:

```text
bernart_can_vei_la_lauzeta_mover.txt
bernart_non_es_meravelha.txt
bernart_tant_ai_mo_cor.txt
bernart_ab_joi_mou_lo_vers.txt
bernart_amics_bernartz.txt
bernart_amors_enquera_us_preyara.txt
bernart_anc_no_gardei.txt
bernart_era_m_cosselhatz.txt
bernart_ara_no_vei_luzir_solelh.txt
bernart_a_tantas_bonas_chansos.txt
```

Reason:

The old packet had one large narrative source, `La Chanson de la croisade
contre les albigeois`. The Bernart packet adds author/register diversity,
earlier lyric material, and a separate Old Occitan style family.

## Re-Ingested Stats

After re-ingesting `middle_french` and `old_occitan`:

```text
middle_french: 11 files, 8663 sequences, 42008 tokens, 7806 unique types
old_occitan: 11 files, 10573 sequences, 74700 tokens, 11113 unique types
```

Interpretation:

- Middle French gained substantial token mass and a second major witness type.
- Old Occitan gained modest token mass but much stronger source diversity.
- Old Occitan remains genre-skewed toward verse. This is acceptable for a
  historical validator if reported, but it should not be mistaken for balanced
  prose.

## Second Pass: Remaining OG Validators

The remaining original intermediate validators were audited after the Middle
French and Old Occitan pass. Three updates were safe to make immediately:

1. Repair `old_french` Roland ingestion.
2. Add a non-Marie witness to `anglo_norman`.
3. Add a verified Adam de la Halle Picard / Arras witness to `langue_d_oil`.

### Repaired Old French Roland Source

```text
corpus_id: old_french
file: data/raw/historical/old_french/chanson_de_roland.txt
source: https://fr.wikisource.org/wiki/La_Chanson_de_Roland/Joseph_B%C3%A9dier/La_Chanson_de_Roland/Texte
start marker: C ARLES li reis
role: high medieval epic witness
```

Reason:

The previous `La Chanson de Roland` source row pointed at the top-level
Wikisource work page. That scrape produced page furniture rather than the poem.
The row now points at the Bédier text page and starts at the first verse.

### Added Anglo-Norman Brandan Source

```text
corpus_id: anglo_norman
file: data/raw/historical/anglo_norman/voyage_saint_brandan.txt
source: https://fr.wikisource.org/wiki/Le_Voyage_de_saint_Brandan
start marker: Donna Aaliz la reïne
role: Benedeit narrative witness
```

Reason:

The Anglo-Norman packet already had several files, but they were all Marie de
France lai/prologue witnesses. `Le Voyage de saint Brandan` adds an independent
Anglo-Norman author/register without merging the row into continental d'oïl.

### Added Langue d'Oil Adam Source

```text
corpus_id: langue_d_oil
file: data/raw/historical/langue_d_oil/conges_adam_de_la_halle.txt
source: https://fr.wikisource.org/wiki/Cong%C3%A9s_%28Adam_de_la_Halle%29
start marker: Comment que men tans aie us
role: Picard / Arras Adam de la Halle witness
```

Reason:

The Adam de la Halle author page identifies him as a Picard-language trouvere
and lists several works. The direct `Jeu de Robin et Marion` and `Jeu de la
Feuillee` URLs do not return parse payloads through the Wikisource API used by
the historical fetcher. `Conges (Adam de la Halle)` does fetch cleanly and adds
a verified late-thirteenth-century Picard / Arras witness to `langue_d_oil`.

### Held After Audit

```text
old_spanish
candidate: Libro de Buen Amor
source page: https://es.wikisource.org/wiki/Libro_de_Buen_Amor
status: hold
reason: available text is dated 1330, outside the current 1140-1270 Old Spanish
        validator window. Adding it would be a date-policy change, not a quiet
        depth patch.
```

### Current OG Validator Snapshot

After re-ingesting `old_french`, `anglo_norman`, and `langue_d_oil`:

```text
old_french:   8 files, 17127 sequences, 106535 tokens, 13630 unique types
anglo_norman: 12 files, 7220 sequences, 37434 tokens, 5693 unique types
langue_d_oil: 5 files, 6165 sequences, 34764 tokens, 5942 unique types
old_spanish:  30 files, 4174 sequences, 29651 tokens, 6286 unique types
```

Interpretation:

- Old French is no longer hiding a broken Roland source.
- Anglo-Norman now has a second author/register family.
- Langue d'oïl now includes a verified Picard / Arras Adam de la Halle witness,
  though the two direct dramatic-work URLs tested remain non-ingestable.
- Old Spanish has many files but remains Berceo-heavy. It should not absorb
  fourteenth-century material unless the validator window is explicitly widened.

## Fetching Policy

The historical source fetcher now supports:

```text
--sleep-seconds
```

This exists because Wikisource may rate-limit rapid API fetches. Historical
fetching should use a pause between rows, especially when fetching many pages
from the same site.

## Updated Artifacts

```text
data/raw/historical/validator_sources.csv
data/raw/historical/validator_bank_manifest.csv
data/raw/historical/old_french/chanson_de_roland.txt
data/raw/historical/anglo_norman/voyage_saint_brandan.txt
data/raw/historical/langue_d_oil/conges_adam_de_la_halle.txt
data/raw/historical/middle_french/farce_maistre_pathelin_ancien_francais.txt
data/raw/historical/old_occitan/bernart_*.txt
data/processed/historical/old_french_tokens.json
data/processed/historical/old_french_manifest.json
data/processed/historical/anglo_norman_tokens.json
data/processed/historical/anglo_norman_manifest.json
data/processed/historical/langue_d_oil_tokens.json
data/processed/historical/langue_d_oil_manifest.json
data/processed/historical/middle_french_tokens.json
data/processed/historical/middle_french_manifest.json
data/processed/historical/old_occitan_tokens.json
data/processed/historical/old_occitan_manifest.json
data/matrices/old_french_*
data/matrices/anglo_norman_*
data/matrices/langue_d_oil_*
data/matrices/middle_french_*
data/matrices/old_occitan_*
data/v2/corpora_manifest.json
```

## Remaining Need

This is not the end of validator deepening. Before final v2 freeze, consider:

- one additional non-Villon Middle French prose witness if a clean source is
  available;
- one Old Occitan prose or administrative/legal witness if available;
- one additional langue d'oïl prose or dramatic witness if a stable premodern
  text page can be fetched cleanly;
- one additional pre-1270 Old Spanish non-Berceo witness if available, or an
  explicit policy decision to widen the Old Spanish date window;
- a per-validator register label in the static feature matrix.
