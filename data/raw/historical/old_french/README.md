# Old French Validator Packet

Date assembled: 2026-04-08

This starter validator packet is a practical first-pass Old French corpus for the
historical comparison workflow.

Contents:

1. `sequence_sainte_eulalie.txt`
2. `serments_de_strasbourg.txt`
3. `la_vie_de_saint_alexis_full.txt`

Sources:

- Sequence de sainte Eulalie
  `https://fr.wikisource.org/wiki/S%C3%A9quence_de_sainte_Eulalie`
- Serments de Strasbourg
  `https://fr.wikisource.org/wiki/Serments_de_Strasbourg_%28Nithard%29`
- La Vie de saint Alexis
  `https://fr.wikisource.org/wiki/La_Vie_de_saint_Alexis`

Notes:

- This packet is intentionally small and local-first so the validator workflow can
  run now.
- The Saint Alexis file is a locally extracted full-text render from Wikisource.
- Orthography follows the visible transcription on the source pages, with minor
  cleanup to remove bracket markup that would tokenize poorly.
