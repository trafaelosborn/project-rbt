#!/usr/bin/env python3
"""
Project RBT — Pipeline Runner
==============================
Runs all ingestion and fingerprinting steps in order.
Steps that have already produced output are skipped unless --force is used.

Usage:
    python pipeline.py                  # run all missing steps
    python pipeline.py --force          # re-run everything
    python pipeline.py --step ingest    # run only ingestion steps
    python pipeline.py --step fingerprint  # run only fingerprinting steps
    python pipeline.py --status         # print what is and isn't done

Network requirements:
    - Wikipedia API (all Romance languages + Portuguese)
    - GitHub API + raw.githubusercontent.com (Latin)
    - oracc.museum.upenn.edu (Sumerian fallback)

    If the ORACC download fails with an SSL error (common on institutional
    networks), download the file manually and set the environment variable:
        ORACC_LOCAL_ZIP=/path/to/oracc_dcclt.zip

Estimated runtime on first run:
    Romance ingestion:   ~60-90 min total (6 languages x ~500 articles)
    Portuguese:          ~15 min
    Latin:               ~10 min (428 files via GitHub)
    Sumerian:            ~5 min download + parse
    Markov:              <1 min
    Fingerprints:        ~2 min
"""

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA = PROJECT_ROOT / "data"
ROMANCE_DIR = DATA / "processed" / "romance"
SEQUESTERED_DIR = DATA / "sequestered"
NULLS_DIR = DATA / "processed" / "nulls"
MATRICES_DIR = DATA / "matrices"

ROMANCE_LANGUAGES = ["italian", "french", "spanish", "romanian", "occitan", "genoese"]


# ---------------------------------------------------------------------------
# Output existence checks
# ---------------------------------------------------------------------------

def _romance_done(lang: str) -> bool:
    return (ROMANCE_DIR / f"{lang}_tokens.json").exists()

def _portuguese_done() -> bool:
    return (SEQUESTERED_DIR / "portuguese" / "portuguese_tokens.json").exists()

def _latin_done() -> bool:
    return (SEQUESTERED_DIR / "latin" / "latin_tokens.json").exists()

def _sumerian_done() -> bool:
    return (NULLS_DIR / "sumerian" / "sumerian_tokens.json").exists()

def _markov_done() -> bool:
    return (NULLS_DIR / "markov" / "markov_tokens.json").exists()

def _fingerprints_done(lang: str) -> bool:
    return (
        (MATRICES_DIR / f"{lang}_cooccurrence.npy").exists()
        and (MATRICES_DIR / f"{lang}_positional.npy").exists()
        and (MATRICES_DIR / f"{lang}_ngram_meta.json").exists()
    )


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_romance(force: bool) -> None:
    from src.ingest.wikipedia import ingest
    for lang in ROMANCE_LANGUAGES:
        if not force and _romance_done(lang):
            log.info("SKIP  romance/%s - already ingested", lang)
            continue
        log.info("RUN   romance/%s", lang)
        ingest(lang)


def step_portuguese(force: bool) -> None:
    from src.ingest.wikipedia import ingest
    if not force and _portuguese_done():
        log.info("SKIP  portuguese - already ingested")
        return
    log.info("RUN   portuguese (sequestered)")
    ingest("portuguese", sequester=True)


def step_latin(force: bool) -> None:
    from src.ingest.latin import ingest
    if not force and _latin_done():
        log.info("SKIP  latin - already ingested")
        return
    log.info("RUN   latin (sequestered, ~10 min)")
    ingest()


def step_sumerian(force: bool) -> None:
    from src.ingest.sumerian import run
    if not force and _sumerian_done():
        log.info("SKIP  sumerian - already ingested")
        return
    log.info("RUN   sumerian")
    local_zip = os.environ.get("ORACC_LOCAL_ZIP")
    local_zip_path = Path(local_zip) if local_zip else None
    if local_zip_path:
        log.info("Using local ORACC ZIP: %s", local_zip_path)
    try:
        run(source="auto", local_zip_path=local_zip_path)
    except Exception as exc:
        if "SSL" in str(exc) or "certificate" in str(exc).lower():
            log.error(
                "Sumerian download failed with SSL error.\n"
                "Download the file manually from:\n"
                "  https://oracc.museum.upenn.edu/json/dcclt.zip\n"
                "Then re-run with:\n"
                "  ORACC_LOCAL_ZIP=/path/to/dcclt.zip python pipeline.py"
            )
            sys.exit(1)
        raise


def step_markov(force: bool) -> None:
    from src.nullmodel.markov import run
    if not force and _markov_done():
        log.info("SKIP  markov - already generated")
        return
    log.info("RUN   markov null model")
    run()


def step_fingerprints(force: bool) -> None:
    from src.fingerprint import cooccurrence, positional, ngram

    corpora = {lang: None for lang in ROMANCE_LANGUAGES}
    corpora["markov"] = NULLS_DIR / "markov" / "markov_tokens.json"
    corpora["sumerian"] = NULLS_DIR / "sumerian" / "sumerian_tokens.json"

    for lang, input_path in corpora.items():
        if not force and _fingerprints_done(lang):
            log.info("SKIP  fingerprint/%s - already built", lang)
            continue
        log.info("RUN   fingerprint/%s", lang)
        kwargs = {"input_path": input_path} if input_path else {}
        cooccurrence.run(lang, **kwargs)
        positional.run(lang, **kwargs)
        ngram.run(lang, **kwargs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PHASE3_LANGUAGES = ["french", "italian", "spanish", "romanian"]
REINFORCED_LANGUAGES = ["french"]
REINFORCED_V2_LANGUAGES = ["french"]
REINFORCED_V3_LANGUAGES = ["french"]


def _retrodiction_done(lang: str) -> bool:
    return (DATA / "retrodiction" / lang / "run_summary.json").exists()


def _reinforced_done(lang: str) -> bool:
    return (
        (DATA / "retrodiction" / lang / "stochastic" / "run_summary.json").exists()
        and (DATA / "retrodiction" / lang / "gradient" / "run_summary.json").exists()
    )


def _reinforced_v2_done(lang: str) -> bool:
    return (DATA / "retrodiction" / lang / "v2" / "run_summary.json").exists()


def _reinforced_v3_done(lang: str) -> bool:
    return (DATA / "retrodiction" / lang / "v3" / "run_summary.json").exists()


def step_retrodiction(force: bool) -> None:
    from src.retrodiction.engine import run as retro_run, RetrodictionConfig
    cfg = RetrodictionConfig()
    for lang in PHASE3_LANGUAGES:
        if not force and _retrodiction_done(lang):
            log.info("SKIP  retrodiction/%s - already complete", lang)
            continue
        log.info("RUN   retrodiction/%s (this may take a while)", lang)
        retro_run(lang, config=cfg)


def step_reinforced(force: bool) -> None:
    from src.retrodiction.engine_reinforced import run as reinforced_run, ReinforcedConfig
    cfg = ReinforcedConfig()
    for lang in REINFORCED_LANGUAGES:
        if not force and _reinforced_done(lang):
            log.info("SKIP  reinforced/%s - already complete", lang)
            continue
        log.info("RUN   reinforced/%s (stochastic + gradient)", lang)
        reinforced_run(lang, algorithm="both", config=cfg)


def step_reinforced_v2(force: bool) -> None:
    from src.retrodiction.engine_reinforced_v2 import run as reinforced_v2_run, ReinforcedV2Config
    cfg = ReinforcedV2Config()
    for lang in REINFORCED_V2_LANGUAGES:
        if not force and _reinforced_v2_done(lang):
            log.info("SKIP  reinforced_v2/%s - already complete", lang)
            continue
        log.info("RUN   reinforced_v2/%s (relational corpus mutation search)", lang)
        reinforced_v2_run(lang, config=cfg)


def step_reinforced_v3(force: bool) -> None:
    from src.retrodiction.engine_reinforced_v3 import run as reinforced_v3_run, ReinforcedV3Config
    cfg = ReinforcedV3Config()
    for lang in REINFORCED_V3_LANGUAGES:
        if not force and _reinforced_v3_done(lang):
            log.info("SKIP  reinforced_v3/%s - already complete", lang)
            continue
        log.info("RUN   reinforced_v3/%s (weird relational search + amplified Latin reward)", lang)
        reinforced_v3_run(lang, config=cfg)


INGEST_STEPS = [step_romance, step_portuguese, step_latin, step_sumerian, step_markov]
FINGERPRINT_STEPS = [step_fingerprints]
RETRODICTION_STEPS = [step_retrodiction]
REINFORCED_STEPS = [step_reinforced]
REINFORCED_V2_STEPS = [step_reinforced_v2]
REINFORCED_V3_STEPS = [step_reinforced_v3]
ALL_STEPS = INGEST_STEPS + FINGERPRINT_STEPS + RETRODICTION_STEPS + REINFORCED_STEPS + REINFORCED_V2_STEPS + REINFORCED_V3_STEPS


def print_status() -> None:
    rows = []

    for lang in ROMANCE_LANGUAGES:
        rows.append((f"romance/{lang}", _romance_done(lang)))
    rows.append(("portuguese (sequestered)", _portuguese_done()))
    rows.append(("latin (sequestered)", _latin_done()))
    rows.append(("sumerian", _sumerian_done()))
    rows.append(("markov", _markov_done()))
    for lang in ROMANCE_LANGUAGES + ["markov", "sumerian"]:
        rows.append((f"fingerprint/{lang}", _fingerprints_done(lang)))
    for lang in PHASE3_LANGUAGES:
        rows.append((f"retrodiction/{lang}", _retrodiction_done(lang)))
    for lang in REINFORCED_LANGUAGES:
        rows.append((f"reinforced/{lang}", _reinforced_done(lang)))
    for lang in REINFORCED_V2_LANGUAGES:
        rows.append((f"reinforced_v2/{lang}", _reinforced_v2_done(lang)))
    for lang in REINFORCED_V3_LANGUAGES:
        rows.append((f"reinforced_v3/{lang}", _reinforced_v3_done(lang)))

    done = sum(1 for _, d in rows if d)
    print(f"\nPipeline status: {done}/{len(rows)} complete\n")
    for name, is_done in rows:
        mark = "+" if is_done else "-"
        print(f"  {mark}  {name}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Project RBT pipeline runner")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run all steps even if output already exists",
    )
    parser.add_argument(
        "--step", choices=["ingest", "fingerprint", "retrodiction", "reinforced", "reinforced_v2", "reinforced_v3"],
        help="Run only a subset of steps",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show which steps are complete and exit",
    )
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    if args.step == "ingest":
        steps = INGEST_STEPS
    elif args.step == "fingerprint":
        steps = FINGERPRINT_STEPS
    elif args.step == "retrodiction":
        steps = RETRODICTION_STEPS
    elif args.step == "reinforced":
        steps = REINFORCED_STEPS
    elif args.step == "reinforced_v2":
        steps = REINFORCED_V2_STEPS
    elif args.step == "reinforced_v3":
        steps = REINFORCED_V3_STEPS
    else:
        steps = ALL_STEPS

    for step in steps:
        step(force=args.force)

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
