"""
Driver Adapter
==============
Internal wiring between RunController and the long_run_v5 driver.

This module is private to src/control and should not be imported directly.
It translates a RunConfig into a LongRunConfig and invokes the long-run
driver, plumbing the optional progress callback through block boundaries.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

from src.control.run_config import RunConfig
from src.retrodiction.long_run_v5 import LongRunConfig, run_long_continuation

log = logging.getLogger(__name__)


def run_driver(
    config: RunConfig,
    output_dir: Path,
    progress_callback: Callable | None = None,
) -> None:
    """
    Translate RunConfig -> LongRunConfig and run the long-run driver.

    Progress callback receives a RunStatus after each block if provided.
    The stop-sentinel file is checked before each block starts.
    """
    long_cfg = _to_long_run_config(config, output_dir)

    if progress_callback is not None:
        _run_with_callback(long_cfg, config, output_dir, progress_callback)
    else:
        run_long_continuation(long_cfg)


def _to_long_run_config(config: RunConfig, output_dir: Path) -> LongRunConfig:
    return LongRunConfig(
        language=config.source_language,
        start_corpus=Path(config.start_corpus),
        output_dir=output_dir,
        total_target_proposals=config.total_target_proposals,
        block_proposals=config.block_proposals,
        starting_proposals=0,
        num_sequences=config.num_sequences,
        n_candidates=config.candidate_count,
        max_accepted_stages=512,
        seed=config.seed,
        min_improvement=0.0001,
        struct_target=config.struct_target,
        form_target=config.form_target,
        use_incremental_scoring=config.use_incremental_scoring,
        use_fortran_cosine=config.use_fortran_cosine,
        use_fortran_batch=config.use_fortran_batch,
        use_semantic_transparency=config.use_semantic_transparency,
        transparency_weight=config.transparency_weight,
        validator_set=list(config.validator_set),
        validator_snapshot_every_blocks=config.validator_snapshot_every_blocks,
        live_event_mode=config.live_event_mode,
        live_event_buffer_size=config.live_event_buffer_size,
    )


def _run_with_callback(
    long_cfg: LongRunConfig,
    run_config: RunConfig,
    output_dir: Path,
    callback: Callable,
) -> None:
    """
    Wrap the long-run driver to emit progress callbacks at block boundaries.

    This wraps run_long_continuation with a polling loop so the TUI can
    refresh without requiring the driver to know about the callback.
    """
    import threading
    import time

    from src.control.run_controller import RunController

    controller = RunController()
    done = threading.Event()

    def _driver_thread():
        try:
            run_long_continuation(long_cfg)
        finally:
            done.set()

    thread = threading.Thread(target=_driver_thread, daemon=True)
    thread.start()

    last_blocks = 0
    while not done.is_set():
        time.sleep(2.0)
        status = controller.status(output_dir)
        if status.blocks_completed != last_blocks:
            last_blocks = status.blocks_completed
            try:
                callback(status)
            except Exception:
                log.debug("Progress callback raised", exc_info=True)

        # honour stop sentinel
        sentinel = output_dir / "stop_requested"
        if sentinel.exists():
            log.info("Stop sentinel detected; waiting for current block to finish.")

    thread.join()
    # Final callback
    status = controller.status(output_dir)
    try:
        callback(status)
    except Exception:
        log.debug("Final progress callback raised", exc_info=True)
