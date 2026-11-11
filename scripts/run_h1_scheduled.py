"""Durable, fail-closed Windows host for the existing H1 matrix queue."""
from __future__ import annotations
import argparse
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_h1_ablation_matrix import run_matrix
from src.control.h1_ablation import validate_registration, job_contract
from src.control import h1_io, h1_windows_host
from src.retrodiction.h1_checkpointed import atomic_json, seal, verify, load_checkpoint, load_state
from src.retrodiction.h1_objective import stable_hash, text_hash
from src.retrodiction.h1_run_lock import run_lock

SOURCES = ('scripts/run_h1_scheduled.py', 'scripts/run_h1_ablation_matrix.py',
           'src/control/h1_io.py', 'src/control/h1_windows_host.py',
           'scripts/manage_h1_task.ps1')
TERMINAL = {'complete', 'paused', 'failed', 'recovery_budget_exhausted'}


def now():
    return datetime.now(timezone.utc).isoformat()


def source_hashes():
    return {p: text_hash(ROOT / p) for p in SOURCES}


def write(path, payload):
    h1_io.retry_access(lambda: atomic_json(path, seal(payload)))


def read_optional(path):
    try:
        return verify(h1_io.read_json(path))
    except FileNotFoundError:
        return None


def configure(root, *, workers=2, session_proposals=100, session_seconds=1800,
              max_interruptions=3):
    if workers < 1 or session_proposals < 1 or session_seconds <= 0 or not 0 <= max_interruptions <= 3:
        raise ValueError('Invalid host limits')
    root = root.resolve()
    registration = validate_registration(root)
    folder = root / 'task_host'
    folder.mkdir(exist_ok=True)
    config = {'schema': 'h1_scheduled_config_v1', 'root': str(root),
              'registration_sha256': registration['sha256'], 'source_hashes_lf': source_hashes(),
              'workers': workers, 'session_proposals': session_proposals,
              'session_seconds': session_seconds, 'max_interruptions': max_interruptions}
    path = folder / 'config.json'
    with run_lock(folder / 'host.lock'):
        existing = read_optional(path)
        if existing is not None:
            if existing != seal(config):
                raise ValueError('Existing host config differs; explicit review required')
        else:
            write(path, config)
    return path


def preflight(root, registration):
    """Exclude all writers while verifying committed chains and current states."""
    with ExitStack() as locks:
        locks.enter_context(run_lock(root / 'matrix_runner.lock'))
        for job in registration['jobs']:
            folder = root / job['job_id']
            folder.mkdir(exist_ok=True)
            locks.enter_context(run_lock(folder / 'run.lock'))
        if (root / 'PAUSE').exists() or any((root / j['job_id'] / 'PAUSE').exists() for j in registration['jobs']):
            return {'paused': True, 'jobs': []}
        rows = []
        for job in registration['jobs']:
            folder = root / job['job_id']
            if not (folder / 'checkpoint.json').exists():
                if any((folder / 'proposals').glob('*.json')):
                    raise ValueError(f"History without checkpoint: {job['job_id']}")
                rows.append({'job_id': job['job_id'], 'proposal_index': 0})
                continue
            cp = h1_io.retry_access(lambda: load_checkpoint(folder, stable_hash(job_contract(registration, job))))
            if not 0 <= cp['proposal_index'] <= job['search_config']['proposal_budget']:
                raise ValueError('Checkpoint exceeds registered budget')
            h1_io.retry_access(lambda: load_state(folder, cp['state_sha256']))
            rows.append({'job_id': job['job_id'], 'proposal_index': cp['proposal_index'],
                         'checkpoint_sha256': cp['sha256'], 'state_sha256': cp['state_sha256']})
        return {'paused': False, 'jobs': rows, 'committed_proposals': sum(j['proposal_index'] for j in rows)}


def run_once(config_path):
    config_path = config_path.resolve()
    config = verify(h1_io.read_json(config_path))
    root = Path(config['root'])
    folder = root / 'task_host'
    if config_path != folder / 'config.json':
        raise ValueError('Host config path does not match its registered root')
    try:
        with run_lock(folder / 'host.lock'):
            return run_locked(root, folder, config)
    except ValueError as exc:
        if str(exc) != 'Another process owns this run':
            raise
        return {'state': 'busy'}


def run_locked(root, folder, config):
    prior = read_optional(folder / 'state.json')
    if prior is not None and prior['config_sha256'] != config['sha256']:
        raise ValueError('Host state belongs to another configuration')
    if prior is not None and prior['state'] in TERMINAL:
        return prior  # Scheduler triggers never override failure, pause, or completion.
    if prior is not None and prior['state'] != 'running':
        raise ValueError('Unknown host state; explicit review required')
    restarts = 0 if prior is None else prior['interruption_restarts'] + 1
    session_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    evidence = folder / 'sessions' / session_id
    evidence.mkdir(parents=True)
    state = {'schema': 'h1_scheduled_state_v1', 'session_id': session_id, 'pid': os.getpid(),
             'config_sha256': config['sha256'], 'created_utc': now(),
             'interruption_restarts': restarts, 'state': 'starting',
             'previous_session_id': prior['session_id'] if prior else None}

    def publish(phase, **extra):
        state.update(state=phase, updated_utc=now(), **extra)
        write(evidence / 'state.json', state)
        write(folder / 'state.json', state)

    with (evidence / 'host.log').open('x', encoding='utf-8') as log, redirect_stdout(log), redirect_stderr(log):
        try:
            if restarts > config['max_interruptions']:
                publish('recovery_budget_exhausted', reason='Persistent interruption budget exhausted; review required')
                return state
            # Check locks before persisting a new attempt. An external writer must
            # never cause checkpoint reads or consume a recovery allowance.
            registration = validate_registration(root)
            if registration['sha256'] != config['registration_sha256'] or source_hashes() != config['source_hashes_lf']:
                raise ValueError('Pinned scientific or operational identity changed')
            try:
                audit = preflight(root, registration)
            except ValueError as exc:
                if str(exc) == 'Another process owns this run':
                    write(evidence / 'deferred.json', {'utc': now(), 'reason': str(exc)})
                    return {'state': 'busy'}
                raise
            write(evidence / 'preflight.json', audit)
            if audit['paused']:
                publish('paused', reason='Existing PAUSE request; no files removed')
                return state
            if (root / 'matrix_status.json').exists():
                previous_matrix = h1_io.read_json(root / 'matrix_status.json')
                shutil.copyfile(root / 'matrix_status.json', evidence / 'previous_matrix_status.json')
                if previous_matrix['status'] == 'failed':
                    raise ValueError('Recorded matrix failure requires explicit review')
                if previous_matrix['status'] == 'complete':
                    publish('complete', reason='Matrix already completed')
                    return state
            for relative in SOURCES:
                dest = evidence / 'source' / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, dest)
            publish('running', starting_committed_proposals=audit['committed_proposals'],
                    recovery_of_interrupted_session=prior is not None)
            h1_windows_host.contain_descendants()
            with h1_windows_host.keep_system_awake():
                result = run_matrix(root, workers=config['workers'],
                                    session_proposals=config['session_proposals'],
                                    session_seconds=config['session_seconds'])
            phase = result['status']
            if phase not in {'complete', 'paused', 'failed'}:
                raise ValueError(f'Unexpected matrix terminal state: {phase}')
            publish(phase, reason=result['reason'], runner_id=result['runner_id'],
                    committed_proposals=sum(result['progress'].values()))
        except BaseException as exc:
            traceback.print_exc()
            publish('failed', reason=f'{type(exc).__name__}: {exc}')
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    setup = sub.add_parser('configure')
    setup.add_argument('--output-root', type=Path, required=True)
    setup.add_argument('--workers', type=int, default=2)
    setup.add_argument('--session-proposals', type=int, default=100)
    setup.add_argument('--session-seconds', type=float, default=1800)
    setup.add_argument('--max-interruptions', type=int, default=3)
    run = sub.add_parser('run')
    run.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'configure':
        print(configure(args.output_root, workers=args.workers, session_proposals=args.session_proposals,
                        session_seconds=args.session_seconds, max_interruptions=args.max_interruptions))
    else:
        result = run_once(args.config)
        print(json.dumps({k: v for k, v in result.items() if k in {'state', 'reason', 'session_id'}}))
        # Diagnosed failures are latched in state, not repeatedly relaunched.


if __name__ == '__main__':
    main()
