"""Real Windows scheduled-task death/recovery and trajectory-equivalence pilot.

Only fresh, explicitly excluded pilot roots are created. Never use production.
"""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_h1_scheduled import configure
from src.control.h1_ablation import register
from src.control.h1_io import read_json
from src.retrodiction.h1_checkpointed import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    if os.name != 'nt':
        parser.error('Actual Windows Task Scheduler is required')
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    pilot, reference = root / 'interrupted', root / 'reference'
    register(pilot, 'pilot')
    register(reference, 'pilot')
    config = configure(pilot, session_proposals=2, session_seconds=60)
    task = 'ProjectRBT-H1-Pilot-' + uuid.uuid4().hex[:10]
    ps = str(Path(os.environ['SystemRoot']) / 'System32/WindowsPowerShell/v1.0/powershell.exe')
    receipt = {'purpose': 'Excluded operational pilot', 'task_name': task, 'complete': False, 'actions': []}
    def save():
        atomic_json(root / 'verification_results.json', receipt)
    def task_action(action):
        command = [ps, '-NoProfile', '-ExecutionPolicy', 'RemoteSigned', '-File', str(ROOT / 'scripts/manage_h1_task.ps1'),
                   '-Action', action, '-Config', str(config), '-TaskName', task]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        receipt['actions'].append({'action': action, 'utc': datetime.now(timezone.utc).isoformat(),
                                   'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
        save()
        if result.returncode:
            raise RuntimeError(f'Task {action} failed: {result.stderr}')
    task_action('Install')
    task_action('Start')  # First run is manual; recovery must come from the timer.
    print(f'Pilot task started: {task}', flush=True)
    deadline = time.monotonic() + 100
    while True:
        state = read_json(pilot / 'task_host/state.json') if (pilot / 'task_host/state.json').exists() else {}
        matrix = read_json(pilot / 'matrix_status.json') if (pilot / 'matrix_status.json').exists() else {}
        if state.get('state') == 'failed':
            raise RuntimeError(state['reason'])
        if matrix.get('active') and any(pilot.glob('*/proposals/000000001.json')):
            break
        if time.monotonic() > deadline:
            raise RuntimeError('Pilot never reached an active committed-proposal boundary')
        time.sleep(.05)
    atomic_json(root / 'before_kill_matrix_status.json', matrix)
    atomic_json(root / 'before_kill_host_state.json', state)
    # The registered pilot's host runs the queue in its own process. Compare both
    # records before targeting this PID; never terminate a production process.
    assert state['pid'] == matrix['runner_pid']
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    api.OpenProcess.restype = wintypes.HANDLE
    api.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    api.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    handles = []
    try:
        host_handle = api.OpenProcess(0x100001, False, state['pid'])  # synchronize + terminate
        if not host_handle:
            raise ctypes.WinError(ctypes.get_last_error())
        handles.append(host_handle)
        for details in matrix['active'].values():
            handle = api.OpenProcess(0x100000, False, details['pid'])
            if handle:
                handles.append(handle)
        assert len(handles) >= 2
        assert api.TerminateProcess(host_handle, 73)
        assert all(api.WaitForSingleObject(handle, 10000) == 0 for handle in handles)
    finally:
        for handle in handles:
            api.CloseHandle(handle)
    receipt['controlled_death'] = {'utc': datetime.now(timezone.utc).isoformat(),
        'host_pid': state['pid'], 'worker_pids': [v['pid'] for v in matrix['active'].values()],
        'all_opened_process_handles_signaled': True, 'termination_exit_code': 73,
        'method': 'TerminateProcess on pilot host only; its private job ended descendants'}
    save()
    print('Pilot host terminated; its workers exited too. Waiting for automatic scheduled recovery.', flush=True)
    deadline = time.monotonic() + 420
    while True:
        state = read_json(pilot / 'task_host/state.json')
        if state['state'] in {'failed', 'paused', 'recovery_budget_exhausted'}:
            raise RuntimeError(f'Pilot recovery stopped: {state}')
        if state['state'] == 'complete':
            break
        if time.monotonic() > deadline:
            raise RuntimeError('Automatic scheduled recovery did not finish within seven minutes')
        time.sleep(1)
    assert state['interruption_restarts'] == 1
    matrix = read_json(pilot / 'matrix_status.json')
    assert matrix['status'] == 'complete' and matrix['completed_jobs'] == 4
    receipt['recovered_host_state'] = state
    receipt['recovered_matrix_status'] = matrix
    receipt['automatic_recovery_without_manual_start'] = True
    save()
    print('Automatic recovery completed all four pilot jobs. Checking uninterrupted references.', flush=True)
    receipt['comparisons'] = []
    for arm in ('full', 'disabled'):
        job = f'french__corpus_42__search_42__{arm}'
        command = [sys.executable, '-u', '-m', 'src.control.h1_ablation', 'run',
                   '--output-root', str(reference), '--job-id', job]
        with (root / f'reference_{arm}.log').open('x') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
                           timeout=180, creationflags=subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
        left, right = pilot / job, reference / job
        a, b = read_json(left / 'checkpoint.json'), read_json(right / 'checkpoint.json')
        for key in ('proposal_index', 'accepted_states', 'stagnation', 'state_sha256', 'rng_state', 'counters'):
            assert a[key] == b[key], (arm, key)
        for index in range(5):
            a = read_json(left / 'proposals' / f'{index:09d}.json')
            b = read_json(right / 'proposals' / f'{index:09d}.json')
            for key in ('proposal_index', 'before_state', 'after_state', 'attempts', 'counters'):
                assert a[key] == b[key], (arm, index, key)
        assert {p.name: p.read_bytes() for p in (left / 'states').glob('*.json')} == {
            p.name: p.read_bytes() for p in (right / 'states').glob('*.json')}
        receipt['comparisons'].append({'arm': arm, 'scientific_trajectory_identical': True,
            'output_root_contract_hashes_intentionally_differ': True})
        save()
    # An additional trigger after completion must not create another run.
    original_session = state['session_id']
    task_action('Start')
    time.sleep(8)
    assert read_json(pilot / 'task_host/state.json')['session_id'] == original_session
    receipt['completion_latch_verified'] = True
    task_action('Remove')
    receipt['complete'] = True
    save()
    print(json.dumps({'complete': True, 'automatic_recoveries': 1, 'jobs_completed': 4,
                      'comparisons': receipt['comparisons']}), flush=True)


if __name__ == '__main__':
    main()
