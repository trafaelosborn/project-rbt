"""Read-only production audit, with all checkpoint writers excluded by locks."""
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.control.h1_ablation import validate_registration, job_contract
from src.retrodiction.h1_checkpointed import verify, load_checkpoint, load_state
from src.retrodiction.h1_objective import stable_hash, text_hash
from src.retrodiction.h1_run_lock import run_lock

root = REPO / 'data/h1_ablation_production_20261001'
evidence = Path(__file__).resolve().parent
prior = json.loads((REPO / 'artifacts/h1_restart_20261031/preflight.json').read_text())
prior_jobs = {j['job_id']: j for j in prior['jobs']}
result = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'jobs': [],
          'action': 'Status audit only; no restart, pause, or scientific edit'}
with ExitStack() as locks:
    locks.enter_context(run_lock(root / 'matrix_runner.lock'))
    registration = validate_registration(root)
    for job in registration['jobs']:
        locks.enter_context(run_lock(root / job['job_id'] / 'run.lock'))
    result['all_67_locks_available'] = True
    result['registration_sha256'] = registration['sha256']
    result['registration_identity_unchanged'] = registration['sha256'] == prior['registration_sha256']
    previous = json.loads((root / 'matrix_status.json').read_text())
    shutil.copyfile(root / 'matrix_status.json', evidence / 'stale_matrix_status.json')
    result['last_heartbeat_utc'] = previous['updated_utc']
    result['stale_status_total'] = sum(previous['progress'].values())
    result['recorded_io_retries'] = len(previous['io_retries'])
    result['recorded_worker_recoveries'] = sum(previous['worker_recoveries'].values())
    result['recorded_shutdown_errors'] = previous['shutdown_errors']
    result['pause_files'] = [str(p) for p in [root / 'PAUSE'] + [root / j['job_id'] / 'PAUSE' for j in registration['jobs']] if p.exists()]
    for name, details in previous['active'].items():
        shutil.copyfile(details['log'], evidence / Path(details['log']).name)
    for name in ['runner_stdout.log', 'runner_stderr.log']:
        shutil.copyfile(REPO / 'artifacts/h1_restart_20261031' / name, evidence / ('previous_' + name))
    for job in registration['jobs']:
        name = job['job_id']
        folder = root / name
        checkpoint = verify(json.loads((folder / 'checkpoint.json').read_text()))
        assert checkpoint['contract_sha256'] == stable_hash(job_contract(registration, job))
        assert 0 <= checkpoint['proposal_index'] <= job['search_config']['proposal_budget']
        changed = checkpoint['sha256'] != prior_jobs[name]['checkpoint_sha256']
        if changed:
            load_checkpoint(folder, stable_hash(job_contract(registration, job)))
            dest = evidence / 'changed_checkpoints' / name
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(folder / 'checkpoint.json', dest / 'checkpoint.json')
        load_state(folder, checkpoint['state_sha256'])
        beyond = sorted(p.name for p in (folder / 'proposals').glob('*.json') if int(p.stem) > checkpoint['proposal_index'])
        result['jobs'].append({'job_id': name, 'proposal_index': checkpoint['proposal_index'],
            'checkpoint_sha256': checkpoint['sha256'], 'state_sha256': checkpoint['state_sha256'],
            'history_check': 'Full committed chain and counter audit today' if changed else 'Checkpoint hash unchanged from October 31 full audit',
            'changed_since_october31_preflight': changed, 'uncommitted_record_filenames': beyond})
    result['committed_proposal_total'] = sum(j['proposal_index'] for j in result['jobs'])
    result['additional_since_restart'] = result['committed_proposal_total'] - prior['committed_proposal_total']
    result['completed_jobs'] = sum(j['proposal_index'] == 1000 for j in result['jobs'])
    result['free_disk_gib'] = shutil.disk_usage(root).free / 2**30
    result['operational_source_sha256_lf'] = {p: text_hash(REPO / p) for p in prior['operational_source_sha256_lf']}
    result['operational_source_unchanged'] = result['operational_source_sha256_lf'] == prior['operational_source_sha256_lf']
    result['complete'] = True
    result['finished_utc'] = datetime.now(timezone.utc).isoformat()
(evidence / 'audit.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
summary = {k:v for k,v in result.items() if k != 'jobs'}
summary['changed_jobs'] = [j for j in result['jobs'] if j['changed_since_october31_preflight']]
print(json.dumps(summary, indent=2))
