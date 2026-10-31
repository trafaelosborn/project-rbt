"""One-time, lock-protected audit before the authorized October 31 resume."""
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.control.h1_ablation import validate_registration, job_contract
from src.retrodiction.h1_checkpointed import load_checkpoint, load_state
from src.retrodiction.h1_objective import stable_hash, text_hash
from src.retrodiction.h1_run_lock import run_lock

root = REPO / 'data/h1_ablation_production_20261001'
evidence = Path(__file__).resolve().parent
receipt = {'started_utc': datetime.now(timezone.utc).isoformat(), 'jobs': []}
try:
    with ExitStack() as locks:
        locks.enter_context(run_lock(root / 'matrix_runner.lock'))
        registration = validate_registration(root)
        assert registration['sha256'] == '012560628d8d5f7c54f74560a711804cf40d962192cf6f79c08b2d987f60eeb0'
        for job in registration['jobs']:
            locks.enter_context(run_lock(root / job['job_id'] / 'run.lock'))
        receipt['all_67_locks_available'] = True
        receipt['registration_sha256'] = registration['sha256']
        receipt['registration_identity_unchanged'] = True
        previous = json.loads((root / 'matrix_status.json').read_text())
        shutil.copyfile(root / 'matrix_status.json', evidence / 'previous_matrix_status.json')
        receipt['prior_status_total'] = sum(previous['progress'].values())
        receipt['prior_heartbeat_utc'] = previous['updated_utc']
        for name, info in previous['active'].items():
            shutil.copyfile(info['log'], evidence / Path(info['log']).name)
        old_logs = REPO / 'artifacts/h1_io_repair_20261022'
        for filename in ['runner_stdout_continuation.log', 'runner_stderr_continuation.log']:
            shutil.copyfile(old_logs / filename, evidence / ('previous_' + filename))
        pauses = [p for p in [root / 'PAUSE'] + [root / j['job_id'] / 'PAUSE' for j in registration['jobs']] if p.exists()]
        receipt['pause_files'] = [str(p) for p in pauses]
        if pauses:
            raise ValueError('Pause request exists; inspect its provenance before resuming')
        expected = {
            'scripts/run_h1_ablation_matrix.py': '9565366e561ffdb6e944b231b08aa01b3a6a4bfb93d69be4d6ff6d2ec0e08f95',
            'src/control/h1_io.py': 'f3d9e91f3e7ba85ef30afff79a0b80842af8d8dc1f9363f6b78a0409d0c6fa47',
        }
        receipt['operational_source_sha256_lf'] = {p: text_hash(REPO / p) for p in expected}
        assert receipt['operational_source_sha256_lf'] == expected
        for job in registration['jobs']:
            folder = root / job['job_id']
            checkpoint = load_checkpoint(folder, stable_hash(job_contract(registration, job)))
            assert 0 <= checkpoint['proposal_index'] <= job['search_config']['proposal_budget']
            load_state(folder, checkpoint['state_sha256'])
            beyond = []
            for record in (folder / 'proposals').glob('*.json'):
                if int(record.stem) > checkpoint['proposal_index']:
                    beyond.append({'filename': record.name, 'sha256_bytes': hashlib.sha256(record.read_bytes()).hexdigest()})
                    dest = evidence / 'uncommitted_records' / job['job_id'] / record.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(record, dest)
            item = {'job_id': job['job_id'], 'proposal_index': checkpoint['proposal_index'],
                    'checkpoint_sha256': checkpoint['sha256'], 'state_sha256': checkpoint['state_sha256'],
                    'uncommitted_records': beyond}
            receipt['jobs'].append(item)
            if job['job_id'] in previous['active']:
                dest = evidence / 'interrupted_checkpoints' / job['job_id']
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(folder / 'checkpoint.json', dest / 'checkpoint.json')
            print(f"Verified {len(receipt['jobs'])}/66: {job['job_id']} at {checkpoint['proposal_index']}", flush=True)
        receipt['committed_proposal_total'] = sum(j['proposal_index'] for j in receipt['jobs'])
        receipt['proposal_budget_total'] = sum(j['search_config']['proposal_budget'] for j in registration['jobs'])
        receipt['completed_jobs'] = sum(j['proposal_index'] == 1000 for j in receipt['jobs'])
        receipt['free_disk_gib'] = shutil.disk_usage(root).free / 2**30
        assert receipt['free_disk_gib'] >= 10
        receipt['git_head'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
        receipt['complete'] = True
except BaseException as exc:
    receipt['complete'] = False
    receipt['error'] = repr(exc)
    raise
finally:
    receipt['finished_utc'] = datetime.now(timezone.utc).isoformat()
    (evidence / 'preflight.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: v for k, v in receipt.items() if k != 'jobs'}, indent=2))
