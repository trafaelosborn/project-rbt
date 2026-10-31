"""Read queue status and immutable filenames; never open active checkpoints."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.control.h1_io import read_json
from src.retrodiction.h1_run_lock import run_lock

root = REPO / 'data/h1_ablation_production_20261001'
evidence = Path(__file__).resolve().parent
status = read_json(root / 'matrix_status.json')
launch = json.loads((evidence / 'process_launch.json').read_text(encoding='utf-8-sig'))
now = datetime.now(timezone.utc)
result = {k: status[k] for k in ['status', 'reason', 'runner_id', 'runner_pid', 'updated_utc',
    'registration_sha256', 'completed_jobs', 'total_jobs', 'progress_policy', 'shutdown_errors']}
result['checked_utc'] = now.isoformat()
result['heartbeat_age_seconds'] = (now - datetime.fromisoformat(status['updated_utc'])).total_seconds()
result['last_exited_session_proposal_total'] = sum(status['progress'].values())
result['io_retry_count'] = len(status['io_retries'])
result['worker_recovery_count'] = sum(status['worker_recoveries'].values())
result['active'] = {}
for name, details in status['active'].items():
    latest = max((int(p.stem) for p in (root / name / 'proposals').glob('*.json')), default=-1)
    result['active'][name] = {**details, 'latest_immutable_record_index': latest}
try:
    with run_lock(root / 'matrix_runner.lock'):
        result['matrix_lock_held_by_runner'] = False
except ValueError as exc:
    if str(exc) != 'Another process owns this run':
        raise
    result['matrix_lock_held_by_runner'] = True
result['runner_stderr_bytes'] = (evidence / 'runner_stderr.log').stat().st_size
result['verified'] = (result['status'] == 'running' and result['runner_pid'] == launch['runner_pid']
    and 0 <= result['heartbeat_age_seconds'] < 15 and result['matrix_lock_held_by_runner']
    and len(result['active']) == 2
    and all(a['latest_immutable_record_index'] > a['starting_proposal'] for a in result['active'].values())
    and result['registration_sha256'] == launch['registration_sha256'])
result['note'] = 'Active record filenames demonstrate new work, not independently validated checkpoint totals.'
(evidence / 'live_verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2))
if not result['verified']:
    raise SystemExit(1)
