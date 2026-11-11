"""Snapshot safe queue/host status and immutable filenames, never checkpoints."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.control.h1_io import read_json
from src.retrodiction.h1_checkpointed import verify
from src.retrodiction.h1_objective import text_hash

root = REPO / 'data/h1_ablation_production_20261001'
base = Path(__file__).resolve().parent
now = datetime.now(timezone.utc)
matrix = read_json(root / 'matrix_status.json')
state = verify(read_json(root / 'task_host/state.json'))
config = verify(read_json(root / 'task_host/config.json'))
result = {'checked_utc': now.isoformat(), 'host_state': state['state'],
          'host_pid': state['pid'], 'host_session': state['session_id'],
          'host_recoveries': state['interruption_restarts'], 'matrix_state': matrix['status'],
          'matrix_session': matrix['runner_id'], 'matrix_pid': matrix['runner_pid'],
          'matrix_heartbeat_utc': matrix['updated_utc'],
          'heartbeat_age_seconds': (now - datetime.fromisoformat(matrix['updated_utc'])).total_seconds(),
          'last_exited_session_total': sum(matrix['progress'].values()),
          'completed_jobs': matrix['completed_jobs'], 'io_retries': len(matrix['io_retries']),
          'worker_recoveries': sum(matrix['worker_recoveries'].values()),
          'shutdown_errors': matrix['shutdown_errors'],
          'completed_worker_sessions': [e for e in matrix['events'] if e['event'] == 'exited'],
          'active': {name: {**details, 'latest_immutable_record': max(
              int(p.stem) for p in (root / name / 'proposals').glob('*.json'))}
              for name, details in matrix['active'].items()}}
assert state['state'] == matrix['status'] == 'running'
assert state['pid'] == matrix['runner_pid']
assert 0 <= result['heartbeat_age_seconds'] < 15
assert all(text_hash(REPO / p) == digest for p, digest in config['source_hashes_lf'].items())
result['operational_sources_unchanged'] = True
result['note'] = 'Active immutable filenames demonstrate work, not independently validated checkpoint totals.'
path = base / ('production_observation_' + now.strftime('%Y%m%dT%H%M%SZ') + '.json')
path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2))
