"""Host recovery never changes the registered scientific queue or hides errors."""
from contextlib import nullcontext
from pathlib import Path
import pytest

from scripts import run_h1_scheduled as host
from src.retrodiction.h1_run_lock import run_lock


@pytest.fixture
def setup(tmp_path, monkeypatch):
    registration = {'sha256': 'registered', 'jobs': []}
    monkeypatch.setattr(host, 'validate_registration', lambda root: registration)
    monkeypatch.setattr(host, 'source_hashes', lambda: {'source': 'fixed'})
    monkeypatch.setattr(host, 'preflight', lambda *args: {'paused': False, 'jobs': [], 'committed_proposals': 11})
    monkeypatch.setattr(host.h1_windows_host, 'contain_descendants', lambda: None)
    monkeypatch.setattr(host.h1_windows_host, 'keep_system_awake', nullcontext)
    calls = []
    def run(root, **options):
        calls.append(options)
        return {'status': 'complete', 'reason': None, 'runner_id': 'matrix', 'progress': {'job': 12}}
    monkeypatch.setattr(host, 'run_matrix', run)
    config = host.configure(tmp_path)
    return tmp_path, config, calls


def interrupted(root, config, count=0):
    host.write(root / 'task_host/state.json', {'state': 'running', 'session_id': 'old',
        'config_sha256': host.read_optional(config)['sha256'], 'interruption_restarts': count})


def test_complete_and_repeat_are_idempotent(setup):
    root, config, calls = setup
    assert host.run_once(config)['state'] == 'complete'
    assert host.run_once(config)['state'] == 'complete'
    assert len(calls) == 1
    assert calls[0] == {'workers': 2, 'session_proposals': 100, 'session_seconds': 1800}
    assert host.read_optional(root / 'task_host/state.json')['committed_proposals'] == 12


def test_interrupted_host_resumes_once_with_persisted_count(setup):
    root, config, calls = setup
    interrupted(root, config, 1)
    result = host.run_once(config)
    assert result['state'] == 'complete' and result['interruption_restarts'] == 2
    assert result['recovery_of_interrupted_session'] and len(calls) == 1


def test_recovery_budget_is_not_reset_by_new_scheduler_invocation(setup):
    root, config, calls = setup
    interrupted(root, config, 3)
    assert host.run_once(config)['state'] == 'recovery_budget_exhausted'
    assert host.run_once(config)['state'] == 'recovery_budget_exhausted'
    assert not calls


def test_scientific_error_is_latched_and_never_retried(setup, monkeypatch):
    root, config, calls = setup
    def bad(*args, **kwargs):
        calls.append('bad')
        raise ValueError('Accepted candidate failed independent score replay')
    monkeypatch.setattr(host, 'run_matrix', bad)
    assert host.run_once(config)['state'] == 'failed'
    assert host.run_once(config)['state'] == 'failed'
    assert calls == ['bad']
    assert 'independent score replay' in host.read_optional(root / 'task_host/state.json')['reason']


def test_returned_matrix_failure_is_latched(setup, monkeypatch):
    _, config, _ = setup
    monkeypatch.setattr(host, 'run_matrix', lambda *a, **k: {'status': 'failed', 'reason': 'child_failed', 'runner_id': 'matrix', 'progress': {}})
    assert host.run_once(config)['state'] == 'failed'


def test_existing_pause_is_preserved(setup, monkeypatch):
    root, config, calls = setup
    (root / 'PAUSE').write_text('user pause')
    monkeypatch.setattr(host, 'preflight', lambda *a: {'paused': True, 'jobs': []})
    assert host.run_once(config)['state'] == 'paused'
    (root / 'PAUSE').unlink()
    assert host.run_once(config)['state'] == 'paused'
    assert not calls


def test_changed_scientific_registration_fails_closed(setup, monkeypatch):
    _, config, calls = setup
    monkeypatch.setattr(host, 'validate_registration', lambda *a: {'sha256': 'different'})
    assert host.run_once(config)['state'] == 'failed'
    assert not calls


def test_changed_operational_source_fails_closed(setup, monkeypatch):
    _, config, calls = setup
    monkeypatch.setattr(host, 'source_hashes', lambda: {'source': 'changed'})
    assert host.run_once(config)['state'] == 'failed'
    assert not calls


def test_recorded_matrix_error_prevents_abrupt_recovery(setup):
    root, config, calls = setup
    interrupted(root, config)
    host.atomic_json(root / 'matrix_status.json', {'status': 'failed'})
    assert host.run_once(config)['state'] == 'failed'
    assert not calls


def test_corrupt_checkpoint_history_prevents_launch(setup, monkeypatch):
    _, config, calls = setup
    def corrupt(*a):
        raise ValueError('Proposal chain is discontinuous')
    monkeypatch.setattr(host, 'preflight', corrupt)
    assert host.run_once(config)['state'] == 'failed'
    assert not calls


def test_live_host_excludes_duplicate_invocation(setup):
    root, config, calls = setup
    with run_lock(root / 'task_host/host.lock'):
        assert host.run_once(config) == {'state': 'busy'}
    assert not calls


def test_external_writer_defers_without_consuming_recovery(setup, monkeypatch):
    root, config, calls = setup
    interrupted(root, config, 1)
    def busy(*a):
        raise ValueError('Another process owns this run')
    monkeypatch.setattr(host, 'preflight', busy)
    assert host.run_once(config) == {'state': 'busy'}
    assert host.read_optional(root / 'task_host/state.json')['interruption_restarts'] == 1
    assert not calls


def test_config_cannot_be_silently_replaced(setup):
    root, config, _ = setup
    assert host.configure(root) == config
    with pytest.raises(ValueError, match='differs'):
        host.configure(root, workers=4)


def test_corrupt_state_is_not_treated_as_fresh_start(setup):
    root, config, calls = setup
    (root / 'task_host/state.json').write_text('{"state":"running"}')
    with pytest.raises(ValueError):
        host.run_once(config)
    assert not calls
