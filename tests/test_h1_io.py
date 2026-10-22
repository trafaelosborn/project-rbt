"""Windows file-sharing regression tests, including real OS handles."""
import ctypes
import errno
import json
import os
from pathlib import Path

import pytest

from src.control import h1_io as io
from src.retrodiction.h1_checkpointed import atomic_json


def test_access_retry_is_bounded_and_reports_each_retry(monkeypatch):
    monkeypatch.setattr(io, "WINDOWS", True)
    sleeps, reports, calls = [], [], []
    monkeypatch.setattr(io.time, "sleep", sleeps.append)
    def denied():
        calls.append(1)
        raise PermissionError(errno.EACCES, "locked")
    with pytest.raises(PermissionError):
        io.retry_access(denied, on_retry=reports.append)
    assert len(calls) == 10 and len(reports) == 9
    assert sum(sleeps) == pytest.approx(5.55)


@pytest.mark.parametrize("error", [FileNotFoundError(errno.ENOENT, "missing"),
                                   OSError(errno.ENOSPC, "disk full"), ValueError("corrupt")])
def test_other_errors_are_not_retried(monkeypatch, error):
    monkeypatch.setattr(io, "WINDOWS", True)
    calls = []
    def fail():
        calls.append(1)
        raise error
    with pytest.raises(type(error)):
        io.retry_access(fail)
    assert len(calls) == 1


def test_non_windows_permission_failure_is_not_retried(monkeypatch):
    monkeypatch.setattr(io, "WINDOWS", False)
    reports = []
    with pytest.raises(PermissionError):
        io.retry_access(lambda: (_ for _ in ()).throw(PermissionError(errno.EACCES, "denied")),
                        on_retry=reports.append)
    assert not reports


def test_invalid_json_is_not_hidden(tmp_path):
    path = tmp_path / "checkpoint.json"
    path.write_text('{"partial":', encoding="utf-8")
    reports = []
    with pytest.raises(json.JSONDecodeError):
        io.read_json(path, on_retry=reports.append)
    assert not reports


@pytest.mark.skipif(os.name != "nt", reason="Real Windows sharing semantics")
def test_reader_blocks_scientific_writer_until_reader_exits(tmp_path):
    path = tmp_path / "checkpoint.json"
    atomic_json(path, {"proposal": 1})
    with path.open(encoding="utf-8"):
        with pytest.raises(PermissionError):
            atomic_json(path, {"proposal": 2})
    assert json.loads(path.read_text()) == {"proposal": 1}
    atomic_json(path, {"proposal": 2})
    assert io.read_json(path) == {"proposal": 2}


@pytest.mark.skipif(os.name != "nt", reason="Real Windows exclusive handle")
def test_exclusive_handle_read_retries_then_succeeds(tmp_path, monkeypatch):
    from ctypes import wintypes
    path = tmp_path / "checkpoint.json"
    atomic_json(path, {"proposal": 7})
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateFileW(str(path), 0x80000000, 0, None, 3, 0x80, None)
    assert handle != ctypes.c_void_p(-1).value
    reports = []
    def release(details):
        nonlocal handle
        reports.append(details)
        kernel.CloseHandle(handle)
        handle = None
    monkeypatch.setattr(io.time, "sleep", lambda _: None)
    try:
        assert io.read_json(path, on_retry=release) == {"proposal": 7}
    finally:
        if handle is not None:
            kernel.CloseHandle(handle)
    assert len(reports) == 1 and reports[0]["errno"] == errno.EACCES


@pytest.mark.skipif(os.name != "nt", reason="Real Windows replacement contention")
def test_status_write_retry_preserves_old_file_until_reader_releases(tmp_path, monkeypatch):
    path = tmp_path / "status.json"
    atomic_json(path, {"status": "old"})
    reader = path.open(encoding="utf-8")
    reports = []
    def release(details):
        reports.append(details)
        assert json.load(reader) == {"status": "old"}
        reader.close()
    monkeypatch.setattr(io.time, "sleep", lambda _: None)
    try:
        io.retry_access(lambda: atomic_json(path, {"status": "new"}), on_retry=release)
    finally:
        reader.close()
    assert len(reports) == 1
    assert io.read_json(path) == {"status": "new"}
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("final,expected", [
    ("PermissionError: [Errno 13] Permission denied: {path}", True),
    ("PermissionError: [WinError 5] Access denied: 'temp' -> {path}", True),
    ("PermissionError: [WinError 32] Sharing violation: {path}", True),
    ("ValueError: checkpoint integrity failure: {path}", False),
    ("PermissionError: [Errno 13] Permission denied: 'other/checkpoint.json'", False),
    ("PermissionError: [Errno 13] Permission denied: {path}\nValueError: corrupt", False),
    ("PermissionError: [WinError 19] Write protected: {path}", False),
])
def test_worker_recovery_requires_exact_terminal_error_and_path(tmp_path, monkeypatch, final, expected):
    monkeypatch.setattr(io, "WINDOWS", True)
    path = tmp_path / "checkpoint.json"
    log = tmp_path / "child.log"
    log.write_text(final.format(path=repr(str(path.resolve()))) + "\n", encoding="utf-8")
    assert io.checkpoint_access_failure(log, path) is expected
