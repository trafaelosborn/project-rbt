"""Operational Windows I/O; does not modify the registered scientific writer."""
import errno
import json
import os
from pathlib import Path
import time

WINDOWS = os.name == "nt"
IO_ATTEMPTS = 10
WORKER_RECOVERIES = 3  # Per job, per matrix invocation; never reset on progress.


def is_access_error(exc):
    return WINDOWS and isinstance(exc, OSError) and (
        getattr(exc, "winerror", None) in {5, 32, 33}
        or (getattr(exc, "winerror", None) is None and exc.errno == errno.EACCES)
    )


def retry_access(action, *, on_retry=None):
    """Retry only Windows access/sharing errors, for at most 5.55s of sleep."""
    for attempt in range(IO_ATTEMPTS):
        try:
            return action()
        except OSError as exc:
            if not is_access_error(exc) or attempt == IO_ATTEMPTS - 1:
                raise
            delay = min(0.05 * 2**attempt, 1.0)
            if on_retry is not None:
                on_retry({"attempt": attempt + 1, "delay_seconds": delay,
                          "errno": exc.errno, "winerror": getattr(exc, "winerror", None)})
            time.sleep(delay)


def read_json(path, *, on_retry=None):
    """Read with bounded access retries; the caller must exclude active writers."""
    return retry_access(lambda: json.loads(Path(path).read_text(encoding="utf-8")),
                        on_retry=on_retry)


def checkpoint_access_failure(log, checkpoint):
    """Classify only the final traceback error for this exact checkpoint path."""
    if not WINDOWS:
        return False
    lines = Path(log).read_text(encoding="utf-8").splitlines()
    final = next((line for line in reversed(lines) if line.strip()), "")
    codes = ("[Errno 13]", "[WinError 5]", "[WinError 32]", "[WinError 33]")
    normalized = final.replace("\\\\", "\\")
    return (final.startswith("PermissionError:") and any(code in final for code in codes)
            and repr(str(Path(checkpoint).resolve())).replace("\\\\", "\\") in normalized)
