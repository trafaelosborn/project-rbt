"""Excluded operational probe: no experiment data is read or changed."""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
p.add_argument('--seconds', type=int, default=90)
a = p.parse_args()
k = ctypes.WinDLL('kernel32', use_last_error=True)
k.GetCurrentProcess.restype = wintypes.HANDLE
k.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
k.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
inside = wintypes.BOOL()
assert k.IsProcessInJob(k.GetCurrentProcess(), None, ctypes.byref(inside))
buf = ctypes.create_string_buffer(144)
used = wintypes.DWORD()
ok = k.QueryInformationJobObject(None, 9, buf, 144, ctypes.byref(used))
flags = int.from_bytes(buf.raw[16:20], 'little') if ok else None
base = {'pid': os.getpid(), 'parent_pid': os.getppid(), 'in_job': bool(inside.value),
        'job_flags': flags, 'started_utc': datetime.now(timezone.utc).isoformat()}
for i in range(a.seconds + 1):
    temp = a.output.with_suffix('.tmp')
    temp.write_text(json.dumps({**base, 'tick': i, 'updated_utc': datetime.now(timezone.utc).isoformat(),
                               'complete': i == a.seconds}, indent=2) + '\n')
    os.replace(temp, a.output)
    if i < a.seconds:
        time.sleep(1)
