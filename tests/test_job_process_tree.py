"""Cancellation stops a real Windows venv launcher and its Python descendant."""

import ctypes
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from audio_studio.jobs import stop_worker_process


@pytest.mark.skipif(os.name != "nt", reason="Windows venv launcher regression")
def test_windows_worker_cancellation_leaves_no_running_python_child(tmp_path: Path):
    marker = tmp_path / "child-pid.txt"
    code = "import subprocess,sys,time;from pathlib import Path; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)']);Path(sys.argv[1]).write_text(str(p.pid));p.wait()"
    process = subprocess.Popen([sys.executable, "-c", code, str(marker)], creationflags=subprocess.CREATE_NO_WINDOW)

    def running(identity):
        kernel = ctypes.windll.kernel32
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x00100000, False, identity)
        if not handle:
            return False
        try:
            return kernel.WaitForSingleObject(ctypes.c_void_p(handle), 0) == 0x00000102
        finally:
            kernel.CloseHandle(ctypes.c_void_p(handle))

    try:
        deadline = time.monotonic() + 15
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(.05)
        assert marker.exists()
        child = int(marker.read_text())
        assert running(child)
        stop_worker_process(process)
        process.wait(timeout=10)
        assert not running(child)
    finally:
        stop_worker_process(process)
