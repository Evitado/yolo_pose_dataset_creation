#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
I/O helpers for YOLO-pose export:

- Shell helpers for gsutil.
- Listing local / GCS .h5 paths.
- Cached open for HDF5 files (local temp files per GCS object).
"""

import os
import tempfile
import subprocess
import shlex
import atexit
from pathlib import Path
from typing import List, Dict

import h5py


def _sh(cmd: str, check: bool = True) -> str:
    """Run a shell command and return stdout (raise if non-zero exit when check=True)."""
    proc = subprocess.run(
        cmd,
        shell=True,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"\n[gsutil ERROR]\ncmd: {cmd}\n--- output ---\n{proc.stdout}\n--------------"
        )
    return proc.stdout


def _ensure_gsutil() -> None:
    """Ensure gsutil is on PATH, otherwise raise."""
    try:
        _sh("command -v gsutil")
    except RuntimeError:
        raise RuntimeError(
            "gsutil not found on PATH. Install Google Cloud SDK and run `gcloud auth login`."
        )


def list_h5_paths(source: str) -> List[str]:
    """
    List all .h5 files under a GCS prefix or local directory.

    - For 'gs://...' prefixes, uses 'gsutil ls -r'.
    - For local paths, uses Path.rglob.
    """
    if source.startswith("gs://"):
        _ensure_gsutil()
        prefix = source.rstrip("/")

        out1 = _sh(f"gsutil ls -r {shlex.quote(prefix + '/**/*.h5')}", check=False)
        out2 = _sh(f"gsutil ls -r {shlex.quote(prefix + '/**/*.H5')}", check=False)

        paths = [
            ln.strip()
            for ln in (out1 + "\n" + out2).splitlines()
            if ln.strip().startswith("gs://") and ln.strip().lower().endswith(".h5")
        ]
        if not paths:
            ver = _sh("gsutil version", check=False).strip()
            print(f"[diag] gsutil: {ver}\n[diag] tried: {prefix}/**/*.h5 and **/*.H5")
        return sorted(set(paths))

    # local directory
    p = Path(source)
    return [str(x) for x in sorted(list(p.rglob("*.h5")) + list(p.rglob("*.H5")))]


# =========================
# Cached open for GCS HDF5
# =========================

_H5_LOCAL_CACHE: Dict[str, str] = {}
_CLEANUP: List[str] = []


@atexit.register
def _cleanup_cache() -> None:
    """Remove all temporary local copies created for GCS .h5 files."""
    for tmp in _CLEANUP:
        try:
            os.remove(tmp)
        except OSError:
            pass
        d = os.path.dirname(tmp)
        try:
            os.rmdir(d)
        except OSError:
            pass


def open_h5_any(path: str) -> h5py.File:
    """
    Open local or GCS .h5. For GCS, download once per object (cached).

    Returns:
        h5py.File opened in read mode.
    """
    if not path.startswith("gs://"):
        return h5py.File(path, "r")

    _ensure_gsutil()
    if path not in _H5_LOCAL_CACHE:
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, Path(path).name)
        _sh(f"gsutil cp {shlex.quote(path)} {shlex.quote(temp_file_path)}")
        _H5_LOCAL_CACHE[path] = temp_file_path
        _CLEANUP.append(temp_file_path)

    return h5py.File(_H5_LOCAL_CACHE[path], "r")
