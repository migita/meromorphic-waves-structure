#!/usr/bin/env python3
"""Run every check in this folder: python3 run_all.py [--verbose]

Requires Python 3 and SymPy. Each script is run from this folder; the
JSON certificates are written beside the scripts.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    "check_bracket.py",
    "check_bracket_audit.py",
    "check_sharpness.py",
    "check_ks_residue.py",
    "check_ks_family.py",
    "check_examples.py",
    "check_text_identities.py",
    "check_linear_terms.py",
    "check_more_examples.py",
    "check_drift_n2.py",
    "sh_dispersion.py",
    "tables_q2.py",
    "network_32_explicit.py",
]


def main():
    verbose = "--verbose" in sys.argv[1:]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    failed = []
    for name in SCRIPTS:
        start = time.time()
        run = subprocess.run([sys.executable, name], cwd=HERE, env=env,
                             capture_output=True, text=True)
        ok = run.returncode == 0
        print(f"{name:28s} {'ok' if ok else f'FAILED (exit {run.returncode})':18s}"
              f"{time.time() - start:6.1f} s", flush=True)
        if verbose or not ok:
            print(run.stdout, run.stderr, sep="", flush=True)
        if not ok:
            failed.append(name)
    if failed:
        print("FAILED:", ", ".join(failed))
        sys.exit(1)
    print(f"All {len(SCRIPTS)} scripts passed.")


if __name__ == "__main__":
    main()
