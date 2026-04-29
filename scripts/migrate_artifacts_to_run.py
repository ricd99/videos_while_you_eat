#!/usr/bin/env python3
"""
Migrate canonical artifacts from artifacts/ into a per-run folder artifacts/run-<timestamp>/
and create artifacts/latest_run.txt pointing to the new run.

Phase 2: This script moves runtime artifacts into a canonical per-run folder and
lets the serving code load from that location by default.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
RUN_ID = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
TARGET_RUN_DIR = ARTIFACTS_ROOT / RUN_ID

def main():
    if not ARTIFACTS_ROOT.exists():
        print(f"Artifacts root not found: {ARTIFACTS_ROOT}")
        return 1
    TARGET_RUN_DIR.mkdir(parents=True, exist_ok=False)

    files_to_move = [
        "nn_model.pkl",
        "embeddings.pkl",
        "df_lookup.pkl",
        "feature_columns.json",
        "preprocessing.pkl",
    ]

    moved = []
    for fname in files_to_move:
        src = ARTIFACTS_ROOT / fname
        if src.exists():
            dst = TARGET_RUN_DIR / fname
            shutil.move(str(src), str(dst))
            moved.append(fname)
        else:
            print(f"Warning: expected artifact not found at {src}")

    # Write latest_run marker
    (ARTIFACTS_ROOT / "latest_run.txt").write_text(RUN_ID)
    print(f"Moved artifacts to {TARGET_RUN_DIR}")
    print(f"New latest_run.txt -> {RUN_ID}")
    if moved:
        print(f"Moved: {', '.join(moved)}")
    else:
        print("No artifacts moved (none found to relocate).")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
