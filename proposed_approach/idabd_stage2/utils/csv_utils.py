from __future__ import annotations
import os
from os import path
import csv
from typing import Sequence

def ensure_csv(csv_path: str, fieldnames: Sequence[str], overwrite: bool = False) -> None:
    if not csv_path:
        return
    out_dir = path.dirname(csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if overwrite and path.exists(csv_path):
        os.remove(csv_path)
    if not path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(fieldnames))
            w.writeheader()
            f.flush()

def append_csv_row(csv_path: str, fieldnames: Sequence[str], row: dict) -> None:
    if not csv_path:
        return
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writerow(row)
        f.flush()
