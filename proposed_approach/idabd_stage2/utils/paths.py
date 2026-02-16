from __future__ import annotations

import sys
from pathlib import Path


def ensure_zoo_importable(script_file: str | None = None) -> None:
    """
    Adds the first parent folder containing `zoo/` to sys.path.
    Call at top of scripts before importing zoo.models.
    """
    base = Path(script_file).resolve().parent if script_file else Path.cwd()
    for cand in [base, base.parent, base.parent.parent, Path.cwd(), Path.cwd().parent]:
        if (cand / "zoo").is_dir():
            sys.path.insert(0, str(cand))
            return
    # If not found, leave sys.path unchanged; import will raise a clear error.


def find_repo_root(start: Path | str | None = None) -> Path:
    """
    Find repo root by searching upward for a folder containing 'zoo/'.
    Returns a Path to the repo root.
    """
    if start is None:
        cur = Path(__file__).resolve()
    else:
        cur = Path(start).resolve()

    # If start is a file, go to its parent
    if cur.is_file():
        cur = cur.parent

    # Walk upward a few levels
    for _ in range(12):
        if (cur / "zoo").is_dir():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent

    # fallback: best guess (package root)
    return Path(__file__).resolve().parents[2]


def ensure_repo_on_syspath(repo_root: Path | str) -> Path:
    """
    Ensure repo root is on sys.path (so 'zoo' and your packages import cleanly).
    Returns repo_root as Path.
    """
    repo_root = Path(repo_root).resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def abs_from_repo(repo_root: Path | str, maybe_rel: str | Path) -> str:
    """
    Convert a repo-relative path to an absolute path string.
    If already absolute, returns it as-is.
    """
    repo_root = Path(repo_root).resolve()
    p = Path(maybe_rel)
    if p.is_absolute():
        return str(p)
    return str((repo_root / p).resolve())
