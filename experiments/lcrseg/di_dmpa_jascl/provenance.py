from __future__ import annotations

import subprocess
from pathlib import Path

from . import UPSTREAM_COMMIT


OFFICIAL_ORIGINS = {
    "https://github.com/prinshul/JASCL",
    "git@github.com:prinshul/JASCL",
}


def _git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def git_revision(repo_root: str | Path) -> str:
    return _git_output(Path(repo_root), "rev-parse", "HEAD")


def assert_upstream_unchanged(reference_root: str | Path, upstream_path: str) -> None:
    """Require an independent, clean checkout of the pinned official JASCL source."""

    root = Path(reference_root).resolve()
    if not (root / ".git").exists():
        raise RuntimeError(f"official JASCL reference checkout is missing: {root}")
    observed_commit = _git_output(root, "rev-parse", "HEAD")
    if observed_commit != UPSTREAM_COMMIT:
        raise RuntimeError(f"official JASCL reference commit mismatch: {observed_commit} != {UPSTREAM_COMMIT}")
    observed_origin = _git_output(root, "remote", "get-url", "origin").removesuffix(".git")
    if observed_origin not in OFFICIAL_ORIGINS:
        raise RuntimeError(f"unexpected JASCL origin: {observed_origin}")
    source_root = (root / upstream_path).resolve()
    if not source_root.is_dir() or root not in source_root.parents:
        raise RuntimeError(f"official JASCL source path is missing or unsafe: {source_root}")
    result = subprocess.run(
        ["git", "-C", str(root), "diff", "--quiet", "HEAD", "--", upstream_path],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("official JASCL source differs from pinned commit 3c93ca7")
