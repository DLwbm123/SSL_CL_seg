"""Review-only modes; real modes remain unconditionally locked."""
from __future__ import annotations

from .contracts import candidate_grid, require_external_review_authorization, review_contract
from .io_guards import static_audit


def synthetic_tests():
    grid = candidate_grid()
    assert len(grid) == 8 and {row["blend_lambda"] for row in grid} == {0.5, 0.75}
    return {"synthetic": True, "candidate_count": len(grid)}


def print_contract():
    return review_contract()


def run_static_audit(repo_root):
    return static_audit(repo_root)


def train(*_args, **_kwargs):
    require_external_review_authorization()


def fit(*_args, **_kwargs):
    require_external_review_authorization()


def evaluate(*_args, **_kwargs):
    require_external_review_authorization()
