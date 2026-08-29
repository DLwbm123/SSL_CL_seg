from __future__ import annotations

import json
from pathlib import Path

from scripts.compile_gate0_reports import audit_log


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_unlabeled_row_does_not_require_lr_field(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    _write_rows(
        path,
        [
            {
                "phase": "supervised",
                "loss_total": 0.2,
                "loss_supervised": 0.2,
                "lr": 1.0e-3,
                "hidden_gt_training_usage": "none",
            },
            {
                "phase": "unlabeled",
                "loss_total": 0.3,
                "loss_supervised": 0.1,
                "optimizer_step_executed": True,
                "teacher_forward_no_grad": True,
                "hidden_gt_training_usage": "none",
            },
        ],
    )
    result = audit_log(path)
    assert result == {"rows": 2, "unlabeled_rows": 1, "errors": []}


def test_supervised_row_requires_finite_lr(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    _write_rows(
        path,
        [
            {
                "phase": "supervised",
                "loss_total": 0.2,
                "loss_supervised": 0.2,
                "hidden_gt_training_usage": "none",
            }
        ],
    )
    assert audit_log(path)["errors"] == ["line 1: missing or non-finite supervised lr"]
