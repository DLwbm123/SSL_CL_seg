from __future__ import annotations

import torch

from lcrseg.engine.checkpoint import checkpoint_payload, load_checkpoint, save_checkpoint
from lcrseg.models import UNet2D


def test_checkpoint_schema_and_reload(tmp_path) -> None:
    model = UNet2D(3, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    payload = checkpoint_payload(
        method_name="test",
        method_version="0.1",
        git_commit="NO_GIT_WORKTREE",
        config_resolved={"seed": 123},
        site_id="SITE",
        site_index=0,
        epoch=0,
        site_step=1,
        global_step=1,
        current_model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state={},
        scaler_state={},
        current_anchor_state={},
        historical_anchor_state={},
        bootstrap_state={},
        method_statistics={},
        data_split_hash="split",
        manifest_hash="manifest",
    )
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, payload)
    restored = load_checkpoint(path)
    assert restored["method_name"] == "test"
    assert restored["current_model_state"].keys() == model.state_dict().keys()
