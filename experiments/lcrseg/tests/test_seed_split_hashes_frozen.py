import hashlib
import os
from pathlib import Path

from lcrseg.methods.lcrseg_v0_3 import FROZEN_FUNDUS_MANIFEST_HASHES, FROZEN_FUNDUS_SPLIT_HASHES


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_seed_split_hashes_frozen() -> None:
    assert set(FROZEN_FUNDUS_SPLIT_HASHES) == {0, 1, 2}
    assert set(FROZEN_FUNDUS_MANIFEST_HASHES) == {0, 1, 2}
    root = Path(os.environ.get("LCRSEG_DATA_ROOT", "/home/jiangsuiyang/SSL_CL"))
    if not root.is_dir():
        return
    for seed in (0, 1, 2):
        assert _sha(root / "splits" / f"fundus_seed{seed}.json") == FROZEN_FUNDUS_SPLIT_HASHES[seed]
        assert _sha(root / "manifests/training" / f"lcrseg_v1_seed{seed}.csv") == FROZEN_FUNDUS_MANIFEST_HASHES[seed]

