from unittest.mock import patch
from . import core
from ..gyro_strength_v1 import run as shared


def main(cfg):
    with patch.object(shared,'core',core):shared.main(cfg)


def qualify(cfg):
    with patch.object(shared,'core',core):shared.qualify(cfg)
