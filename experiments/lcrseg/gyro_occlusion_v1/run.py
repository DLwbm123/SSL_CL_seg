from unittest.mock import patch
from . import core
from ..gyro_fixed_projector_v1 import run as fixed

def main(cfg):
    with patch.object(fixed,'core',core):fixed.main(cfg)

def qualify(cfg):
    with patch.object(fixed,'core',core):fixed.qualify(cfg)
