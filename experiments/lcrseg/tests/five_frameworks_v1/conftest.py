import os
from pathlib import Path
import pytest
import torch
from experiments.lcrseg.five_frameworks_v1.losses import CWMI,MissingBackend


def pytest_sessionstart(session):
    torch.set_num_threads(1)
    from experiments.lcrseg.tests.five_frameworks_v1.cost_audit import install
    install()


@pytest.fixture(scope='session')
def dependency_root():
    root=os.environ.get('SSLCL5_DEP_ROOT')
    if not root:pytest.fail('SSLCL5_DEP_ROOT must point to locked author sources; backend tests are not skipped')
    return Path(root)


@pytest.fixture(scope='session')
def cwmi(dependency_root):return CWMI(dependency_root/'CWMI')
