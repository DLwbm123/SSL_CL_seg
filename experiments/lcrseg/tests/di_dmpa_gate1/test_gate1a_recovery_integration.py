"""Opt-in full known-failure audit, bound to a published exact recovery commit."""
import os
from pathlib import Path

import pytest

from di_dmpa_gate1.recovery import known_failure_localization


def test_known_failure_checkpoint_full_registered_coordinate_audit():
    destination=os.environ.get('GATE1A_RECOVERY_LOCALIZATION_OUTPUT')
    if not destination:
        pytest.skip('Explicit published recovery commit and unique localization destination required')
    root=Path(__file__).resolve().parents[2]
    audit=known_failure_localization(root,'/root/LCRSeg',destination,os.environ['GATE1A_CODE_COMMIT'])
    assert audit['complete_registered_coordinate_coverage'] and audit['bitwise_unchanged']
    assert audit['model_optimizer_steps']==audit['transport_optimizer_steps']==audit['clustering_jobs']==0
    assert audit['checkpoint_sha256_before']==audit['checkpoint_sha256_after']
    assert audit['localization_status'] in ('PASS_FALSE_POSITIVE_FULL_MAP_SCOPE_CONFIRMED',
        'BLOCKED_REGISTERED_ZERO_FEATURE','BLOCKED_NUMERICAL_FAILURE')
    # A correctly detected real numerical block is not a test implementation
    # failure and NEVER authorizes formal attempt2. Preserve the audit verdict.
    assert audit['attempt2_authorized']==(audit['localization_status']=='PASS_FALSE_POSITIVE_FULL_MAP_SCOPE_CONFIRMED')
