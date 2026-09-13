"""Focused review regressions: R06 helpers, finite checker, new R07 metadata aliases.

Default: imports reviewed repository modules normally. This is the Codex mode.
SSLCL_R2_SNAPSHOT: reviewer-only, hash-verified snapshot AST helper mode.
The permit is a UNIT-TEST FIXTURE, like the existing repository tests. No real
approval is written, no real runner is registered, and readers only record strings.
This does not test resistance to malicious code running in the same Python process.
"""
from __future__ import annotations
import copy
import os
from pathlib import Path
import numpy as np
import pytest
import torch

if os.environ.get('SSLCL_R2_SNAPSHOT'):
    from audit_loader import load_snapshot
    mods=load_snapshot(Path(os.environ['SSLCL_R2_SNAPSHOT']))
    integ,metrics,numerics=mods['integration'],mods['evaluate'],mods['numerics']
else:
    from experiments.lcrseg.five_frameworks_v1 import integration as integ
    from experiments.lcrseg.five_frameworks_v1 import evaluate as metrics
    from experiments.lcrseg.five_frameworks_v1 import numerics


def current_manifest(domain='current',suffix='0'):
    return {'domain':domain,'seed':161,'order':1,'stage':1,'split_id':'synthetic',
            'L':[{'image':'approved-L-'+suffix,'label':'approved-GT-'+suffix,
                  'patient_id':'l'+suffix,'domain':domain}],
            'U':[{'image':'approved-U-'+suffix,'geometry':'geometry-'+suffix,
                  'source_id':'u'+suffix,'domain':domain}]}


def permit_for(manifest):
    sha=integ.validate_current_manifest(manifest)
    bindings={'authorized_manifest_digests':[sha]}
    budget={'max_formal_optimizer_updates':0}
    # This is not an operator approval, and is never given to a native runner.
    p=integ.ExecutionPermit(bindings,('B',),budget,integ._PERMIT_SEAL)
    return p,bindings,budget


def test_ignore_changes_do_not_change_any_reported_metric():
    y=np.array([[1,255],[0,0]])
    a=np.array([[1,0],[0,0]])
    b=np.array([[1,2],[0,0]])
    assert metrics.segmentation_metrics(a,y)==metrics.segmentation_metrics(b,y)
    assert metrics.segmentation_metrics(a,y)['rim']['Dice']==1.


def test_all_ignore_has_no_valid_score():
    result=metrics.segmentation_metrics(np.zeros((2,3),int),np.full((2,3),255))
    for value in result.values():
        assert value['Dice'] is None and value['valid_pixels']==0
        assert value['HD95'] is None and value['ASSD'] is None


def test_explicit_valid_mask_intersects_ignore_mask():
    y=np.array([[1,255],[0,1]])
    valid=np.array([[True,True],[True,False]])
    result=metrics.segmentation_metrics(np.array([[1,2],[0,0]]),y,valid)
    assert result['rim']['Dice']==1. and result['rim']['valid_pixels']==2
    assert result['rim']['pred_components'] is None


def test_fully_valid_behavior_preserved():
    y=np.array([[1,1,0],[1,2,0],[0,0,0]])
    result=metrics.segmentation_metrics(y,y)
    assert all(v['Dice']==1. and v['HD95']==0 and v['ASSD']==0 for v in result.values())


def test_finite_checker_rejects_nonfinite_loss_even_with_finite_gradient():
    x=torch.tensor(1.,requires_grad=True)
    loss=x.square()+float('inf')
    g,=torch.autograd.grad(loss,x)
    assert torch.isfinite(g)
    with pytest.raises(FloatingPointError):numerics.finite(loss,'loss')


def test_finite_checker_rejects_overflowed_effective_product():
    a=torch.full((2,2),1e30)
    b=torch.full((2,2),1e30)
    numerics.finite((a,b),'factors')
    with pytest.raises(FloatingPointError):numerics.finite(b@a,'effective')


def test_unapproved_manifest_rejected_without_reader_call():
    m=current_manifest();p,_,_=permit_for(m);calls=[]
    other=current_manifest('other','9')
    with pytest.raises((ValueError,PermissionError,integ.ReviewRequired)):
        integ.CurrentDomainDataAdapter(other,p,lambda *v:calls.append(v),lambda *v:calls.append(v))
    assert calls==[]


@pytest.mark.parametrize('branch',['L','U'])
def test_R07_mutating_original_manifest_cannot_change_reader_arguments(branch):
    m=current_manifest();p,_,_=permit_for(m);calls=[]
    adapter=integ.CurrentDomainDataAdapter(m,p,lambda *v:calls.append(v),lambda *v:calls.append(v))
    approved=copy.deepcopy(m[branch][0])
    m[branch][0]['image']='unapproved-history-or-future-image'
    m[branch][0]['domain']='not-current'
    if branch=='L':m[branch][0]['label']='unapproved-label'
    try:
        getattr(adapter,'labeled' if branch=='L' else 'unlabeled')(0)
    except (ValueError,PermissionError,integ.ReviewRequired):
        assert calls==[]
        return
    expected=((approved['image'],approved['label'],approved['patient_id']) if branch=='L'
              else (approved['image'],approved['geometry'],approved['source_id']))
    assert calls==[expected], 'reader consumed metadata different from the admitted manifest'


def test_R07_mutating_external_bindings_cannot_expand_permit():
    m=current_manifest();p,bindings,_=permit_for(m)
    other=current_manifest('other','9')
    other_hash=integ.validate_current_manifest(other)
    bindings['authorized_manifest_digests'].append(other_hash)
    calls=[]
    with pytest.raises((ValueError,PermissionError,integ.ReviewRequired)):
        adapter=integ.CurrentDomainDataAdapter(other,p,lambda *v:calls.append(v),lambda *v:calls.append(v))
        adapter.unlabeled(0)
    assert calls==[]


def test_R07_mutating_external_budget_cannot_change_permit_cap():
    m=current_manifest();p,_,budget=permit_for(m)
    budget['max_formal_optimizer_updates']=1000000
    assert p.budget['max_formal_optimizer_updates']==0
