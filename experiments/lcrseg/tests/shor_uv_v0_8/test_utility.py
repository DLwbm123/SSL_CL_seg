import inspect
import json
import numpy as np
import pytest
from shor_uv_v0_8.features import extract,choose,deploy,NAMES,predict
from shor_uv_v0_8.learning import (fit,patient_weights,inner_folds,train_nested,
                                 select_threshold,selection_metrics,LAMBDAS)


def probability(hard):
    return np.eye(3,dtype=np.float32)[hard].transpose(2,0,1)


def synthetic(n=90):
    rng=np.random.default_rng(713)
    x=rng.normal(size=(n,2,18))
    y=np.zeros((n,2,3));y[:,:,0]=.08*x[:,:,0];y[:,:,1]=.08*x[:,:,1]
    y[:,:,2]=np.maximum(0,-y[:,:,:2].min(2))
    patients=np.array(['p%03d'%i for i in range(n)])
    routes=np.arange(n)%3
    seeds=np.arange(n)%3;domains=np.arange(n)//3%3
    return x,y,patients,routes,seeds,domains,np.ones(n,dtype=bool)


def test_independent_feature_witness():
    c=probability(np.array([[0,1],[2,0]]))
    h=probability(np.array([[1,1],[2,0]]))
    a=np.array([.6,.1,.3])
    x=extract(c,h,a,0)
    assert len(NAMES)==18 and x.dtype==np.float64
    np.testing.assert_allclose(x[:6],[.25,.25,.5,.25,np.log(3/2),0])
    np.testing.assert_allclose(x[6:12],[0,0,1,1,1,1])
    assert x[12]==pytest.approx(np.log(2)/3)
    assert x[13]==pytest.approx(1/3)
    np.testing.assert_allclose(x[14:16],[2/3,1])
    assert x[16]==pytest.approx(np.log(.6+1e-12)-np.log(.3+1e-12))
    assert x[17]==pytest.approx(.3)


def test_empty_foreground_and_soft_zero():
    p=probability(np.zeros((3,4),int))
    x=extract(p,p,[.1,.2,.7],0)
    assert np.isfinite(x).all()
    np.testing.assert_array_equal(x[14:16],[1,1])
    assert x[13]==0


@pytest.mark.parametrize('bad',['nan','negative','sum','alpha'])
def test_invalid_probability_rejected(bad):
    p=probability(np.zeros((2,2),int));h=p.copy();a=[.2,.2,.6]
    if bad=='nan':h[0,0,0]=np.nan
    if bad=='negative':h[1,0,0]=-.1
    if bad=='sum':h[0,0,0]=.5
    if bad=='alpha':a=[.2,.2,.2]
    with pytest.raises(ValueError):extract(p,h,a,0)


def test_deployment_rejects_identity_gt_and_noop_preserves_object():
    p=probability(np.zeros((2,2),int))
    for key in ('GT','domain','patient_id','seed','case_id','fold','path'):
        with pytest.raises(TypeError):extract(p,p,[.2,.2,.6],0,**{key:1})
        with pytest.raises(TypeError):choose(0,np.zeros(18),None,None,**{key:1})
    assert deploy(p,p,[.2,.2,.6],2,None,None) is p
    assert choose(1,np.zeros(18),None,None)==2
    assert choose(2,None,None,dict(kind='confidence',threshold=None))==2


def test_threshold_strict_gain_inclusive_harm_and_confidence():
    m=dict(mean=[0]*18,scale=[1]*18,coef=np.zeros((18,3)).tolist(),intercept=[.005,.005,.025])
    c=dict(kind='cf',epsilon=.005,harm_limit=.025)
    assert choose(0,np.zeros(18),m,c)==2
    c['epsilon']=0
    assert choose(0,np.zeros(18),m,c)==0
    c['harm_limit']=.01
    assert choose(0,np.zeros(18),m,c)==2
    assert choose(1,np.zeros(18),None,dict(kind='confidence',threshold=0))==1


def test_weighted_ridge_independent_augmented_solution():
    rng=np.random.default_rng(2);x=rng.normal(size=(12,18));x[:,3]=7
    y=rng.normal(size=(12,3));ids=np.array(['a']*4+['b']*5+['c']*3);w=patient_weights(ids)
    m=fit(x,y,w,10.)
    assert w.sum()==pytest.approx(3)
    z=(x-np.array(m['mean']))/np.array(m['scale'])
    design=np.column_stack([np.ones(12),z])
    penalty=np.diag([0]+[10.]*18)
    expected=np.linalg.solve(np.einsum('ni,n,nj->ij',design,w,design)+penalty,np.einsum('ni,n,nj->ij',design,w,y))
    np.testing.assert_allclose(m['intercept'],expected[0],atol=1e-12)
    np.testing.assert_allclose(m['coef'],expected[1:],atol=1e-12)
    assert m['scale'][3]==1


def test_patient_duplicate_invariance():
    rng=np.random.default_rng(3);x=rng.normal(size=(5,18));y=rng.normal(size=(5,3))
    ids=np.array(list('abcde'))
    first=fit(x,y,patient_weights(ids),1.)
    ix=np.array([0,0,0,1,2,3,4])
    second=fit(x[ix],y[ix],patient_weights(ids[ix]),1.)
    np.testing.assert_allclose(first['coef'],second['coef'],atol=1e-12)
    np.testing.assert_allclose(first['mean'],second['mean'],atol=1e-12)


def test_patient_inner_assignment_and_training_scaler_exclusion():
    ids=np.array(['b','a','b','c','d','e','a'])
    f=inner_folds(ids)
    assert f[0]==f[2] and f[1]==f[-1]
    x,y,p,*_=synthetic()
    train=np.arange(60)
    before=fit(x.reshape(-1,18)[:60],y.reshape(-1,3)[:60],np.ones(60),1)
    x.reshape(-1,18)[60:]=1e9;y.reshape(-1,3)[60:]=-1e9
    after=fit(x.reshape(-1,18)[:60],y.reshape(-1,3)[:60],np.ones(60),1)
    assert before==after


def test_inner_fit_patient_and_target_isolation(monkeypatch):
    import shor_uv_v0_8.learning as mod
    x,y,p,r,s,d,e=synthetic()
    x[:,:,17]=np.arange(len(x))[:,None]
    original=mod.fit;calls=[]
    def observed(xx,yy,w,lam):
        calls.append(set(xx[:,17].astype(int)))
        return original(xx,yy,w,lam)
    monkeypatch.setattr(mod,'fit',observed)
    result=train_nested(x,y,p,r,s,d,e)
    f=inner_folds(p)
    assert len(calls)==17
    for k,seen in enumerate(calls[:16]):assert seen==set(np.flatnonzero(f!=k%4))
    assert calls[-1]==set(range(len(x)))
    assert sum(z['actual_design_solves'] for z in result['fits'])==17


def test_lambda_tie_larger_and_gain_shared():
    x,y,p,r,s,d,e=synthetic();x[:]=0;y[:]=0
    result=train_nested(x,y,p,r,s,d,e)
    assert result['selected_lambda']==100
    assert len(result['fits'])==17 and set(result['choices'])=={'cf','gain'}


def test_routed_only_no_data_fallback_and_all_ignore():
    x,y,p,r,s,d,e=synthetic();r[:]=2
    result=train_nested(x,y,p,r,s,d,e,True)
    assert result['model'] is None
    assert sum(f['actual_design_solves'] for f in result['fits'])==0
    e[:]=False
    result=train_nested(x,y,p,r,s,d,e)
    assert result['model'] is None


def test_missing_group_and_no_safe_candidate_retained():
    x,y,p,r,s,d,e=synthetic();r[:]=0;y[:,:,:2]=-.3;y[:,:,2]=.3
    pred=np.zeros_like(y);pred[:,:,:2]=1;pred[:,:,2]=0
    chosen,allc=select_threshold(pred,x,y,r,s,d,e,'cf')
    assert chosen is None and len(allc)==8
    assert not any(c['metrics']['safe'] for c in allc)
    assert not selection_metrics(np.zeros((len(x),3)),s,np.zeros(len(x),int),e)['support']


def test_fixed_grid_and_target_outer_not_in_training_signature():
    assert LAMBDAS==(.1,1.,10.,100.)
    assert list(inspect.signature(fit).parameters)==['x','y','weights','regularization']
    assert 'outer_targets' not in inspect.signature(train_nested).parameters


def test_seal_rejects_mutation(tmp_path,monkeypatch):
    import shor_uv_v0_8.pipeline as pipe
    monkeypatch.setattr(pipe,'source_state',lambda:'synthetic')
    p=tmp_path/'prediction.json';p.write_text('[]')
    pipe.seal(tmp_path,'seal.json',[p],GT_reads=0)
    pipe.verify(tmp_path,'seal.json')
    p.write_text('[1]')
    with pytest.raises(ValueError):pipe.verify(tmp_path,'seal.json')


def test_nested_deployment_complete_synthetic_patient_flow():
    from care_hr_v0_7_1.scoring_r1 import case_metrics
    x,y,p,r,s,d,e=synthetic()
    outer=np.arange(len(p))%5
    decisions=np.full(len(p),-1)
    for f in range(5):
        train=outer!=f;test=~train
        model=train_nested(x[train],y[train],p[train],r[train],s[train],d[train],e[train])
        assert not set(p[train])&set(p[test])
        for i in np.flatnonzero(test):
            h=r[i]
            decisions[i]=choose(h,x[i,h] if h<2 else np.zeros(18),model['model'],model['choices']['cf'])
    assert np.isin(decisions,[0,1,2]).all()
    assert np.all((decisions==2)|(decisions==r))


def test_all_history_hashes_protected():
    from pathlib import Path
    from care_hr_v0_7_1.io_r1 import read_json,verify_files
    root=Path(__file__).resolve().parents[4]
    protected=read_json(root/'experiments/lcrseg/docs/shor_uv_v0_8/HISTORY_PROTECTION.json')
    assert len(protected)==193
    verify_files(root,protected)


def test_cross_seed_outer_patient_leak_rejected():
    from shor_uv_v0_8.pipeline import validate_folds
    rows=[dict(patient_id=str(i),fold=i) for i in range(5)]
    rows.append(dict(patient_id='0',fold=1))
    with pytest.raises(ValueError,match='patient crosses'):validate_folds(rows)
