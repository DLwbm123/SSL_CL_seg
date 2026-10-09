"""Matched13-action fits and real trajectories using sealed OFF-prefix coverage."""
import json
import os
import time
import traceback
from pathlib import Path
import numpy as np
import torch


def fresh(v,seed):
    class Thirteen(N.c.Actor):
        def forward(self,z,memory=True):
            logits=self.net(z)
            if not memory:logits=logits.masked_fill(torch.arange(13,device=z.device)>=3,-torch.inf)
            return logits
    rng=torch.get_rng_state();actor=Thirteen(seed)
    with torch.random.fork_rng(devices=[]):actor.net[-1]=torch.nn.Linear(32,13)
    torch.nn.init.zeros_(actor.net[-1].weight);torch.nn.init.zeros_(actor.net[-1].bias)
    assert torch.equal(rng,torch.get_rng_state()) and sum(p.numel() for p in actor.parameters())==1229
    return actor


def validate(data,keys):
    assert len(keys)==len({tuple(k) for k in keys})==120 and len({k[0] for k in keys})==20
    expected={(s,t,p) for s in (1,2) for t,p in ((100,'ORIGINAL'),(200,'ORIGINAL'),(200,'OFF'))}
    for context in {k[0] for k in keys}:assert {(k[1],k[2],k[3]) for k in keys if k[0]==context}==expected
    for name,shape in [('original',(120,24)),('probes',(120,4,24)),('returns',(120,13))]:assert data[name].shape==shape and np.isfinite(data[name]).all()


def selfcheck():
    keys=[(str(i),s,t,p) for i in range(20) for s in (1,2) for t,p in ((100,'ORIGINAL'),(200,'ORIGINAL'),(200,'OFF'))]
    data=dict(original=np.zeros((120,24)),probes=np.zeros((120,4,24)),returns=np.zeros((120,13)));validate(data,keys)
    for bad in (keys[:-1],keys[:-1]+[keys[0]]):
        try:validate(data,bad)
        except AssertionError:pass
        else:raise AssertionError('invalid keys accepted')
    data['returns'][0,0]=np.nan
    try:validate(data,keys)
    except AssertionError:pass
    else:raise AssertionError('nonfinite return accepted')
    N.selfcheck();a=fresh(N.V,601);other=fresh(N.V,601);old=N.V.model12(601)
    assert all(torch.equal(p,other.state_dict()[k]) for k,p in a.state_dict().items()) and torch.equal(a.net[0].weight,old.net[0].weight)
    assert a(torch.zeros(24)).shape==(13,) and torch.isneginf(a(torch.zeros(24),memory=False)[3:]).all()
    pred=dict(mean=np.zeros(24),scale=np.ones(24),time_actions=np.array([12,11]));state=torch.zeros(24);g=torch.Generator()
    state[0]=1/3;assert N.select(state,'TIME',{},pred,11,g)[0]==12
    state[0]=2/3;assert N.select(state,'TIME',{},pred,11,g)[0]==11 and N.select(state,'OFF',{},pred,11,g)[0]==12
    assert N.distribution(state,'UNIFORM',{},pred,11).shape==(13,) and len(N.METHODS)==22 and N.CAPS['development']==52800
    x=torch.zeros((3,24));x[1]=1;returns=torch.arange(39,dtype=torch.float32).reshape(3,13)/100
    mean,std,adv,scale=N.H.prepare(x,returns);changed=returns.clone();changed[:,12]+=100
    m2,s2,_,scale2=N.H.prepare(x,changed);assert torch.equal(mean,m2) and torch.equal(std,s2) and torch.equal(scale,scale2)


def training(root,cfg):
    src=Path(cfg['v110']);assert N.D.read(src/'FINAL.json')['status']=='COMPLETE' and N.D.read(src/'DATASET_LOCK.json')['status']=='SEALED'
    data=np.load(src/'DATASET.private.npz');keys=N.D.read(src/'KEYS.private.json');validate(data,keys)
    assert np.array_equal(data['keys'],np.array(keys,dtype=str))
    old=np.load(Path(cfg['v107'])/'DATASET.private.npz');oldreturns=np.load(Path(cfg['v109'])/'RETURNS13.private.npz')
    assert np.array_equal(data['returns'][:80],oldreturns['returns']) and np.array_equal(data['original'][:80],old['original']) and np.array_equal(data['probes'][:80],old['probes'])
    N.D.write(root/'DATASET_ADMISSION.json',dict(status='PASS',states=120,actions=13,returns=1560,old80_arrays_preserved=True,fit_inputs='training_only',state_weight=1/120,step100_total_weight=1/3,step200_total_weight=2/3,advantage_baseline_columns=list(range(9)),query_calls=0))
    return keys,data['probes'].mean(1),data['returns']


def configure(cfg):
    global N
    import importlib.util
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
    N=load('sequential',cfg['sequential_entry']);N.STAGE='V111';N.ACTION_COUNT=13
    N.METHODS+=('TIME','OFF');N.CONTROL_NAMES+=('TIME','OFF');N.TRADE_CONTROLS=N.CONTROL_NAMES
    N.CAPS['development']=52800;N.QUALIFICATION_ACTIONS=((0,9),(7,12))
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior import core,stage_b
    N.c=core;N.b=stage_b;core.CAPS['development']=4400
    for name,key in [('B','base_entry'),('D','diagnostic_entry'),('V','v99_entry'),('H','v104_entry'),('M','v105_entry'),('P','probe_entry'),('E','expanded_entry')]:setattr(N,name,load(name,cfg[key]))
    N.H.fresh=fresh;N.EXTRA_ACTION_INSTALLER=load('off_action',cfg['off_entry']).install_off


if __name__=='__main__':
    cfg=json.loads(Path(os.environ['EXEC_CONFIG']).read_text());root=Path(os.environ['EXEC_RUN']);configure(cfg)
    if os.environ.get('EXEC_SELFCHECK')=='1':selfcheck();print('PASS13-action validation/constructors/control routing/unchanged baseline/decision gates; zero model updates or queries')
    else:
        N.D.write(root/'STARTED.json',dict(pid=os.getpid(),time=time.time(),commit=cfg['commit']))
        try:
            if os.environ.get('EXEC_MODE')=='job':
                if cfg['job']=='fit':N.fit(root,cfg,training=training(root,cfg))
                else:N.gpu_job(root,cfg)
            else:selfcheck();N.coordinator(root,cfg)
        except BaseException as exc:N.D.write(root/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=time.time()));raise
