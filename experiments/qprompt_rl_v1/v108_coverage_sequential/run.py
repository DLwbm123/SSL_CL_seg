"""Same sequential experiment with the frozen V107 training table."""
import json
import os
import time
import traceback
from pathlib import Path
import numpy as np


def validate(data,keys):
    assert len(keys)==80 and len({tuple(k) for k in keys})==80
    assert len({k[0] for k in keys})==20 and set(k[1] for k in keys)=={1,2} and set(k[2] for k in keys)=={100,200}
    for context in {k[0] for k in keys}:
        assert {(k[1],k[2]) for k in keys if k[0]==context}=={(s,t) for s in (1,2) for t in (100,200)}
    for name,shape in [('original',(80,24)),('probes',(80,4,24)),('returns',(80,12))]:
        assert data[name].shape==shape and np.isfinite(data[name]).all()


def selfcheck():
    keys=[(str(i),s,t) for i in range(20) for s in (1,2) for t in (100,200)]
    data=dict(original=np.zeros((80,24)),probes=np.zeros((80,4,24)),returns=np.zeros((80,12)));validate(data,keys)
    for bad in (keys[:-1],keys[:-1]+[keys[0]]):
        try:validate(data,bad)
        except AssertionError:pass
        else:raise AssertionError('invalid keys accepted')
    data['returns'][0,0]=np.nan
    try:validate(data,keys)
    except AssertionError:pass
    else:raise AssertionError('nonfinite returns accepted')
    N.selfcheck()


def training(root,cfg):
    src=Path(cfg['v107']);assert N.D.read(src/'FINAL.json')['status']=='COMPLETE' and N.D.read(src/'DATASET_LOCK.json')['status']=='SEALED'
    data=np.load(src/'DATASET.private.npz');keys=N.D.read(src/'KEYS.private.json');validate(data,keys)
    assert np.array_equal(data['keys'],np.array(keys,dtype=str))
    oldkeys,oldoriginal,oldreturns,_,_=N.D.dataset(cfg,'V101');old=np.load(Path(cfg['v103'])/'FEATURES.private.npz')
    assert [tuple(k) for k in keys[:32]]==oldkeys and np.array_equal(data['returns'][:32],oldreturns)
    assert np.array_equal(data['original'][:32],oldoriginal) and np.array_equal(data['probes'][:32],old['probes'])
    N.D.write(root/'DATASET_ADMISSION.json',dict(status='PASS',states=80,returns=960,old_arrays_preserved=True,fit_inputs='training_only',query_calls=0))
    return keys,data['probes'].mean(1),data['returns']


if __name__=='__main__':
    import importlib.util
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
    cfg=json.loads(Path(os.environ['EXEC_CONFIG']).read_text());root=Path(os.environ['EXEC_RUN'])
    N=load('sequential',cfg['sequential_entry']);N.STAGE='V108'
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior import core,stage_b
    N.c=core;N.b=stage_b
    for name,key in [('B','base_entry'),('D','diagnostic_entry'),('V','v99_entry'),('H','v104_entry'),('M','v105_entry'),('P','probe_entry'),('E','expanded_entry')]:setattr(N,name,load(name,cfg[key]))
    if os.environ.get('EXEC_SELFCHECK')=='1':selfcheck();print('PASS dataset validation and inherited decision/matrix/action checks; zero optimizer/query')
    else:
        N.D.write(root/'STARTED.json',dict(pid=os.getpid(),time=time.time(),commit=cfg['commit']))
        try:
            if os.environ.get('EXEC_MODE')=='job':
                if cfg['job']=='fit':N.fit(root,cfg,training=training(root,cfg))
                else:N.gpu_job(root,cfg)
            else:selfcheck();N.coordinator(root,cfg)
        except BaseException as exc:N.D.write(root/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=time.time()));raise
