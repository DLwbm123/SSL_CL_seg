import argparse,gc,json,os,platform,time,unittest
from pathlib import Path
import torch
from .core import *
from .freeze import verify
from .telemetry import Operations,flush

def synthetic(output,device,reference,development=False):
    source='development' if development else verify();root=Path(output);root.mkdir(parents=True)
    os.environ['SF_DEVICE']=device;os.environ['SF_REFERENCE']=reference
    from experiments.lcrseg.tests.ssl_anchored_mix_v0_1.test_contract import Contract
    start=time.time()
    with Operations(root,update_cap=200) as op:
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contract))
    receipt=dict(status='PASS' if result.wasSuccessful() else 'FAIL',source=source,tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),counts=dict(op.counts),synthetic_optimizer_updates=op.counts['optimizer_steps'],device=device,python=platform.python_version(),torch=torch.__version__,seconds=time.time()-start)
    write_json(root/'qualification.json',receipt);print(json.dumps(receipt))
    if not result.wasSuccessful():raise SystemExit(1)

def smoke(output,device,reference,data):
    source=verify();root=Path(output);root.mkdir();device=torch.device(device);rows=[];ct=Counter()
    with Operations(root,update_cap=8) as op:
        for d in DOMAINS:
            ds=DomainData(data,d,'train_labeled');lb=batch(ds,[0,1],device,31,21,0,'smoke_l');ub={k:v for k,v in lb.items() if k!='label'}
            # Use singleton pseudo-U to exercise the real-resolution odd-batch path within 8 updates.
            ub={k:v[:1] for k,v in ub.items()}
            for a in ARMS:
                m=build(reference,device,31,d,a);t=ema_from(m);opt=optimizer_for(m)
                row=train_step(m,t,opt,lb,ub if config(a)['ssl'] else None,None,None,a,31,d,21,0,ct)
                assert math.isfinite(row['loss']) and any(p.grad is not None for p in m.parameters())
                rows.append(dict(domain=d,arm=a,**row));append(root/'smoke_steps.jsonl',rows[-1]);flush('smoke_step');del m,t,opt;gc.collect()
    receipt=dict(status='PASS',source=source,updates=ct['optimizer_steps'],maximum=8,counts=dict(op.counts),training_counters=dict(ct),rows=rows,real_U_images_opened=0,GT_scope='first two sorted current labeled cases per domain; singleton pseudo-U has labels stripped',initializations_reused=False)
    write_json(root/'receipt.json',receipt);print(json.dumps(receipt))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['synthetic','smoke'])
    for k in ['output','device','reference']:p.add_argument('--'+k,required=True)
    p.add_argument('--data');p.add_argument('--development',action='store_true');a=vars(p.parse_args());mode=a.pop('mode')
    if mode=='synthetic':a.pop('data');synthetic(**a)
    else:a.pop('development');smoke(**a)
