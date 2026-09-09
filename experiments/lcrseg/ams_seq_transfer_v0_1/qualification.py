import argparse,gc,json,math,os,platform,time,unittest
from pathlib import Path
from collections import Counter
import torch
from . import core as c
from .contract import verify
from experiments.lcrseg.ssl_anchored_mix_v0_1.telemetry import Operations

def synthetic(output,device,reference,development=False):
    source='development' if development else verify();root=Path(output);root.mkdir(parents=True,exist_ok=False)
    os.environ['SF_DEVICE']=device;os.environ['SF_REFERENCE']=reference
    from experiments.lcrseg.tests.ams_seq_transfer_v0_1.test_contract import Contract
    from experiments.lcrseg.tests.ssl_anchored_mix_v0_1.test_contract import Contract as MathContract
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(Contract),MathContract('test_dice_and_ce'),MathContract('test_complementary_sources_odd_and_gradient')])
    started=time.time()
    with Operations(root/'operations',update_cap=600) as op:result=unittest.TextTestRunner(verbosity=2).run(suite)
    r=dict(status='PASS' if result.wasSuccessful() else 'FAIL',source=source,tests=result.testsRun,errors=len(result.errors),failures=len(result.failures),counts=dict(op.counts),device=device,python=platform.python_version(),torch=torch.__version__,seconds=time.time()-started)
    c.write_json(root/'qualification.json',r);print(json.dumps(r))
    if not result.wasSuccessful():raise SystemExit(1)

def smoke(output,device,reference,data):
    source=verify();root=Path(output);root.mkdir(exist_ok=False);dev=torch.device(device);rows=[];peak=0;reserved=0
    with Operations(root/'operations',update_cap=8) as op:
        for d in c.DOMAINS:
            with c.training_access(d):
                ds=c.DomainData(data,d,'train_labeled');patients=c.patients_for(data,d,(c.MANIFEST_SHA,c.SPLIT_SHA));lb=c.batch(ds,[0,1],dev,61,21,0,'labeled');ub={k:v[:1] for k,v in lb.items() if k!='label'}
                for a in ('T_CED','T_LCTX','T_UCTX','T_AMS'):
                    m=c.build(reference,dev,61,d,'SUP_CE');t=c.ema_from(m);opt=c.optimizer_for(m);torch.cuda.reset_peak_memory_stats(dev)
                    row=c.step(m,t,opt,lb,ub if a in ('T_UCTX','T_AMS') else None,a,61,d,21,0,Counter(),patients[:2]);peak=max(peak,torch.cuda.max_memory_allocated(dev));reserved=max(reserved,torch.cuda.max_memory_reserved(dev));rows.append(dict(domain=d,arm=a,loss=row['loss']));del m,t,opt;gc.collect();torch.cuda.empty_cache()
    r=dict(status='PASS',source=source,counts=dict(op.counts),updates=op.counts['optimizer_steps'],peak_allocated=peak,peak_reserved=reserved,admission_mib=max(2048,math.ceil(max(peak,reserved)/1024**2*1.25+512)),real_U_image_reads=0,GT_scope='first two authorized L cases per domain; pseudo-U label stripped; synthetic-role image reuse only',reused_for_formal=False,rows=rows)
    c.write_json(root/'receipt.json',r);print(json.dumps(r))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('synthetic','smoke'))
    for k in ('output','device','reference'):p.add_argument('--'+k,required=True)
    p.add_argument('--development',action='store_true');p.add_argument('--data');a=vars(p.parse_args());mode=a.pop('mode')
    if mode=='synthetic':a.pop('data');synthetic(**a)
    else:a.pop('development');smoke(**a)
