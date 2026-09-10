"""Bounded numerical, exact-parent and resume qualification; no validation selection."""
import argparse,gc,json,math,os,tempfile,time,platform,unittest
from pathlib import Path
from collections import Counter
import torch
from . import core as c,engine as e
from .contract import verify,protocol,task_by_id
from experiments.lcrseg.ssl_anchored_mix_v0_1.telemetry import Operations

def suite(reference,device):
    os.environ['SF_REFERENCE']=reference;os.environ['SF_DEVICE']=str(device)
    from experiments.lcrseg.tests.ams_seq_transfer_v0_1.test_contract import fixture
    class Checks(unittest.TestCase):
        def test_parameters_and_parent(self):
            with tempfile.TemporaryDirectory() as tmp:
                expected=fixture(tmp);d=c.DOMAINS[1];ds=c.DomainData(tmp,d,'train_labeled',expected=expected,shape=(24,24));lb=c.batch(ds,[0,1],device,61,21,0,'labeled');patients=['p0','p1']
                for arm in c.ARMS[1:]:
                    m=c.build(reference,device,61,c.DOMAINS[0],'SUP_CE');t=c.ema_from(m)
                    tensors={n:c.basis(m.get_parameter(n),61,c.DOMAINS[0],n)[0] for n in c.LAYERS};load=lambda n:tensors[n]
                    with torch.no_grad():initial=m(lb['image'],stochastic_classifier=False)[0]
                    c.configure(m,arm,load);frozen=e.frozen_hash(m);opt=c.optimizer_for(m)
                    with torch.no_grad():self.assertTrue(torch.equal(initial,m(lb['image'],stochastic_classifier=False)[0]))
                    for i in range(1 if arm=='F_CONV' else 3):
                        row=c.step(m,t,opt,lb,61,d,21,i,Counter(),patients,arm)
                        self.assertEqual(row['U_image_records'],0);self.assertEqual(row['anchor_label_records'],2);self.assertEqual(row['labeled_source_scoring_multiplicity'],1)
                        if arm.startswith('LR_'):
                            for n in c.LAYERS:
                                layer=m.get_submodule(n[:-7]);self.assertGreater(float(layer.B.grad.norm()),0)
                                if i>0:self.assertGreater(float(layer.A.grad.norm()),0)
                    self.assertEqual(e.frozen_hash(m),frozen);self.assertIsNone(m.decoder.conv_logit.mu.weight.grad)
                    c.diagnostics(m,load,20,arm)
                    with torch.no_grad():
                        before=m(lb['image'],stochastic_classifier=False)[0];c.merge(m);after=m(lb['image'],stochastic_classifier=False)[0]
                    self.assertLessEqual(float((before-after).abs().max()),1e-5);del m,t,opt,tensors;gc.collect()
                results=[]
                for new in [False,True]:
                    m=c.build(reference,device,61,c.DOMAINS[0],'SUP_CE');t=c.ema_from(m);opt=c.optimizer_for(m);rows=[]
                    for ep in [1,21]:
                        before_rng=torch.get_rng_state().clone()
                        row=c.step(m,t,opt,lb,61,d,ep,0,Counter(),patients,'F_FULL') if new else c.parent.step(m,t,opt,lb,None,'T_LCTX',61,d,ep,0,Counter(),patients)
                        self.assertTrue(torch.equal(before_rng,torch.get_rng_state()));rows.append(row)
                    results.append((rows,c.state_hash(m),c.state_hash(t),c.optimizer_hash(opt),c.hash_state({n:p.grad for n,p in m.named_parameters() if p.grad is not None})))
                    del m,t,opt;gc.collect()
                self.assertEqual(results[0],results[1])
                m=c.build(reference,device,61,d,'SUP_CE');t=c.ema_from(m);opt=c.optimizer_for(m)
                loss=[c.step(m,t,opt,lb,61,d,1,i,Counter(),patients,'F_FULL')['loss'] for i in range(5)]
                self.assertLess(loss[-1],loss[0]);del m,t,opt;gc.collect()
                with c.training_access(d):
                    for dom,role in [(d,'train_unlabeled'),(c.DOMAINS[0],'train_labeled'),(d,'val')]:
                        with self.assertRaises(PermissionError):c.DomainData(tmp,dom,role,evaluator=role=='val',expected=expected)
        def test_resume_and_registration(self):
            with tempfile.TemporaryDirectory() as tmp:
                b=Path(tmp);data=b/'data';expected=fixture(data)
                for arm in ['F_FULL','LR_SRC_AB']:
                    def kw(tid):
                        task=dict(task_id=tid,arm=arm,seed=61,order='O1',source_domain=c.DOMAINS[0],domain=c.DOMAINS[1],source_task_id='synthetic',updates=4)
                        return dict(base=b,task_id=tid,data=data,reference=reference,device=device,fixture=dict(task=task,epochs=2,steps=2,expected=expected))
                    full=e.train(**kw(arm+'_full'));gc.collect()
                    e.train(**kw(arm+'_split'),stop_after=1);gc.collect();split=e.train(**kw(arm+'_split'),resume=True);gc.collect()
                    for k in ['student_hash','EMA_hash','optimizer_hash','label_order_hash','counts','L_opens','U_opens']:self.assertEqual(full[k],split[k],k)
                    self.assertEqual((b/'tasks'/(arm+'_full')/'steps.jsonl').read_text(),(b/'tasks'/(arm+'_split')/'steps.jsonl').read_text())
            p=protocol();self.assertEqual(len(p['tasks']),36);self.assertEqual(sum(r['updates'] for r in p['tasks']),95400)
            with self.assertRaises(PermissionError):task_by_id('unregistered')
            from .results import gate,COMPARISONS
            import numpy as np
            vals={a:np.zeros((2,3,4)) for a in c.ARMS};vals['LR_SRC_A'][:]=[.015,.005,.025,-.025]
            costs=[dict(arm='LR_SRC_A',baseline=a,order=o,role=r,metric=m,delta=.01) for a in c.ARMS if a!='LR_SRC_A' for o in ['O1','O2'] for r in ['old','current'] for m in ['rim_dice','cup_dice']]
            self.assertTrue(gate(vals,costs,'F_FULL','VALUE')['passed']);self.assertTrue(gate(vals,costs,'LR_FREE','SOURCE_GEOMETRY')['passed'])
            vals['LR_SRC_A'][:,:,1]=-.006
            self.assertFalse(gate(vals,costs,'F_FULL','VALUE')['passed'])
            self.assertIn(('LR_SRC_A','LR_SRC_AB'),COMPARISONS)

    return unittest.defaultTestLoader.loadTestsFromTestCase(Checks)

def synthetic(output,reference,device,development=False):
    source='development' if development else verify();b=Path(output);b.mkdir(parents=True,exist_ok=False);started=time.time()
    with Operations(b/'operations',update_cap=40) as op:result=unittest.TextTestRunner(verbosity=2).run(suite(reference,torch.device(device)))
    r=dict(status='PASS' if result.wasSuccessful() else 'FAIL',source=source,tests=result.testsRun,errors=len(result.errors),failures=len(result.failures),counts=dict(op.counts),device=device,python=platform.python_version(),torch=torch.__version__,seconds=time.time()-started)
    c.write_json(b/'qualification.json',r);print(json.dumps(r))
    if not result.wasSuccessful():raise SystemExit(1)

def smoke(output,base,data,reference,device):
    source=verify();b=Path(output);b.mkdir();dev=torch.device(device);peak=reserved=0;rows=[]
    with Operations(b/'operations',update_cap=24) as op:
        for domain in c.DOMAINS:
            tasks=[t for t in protocol()['tasks'] if t['seed']==61 and t['domain']==domain]
            with c.training_access(domain):
                ds=c.DomainData(data,domain,'train_labeled');patients=c.patients_for(data,domain,(c.MANIFEST_SHA,c.SPLIT_SHA));lb=c.batch(ds,[0,1],dev,61,21,0,'labeled')
                for task in tasks:
                    torch.cuda.reset_peak_memory_stats(dev);m,t,opt,load,entry,bound=e.initial(base,task,reference,dev)
                    for i in range(2):row=c.step(m,t,opt,lb,61,domain,21,i,Counter(),patients[:2],task['arm'])
                    peak=max(peak,torch.cuda.max_memory_allocated(dev));reserved=max(reserved,torch.cuda.max_memory_reserved(dev));rows.append(dict(arm=task['arm'],domain=domain,loss=row['loss']))
                    del m,t,opt,load;gc.collect();torch.cuda.empty_cache()
    r=dict(status='PASS',source=source,counts=dict(op.counts),updates=op.counts['optimizer_steps'],U_image_reads=0,discarded=True,peak_allocated=peak,peak_reserved=reserved,admission_mib=max(2048,math.ceil(max(peak,reserved)/2**20*1.25+512)),rows=rows)
    c.write_json(b/'receipt.json',r);print(json.dumps(r))
def cold(output,reference,device):
    source=verify();b=Path(output);b.mkdir(parents=True,exist_ok=False);dev=torch.device(device)
    if dev.type=='cuda' and torch.cuda.is_initialized():raise RuntimeError('cold-process regression requires uninitialized CUDA')
    os.environ['SF_REFERENCE']=reference;os.environ['SF_DEVICE']=device
    from experiments.lcrseg.tests.ams_seq_transfer_v0_1.test_contract import fixture
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);data=root/'data';expected=fixture(data)
        task=dict(task_id='cold_start',arm='F_FULL',seed=61,order='O1',source_domain=c.DOMAINS[0],domain=c.DOMAINS[1],source_task_id='synthetic',updates=2)
        with Operations(b/'operations',update_cap=2) as op:
            r=e.train(root,task['task_id'],data,reference,dev,fixture=dict(task=task,epochs=1,steps=2,expected=expected))
        assert r['updates']==2 and r['merge_max_abs']<=1e-5 and r['U_opens']==0
    receipt=dict(status='PASS',source=source,tests=1,counts=dict(op.counts),device=device,cold_start=True,optimizer_updates=2,python=platform.python_version(),torch=torch.__version__,inherited_full_regression_source='c097633ee08837bb4fc07ad3cc60cec71c463128',scope='current exact-source cold engine execution; prior complete math/resume qualification remains separately attributed')
    c.write_json(b/'qualification.json',receipt);print(json.dumps(receipt))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['synthetic','smoke','cold'])
    for k in ['output','reference','device']:p.add_argument('--'+k,required=True)
    p.add_argument('--development',action='store_true');p.add_argument('--base');p.add_argument('--data');a=vars(p.parse_args());mode=a.pop('mode')
    if mode=='cold':a.pop('base');a.pop('data');a.pop('development');cold(**a)
    elif mode=='synthetic':a.pop('base');a.pop('data');synthetic(**a)
    else:a.pop('development');smoke(**a)
