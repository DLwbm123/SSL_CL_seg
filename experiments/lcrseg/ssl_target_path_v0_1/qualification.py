import argparse,gc,json,os,platform,time,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from .core import *
from .freeze import verify

def qualify(output,device,reference,development=False):
    source='development' if development else verify();root=Path(output);root.mkdir(parents=True)
    os.environ['SF_DEVICE']=device;os.environ['SF_REFERENCE']=reference
    from experiments.lcrseg.tests.ssl_target_path_v0_1.test_contract import Contract
    count=[0];original=torch.optim.Adam.step
    from experiments.lcrseg.di_dmpa_jascl.modeling import LCRSegUNet2DJASCL
    operation_counts=Counter();original_forward=LCRSegUNet2DJASCL.forward;original_backward=torch.Tensor.backward;original_grad=torch.autograd.grad
    def forwarded(self,x,**kw):
        operation_counts['model_forward']+=1;operation_counts['image_forward']+=len(x);return original_forward(self,x,**kw)
    def backwarded(*a,**kw):
        operation_counts['backward']+=1;return original_backward(*a,**kw)
    def graded(*a,**kw):
        operation_counts['autograd_grad_calls']+=1;return original_grad(*a,**kw)
    def counted(*a,**kw):
        if count[0]>=200:raise RuntimeError('per-attempt synthetic budget exhausted')
        result=original(*a,**kw);count[0]+=1;append(root/'optimizer_events.jsonl',dict(successful_update=count[0]));return result
    start=time.time()
    with patch.object(torch.optim.Adam,'step',counted),patch.object(LCRSegUNet2DJASCL,'forward',forwarded),patch.object(torch.Tensor,'backward',backwarded),patch.object(torch.autograd,'grad',graded):result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contract))
    out=dict(status='PASS' if result.wasSuccessful() else 'FAIL',source=source,tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),synthetic_optimizer_updates=count[0],operation_counts=dict(operation_counts),real_diagnostic_updates=0,formal_updates=0,python=platform.python_version(),torch=torch.__version__,device=device,seconds=time.time()-start)
    write_json(root/'qualification.json',out);print(json.dumps(out),flush=True)
    if not result.wasSuccessful():raise SystemExit(1)

def smoke(output,data,reference,device):
    source=verify();root=Path(output);root.mkdir();rows=[];counter=Counter()
    for domain in DOMAINS:
        l=DomainData(data,domain,'train_labeled')
        lb=batch(l,[0,1],device,31,21,0,'smoke_l');ub={k:v for k,v in lb.items() if k!='label'}
        for arm in ARMS:
            student=build(reference,device,31,domain,arm);teacher=ema_from(student);opt=optimizer_for(student)
            row=train_step(student,teacher,opt,lb,ub if arm!='SUP' else None,None,None,arm,31,domain,21,0,counter)
            norm=sum(float(p.grad.detach().double().square().sum()) for p in student.parameters() if p.grad is not None)**.5
            assert math.isfinite(norm) and norm>0
            rows.append(dict(domain=domain,arm=arm,existing_backward_gradient_norm=norm,**row));append(root/'updates.jsonl',rows[-1]);del student,teacher,opt;gc.collect()
    out=dict(status='PASS',source=source,updates=counter['optimizer_steps'],maximum_total=10,actual_per_domain=5,counts=dict(counter),rows=rows,real_U_images_opened=0,GT_scope='only first two sorted current labeled cases per domain; labels stripped for pseudo-U',formal_initializations_reused=False)
    write_json(root/'receipt.json',out);print(json.dumps(out),flush=True)

if __name__=='__main__':
    import gc
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('synthetic','smoke'))
    for k in ('output','device','reference'):p.add_argument('--'+k,required=True)
    p.add_argument('--data');p.add_argument('--development',action='store_true');a=vars(p.parse_args());mode=a.pop('mode');a['device']=a['device']
    if mode=='synthetic':a.pop('data');qualify(**a)
    else:a.pop('development');a['device']=torch.device(a['device']);smoke(**a)
