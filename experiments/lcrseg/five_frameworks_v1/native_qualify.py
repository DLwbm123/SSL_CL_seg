"""Bounded CUDA qualification and disposable current-L smoke; no score selection."""
import os,json,gc,time
from pathlib import Path
import torch
from .native_runner import admit,read,write,Counter
from .native_parent import build,NativeLRParent
from .native_data import NativeCurrentDomain
from .model import Model
from .train_stage import StageTrainer
from .losses import CWMI


def cuda(config):
    admit(config)
    root=Path(config['run_root']);counter=Counter(root/'cuda_qualification_physical.jsonl',256)
    original=torch.optim.Adam.step
    def measured(opt,*a,**kw):counter.call(counter.count+1);return original(opt,*a,**kw)
    torch.optim.Adam.step=measured
    os.environ.update(NATIVE_TEST_DEVICE='cuda:0',NATIVE_TEST_SIZE='384',NATIVE_REFERENCE=config['reference'],SSLCL5_DEP_ROOT=config['dependencies'])
    torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
    import pytest
    started=time.time()
    code=pytest.main(['-q','experiments/lcrseg/tests/five_frameworks_v1/test_native.py','--junitxml='+str(root/'cuda_qualification.xml')])
    write(root/'CUDA_QUALIFICATION.json',{'status':'PASS' if code==0 else 'FAIL','execution_commit':config['execution_commit'],
        'optimizer_calls_cumulative':counter.count,'cap':256,'seconds':time.time()-started,
        'device':torch.cuda.get_device_name(),'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()})
    if code:raise RuntimeError('native CUDA qualification failed')


def smoke(config):
    permit,_=admit(config);root=Path(config['run_root']);device=torch.device('cuda:0')
    q=read(root/'CUDA_QUALIFICATION.json')
    if q['status']!='PASS' or q['execution_commit']!=config['execution_commit']:raise PermissionError('CUDA qualification required')
    counter=Counter(root/'smoke_physical.jsonl',24);rows=[]
    backend=CWMI(Path(config['dependencies'])/'CWMI')
    for family,steps in {'F1':4,'F2':8,'F3':4,'F4':4,'F5':4}.items():
        if counter.count+steps>24:raise RuntimeError('smoke remaining cap insufficient; prior attempts retained')
        source={'kind':'disposable_random_native_smoke','domain':'REFUGE','seed':161,'never_formal_predecessor':True}
        provider=NativeCurrentDomain(config['data'],161,1,1,source,device,permit,allow_u=False)
        parent=NativeLRParent(build(config['reference'],device,161),161,source)
        model=Model(parent,family).to(device)
        # Match actual warmup: short L-only smoke; never compress warmup to read U.
        options={'lr':.001,'weight_decay':4e-5,'total_steps':3200}
        t=StageTrainer(model,provider,options,backend if family=='F2' else None,execution=permit)
        counter.wrap(t.optimizer);before=counter.count;started=time.time()
        for _ in range(steps):t.update()
        rows.append({'family':family,'successful_updates':t.step,'physical_calls':counter.count-before,
                     'L_batches':provider.l_reads,'U_batches':provider.u_reads,'telemetry':t.telemetry,
                     'probe':t.probe,'seconds':time.time()-started,'weights_discarded':True})
        if provider.u_reads:raise PermissionError('smoke read U')
        del t,model,parent,provider;gc.collect();torch.cuda.empty_cache()
    write(root/'SMOKE.json',{'status':'PASS','execution_commit':config['execution_commit'],
         'optimizer_calls_cumulative':counter.count,'cap':24,'rows':rows,
         'initialization':'independent random designated native weights, not a trained source; source students train only after qualification',
         'no_score_selection':True,'inherited_by_formal':False})


if __name__=='__main__':
    config=read(os.environ['EXEC_CONFIG'])
    if os.environ['QUALIFICATION_MODE']=='cuda':cuda(config)
    elif os.environ['QUALIFICATION_MODE']=='smoke':smoke(config)
    else:raise ValueError('unknown qualification mode')
