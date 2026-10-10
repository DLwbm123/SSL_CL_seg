"""New user-authorized finite pilot; reuses StageTrainer, not an old reviewed DAG."""
import copy
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
import torch
from .protocol import plan,read,write,jobs,OPTIONS,STEPS,CAPS,metrics,compare,MANIFEST_SHA,SPLIT_SHA
from .modules import ARMS,PilotTrainer,make_model
from ..five_frameworks_v1.gate import digest
from ..five_frameworks_v1.integration import ExecutionPermit,_PERMIT_SEAL
from ..five_frameworks_v1.native_data import NativeCurrentDomain,ORDERS,inspect
from ..five_frameworks_v1.native_parent import build,NativeLRParent
from ..five_frameworks_v1.native_runner import Counter,evaluate
from ..five_frameworks_v1.native_operations import NativeOperations
from ..five_frameworks_v1.recipes import generator
from ..five_frameworks_v1 import checkpoint
from ..five_frameworks_v1.numerics import finite


def admission(config,scope='formal'):
    root=Path(config['run_root']).resolve();canonical=Path(os.environ['SSLCL_STORAGE_ROOT']).resolve()
    if canonical not in root.parents or root==Path(config['reuse_root']).resolve():
        raise PermissionError('create-only isolated NAS output required')
    if read(root/'PLAN.json')!=json.loads(json.dumps(plan())):
        raise PermissionError('frozen pilot plan changed')
    auth=read(root/'USER_AUTHORIZATION.json')
    if auth!={'instruction':plan()['authority'],'plan_id':plan()['plan_id'],'execution_commit':config['execution_commit']}:
        raise PermissionError('current user authorization does not bind this pilot')
    if Path(config['code'],'CODE_COMMIT').read_text().strip()!=config['execution_commit']:
        raise PermissionError('execution code commit changed')
    manifests=[digest(dict(domain=stage['domain'],seed=j['seed'],order=j['order'],stage=stage['stage'],
                           manifest=MANIFEST_SHA,split=SPLIT_SHA)) for j in jobs() for stage in j['stages']]
    return ExecutionPermit(dict(execution_scope=scope,execution_commit=config['execution_commit'],
                                plan_id=plan()['plan_id'],authorized_manifest_digests=manifests),
                           ('pilot',),CAPS,_PERMIT_SEAL)


def source(config,seed,device):
    sr=Path(config['reuse_root'])/f'SOURCE_S{seed}'
    receipt=read(sr/'receipt.json')
    if receipt['identity']['seed']!=seed or receipt['identity']['kind']!='source' or receipt['step']!=8000:
        raise ValueError('wrong source receipt')
    payload=torch.load(sr/'student.pt',map_location=device,weights_only=False)
    if payload['identity']!=receipt['identity'] or payload['transform'] is not None:
        raise ValueError('source payload identity mismatch')
    native=build(config['reference'],device,seed);native.load_state_dict(payload['student'],strict=True)
    del payload
    return native,receipt


def construct(config,j,stage,device,permit,previous=None,native=None,source_id=None,allow_u=True):
    if native is None:
        native,sr=source(config,j['seed'],device)
        source_id=dict(node_id=sr['node_id'],student_hash=sr['student_hash'],seed=j['seed'],domain='REFUGE')
    provider=NativeCurrentDomain(config['data'],j['seed'],j['order'],stage,source_id,device,permit,allow_u=allow_u)
    options={**OPTIONS,'total_steps':STEPS[provider.domain]}
    model=make_model(NativeLRParent(native,j['seed'],source_id),j['arm'],options,previous,
                     generator(j['seed'],j['order'],stage,0,'pilot_adapter_initialization')).to(device)
    return PilotTrainer(model,provider,options,arm=j['arm'],execution=permit)


def train_job(config,j):
    root=Path(config['run_root']);jr=root/'jobs'/j['id'];jr.mkdir(parents=True,exist_ok=True)
    if (jr/'EXIT.json').exists():raise FileExistsError('job already attempted; no retry')
    permit=admission(config)
    q=read(root/'QUALIFICATION.json')
    if q.get('status')!='PASS' or q.get('execution_commit')!=config['execution_commit']:
        raise PermissionError('current native qualification required')
    torch.set_num_threads(2);device=torch.device('cuda:0');torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats();started=time.time();previous=None;native=None;source_id=None
    try:
        for st in j['stages']:
            stage=st['stage'];out=jr/f'stage{stage}';out.mkdir(exist_ok=False)
            with NativeOperations(out/'operations') as operations:
                trainer=construct(config,j,stage,device,permit,previous,native,source_id)
                del native;counter=Counter(out/'physical.jsonl',st['steps']);counter.wrap(trainer.optimizer)
                diagnostics=[];stat_rows=[]
                for _ in range(st['steps']):
                    trainer.update()
                    if trainer.step==1 or trainer.step%trainer.provider.steps_per_epoch==0:
                        write(out/'STATUS.json',dict(status='TRAINING',successful=trainer.step,attempts=counter.count,
                                                    cap=st['steps'],module=trainer.module_stats,loss=trainer.last))
                    if trainer.step in {max(1,int(st['steps']*p)) for p in (.25,.5,.75,1.)}:
                        stat_rows.append(dict(step=trainer.step,**copy.deepcopy(trainer.module_stats)))
                    if trainer.step%400==0 or trainer.step==st['steps']:
                        checkpoint.save(trainer,out/'latest.pt',dict(family='F5',seed=j['seed'],order=j['order'],stage=stage,arm=j['arm']))
                costs=copy.deepcopy(trainer.telemetry);diagnostics=trainer.gradient_rows
                model=trainer.model;del trainer
                deployment=model.deploy();del model
                identity=dict(arm=j['arm'],seed=j['seed'],order=j['order'],stage=stage,
                              execution_commit=config['execution_commit'],job_id=j['id'])
                checkpoint.atomic_save(dict(identity=identity,student=deployment.parent.native.state_dict(),
                                            transform=deployment.transform.detach(),step=st['steps']),out/'student.pt')
                # Sequential stages inherit their own sealed deployed student, never another arm's prefix.
                native=deployment.parent.native;previous=deployment.transform.detach().clone()
                source_id=dict(node_id=j['id']+f'/stage{stage}',seed=j['seed'],domain=st['domain'],arm=j['arm'])
                del deployment
            write(out/'SEALED.json',dict(identity=identity,steps=st['steps'],attempts=counter.count,cost=costs,
                                        module_diagnostics=stat_rows,gradient_diagnostics=diagnostics,
                                        operation_counts=dict(operations.counts),evaluation='NOT_READ_YET'))
        write(jr/'EXIT.json',dict(exit_code=0,seconds=time.time()-started,
                                 peak_cuda_allocated=torch.cuda.max_memory_allocated(),peak_cuda_reserved=torch.cuda.max_memory_reserved()))
    except BaseException as e:
        write(jr/'FAILURE.private.json',dict(error=repr(e),traceback=traceback.format_exc()))
        write(jr/'EXIT.json',dict(exit_code=1,error_type=type(e).__name__,seconds=time.time()-started))
        raise


def final_readout(config):
    root=Path(config['run_root']);rows=[];stage_rows=[]
    for j in jobs():
        if read(root/'jobs'/j['id']/'EXIT.json')['exit_code']!=0:
            raise RuntimeError('full sealed matrix required before evaluation')
        for st in j['stages']:
            if not (root/'jobs'/j['id']/f'stage{st["stage"]}'/'SEALED.json').exists():
                raise RuntimeError('unsealed target')
    write(root/'READOUT_SEAL.json',dict(targets=32,all_sealed=True,time=time.time()))
    from ..five_frameworks_v1.model import Deployment
    device=torch.device('cuda:0');torch.set_num_threads(2)
    with NativeOperations(root/'readout_operations') as operations:
        for j in jobs():
            values=[]
            for st in j['stages']:
                out=root/'jobs'/j['id']/f'stage{st["stage"]}'
                payload=torch.load(out/'student.pt',map_location=device,weights_only=False)
                native=build(config['reference'],device,j['seed']);native.load_state_dict(payload['student'])
                deployment=Deployment(NativeLRParent(native,j['seed'],payload['identity'],adapt=False),payload['transform']).to(device)
                del payload,native
                seen=list(ORDERS[j['order']-1][:st['stage']+1])
                scores,private=evaluate(deployment,config['data'],seen,device);del deployment
                write(out/'EVALUATION.private.json',private)
                row=dict(arm=j['arm'],seed=j['seed'],order=j['order'],stage=st['stage'],seen=seen,scores=scores)
                write(out/'EVALUATION.json',row);values.append(row);stage_rows.append(row)
            sr=read(Path(config['reuse_root'])/f'SOURCE_S{j["seed"]}'/'receipt.json')
            rows.append(dict(arm=j['arm'],seed=j['seed'],order=j['order'],scores=values[-1]['scores'],metrics=metrics(values,sr)))
    refs=[]
    historical=read(Path(config['reuse_root'])/'PUBLIC_RESULTS.json')
    for arm in (() if config.get('reuse_comparison_results') else ('B0_PARENT_LCTX','B2_PARENT_PAS_KL')):
        for seed in (163,164):
            for order in (1,2):
                stages=sorted([r for r in historical['rows'] if r['identity']['family']==arm and r['identity']['seed']==seed and r['identity']['order']==order],key=lambda r:r['identity']['stage'])
                sr=read(Path(config['reuse_root'])/f'SOURCE_S{seed}'/'receipt.json')
                refs.append(dict(arm=arm,seed=seed,order=order,scores=stages[-1]['scores'],metrics=metrics([
                    dict(scores=r['scores'],seen=list(ORDERS[order-1][:r['identity']['stage']+1])) for r in stages],sr),origin='historical aggregate import'))
    if config.get('reuse_comparison_results'):
        previous_results=read(config['reuse_comparison_results'])
        refs=[{**r,'origin':('matched prior-round F5' if r['arm']=='F5' else 'historical aggregate import')}
              for r in previous_results['rows']+previous_results['historical_refs']
              if r['arm'] in ('F5','B0_PARENT_LCTX','B2_PARENT_PAS_KL')]
    output=dict(status='COMPLETE_DEVELOPMENT_ONLY',rows=rows,historical_refs=refs,stages=stage_rows,
                vs_F5=compare(rows+refs,'F5'),vs_B0=compare(rows+refs,'B0_PARENT_LCTX'),vs_B2=compare(rows+refs,'B2_PARENT_PAS_KL'),
                readout_operations=dict(operations.counts),automatic_followup=False)
    write(root/'RESULTS.json',output)
    lines=['# '+plan()['study_id']+': completed development experiment','',
           f'Two optimization seeds, two orders; all {len(stage_rows)} target students sealed before readout. No independent patient confirmation.','',
           '| Arm | Final | Old | Incoming | Forget |','|---|---:|---:|---:|---:|']
    for arm in ARMS:
        selected=[r for r in rows+refs if r['arm']==arm]
        v={m:sum(r['metrics'][m] for r in selected)/len(selected) for m in ('Final','Old','Incoming','Forget')}
        lines.append('|'+arm+'|'+'|'.join(f'{v[m]:.6f}' for m in ('Final','Old','Incoming','Forget'))+'|')
    lines+=['','All paired seed/order/class deltas, including negative outcomes, are in RESULTS.json.',
            'Paper adaptations and fixed coefficients are in PREREG.md and PLAN.json. No module combination or tuning was run.',
            'Private weights, patient rows, images, labels and runtime configuration remain on NAS.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(root/'FINAL.json',dict(status=output['status'],plan_id=plan()['plan_id'],execution_commit=config['execution_commit'],
                               formal_updates=CAPS['formal_updates'],source_updates=0,automatic_followup=False))


def coordinator(config):
    root=Path(config['run_root']);permit=admission(config);inspect(config['data']);permit.validate()
    lock=(root/'.controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'CONTROLLER_EXIT.json').exists():raise FileExistsError('controller already attempted; no automatic retry')
    started=time.time()
    try:
        # Each batch ends before the next-priority module starts; no efficacy-based pruning.
        batches=plan().get('execution_batches') or [[j['id'] for j in jobs() if j['arm']==arm] for arm in ARMS]
        for group in batches:
            batch=[j for jid in group for j in jobs() if j['id']==jid];children=[];arm=batch[0]['arm']
            write(root/'STATUS.json',dict(status='RUNNING',phase='train',arm=arm,plan_id=plan()['plan_id']))
            free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
            if any(int(free[g])<4096 for g in (4,5,6,7)):
                raise RuntimeError('RESOURCE_WAIT: need 4 GiB free on GPU 4-7; no work was killed')
            for gpu,j in zip((4,5,6,7),batch):
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),EXEC_MODE='job',EXEC_JOB=j['id'])
                jr=root/'jobs'/j['id'];jr.mkdir(parents=True,exist_ok=False)
                log=(jr/'worker.log').open('x')
                child=subprocess.Popen([sys.executable,'-c','import os,runpy; runpy.run_path(os.environ["EXEC_ENTRY"],run_name="__main__")'],
                                       env=env,cwd=config['code'],stdout=log,stderr=subprocess.STDOUT)
                log.close();children.append((j,child))
                write(jr/'STARTUP.json',dict(pid=child.pid,gpu=gpu,job_id=j['id'],time=time.time()))
            failures=[]
            for j,child in children:
                code=child.wait()
                if code!=0:failures.append(dict(job=j['id'],exit_code=code))
            if failures:raise RuntimeError('ENGINEERING_STOP: '+json.dumps(failures))
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='4',EXEC_MODE='readout')
        write(root/'STATUS.json',dict(status='RUNNING',phase='sealed_readout'))
        with (root/'readout.log').open('x') as log:
            subprocess.run([sys.executable,'-c','import os,runpy; runpy.run_path(os.environ["EXEC_ENTRY"],run_name="__main__")'],
                           env=env,cwd=config['code'],stdout=log,stderr=subprocess.STDOUT,check=True)
        qualification=read(root/'QUALIFICATION.json')
        total=sum(sum(1 for _ in (root/'jobs'/j['id']/f'stage{st["stage"]}'/'physical.jsonl').open()) for j in jobs() for st in j['stages'])
        if total!=CAPS['formal_updates']:raise RuntimeError('formal ledger mismatch')
        write(root/'COSTS.json',dict(formal_updates=total,source_updates=0,qualification=qualification['costs'],
                                   elapsed_seconds=time.time()-started))
        write(root/'COMPLETION_AUDIT.json',dict(status='PASS',jobs=len(jobs()),target_stages=sum(len(j['stages']) for j in jobs()),formal_updates=total,
                                              readout_seal=read(root/'READOUT_SEAL.json')))
        write(root/'STATUS.json',dict(status='COMPLETE_PENDING_PUBLIC_DELIVERY',phase='complete',plan_id=plan()['plan_id']))
        write(root/'CONTROLLER_EXIT.json',dict(exit_code=0,seconds=time.time()-started))
    except BaseException as e:
        write(root/'CONTROLLER_FAILURE.private.json',dict(error=repr(e),traceback=traceback.format_exc()))
        write(root/'STATUS.json',dict(status='STOPPED',phase='engineering_or_resource',error_type=type(e).__name__))
        write(root/'CONTROLLER_EXIT.json',dict(exit_code=1,seconds=time.time()-started))
        raise


def main():
    config=read(os.environ['EXEC_CONFIG']);mode=os.environ.get('EXEC_MODE','coordinator')
    if mode=='job':train_job(config,next(j for j in jobs() if j['id']==os.environ['EXEC_JOB']))
    elif mode=='readout':admission(config);final_readout(config)
    elif mode=='qualify':
        from .tests import qualify
        qualify(config)
    elif mode=='coordinator':coordinator(config)
    else:raise ValueError('unregistered execution mode')


if __name__=='__main__':main()
