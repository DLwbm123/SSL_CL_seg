"""Finite counterfactual action grid; never a deployable policy or training target."""
import fcntl
import importlib.util
import json
import os
import random
import subprocess
import time
import traceback
from pathlib import Path

CAPS = dict(qualification=8, audit=36000)


def read(path): return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def feasibility(rows,controls):
    import numpy as np
    groups=[sorted((r for r in rows if r['context']==f'dev{i}'),key=lambda r:(r['first'],r['second'])) for i in range(4)]
    assert all(groups)
    values=[np.array([[r['new']['macro'],r['old']['macro'],r['utility']] for r in g],dtype=np.float64) for g in groups]
    assert all(np.isfinite(v).all() for v in values)
    left=(values[0][:,None,:]+values[1][None,:,:]).reshape(-1,3)
    right=(values[2][:,None,:]+values[3][None,:,:]).reshape(-1,3)
    threshold=max(v['utility'] for v in controls.values())+.0005
    best=None;best_feasible=None;count=0
    for start in range(0,len(left),64):
        mean=(left[start:start+64,None,:]+right[None,:,:])/4
        valid=mean[:,:,2]>=threshold
        for name in ('UNIFORM_ACTION','CONTINUE_CE','POOLED_RL'):
            c=controls[name];dn=mean[:,:,0]-c['new'];do=mean[:,:,1]-c['old']
            valid&=((dn>=.002)&(do>=-.0025))|((do>=.005)&(dn>=-.0025))
        count+=int(valid.sum())
        for mask,which in ((None,'all'),(valid,'feasible')):
            scores=mean[:,:,2] if mask is None else np.where(mask,mean[:,:,2],-np.inf)
            flat=int(scores.argmax());i,j=np.unravel_index(flat,scores.shape);value=float(scores[i,j])
            current=best if which=='all' else best_feasible
            if np.isfinite(value) and (current is None or value>current['utility']):
                indices=((start+i)//len(groups[1]),(start+i)%len(groups[1]),j//len(groups[3]),j%len(groups[3]))
                record=dict(new=float(mean[i,j,0]),old=float(mean[i,j,1]),utility=value,
                    sequences=[dict(context=f'dev{k}',first=groups[k][ix]['first'],second=groups[k][ix]['second']) for k,ix in enumerate(indices)])
                if which=='all':best=record
                else:best_feasible=record
    return dict(status='ACTION_SEQUENCE_MARGIN_EXISTS_ON_FROZEN_D1_ONLY' if count else 'NO_ACTION_SEQUENCE_MARGIN_ON_FROZEN_D1_STREAMS',
        combinations=len(left)*len(right),feasible_combinations=count,utility_threshold=threshold,
        unconstrained_best=best,best_feasible=best_feasible,controls=controls,
        label='Post-hoc empirical finite-grid reference; not a deployable policy or independent upper bound',
        actor_training_from_development=False,V84='NOT_RUN',independent_confirmation=False)


def selfcheck():
    controls={k:dict(new=.8,old=.8,utility=0.) for k in ('UNIFORM_ACTION','CONTINUE_CE','POOLED_RL')}
    rows=[dict(context=f'dev{i}',first=a,second=a,new={'macro':.8+.003*a},old={'macro':.8},utility=.003*a) for i in range(4) for a in (0,1)]
    result=feasibility(rows,controls)
    assert result['combinations']==16 and result['feasible_combinations']>0
    assert all(s['first']==s['second']==1 for s in result['best_feasible']['sequences'])
    assert abs(result['best_feasible']['utility']-.003)<1e-12
    assert feasibility([dict(r,utility=0.) for r in rows],controls)['feasible_combinations']==0
    assert feasibility([dict(r,new={'macro':.8}) for r in rows],controls)['feasible_combinations']==0


def gpu_job(root,cfg):
    import numpy as np
    import torch
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior import core as c,stage_b as b
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.worker import scores
    torch.set_num_threads(2);torch.cuda.set_device(0)
    random.seed(168);np.random.seed(168);torch.manual_seed(168)
    roles=c.split_roles(cfg['data']);assert roles==read(Path(cfg['action_root'])/'ROLES.private.json')
    ledger=b.JobLedger(root,cfg['caps']);queries=0;campaign=Path(cfg['campaign']);i=cfg['context_index']
    m,n,kind,factor=read(Path(cfg['code'])/'experiments/qprompt_rl_v1/v8_grpo_transfer_prior/ACTION_AUDIT_CONFIG.json')['dev_contexts'][i]
    ctx=(m,n,(kind,factor),f'dev{i}');t=b.make(cfg,roles,ledger,ctx,300)
    def restore(path,step):
        value=torch.load(path,map_location='cpu',weights_only=False)
        assert value['step']==step and value['options']['total_steps']==300
        c.restore(t,value);return value
    def query(role,condition):
        nonlocal queries
        queries+=len(roles[role]);event=dict(role=role,step=t.step,image_calls_upper_bound=queries)
        c.e.append(root/'QUERY_LEDGER.jsonl',dict(event='attempt',**event))
        try:result=scores(t,roles,role,condition)
        except BaseException:
            c.e.append(root/'QUERY_LEDGER.jsonl',dict(event='failure',**event));raise
        c.e.append(root/'QUERY_LEDGER.jsonl',dict(event='success',**event));return result
    if cfg['job']=='train':
        assert read(campaign/'jobs/qualification/QUALIFICATION.json')['status']=='PASS'
        entry=restore(Path(cfg['original'])/f'jobs/development/dev{i}/ENTRY.private.pt',100)
        t.category='audit';sealed=[]
        for first in range(9):
            c.restore(t,entry);t.action=first;t.key=f'dev{i}/prefix{first}'
            for _ in range(100):
                t.update()
                if t.step==150:c.e.atomic_save(c.snapshot(t),root/f'FIRST_{first}_MID.private.pt')
            assert t.step==200;prefix=c.snapshot(t);c.e.atomic_save(prefix,root/f'FIRST_{first}_PREFIX.private.pt')
            for second in range(9):
                c.restore(t,prefix);t.action=second;t.key=f'dev{i}/{first}_{second}'
                for _ in range(100):t.update()
                assert t.step==300
                c.e.atomic_save(c.snapshot(t),root/f'PAIR_{first}_{second}_FINAL.private.pt')
                sealed.append(dict(context=f'dev{i}',first=first,second=second))
                write(root/'STATUS.json',dict(status='RUNNING',endpoints=len(sealed),physical=dict(ledger.count)))
        assert dict(ledger.count)=={'audit':9000} and queries==0 and len(sealed)==81
        write(root/'SEALED.json',dict(status='SEALED',endpoints=sealed,midpoints=9,physical=dict(ledger.count),time=time.time()))
    elif cfg['job']=='evaluate':
        assert read(campaign/'ENDPOINT_LOCK.json')['endpoints']==324
        previous=[r for r in read(campaign/'REUSED_RESULTS.json') if r['context']==f'dev{i}']
        assert len({r['new_entry'] for r in previous})==len({r['old_memory_reference'] for r in previous})==1
        newentry,ref=previous[0]['new_entry'],previous[0]['old_memory_reference'];src=campaign/f'jobs/train{i}';rows=[]
        for first in range(9):
            restore(src/f'FIRST_{first}_MID.private.pt',150);mid=query('Q_dev_new',ctx[2])
            for second in range(9):
                restore(src/f'PAIR_{first}_{second}_FINAL.private.pt',300)
                new=query('Q_dev_new',ctx[2]);old=query('Q_dev_old',('identity',1.))
                value=B.reward(mid['macro'],new['macro'],newentry,old['macro'],ref)
                rows.append(dict(context=f'dev{i}',first=first,second=second,new50=mid,new=new,old=old,new_entry=newentry,
                    old_memory_reference=ref,gain=value['gain'],forget_penalty=value['forget_penalty'],utility=value['reward'],absolute_forgetting=ref-old['macro']))
        assert queries==684 and not ledger.count and len(rows)==81
        write(root/'RESULTS.json',rows)
        replay=[];lookup={(r['first'],r['second']):r for r in rows}
        for old in previous:
            new=lookup[tuple(old['actions'])]
            differences=[abs(new[k][ch]-old[k][ch]) for k in ('new50','new','old') for ch in new[k]]
            differences += [abs(new[k]-old[k]) for k in ('gain','forget_penalty','utility','absolute_forgetting')]
            delta=max(differences)
            replay.append(dict(context=f'dev{i}',method=old['method'],actions=old['actions'],max_abs_difference=delta,pass_tolerance=delta<=1e-7))
        write(root/'REPLAY_AUDIT.json',replay)
        assert all(r['pass_tolerance'] for r in replay),'historical sequence metric replay mismatch'
    else:raise ValueError('unknown job')
    write(root/'FINAL.json',dict(status='COMPLETE',physical=dict(ledger.count),query_image_evaluations=queries,time=time.time()))


def schedule(root,cfg,jobs,account):
    pending=list(jobs);active={};failed=[];done=[]
    while pending or active:
        for slot,(p,spec) in list(active.items()):
            if p.poll() is None:continue
            dest=root/'jobs'/spec['id'];write(dest/'PROCESS_EXIT.json',dict(exit_code=p.returncode,time=time.time()))
            (failed if p.returncode or not (dest/'FINAL.json').exists() else done).append(spec['id']);del active[slot]
        if not failed:
            for slot in (5,6,7):
                if not pending:break
                if slot in active:continue
                if int(subprocess.check_output(['nvidia-smi','--id='+str(slot),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())<12000:continue
                spec=pending.pop(0);dest=root/'jobs'/spec['id'];dest.mkdir(exist_ok=False)
                write(dest/'CONFIG.private.json',dict(cfg,**spec,campaign=str(root),gpu=slot))
                env=dict(os.environ,EXEC_RUN=str(dest),EXEC_CONFIG=str(dest/'CONFIG.private.json'),EXEC_ENTRY=cfg['entry'],EXEC_MODE='job',PYTHONPATH=cfg['code'],CUDA_VISIBLE_DEVICES=str(slot),CUBLAS_WORKSPACE_CONFIG=':4096:8')
                with (dest/'worker.log').open('x') as log:
                    p=subprocess.Popen(['bash','./with_nas_storage.sh',cfg['python'],'-c','import os,runpy;runpy.run_path(os.environ["EXEC_ENTRY"],run_name="__main__")'],cwd=cfg['code']+'/experiments/lcrseg/scripts',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                write(dest/'LAUNCH.json',dict(pid=p.pid,gpu=slot,time=time.time(),commit=cfg['commit']));active[slot]=p,spec
        account.refresh()
        write(root/'COSTS.json',dict(scope='V97_NEW_ONLY',attempts=dict(account.count),success=dict(account.success),failures=dict(account.failure),student_cap=36008,actor_cap=0))
        write(root/'STATUS.json',dict(status='ENGINEERING_STOP' if failed else 'RUNNING',active=[s['id'] for _,s in active.values()],pending=[s['id'] for s in pending],completed=done,failed=failed,physical=dict(account.count),time=time.time()))
        if failed and not active:raise RuntimeError('failed jobs: '+','.join(failed))
        if pending or active:time.sleep(10)


def coordinator(root,cfg):
    from experiments.qprompt_rl_v1.v8_grpo_transfer_prior.run_b import Accounting
    lock=(root/'COORDINATOR.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=Path(cfg['v96']);assert read(previous/'FINAL.json')['status']=='COMPLETE'
    assert read(previous/'REUSE_AUDIT.json')['status']=='PASS' and read(previous/'ENDPOINT_LOCK.json')['endpoints']==168
    previous_rows=read(previous/'DEVELOPMENT_RESULTS.json')['rows'];assert len(previous_rows)==len({(r['context'],r['method']) for r in previous_rows})==168
    assert all(read(previous/f'jobs/eval{i}/PROCESS_EXIT.json')['exit_code']==0 for i in range(4))
    write(root/'REUSED_RESULTS.json',previous_rows)
    write(root/'REUSE_AUDIT.json',dict(status='PASS',historical_endpoints=168,purpose='reference metrics only; all grid endpoints newly trained',new_updates=0))
    quals=[dict(id='qualification',job='qualification',caps=dict(qualification=8))]
    trains=[dict(id=f'train{i}',job='train',context_index=i,caps=dict(audit=9000)) for i in range(4)]
    evals=[dict(id=f'eval{i}',job='evaluate',context_index=i,caps={}) for i in range(4)]
    jobs=quals+trains+evals;(root/'jobs').mkdir(exist_ok=False);(root/'PHYSICAL_LEDGER.jsonl').touch(exist_ok=False)
    account=Accounting(root,caps=CAPS,jobs=jobs);schedule(root,cfg,quals,account);schedule(root,cfg,trains,account)
    assert dict(account.count)==CAPS;sealed=[]
    for i in range(4):
        src=root/f'jobs/train{i}';receipt=read(src/'SEALED.json');assert len(receipt['endpoints'])==81
        for a in range(9):
            assert (src/f'FIRST_{a}_MID.private.pt').is_file() and (src/f'FIRST_{a}_PREFIX.private.pt').is_file()
            for b in range(9):assert (src/f'PAIR_{a}_{b}_FINAL.private.pt').is_file()
        sealed+=receipt['endpoints']
    write(root/'ENDPOINT_LOCK.json',dict(endpoints=324,midpoints=36,sealed=sealed,time=time.time(),all_before_new_evaluation=True))
    schedule(root,cfg,evals,account);rows=[];replay=[]
    for i in range(4):
        rows+=read(root/f'jobs/eval{i}/RESULTS.json');replay+=read(root/f'jobs/eval{i}/REPLAY_AUDIT.json')
    assert len(rows)==len({(r['context'],r['first'],r['second']) for r in rows})==324
    assert len(replay)==168 and all(r['pass_tolerance'] for r in replay)
    write(root/'REPLAY_AUDIT.json',replay)
    controls={m:v for m,v in read(previous/'DECISION.json')['means'].items() if not m.endswith(('_601','_602'))}
    decision=feasibility(rows,controls);assert decision['combinations']==81**4
    write(root/'GRID_RESULTS.json',dict(rows=rows,decision=decision));write(root/'DECISION.json',decision)
    B.csvfile(root/'ACTION_GRID.csv',[dict(context=r['context'],first=r['first'],second=r['second'],**{k:r[k] for k in ('gain','forget_penalty','utility','absolute_forgetting')},**{f'{phase}_{k}':v for phase in ('new50','new','old') for k,v in r[phase].items()}) for r in rows])
    qd=sum(read(root/f'jobs/eval{i}/FINAL.json')['query_image_evaluations'] for i in range(4));assert qd==2736
    write(root/'COSTS.json',dict(attempts=dict(account.count),success=dict(account.success),failures=dict(account.failure),student_total=36008,actor_total=0,training_query_image_evaluations=0,development_query_image_evaluations=qd,new_unique_images=0,grid_endpoints=324,shared_prefixes=36,historical_reference_endpoints=168))
    (root/'REPORT.md').write_text('# V97 finite action-sequence feasibility\n\n'+decision['status']+'\n\n'+json.dumps(decision,indent=2)+'\n\nAll324new endpoints sealed before evaluation. Post-hoc finite-grid reference only; no selected action sequence is deployed or used to train an actor. Repeated D1 development, not independent validation. All prior STOPs preserved; V84 NOT_RUN.\n')
    write(root/'FINAL.json',dict(status='COMPLETE',decision=decision['status'],physical=dict(account.count),publication='PENDING',time=time.time()));write(root/'STATUS.json',read(root/'FINAL.json'))


if __name__=='__main__':
    selfcheck()
    if os.environ.get('EXEC_SELFCHECK')=='1':print('PASS finite grid feasibility, indexing and threshold checks')
    else:
        root=Path(os.environ['EXEC_RUN']);cfg=read(os.environ['EXEC_CONFIG'])
        spec=importlib.util.spec_from_file_location('previous_worker',cfg['base_entry']);B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
        with (root/'STARTED.json').open('x') as f:json.dump(dict(pid=os.getpid(),time=time.time(),commit=cfg['commit']),f)
        try:
            if os.environ.get('EXEC_MODE')!='job':coordinator(root,cfg)
            elif cfg['job']=='qualification':B.gpu_job(root,cfg)
            else:gpu_job(root,cfg)
        except BaseException as exc:
            write(root/'STATUS.json',dict(status='ENGINEERING_STOP',error=repr(exc),traceback=traceback.format_exc(),time=time.time()));raise
