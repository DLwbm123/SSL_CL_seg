"""Durable finite DAG, one attempt per node, branch failures preserved."""
import os,json,time,subprocess
from pathlib import Path
from .protocol import nodes
from .runtime import write


def summarize(root,plan):
    summary={}
    for ds in ('fundus','prostate','mnms'):
        trajectories=[]
        for seed in (163,164):
            orders=(1,2) if ds=='fundus' else (1,)
            for order in orders:
                chain=[n for n in plan if n['dataset']==ds and n['seed']==seed and (n['stage']==0 or n['order']==order)]
                chain.sort(key=lambda n:n['stage']);matrix=[]
                for n in chain:
                    path=root/'jobs'/n['id']/'VALIDATION.json'
                    matrix.append(json.loads(path.read_text()) if path.exists() else None)
                final=matrix[-1];entry=dict(seed=seed,order=order,matrix=matrix,status='COMPLETE' if all(m is not None for m in matrix) else 'INCOMPLETE')
                if final is not None:
                    domains=list(final);entry['Final']=sum(final[d]['primary_dice'] for d in domains)/len(domains)
                    entry['Old']=sum(final[d]['primary_dice'] for d in domains[:-1])/len(domains[:-1]);entry['Incoming']=final[domains[-1]]['primary_dice']
                    entry['Forgetting']=sum(matrix[i][d]['primary_dice']-final[d]['primary_dice'] for i,d in enumerate(domains[:-1]))/len(domains[:-1])
                    entry['signed_domain_changes']={d:final[d]['primary_dice']-matrix[i][d]['primary_dice'] for i,d in enumerate(domains[:-1])}
                trajectories.append(entry)
        summary[ds]=trajectories
    write(root/'SUMMARY.json',summary)
    return summary


def controller(config):
    root=Path(config['root']);plan=nodes();gpus=list(config['gpus']);running={};states={};started=time.time()
    qualification=json.loads((Path(config['output'])/'RESULT.json').read_text())
    if qualification.get('status')!='PASS' or qualification.get('code_commit')!=config['code_commit']:
        raise RuntimeError('engineering admission did not PASS on the execution commit')
    if json.loads((root/'CPU_ADMISSION.json').read_text()).get('status')!='PASS_REQUIRED_SCOPE':
        raise RuntimeError('required CPU regressions did not PASS')
    (root/'jobs').mkdir(exist_ok=False);(root/'configs').mkdir(exist_ok=False)
    write(root/'PLAN.json',dict(nodes=plan,total_optimizer_updates=511600,ablation=False,test_accessor=False,selection='fixed final checkpoints only',code_commit=config['code_commit']))
    while len(states)<len(plan):
        for key,(process,gpu,log) in list(running.items()):
            code=process.poll()
            if code is None:continue
            log.close();gpus.append(gpu);del running[key]
            receipt=root/'jobs'/key/'STATUS.json'
            status=json.loads(receipt.read_text()).get('status') if receipt.exists() else 'NO_RECEIPT'
            states[key]='COMPLETE' if code==0 and status=='COMPLETE' else 'FAILED'
        for node in plan:
            key=node['id'];pred=node['predecessor']
            if key in states or key in running:continue
            if pred and states.get(pred) in ('FAILED','BLOCKED'):
                states[key]='BLOCKED';out=root/'jobs'/key;out.mkdir()
                write(out/'STATUS.json',dict(status='BLOCKED',reason='predecessor failed; no automatic retry',node=node));continue
            if not gpus or pred and states.get(pred)!='COMPLETE':continue
            gpu=gpus.pop(0);output=root/'jobs'/key
            child=dict(node=node,output=str(output),data=config['data'],reference=config['reference'],predecessor=None if pred is None else str(root/'jobs'/pred/'sealed.pt'))
            cfg=root/'configs'/(key+'.json');write(cfg,child)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TASK_CONFIG=str(cfg),TASK_MODULE='experiments.lcrseg.main_style_ssl_v1.entry',TASK_ACTION='worker')
            log=(root/'configs'/(key+'.log')).open('x')
            process=subprocess.Popen(['bash','./s',config['python'],'./n.py'],cwd=config['ops'],env=env,stdout=log,stderr=subprocess.STDOUT)
            running[key]=(process,gpu,log)
        write(root/'STATUS.json',dict(status='RUNNING',completed=states,running={k:dict(pid=p.pid,gpu=g) for k,(p,g,_) in running.items()},elapsed_seconds=time.time()-started,total_nodes=len(plan)))
        if len(states)<len(plan):time.sleep(10)
    summary=summarize(root,plan)
    failed={k:v for k,v in states.items() if v!='COMPLETE'}
    write(root/'STATUS.json',dict(status='COMPLETE' if not failed else 'COMPLETE_WITH_FAILURES',nodes=states,elapsed_seconds=time.time()-started))
    write(root/'FINAL.json',dict(completed=len(states),failures=failed,summary=summary,public_delivery='PENDING_USER_STATUS_QUERY_OR_AUTHORIZED_DELIVERY',scientific_success='NOT_ASSUMED'))
