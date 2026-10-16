"""Finite six-target M&Ms DAG, explicit future authority and one attempt per node."""
import json,os,time,subprocess
from pathlib import Path
from .protocol import contract,VERSION
from .runtime import authorize,destination,bootstrap_source
from ..main_style_ssl_v1.runtime import write


def birth(pid):
    return int((Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1].split()[19])


def summarize(root,plan):
    rows=[]
    for seed in (163,164):
        chain=sorted([n for n in plan if n['seed']==seed],key=lambda n:n['stage'])
        matrix=[]
        for n in chain:
            p=root/'jobs'/n['id']/'VALIDATION.json';matrix.append(json.loads(p.read_text()) if p.exists() else None)
        row=dict(seed=seed,matrix=matrix,status='COMPLETE' if all(m is not None for m in matrix) else 'INCOMPLETE')
        if all(m is not None for m in matrix):
            final=matrix[-1];domains=list(final)
            row.update(Final=sum(final[d]['primary_dice'] for d in domains)/4,Old=sum(final[d]['primary_dice'] for d in domains[:-1])/3,Incoming=final[domains[-1]]['primary_dice'],Forgetting=sum(matrix[i][d]['primary_dice']-final[d]['primary_dice'] for i,d in enumerate(domains[:-1]))/3)
        rows.append(row)
    write(root/'SUMMARY.json',dict(mnms=rows))


def controller(config):
    authorize(config);root=destination(config['root'])
    if root.exists():raise FileExistsError('create-only formal root; no automatic resume/retry')
    gpus=config['gpus']
    if set(gpus)!= {'163','164'} or len(set(gpus.values()))!=2:raise ValueError('two distinct fixed GPUs required')
    root.mkdir(parents=True);(root/'jobs').mkdir();(root/'configs').mkdir()
    (root/'scratch').mkdir();(root/'cache').mkdir()
    write(root/'CONTROLLER_PID.json',dict(pid=os.getpid(),birth=birth(os.getpid())))
    plan=contract()['nodes'];write(root/'PLAN.json',contract());states={};running={}
    for node in plan:
        if node['stage']:continue
        out=root/'jobs'/node['id'];out.mkdir()
        try:
            costs=bootstrap_source(Path(config['source_root'])/'jobs'/node['id']/'sealed.pt',node,config['data'],config['reference'],'cpu',out/'sealed.pt',engineering=False,progress=lambda x:write(out/'COLLECTOR_STATUS.json',x))
            for name in ('VALIDATION.json',):(out/name).write_bytes((Path(config['source_root'])/'jobs'/node['id']/name).read_bytes())
            states[node['id']]='REUSED_SOURCE';write(out/'COSTS.json',dict(physical_optimizer_updates=0,inherited_updates=node['steps'],collector=costs,source_bank_L_batch_reads=1,source_bank_L_image_exposures=2));write(out/'EXIT.json',dict(exit_code=0))
        except BaseException as exc:
            states[node['id']]='FAILED';write(out/'EXIT.json',dict(exit_code=1,error=str(exc)))
        write(out/'STATUS.json',dict(status=states[node['id']],node=node))
    while len(states)<len(plan):
        for key,(p,gpu,log) in list(running.items()):
            code=p.poll()
            if code is None:continue
            log.close();del running[key];status=root/'jobs'/key/'STATUS.json'
            states[key]='COMPLETE' if code==0 and status.exists() and json.loads(status.read_text()).get('status')=='COMPLETE' else 'FAILED'
        for node in plan:
            key=node['id'];pred=node['predecessor']
            if key in states or key in running:continue
            if states.get(pred) in ('FAILED','BLOCKED'):
                states[key]='BLOCKED';out=root/'jobs'/key;out.mkdir();write(out/'STATUS.json',dict(status='BLOCKED',node=node,reason='predecessor failed; no retry'));continue
            gpu=gpus[str(node['seed'])]
            if states.get(pred) not in ('REUSED_SOURCE','COMPLETE') or any(g==gpu for _,g,_ in running.values()):continue
            free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
            if free<14000:continue
            child=dict(node=node,output=str(root/'jobs'/key),data=config['data'],reference=config['reference'],predecessor=str(root/'jobs'/pred/'sealed.pt'),authority_receipt=config['authority_receipt'],storage_root=config['storage_root'])
            cfg=root/'configs'/(key+'.json');write(cfg,child)
            env=dict(os.environ,TASK_ACTION='worker',TASK_CONFIG=str(cfg),CUDA_VISIBLE_DEVICES=str(gpu),CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONUTF8='1',TMPDIR=str(root/'scratch'),CUDA_CACHE_PATH=str(root/'cache/cuda'),TORCH_HOME=str(root/'cache/torch'),XDG_CACHE_HOME=str(root/'cache'),HF_HOME=str(root/'cache/hf'))
            log=(root/'configs'/(key+'.log')).open('x');p=subprocess.Popen([config['python'],str(Path(config['ops'])/'n.py')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=config['ops']);running[key]=(p,gpu,log)
        write(root/'STATUS.json',dict(status='RUNNING',completed=states,running={key:dict(pid=p.pid,birth=birth(p.pid),gpu=gpu) for key,(p,gpu,_) in running.items()}))
        if len(states)<len(plan):time.sleep(5)
    summarize(root,plan)
    costs=[]
    for n in plan:
        p=root/'jobs'/n['id']/'COSTS.json'
        if p.exists():costs.append(json.loads(p.read_text()))
    write(root/'COSTS.json',dict(new_formal_optimizer_updates=sum(c.get('physical_optimizer_updates',0) for c in costs),formal_budget=209800,reused_source_updates=sum(c.get('inherited_updates',0) for c in costs),collector_L_batch_reads=sum(c.get('collector',c.get('final_history_collection',{})).get('L_batch_reads',0) for c in costs),source_bank_L_image_exposures=sum(c.get('source_bank_L_image_exposures',0) for c in costs),missing_node_cost_receipts=[n['id'] for n in plan if states[n['id']] not in ('BLOCKED',) and not (root/'jobs'/n['id']/'COSTS.json').exists()]))
    status='COMPLETE' if all(s in ('COMPLETE','REUSED_SOURCE') for s in states.values()) else 'COMPLETE_WITH_FAILURES'
    write(root/'FINAL.json',dict(status=status,nodes=states,public_delivery='PENDING',scientific_success='NOT_ASSUMED'))
    write(root/'EXIT.json',dict(status=status,exit_code=0 if status=='COMPLETE' else 1))
