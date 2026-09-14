"""Finite user-delegated execution. No dynamic imports from approval JSON."""
import copy,json,os,time,subprocess,sys,traceback,functools
from pathlib import Path
import numpy as np
import torch
from . import planner,checkpoint
from .gate import digest,code_manifest
from .integration import ExecutionPermit,_PERMIT_SEAL
from .native_parent import IDENTITY,NativeLRParent,build
from .native_data import NativeCurrentDomain,ORDERS,inspect,evaluation_data
from .model import Model,Deployment
from .train_stage import StageTrainer,NO_U
from .native_state import resume
from .semantics import tensor_fingerprint,resolve_options
from .numerics import finite
from .losses import CWMI
from .analyze import select_candidate
from .evaluate import segmentation_metrics

STEPS={'REFUGE':8000,'RIM_ONE_r3':3200,'Drishti_GS':2100}


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(temp,path)


def read(path):return json.loads(Path(path).read_text())


def admit(config):
    """Bind the user's new authority without changing historical external decisions."""
    root=Path(config['code']);doc=root/'experiments/lcrseg/docs/five_frameworks_v1/native_execution'
    authority=read(doc/'USER_AUTHORIZATION.json');parent=read(doc/'PARENT_BINDING.json');plan=read(doc/'RESOLVED_PROTOCOL.json')
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    if actual!=config['execution_commit']:raise PermissionError('execution commit differs from pinned checkout')
    subprocess.run(['git','diff','--quiet','HEAD'],cwd=root,check=True)
    if authority['mode']!='USER_DELEGATED_EXECUTION' or authority['user_confirmed'] is not True or authority['external_review_of_production'] is not False:
        raise PermissionError('user-delegated authorization required')
    if parent['real_parent_identity']!=IDENTITY or parent['status']!='BOUND_IMPLEMENTATION_VERIFIED':raise PermissionError('native implementation binding required')
    if code_manifest(root)['reviewed_code_tree_sha256']!=read(doc/'CODE_MANIFEST.json')['reviewed_code_tree_sha256']:
        raise PermissionError('execution code tree differs')
    if digest(plan)!=authority['plan_sha256'] or digest(parent)!=authority['parent_binding_sha256']:
        raise PermissionError('execution protocol/binding differs')
    if authority['designation_sha256']!=digest(read(doc/'PARENT_DESIGNATION.json')):raise PermissionError('designation differs')
    bindings={'execution_scope':'formal','code_commit':actual,'tree':code_manifest(root)['reviewed_code_tree_sha256'],
              'authorized_manifest_digests':plan['authorized_manifest_digests']}
    permit=ExecutionPermit(bindings,('B','C','D'),plan['budget'],_PERMIT_SEAL)
    inspect(config['data'])
    from .integration import REAL_RUNNERS
    REAL_RUNNERS[IDENTITY]=NativeRunner
    return permit,plan


class Counter:
    """Durable physical calls, including exceptions. No automatic failed-node retry."""
    def __init__(self,path,cap):self.path=Path(path);self.cap=cap;self.count=sum(1 for _ in self.path.open()) if self.path.exists() else 0
    def call(self,step):
        if self.count>=self.cap:raise RuntimeError('physical optimizer budget exhausted')
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open('a') as f:
            f.write(json.dumps({'invocation':self.count+1,'scientific_step_attempt':step,'time':time.time()})+'\n');f.flush();os.fsync(f.fileno())
        self.count+=1
    def wrap(self,optimizer):
        fn=optimizer.step
        @functools.wraps(fn)
        def step(*a,**kw):self.call(self.count+1);return fn(*a,**kw)
        # Preserve scheduler's method wrapping contract by installing after construction.
        optimizer.step=step


@torch.no_grad()
def evaluate(student,data,seen,device):
    student.eval().requires_grad_(False);aggregates={};private={}
    for domain in seen:
        ds=evaluation_data(data,domain,seen);rows=[]
        for i in range(len(ds)):
            b=ds[i];pred=student(b['image'][None].to(device)).argmax(1)[0].cpu().numpy()
            result=segmentation_metrics(pred,b['label'].numpy())
            rows.append({'case_id':ds.rows[i]['case_id'],'rim':result['rim']['Dice'],'cup':result['cup']['Dice'],
                         'disc_union':result['disc_union']['Dice']})
        if any(r[c] is None for r in rows for c in ('rim','cup')):raise ValueError('no valid evaluation support')
        aggregates[domain]={c:float(np.mean([r[c] for r in rows])) for c in ('rim','cup','disc_union')}
        aggregates[domain]['macro_Dice']=(aggregates[domain]['rim']+aggregates[domain]['cup'])/2
        private[domain]=rows
    return aggregates,private


def options_for(node,study,receipts):
    opts={'lr':.001,'weight_decay':4e-5,'total_steps':STEPS[node['domain']],
          'warmup_fraction':.2,'U_ramp_fraction':.2}
    parents={r['id']:r for r in study['parent_configs']}
    selected=node['candidate_id'] if node['family']=='PARENT' else receipts['SELECT_PARENT']['candidate_id']
    p=parents[selected];opts.update(parent_lr_multiplier=p['lr_multiplier'],lr_B_over_A=p['lr_B_over_A'])
    cid=node['candidate_id']
    if node['phase']=='D':cid=receipts['SELECT_'+node['family']]['candidate_id']
    candidates=planner.expand(study)+planner.expand_baselines(study)
    row=next((r for r in candidates if r['candidate_id']==cid),None)
    if row:
        fixed={k:v for k,v in row.get('fixed',{}).items() if k in ('rank_ratio','lambda_U','inner_steps')}
        opts.update(fixed);opts.update(row['hyperparameters'])
    return resolve_options(opts)


def source_task(config,node,permit,root,device):
    total=STEPS['REFUGE'];identity={'kind':'source','seed':node['seed'],'domain':'REFUGE','node_id':node['id'],'execution_commit':config['execution_commit']}
    provider=NativeCurrentDomain(config['data'],node['seed'],1,0,identity,device,permit,allow_u=False)
    model=build(config['reference'],device,node['seed']);model.train()
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=4e-5)
    counter=Counter(root/'physical.jsonl',total);position=0
    if (root/'latest.pt').exists():
        v=torch.load(root/'latest.pt',map_location=device,weights_only=False)
        if v['identity']!=identity:raise ValueError('source resume identity')
        model.load_state_dict(v['student']);opt.load_state_dict(v['optimizer']);position=v['step'];del v
    started=time.time()
    for step in range(position,total):
        for g in opt.param_groups:g['lr']=.001*(1-step/total)**.9
        x,y,_=provider.labeled(step);opt.zero_grad(set_to_none=True)
        logits=model(x,stochastic_classifier=False)[0]
        loss=torch.nn.functional.cross_entropy(logits,y,ignore_index=255);finite(loss,'source CE');loss.backward()
        finite([p.grad for p in model.parameters() if p.grad is not None],'source gradients')
        counter.call(step+1);opt.step();finite(model.state_dict(),'source updated state');finite(opt.state_dict(),'source optimizer state')
        if step==0 or (step+1)%provider.steps_per_epoch==0:
            write(root/'progress.json',{'node_id':node['id'],'successful_updates':step+1,'physical_updates':counter.count,'loss':float(loss.detach()),'execution_commit':config['execution_commit'],'time':time.time()})
            checkpoint.atomic_save({'identity':identity,'student':model.state_dict(),'optimizer':opt.state_dict(),'step':step+1},root/'latest.pt')
    del opt,logits,loss,x,y
    model.eval().requires_grad_(False)
    deployed=Deployment(NativeLRParent(model,node['seed'],identity,adapt=False),torch.eye(16,device=device)).to(device)
    scores,private=evaluate(deployed,config['data'],['REFUGE'],device);write(root/'private_val.json',private)
    checkpoint.atomic_save({'identity':identity,'student':model.state_dict(),'transform':None,'step':total},root/'student.pt')
    return {'node_id':node['id'],'status':'SEALED','identity':identity,'step':total,'student_hash':tensor_fingerprint(model.state_dict()),
            'seen':['REFUGE'],'scores':scores,'seconds':time.time()-started,'physical_optimizer_calls':counter.count,
            'cost':{'source_forward':total,'source_backward':total,'source_optimizer':counter.count},'data_access':{'L_batches':provider.l_reads,'U_batches':provider.u_reads}}


def target_task(config,node,permit,study,receipts,root,device):
    predecessor=receipts[node['parent_checkpoint']]
    if predecessor['identity']['seed']!=node['seed']:raise ValueError('source seed mismatch')
    if node['stage']==2:
        for key in ('family','candidate_id','seed','order','sequence_id'):
            if predecessor['identity'][key]!=('B0_PARENT_LCTX' if key=='family' and node[key]=='PARENT' else node[key]):raise ValueError('trajectory predecessor mismatch')
    source_id={'node_id':predecessor['node_id'],'student_hash':predecessor['student_hash'],
               'transform_hash':predecessor.get('transform_hash'),'seed':node['seed'],'domain':ORDERS[node['order']-1][node['stage']-1]}
    identity={k:node[k] for k in ('family','candidate_id','seed','order','stage','sequence_id','domain')}
    family='B0_PARENT_LCTX' if node['family']=='PARENT' else node['family'];identity['family']=family
    identity.update(execution_commit=config['execution_commit'],node_id=node['id'])
    opts=options_for(node,study,receipts)
    provider=NativeCurrentDomain(config['data'],node['seed'],node['order'],node['stage'],source_id,device,permit,allow_u=family not in NO_U)
    backend=CWMI(Path(config['dependencies'])/'CWMI') if family=='F2' else None
    started=time.time()
    if (root/'latest.pt').exists():trainer=resume(root/'latest.pt',config['reference'],device,provider,opts,backend,identity,permit)
    else:
        payload=torch.load(Path(config['run_root'])/node['parent_checkpoint']/'student.pt',map_location=device,weights_only=False)
        if payload['identity']!=predecessor['identity'] or tensor_fingerprint(payload['student'])!=predecessor['student_hash']:
            raise ValueError('predecessor weights/receipt mismatch')
        if payload['transform'] is not None and tensor_fingerprint({'F':payload['transform']})!=predecessor['transform_hash']:
            raise ValueError('predecessor feature transform mismatch')
        native=build(config['reference'],device,node['seed']);native.load_state_dict(payload['student'])
        previous=payload['transform'];del payload
        parent=NativeLRParent(native,node['seed'],source_id)
        model=Model(parent,family,ratio=opts.get('rank_ratio',.5),previous=previous).to(device)
        trainer=StageTrainer(model,provider,opts,backend,execution=permit)
    counter=Counter(root/'physical.jsonl',opts['total_steps']);counter.wrap(trainer.optimizer)
    for step in range(trainer.step,opts['total_steps']):
        trainer.update()
        if step==0 or (step+1)%provider.steps_per_epoch==0:
            write(root/'progress.json',{'node_id':node['id'],'successful_updates':trainer.step,'physical_updates':counter.count,'loss':trainer.last['labeled_loss'],'execution_commit':config['execution_commit'],'time':time.time()})
            checkpoint.save(trainer,root/'latest.pt',identity)
    costs=dict(trainer.telemetry);probe=copy.deepcopy(trainer.probe);spectral=copy.deepcopy(trainer.model.spectral)
    # Release EMA/optimizer before constructing the deploy copy: at most two full models.
    model=trainer.model;del trainer,backend
    deploy=model.deploy();del model
    seen=list(ORDERS[node['order']-1][:node['stage']+1])
    scores,private=evaluate(deploy,config['data'],seen,device);write(root/'private_val.json',private)
    checkpoint.atomic_save({'identity':identity,'student':deploy.parent.native.state_dict(),'transform':deploy.transform.detach(),
                            'step':opts['total_steps']},root/'student.pt')
    # Source/stage1 scores are retained as receipts; no old training data is opened.
    timeline=copy.deepcopy(predecessor.get('timeline',{'0':predecessor['scores']}));timeline[str(node['stage'])]=scores
    result={'node_id':node['id'],'status':'SEALED','identity':identity,'step':opts['total_steps'],
            'student_hash':tensor_fingerprint(deploy.parent.native.state_dict()),'transform_hash':tensor_fingerprint({'F':deploy.transform}),'seen':seen,'scores':scores,'timeline':timeline,
            'seconds':time.time()-started,'trajectory_seconds':time.time()-started+predecessor.get('trajectory_seconds',0.),'physical_optimizer_calls':counter.count,'cost':costs,'probe':probe,'spectral':spectral,
            'data_access':{'L_batches':provider.l_reads,'U_batches':provider.u_reads},'resolved_options':opts}
    if node['stage']==2:
        values=[scores[d]['macro_Dice'] for d in seen]
        result.update(Final=sum(values)/3,Old=sum(values[:2])/2,Incoming=values[2],
                      Forget=(timeline['0'][seen[0]]['macro_Dice']-values[0]+timeline['1'][seen[1]]['macro_Dice']-values[1])/2)
    return result


class NativeRunner:
    @staticmethod
    def worker(config,node_id):
        permit,plan=admit(config);nodes={n['id']:n for n in plan['nodes']};node=nodes[node_id]
        root=Path(config['run_root'])/node_id;root.mkdir(parents=True,exist_ok=True)
        if (root/'receipt.json').exists():raise FileExistsError('sealed node protected')
        torch.set_num_threads(2);device=torch.device('cuda:0');torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()
        receipts={n['id']:read(Path(config['run_root'])/n['id']/'receipt.json') for n in plan['nodes'] if (Path(config['run_root'])/n['id']/'receipt.json').exists()}
        if not set(node['dependencies'])<=set(receipts):raise PermissionError('unsealed prerequisites')
        try:
            from .native_operations import NativeOperations
            with NativeOperations(root/'operations') as operations:
                result=source_task(config,node,permit,root,device) if node['kind']=='source' else target_task(config,node,permit,plan['study'],receipts,root,device)
            result['operation_counts']=dict(operations.counts)
            result.update(peak_cuda_allocated=torch.cuda.max_memory_allocated(),peak_cuda_reserved=torch.cuda.max_memory_reserved())
            write(root/'receipt.json',result)
        except BaseException as e:
            write(root/'failure.json',{'node_id':node_id,'error':repr(e),'status':'ENGINEERING_STOP','traceback':traceback.format_exc()});raise

    @staticmethod
    def selection(node,receipts):
        if node['id']=='FREEZE_C_REFERENCES':
            return {'node_id':node['id'],'status':'RESOLVED','strong_L_only':select_candidate([{**receipts['SELECT_'+f],'candidate_id':f} for f in ['B0_PARENT_LCTX','B3_PARENT_LCTX_DENSEG']])['candidate_id'],
                    'strong_SSL':select_candidate([{**receipts['SELECT_'+f],'candidate_id':f} for f in ['B1_PARENT_CONF_KL','B2_PARENT_PAS_KL','B4_PARENT_PAS_KL_DENSEG_JOINT']])['candidate_id']}
        if node.get('reuse_parent'):return {**receipts['SELECT_PARENT'],'node_id':node['id']}
        groups={}
        for dep in node['dependencies']:
            r=receipts[dep];groups.setdefault(r['identity']['candidate_id'],[]).append(r)
        rows=[]
        for cid,values in groups.items():
            if {r['identity']['order'] for r in values}!={1,2}:raise ValueError('selection lacks complete paired orders')
            rows.append({'candidate_id':cid,'Final':float(np.mean([r['Final'] for r in values])),
                         'Old':float(np.mean([r['Old'] for r in values])),
                         'compute':sum(r['trajectory_seconds'] for r in values),'compute_unit':'two_target_stage_wall_seconds', 'sealed_all_orders':True})
        chosen=select_candidate(rows)
        return {'node_id':node['id'],'status':'RESOLVED',**chosen,'all_candidates':rows}

    @staticmethod
    def run_finite(config):
        _,plan=admit(config);root=Path(config['run_root']);root.mkdir(parents=True,exist_ok=True)
        # Qualification receipts are code-bound; an empty process is not RUNNING.
        for name in ('CPU_QUALIFICATION','CUDA_QUALIFICATION','SMOKE'):
            r=read(root/(name+'.json'))
            if r['status']!='PASS' or r['execution_commit']!=config['execution_commit']:raise PermissionError('qualification missing/mismatched '+name)
        active={};nodes=plan['nodes'];receipts={}
        while len(receipts)<len(nodes):
            receipts={n['id']:read(root/n['id']/'receipt.json') for n in nodes if (root/n['id']/'receipt.json').exists()}
            for ident,(process,gpu,log) in list(active.items()):
                if process.poll() is not None:
                    log.close();del active[ident]
                    if process.returncode or ident not in receipts:
                        write(root/'status.json',{'status':'ENGINEERING_STOP','failed_node':ident,'active_remaining':list(active)})
                        for process,_,log in active.values():process.wait();log.close()
                        return
            if len(receipts)==len(nodes):break
            pending=[n for n in nodes if n['id'] not in receipts and n['id'] not in active and set(n['dependencies'])<=set(receipts)]
            for node in [n for n in pending if n['kind']=='selection']:
                receipt=NativeRunner.selection(node,receipts);write(root/node['id']/'receipt.json',receipt);receipts[node['id']]=receipt
            used={v[1] for v in active.values()}
            free={int(line.split(',')[0]):int(line.split(',')[1]) for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()}
            for node in [n for n in pending if n['kind']!='selection']:
                gpu=next((g for g in (5,6,7) if g not in used and free.get(g,0)>=12000),None)
                if gpu is None:break
                nr=root/node['id'];nr.mkdir(parents=True,exist_ok=True)
                if (nr/'failure.json').exists():raise RuntimeError('failed node requires explicit engineering recovery')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),NODE_ID=node['id'],EXEC_MODULE=__name__)
                log=(nr/'worker.log').open('a')
                process=subprocess.Popen([sys.executable,'-c','import os,runpy;runpy.run_module(os.environ["EXEC_MODULE"],run_name="__main__")'],cwd=config['code'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[node['id']]=(process,gpu,log);used.add(gpu)
                write(nr/'launch.json',{'pid':process.pid,'gpu':gpu,'execution_commit':config['execution_commit'],'node_id':node['id'],'time':time.time()})
            write(root/'status.json',{'status':'EXECUTOR_ACTIVE','sealed':len(receipts),'total':len(nodes),'active':{k:{'pid':v[0].pid,'gpu':v[1]} for k,v in active.items()}})
            time.sleep(3)
        write(root/'aggregate_results.json',{'status':'COMPLETE','execution_commit':config['execution_commit'],'receipts':list(receipts.values()),'population':'development patients; new optimization seeds, not independent patients'})
        summarize(root,receipts,plan['study'])
        write(root/'status.json',{'status':'COMPLETE','sealed':len(receipts),'total':len(nodes)})


def summarize(root,receipts,study):
    from .analyze import paired_patient_bootstrap
    table={};pairs={}
    for family in study['baselines']+study['priority_order']:
        rows=[r for r in receipts.values() if r.get('identity',{}).get('family')==family
              and r.get('identity',{}).get('stage')==2 and r.get('node_id','').startswith('D__')]
        if len(rows)!=6:raise ValueError('incomplete replication summary')
        table[family]={k:float(np.mean([r[k] for r in rows])) for k in ('Final','Old','Incoming','Forget')}
        table[family]['seed_Final']={str(seed):float(np.mean([r['Final'] for r in rows if r['identity']['seed']==seed])) for seed in (162,163,164)}
        table[family]['class_macro']={c:float(np.mean([r['scores'][d][c] for r in rows for d in ORDERS[0]])) for c in ('rim','cup','disc_union')}
        pairs[family]={(r['identity']['seed'],r['identity']['order']):read(root/r['node_id']/'private_val.json') for r in rows}
    comparisons={}
    for family in study['priority_order']:
        for reference in (receipts['FREEZE_C_REFERENCES']['strong_L_only'],receipts['FREEZE_C_REFERENCES']['strong_SSL']):
            differences={}
            for domain in ORDERS[0]:
                data=[]
                for seed in (162,163,164):
                    orders=[]
                    for order in (1,2):
                        a=pairs[family][seed,order][domain];b=pairs[reference][seed,order][domain]
                        if [r['case_id'] for r in a]!=[r['case_id'] for r in b]:raise ValueError('unpaired patient identities')
                        orders.append([[((x['rim']+x['cup'])-(y['rim']+y['cup']))/2 for x,y in zip(a,b)]])
                    data.append(orders)
                differences[domain]=np.asarray(data)
            result=paired_patient_bootstrap(differences)
            comparisons[family+'__vs__'+reference]={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in result.items() if k!='replicates'}
    finalists=sorted(study['priority_order'],key=lambda f:(-table[f]['Final'],-table[f]['Old'],f))[:2]
    write(root/'replication_summary.json',{'table':table,'paired_comparisons':comparisons,'at_most_two_research_candidates':finalists,
         'claims':'Reference-specific development-patient screening, not original KI or SOTA; no positive-result gate.',
         'phase_E_started':False})


if __name__=='__main__':
    config=read(os.environ['EXEC_CONFIG'])
    if os.environ.get('NODE_ID'):NativeRunner.worker(config,os.environ['NODE_ID'])
    else:NativeRunner.run_finite(config)
