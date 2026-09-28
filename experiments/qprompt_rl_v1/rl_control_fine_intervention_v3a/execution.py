"""Sequential bounded execution, immutable branch receipts and no retained student."""
import os,json,time,hashlib,copy,fcntl,traceback,sys,resource,gc
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from r1_12h import core as c,runner as old
from rl_control_bandit_v2.runner import Worker,cpu_copy,setup,CODE
from rl_control_bandit_v2.method import LocalSchedule
from rl_control_bandit_v2.data import UImages
from rl_control_bandit_v2.qualification import same
from qprompt.state import make_reference,ema_update
from . import control
ROOT=Path(os.environ['EXEC_RUN']);PREVIOUS=Path(os.environ['EXEC_V2']);CONFIG=json.loads(Path(__file__).with_name('CONFIG.json').read_text());CONFIG_SHA=c.file_sha(Path(__file__).with_name('CONFIG.json'))

def read(p):return json.loads(Path(p).read_text())
def write(p,v):c.atomic(Path(p),v)
def records(p):return c.events(Path(p))

def compatible(path):
    value=read(path)
    if value['code_commit']==CODE:return value
    binding=read(ROOT/'RECOVERY_BINDINGS.private.json')
    assert binding['new_commit']==CODE and binding['old_commit']==value['code_commit']
    assert binding['config_sha']==CONFIG_SHA and binding['receipts'][str(Path(path).relative_to(ROOT))]==c.file_sha(path)
    return value


def fingerprint(value):
    h=hashlib.sha256()
    def add(v):
        if torch.is_tensor(v):h.update(str((str(v.dtype),tuple(v.shape))).encode());h.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v,np.ndarray):h.update(str((str(v.dtype),v.shape)).encode());h.update(v.tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=repr):h.update(repr(k).encode());add(v[k])
        elif isinstance(v,(list,tuple)):
            for x in v:add(x)
        else:h.update(repr(v).encode())
    add(value);return h.hexdigest()

class Ledger:
    def __init__(self):
        self.lock=(ROOT/'OWNER.lock').open('a');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB);self.count=Counter();self.seen=set()
        for x in records(ROOT/'PHYSICAL_LEDGER.jsonl'):
            if x['event']=='attempt':self.count[x['category']]+=1;self.seen.add(x['key'])
    def step(self,opt,kind,key):
        assert time.time()<read(ROOT/'SESSION.json')['deadline'],'DEADLINE';key=kind+'|'+key;cat=('student_replay' if kind.startswith('student') else 'controller_replay') if key in self.seen else kind
        assert self.count[cat]<CONFIG['caps'][cat],f'BUDGET_{cat}'
        assert all(p.grad is None or torch.isfinite(p.grad).all() for g in opt.param_groups for p in g['params'])
        row=dict(key=key,category=cat,kind=kind,code_commit=CODE,config_sha=CONFIG_SHA);c.append(ROOT/'PHYSICAL_LEDGER.jsonl',dict(event='attempt',**row));self.count[cat]+=1;self.seen.add(key)
        try:opt.step()
        except BaseException:c.append(ROOT/'PHYSICAL_LEDGER.jsonl',dict(event='failed',**row));raise
        c.append(ROOT/'PHYSICAL_LEDGER.jsonl',dict(event='success',**row))
        if sum(self.count.values())%50==0:write(ROOT/'BUDGET.json',dict(self.count))

class PanelWorker(Worker):
    def __init__(self,cell,ledger,synthetic=False):
        self.task=cell;self.folder=ROOT/'panels'/cell;self.folder.mkdir(parents=True,exist_ok=True);self.ledger=ledger;self.synthetic=synthetic;self.kind='student_synthetic' if synthetic else 'student_main';self.arm='P1';backbone,domain=cell.split('__');setup(261)
        self.meta=dict(task=cell,code_commit=CODE,config_sha=CONFIG_SHA,seed=261,backbone=backbone,domain=domain);self.model,_=c.build(backbone,False,old.ASSETS,old.SOURCE);self.model.cuda().train();self.opt,source_sch=c.optimizer_for(self.model,backbone)
        if not synthetic:
            binding=read(ROOT/'CONTEXT_MANIFEST.private.json')[cell];self.meta.update(prefix_sha=binding['prefix']['sha256'],schedule_sha=binding['schedule_sha']);p=binding['prefix'];assert c.file_sha(p['path'])==p['sha256'];state=torch.load(p['path'],map_location='cpu',weights_only=False);assert state['global_step']==state['data_cursor']==state['scheduler']['last_epoch']==2000 and state['backbone']==backbone and state['domain']==domain and state['arm']=='QUERY_PREFIX' and state['bank'] is None and state['reference'] is None and state['code_commit']==p['source_commit'];assert all(int(v['step'])==2000 for v in state['optimizer']['state'].values());c.restore(state,self.model,self.opt,source_sch);del state
            ds=old.dataset(domain,'train_labeled');assert ds.rows==binding['l_rows'];self.L=[ds[i] for i in range(len(ds))];u=UImages(os.environ['EXEC_U'],domain);assert u.rows==binding['u_rows'];self.U=[u[i] for i in range(len(u))]
        self.source_scheduler=copy.deepcopy(source_sch.state_dict());self.sch=LocalSchedule(self.opt);self.teacher=make_reference(self.model);self.policy=self.reference=self.popt=None;self.t=self.retained=0;self.elapsed=0.;self.root=self.snapshot();self.root_fp=fingerprint(self.root)
    def charge_step(self,opt,kind,transaction):self.ledger.step(opt,kind,self.task+'/'+transaction)
    def reset(self):
        self.restore(self.root);assert same(self.snapshot(),self.root) and fingerprint(self.root)==self.root_fp
        for i,live in self.opt.state_dict()['state'].items():
            for k,v in live.items():
                stored=self.root['optimizer']['state'][i][k]
                if torch.is_tensor(v) and v.device.type=='cpu':assert v.untyped_storage().data_ptr()!=stored.untyped_storage().data_ptr()
    def advance(self,x,y,us,q,v,a,key):
        info=self.student_step(x,y,us,q,v,a,key,temporary=True);ema_update(self.model,self.teacher,.99);self.t+=1;self.retained+=1;assert self.sch.t==self.t
        entry=0 if self.synthetic else 2000;assert all(int(x['step'])==entry+self.t for x in self.opt.state.values());return info
    def clean_score(self,indices,purpose):
        before=self.snapshot();val=self.score(indices,purpose);assert same(before,self.snapshot()),'feedback changed state';return val
    def branch(self,j,a,repeat,scene):
        key=f'{j}/{a}/{repeat}';target=self.folder/(key.replace('/','_')+'.private.json')
        if target.exists():
            row=compatible(target);assert row['config_sha']==CONFIG_SHA;return row
        self.reset();row=dict(code_commit=CODE,config_sha=CONFIG_SHA,prefix_sha=self.meta['prefix_sha'],schedule_sha=self.meta['schedule_sha'],scene=j,action=a,repeat=repeat,root_fingerprint=self.root_fp,horizons={},steps=[],state=None)
        for h,item in enumerate(scene['items'],1):
            x,y=old.load_batch(self.L,item,torch.device('cuda:0'));us,q,v,z,flags=self.context(item,x,y)
            if h==1:row['state']=z.tolist()
            info=self.advance(x,y,us,q,v,a if h==1 else 2,key+'/'+str(h));row['steps'].append(dict(h=h,info=info,teacher_student_l2=float(sum((p.detach().float()-q.detach().float()).square().sum() for p,q in zip(self.model.parameters(),self.teacher.parameters())).sqrt()),local_scheduler=self.sch.t,adam_step=2000+h,teacher_updated=True))
            if h in (1,5):row['horizons'][str(h)]=dict(online=self.clean_score(item['online'],'online'),audit=self.clean_score(item['audit'],'audit'))
        self.reset();row['root_restored']=True;row['retained_after_discard']=0;write(target,row);return row

def prepare():
    # Freeze grouping and all schedules before opening any medical feedback/rewards.
    bindings=read(PREVIOUS/'PREFIX_BINDINGS.private.json');data=read(PREVIOUS/'DATA_BINDING.json');manifest={};oof={}
    for b in CONFIG['backbones']:
        for d in CONFIG['domains']:
            cell=b+'__'+d;p=PREVIOUS/'schedules'/f'261__{d}.private.json';assert c.file_sha(p)==data['schedules'][f'261__{d}'];doc=read(p);pref=bindings.get(f'FROZEN_PREFIX__S261__{cell}',{'valid':False});scenes=[]
            if not pref['valid'] or not Path(pref['path']).is_file():
                manifest[cell]=dict(eligible=False,reason='missing verified prefix');oof[cell]=dict(eligible=False,reason='missing verified prefix');continue
            for j in range(16):
                origin=20*(60*j//16);items=doc['steps'][origin:origin+5];assert all(all(x[k]==items[0][k] for k in ('fit','online','audit')) for x in items);u=doc['u_rows'][items[0]['u']];scenes.append(dict(j=j,origin_t=origin,items=items,U_identity=u['sha256']))
            manifest[cell]=dict(eligible=True,prefix=pref,schedule_sha=c.file_sha(p),scenes=scenes,l_rows=doc['l_rows'],u_rows=doc['u_rows'])
            try:oof[cell]=dict(eligible=True,groups=[s['U_identity'] for s in scenes],fold=control.folds([s['U_identity'] for s in scenes]))
            except AssertionError as e:oof[cell]=dict(eligible=False,reason=str(e))
    for name,doc in [('CONTEXT_MANIFEST.private.json',manifest),('OOF_SPLITS.private.json',oof)]:
        p=ROOT/name
        if p.exists():assert read(p)==doc
        else:write(p,doc)
    write(ROOT/'PROVENANCE.json',dict(code_commit=CODE,config_sha=CONFIG_SHA,base_commit=CONFIG['base_commit'],manifest_sha=c.file_sha(ROOT/'CONTEXT_MANIFEST.private.json'),oof_sha=c.file_sha(ROOT/'OOF_SPLITS.private.json'),data_manifest_sha=data['manifest_sha'],split_sha=data['split_sha']))

def synthetic(ledger):
    receipt=ROOT/'SYNTHETIC_CONTROLLABILITY.json'
    if receipt.exists():compatible(receipt);return
    results=[]
    for seed in CONFIG['synthetic_seeds']:
        rng=np.random.default_rng(seed);z=rng.uniform(-1,1,(64,16));d_zero=np.zeros((64,3));d_fine=np.tile([-1.,-.5,0.],(64,1));switch=np.where(z[:,0]>0,.5,-.5);d_switch=np.column_stack((switch,-switch,np.zeros(64)))
        for name,d in [('fine_best',d_fine),('state_switch',d_switch),('zero',d_zero)]:
            model,diag=control.fit(z,d,seed,ledger.step,f'synthetic/{seed}/{name}',kind='controller_synthetic');p=model(torch.tensor(z,dtype=torch.float32)).detach().numpy();value=float((p*d).sum(1).mean());prior=float((control.PRIOR*d).sum(1).mean());passed=value>prior if name!='zero' else np.max(np.abs(p-control.PRIOR))<1e-5
            results.append(dict(seed=seed,problem=name,value=value,prior_value=prior,behavior_pass=bool(passed),**diag))
    write(receipt,dict(code_commit=CODE,config_sha=CONFIG_SHA,optimizer_calls=3072,results=results,interpretation='Learning insufficiency is recorded, not a scientific stop gate.'))

def native(cell,ledger):
    p=ROOT/'qualification'/f'{cell}.json'
    if p.exists():compatible(p);return
    w=PanelWorker(cell,ledger,True);torch.manual_seed(261);x=torch.rand(2,3,384,384,device='cuda');y=torch.zeros(2,384,384,device='cuda',dtype=torch.long);y[:,64:300,64:300]=1;y[:,128:224,128:224]=2;q=torch.zeros(1,3,384,384,device='cuda');q[:,0]=.95;q[:,1:]=.025;v=torch.ones(1,384,384,device='cuda',dtype=torch.bool);states=[]
    for repeat in range(2):
        w.reset()
        for h in range(5):w.advance(x,y,x[:1],q,v,0 if h==0 else 2,f'qualification/{repeat}/{h}')
        states.append(cpu_copy(w.model.state_dict()));w.reset()
    assert all(torch.isfinite(v).all() for state in states for v in state.values())
    repeat_max=max(float((states[0][k].float()-states[1][k].float()).abs().max()) for k in states[0])
    write(p,dict(code_commit=CODE,student_synthetic_calls=10,repeat_student_max_abs=repeat_max,repeat_difference_is_diagnostic=True,full_root_restored=True,storage_isolated=True,teacher_each_step=True,adam_step=5,local_scheduler=5));del w;torch.cuda.empty_cache()

def run():
    torch.set_num_threads(2)
    auth=read(ROOT/'AUTHORIZATION.private.json');assert auth['run_authorized']
    old.provenance();ROOT.mkdir(exist_ok=True);ledger=Ledger()
    if not (ROOT/'SESSION.json').exists():write(ROOT/'SESSION.json',dict(start=time.time(),deadline=time.time()+43200,code_commit=CODE))
    else:compatible(ROOT/'SESSION.json')
    write(ROOT/'PROCESS.json',dict(pid=os.getpid(),identity=__import__('r1_6_readout_v1.runtime',fromlist=['process_identity']).process_identity(os.getpid()),code_commit=CODE))
    prepare();synthetic(ledger)
    scale_bindings={}
    for cell in read(ROOT/'CONTEXT_MANIFEST.private.json'):
        p=PREVIOUS/'tasks'/('R3A__'+cell)/'R3A_DONE.json';doc=read(p);assert np.isfinite(doc['scale']) and doc['scale']>0;scale_bindings[cell]=dict(scale=doc['scale'],sha256=c.file_sha(p))
    target=ROOT/'SCALE_BINDINGS.json'
    if target.exists():assert read(target)==scale_bindings
    else:write(target,scale_bindings)
    from .analysis import p0,panels,p2,report
    p0();manifest=read(ROOT/'CONTEXT_MANIFEST.private.json');failures=[]
    for b in CONFIG['backbones']:
        try:native(b+'__RIM_ONE_r3',ledger)
        except Exception as e:failures.append(dict(stage='qualification',backbone=b,error=repr(e)));continue
        for d in CONFIG['domains']:
            cell=b+'__'+d
            if not manifest[cell]['eligible']:failures.append(dict(stage='prefix',cell=cell,error=manifest[cell]['reason']));continue
            try:
                w=PanelWorker(cell,ledger)
                for scene in manifest[cell]['scenes']:
                    for a in (0,1,2):w.kind='student_main';w.branch(scene['j'],a,0,scene)
                    if scene['j'] in (0,8):
                        for a in (0,2):w.kind='student_null';w.branch(scene['j'],a,1,scene)
                del w;torch.cuda.empty_cache()
            except Exception as e:failures.append(dict(stage='P1',cell=cell,error=repr(e),traceback=traceback.format_exc()))
            finally:
                if 'w' in locals():del w
                gc.collect();torch.cuda.empty_cache()
    panels();p2(ledger);report();write(ROOT/'BUDGET.json',dict(ledger.count));write(ROOT/'FINAL.json',dict(status='COMPLETE' if not failures and len(list((ROOT/'panels').glob('*/*.private.json')))==208 and len(read(ROOT/'P1_VALUES.private.json'))==4 and len(list((ROOT/'fits').glob('*.private.json')))==64 else 'INCOMPLETE',failures=failures,code_commit=CODE,config_sha=CONFIG_SHA,finished=time.time(),persistent_student_updates=0,student_endpoints=0))

def main():
    started=time.time()
    try:run()
    except BaseException:
        write(ROOT/'FAILURE.private.json',dict(code_commit=CODE,error=traceback.format_exc(),time=time.time()));raise
    finally:
        ev=records(ROOT/'PHYSICAL_LEDGER.jsonl');counts=Counter(x['category'] for x in ev if x['event']=='attempt');success=sum(x['event']=='success' for x in ev);failed=sum(x['event']=='failed' for x in ev)
        write(ROOT/'BUDGET.json',dict(counts));write(ROOT/'reports/RESOURCE_REPORT.json',dict(code_commit=CODE,physical_attempts=dict(counts),successful_calls=success,failed_calls=failed,unresolved_calls=sum(counts.values())-success-failed,process_wall_seconds=time.time()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,persistent_student_updates=0,student_endpoints=0,real_L_smoke=0))
        (ROOT/'PATCH_LOG.jsonl').touch(exist_ok=True)

if __name__=='__main__':main()
