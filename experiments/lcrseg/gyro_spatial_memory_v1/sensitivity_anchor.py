"""Complete reliable margin with reliable source-sensitivity parameter anchoring on current L only."""
import copy,json,math,os,subprocess,traceback
from pathlib import Path
from unittest.mock import patch
import torch
from torch.nn import functional as F
from . import source_decision as base
from .source_decision import HEAD,Calls,Current,build,setup as native_setup,update_ema,train,qualify,evaluate,preflight,geometry_check
from . import core,target,runtime,spectrum_alignment as shared
from .bounded_runtime import install
from ..main_style_ssl_v1.runtime import write,append
from ..siemens_ssl_reproduction_v1.methods import pixel_ce
from ..five_frameworks_v1.recipes import generator
SEEDS=(20271301,20271302,20271303)
METHODS=('B','R','I','S','C')
CAPS={f'{a}_{b}':n for a,ns in [('p0',(3200,1200)),('detach',(1024,128)),('engineering',(6000,1200)),('formal',(180000,30400)),('evaluation',(43380,0))] for b,n in zip(('native','reverse'),ns)}

from .reliable_margin import weak,margins,references,choose,take,hinge,terms,margin_self_check

def setup(cfg):
    result=native_setup(cfg)
    path=Path(cfg['root'])/'SENSITIVITY.private.pt'
    if path.exists():
        saved=torch.load(path,map_location='cuda',weights_only=False)
        object.__setattr__(result[1],'sensitivity',saved['diagonal'])
    return result

def parameter_anchor(m):
    source=dict(m.reference.named_parameters())
    pairs=[(p,source[n].detach()) for n,p in m.named_parameters() if p.requires_grad]
    return sum((p-q).square().sum() for p,q in pairs)/(sum(q.square().sum() for p,q in pairs).detach()+1e-12)

def sensitivity_anchor(m):
    source=dict(m.reference.named_parameters())
    pairs=[(p,source[n].detach(),m.sensitivity[n]) for n,p in m.named_parameters() if p.requires_grad]
    return sum((w*(p-q).square()).sum() for p,q,w in pairs)/(sum((w*q.square()).sum() for p,q,w in pairs).detach()+1e-12)

def calibrate_sensitivity(cfg):
    source,m,ema,opt=setup(cfg);m.eval();data=Current(dict(cfg,target_seed=20271393))
    before=copy.deepcopy(m.state_dict());reference=copy.deepcopy(source.state_dict());rng=(torch.get_rng_state(),torch.cuda.get_rng_state_all())
    names=[n for n,p in m.named_parameters() if p.requires_grad];pars=[p for p in m.parameters() if p.requires_grad]
    diagonal={n:torch.zeros_like(p) for n,p in zip(names,pars)};support_counts=[0]*4;denom=[0]*4;patients=[set() for _ in range(4)]
    for step in range(16):
        x,y,ids=data.batch('L',step);z=m(x)
        _,_,_,_,_,support,reliability=references(m,x,y,data.cfg['target_seed'],step)
        ix=choose(support,data.cfg['target_seed'],step);labels=take(y[...,None],ix).squeeze(-1)
        logits=take(z.permute(0,2,3,1),ix);rel=take(reliability[...,None],ix).squeeze(-1)
        values=F.cross_entropy(logits,labels.long(),reduction='none')*rel
        cls=[values[labels==c].mean() for c in range(4) if (labels==c).any()]
        loss=torch.stack(cls).mean() if cls else z.sum()*0
        gs=torch.autograd.grad(loss,pars,allow_unused=True)
        for n,g in zip(names,gs):
            if g is not None:diagonal[n].add_(g.detach().square()/16)
        for c in range(4):
            mask=support&(y==c);support_counts[c]+=int(mask.sum());denom[c]+=int((y==c).sum())
            for slot,pid in enumerate(ids):
                if mask[slot].any():patients[c].add(pid)
    assert runtime.exact(before,m.state_dict()) and runtime.exact(reference,source.state_dict()) and runtime.exact(rng,(torch.get_rng_state(),torch.cuda.get_rng_state_all()))
    assert all(p.grad is None for p in source.parameters()) and all(p.grad is None for p in pars)
    reference_parameters=dict(source.named_parameters());mass=sum(float(w.double().sum()) for w in diagonal.values());den=sum(float((w.double()*reference_parameters[n].detach().double().square()).sum()) for n,w in diagonal.items())
    coverage=[a/b if b else 0. for a,b in zip(support_counts,denom)]
    gate=all(torch.isfinite(w).all() and (w>=0).all() for w in diagonal.values()) and mass>1e-12 and den>1e-12 and all(coverage[c]>=.05 and len(patients[c])>=3 for c in (1,2,3))
    torch.save(dict(diagonal={n:w.cpu() for n,w in diagonal.items()}),Path(cfg['root'])/'SENSITIVITY.private.pt')
    write(Path(cfg['root'])/'SENSITIVITY_RESULT.json',dict(status='PASS' if gate else 'STOP',train_seed=20271393,batches=16,gradient_mass=mass,source_weighted_denominator=den,coverage_by_class=coverage,covered_patient_counts=[len(v) for v in patients],measurement_student_at_source=True,reference_no_gradient=True,model_reference_RNG_unchanged=True,is_fisher_information=False))

def losses(arm,m,data,step):
    l,aux=terms(m,data,step);p=parameter_anchor(m);q=sensitivity_anchor(m) if hasattr(m,'sensitivity') else p*0;w=m.cfg.get('weights',{});v=m.cfg['variant']
    if v in ('S','C') or m.cfg['bucket']=='formal':assert hasattr(m,'sensitivity')
    u=l*0
    if v in ('R','I','C'):u=u+w.get('R',0.)*aux['R']
    if v=='I':u=u+w.get('P',0.)*p
    if v in ('S','C'):u=u+w.get('S',0.)*q
    if m.cfg['bucket']=='formal' and step in (0,999,1999):
        pars=[p for p in m.parameters() if p.requires_grad]
        gs={k:torch.autograd.grad(t,pars,retain_graph=True,allow_unused=True) for k,t in [('CE',l),('R',aux['R']),('P',p),('S',q)]}
        squared={k:sum(float(g.double().square().sum()) for g in vv if g is not None) for k,vv in gs.items()}
        dot=sum(float((a.double()*b.double()).sum()) for a,b in zip(gs['S'],gs['P']) if a is not None and b is not None)
        append(Path(m.cfg['output'])/'gradient_diagnostics.jsonl',dict(step=step,squared=squared,S_P_dot=dot,parameter_anchor=float(p.detach()),sensitivity_anchor=float(q.detach()),active_R=m.detail['active']['R']))
    stats=dict(supervised=float(l.detach()),unsupervised=float(u.detach()),R=float(aux['R'].detach()),P=float(p.detach()),S=float(q.detach()),accepted_class_pixels=[0]*4,U_pixels=0,historical_raw_reads=0)
    m.detail=None;return l,u,stats

def calibrate_scales(cfg):
    source,m,ema,opt=setup(cfg);data=Current(dict(cfg,target_seed=20271391));values=[[] for _ in range(4)]
    for step in range(16):
        x,y,_=data.batch('L',step);_,_,_,a,_,support,_=references(m,x,y,data.cfg['target_seed'],step)
        ix=choose(support,data.cfg['target_seed'],step);labels=take(y[...,None],ix).squeeze(-1);v=take(a,ix)
        for c in range(4):
            v_c=v[labels==c][:,[j for j in range(4) if j!=c]];values[c].append(v_c[v_c>0].cpu())
    scales=[float(torch.cat(v).median()) if sum(len(t) for t in v) else 0. for v in values]
    write(Path(cfg['root'])/'SCALES.json',dict(scales=scales,status='PASS' if all(math.isfinite(s) and s>1e-6 for s in scales) else 'STOP',source_only=True,train_seed=20271391))

def measure(cfg):
    scales=json.loads((Path(cfg['root'])/'SCALES.json').read_text())
    if scales['status']!='PASS' or json.loads((Path(cfg['root'])/'SENSITIVITY_RESULT.json').read_text())['status']!='PASS':write(Path(cfg['root'])/'P0_RESULT.json',dict(status='STOP',gates={'scales':False}));return
    cfg=dict(cfg,scales=scales['scales']);source,m,ema,opt=setup(cfg);data=Current(dict(cfg,target_seed=20271392))
    saved=torch.load(Path(cfg['root'])/'probe/final.pt',map_location='cpu',weights_only=False);assert saved['step']==256;m.load_state_dict(saved['model'])
    before=copy.deepcopy(m.state_dict());ref=copy.deepcopy(source.state_dict());rng=(torch.get_rng_state(),torch.cuda.get_rng_state_all())
    names=[n for n,p in m.named_parameters() if p.requires_grad];pars=[p for p in m.parameters() if p.requires_grad]
    squares={k:0. for k in ('CE','R','P','S')};dot=0.;r_s_dot=0.;covered=[0]*4;denom=[0]*4;patients=[set() for _ in range(4)];energy=[0.]*4;groups={k:dict(encoder=0.,decoder=0.) for k in ('R','P','S')};active=[];anchor_values=[]
    for step in range(16):
        l,aux=terms(m,data,step);d=m.detail;p=parameter_anchor(m);q=sensitivity_anchor(m);anchor_values.append(float(q.detach()))
        gs={k:torch.autograd.grad(t,pars,retain_graph=True,allow_unused=True) for k,t in [('CE',l),('R',aux['R']),('P',p),('S',q)]}
        for k,gg in gs.items():
            assert all(g is None or torch.isfinite(g).all() for g in gg);squares[k]+=sum(float(g.double().square().sum()) for g in gg if g is not None)
        dot+=sum(float((a.double()*b.double()).sum()) for a,b in zip(gs['S'],gs['P']) if a is not None and b is not None)
        r_s_dot+=sum(float((a.double()*b.double()).sum()) for a,b in zip(gs['R'],gs['S']) if a is not None and b is not None)
        for k in ('R','P','S'):
            for name,g in zip(names,gs[k]):
                if g is not None:groups[k]['decoder' if 'native.decoder.' in name else 'encoder']+=float(g.double().square().sum())
        for c in range(4):
            mask=d['support']&(d['y']==c);covered[c]+=int(mask.sum());denom[c]+=int((d['y']==c).sum())
            for slot,pid in enumerate(d['ids']):
                if mask[slot].any():patients[c].add(pid)
            chosen=d['values']['R'][d['labels']==c];energy[c]+=float(chosen.mean().detach())/max(1,len(d['labels'].unique())) if len(chosen) else 0.
        active.append(d['active']['R']);m.detail=None
    cosine=dot/math.sqrt(squares['S']*squares['P']) if squares['S']*squares['P']>0 else None;distance=math.sqrt(max(0.,2-2*cosine)) if cosine is not None else 0.
    weights={k:.25*math.sqrt(squares['CE']/squares[k]) if squares[k]>0 else None for k in ('R','P','S')}
    coverage=[a/b if b else 0. for a,b in zip(covered,denom)];share=sum(energy[1:])/max(sum(energy),1e-30)
    combo=weights['R']**2*squares['R']+weights['S']**2*squares['S']+2*weights['R']*weights['S']*r_s_dot if all(v is not None for v in weights.values()) else 0.
    gates=dict(scales=True,gradient_calibration=min(squares.values())>=1e-12 and all(v is not None and math.isfinite(v) and 1e-6<=v<=1e4 for v in weights.values()),foreground_coverage=all(coverage[c]>=.05 and len(patients[c])>=3 for c in (1,2,3)),R_foreground_loss_share=share>=.05,gradient_separation=distance>=.05,full_path_gradient=all(v>0 and math.isfinite(v) for gg in groups.values() for v in gg.values()),combination_nonzero=math.isfinite(combo) and combo>=1e-12,anchor_nonzero=all(math.isfinite(v) and v>0 for v in anchor_values))
    assert runtime.exact(before,m.state_dict()) and runtime.exact(ref,source.state_dict()) and runtime.exact(rng,(torch.get_rng_state(),torch.cuda.get_rng_state_all()))
    assert all(p.grad is None for p in source.parameters()) and m.get_parameter(HEAD).grad is None
    write(Path(cfg['root'])/'P0_RESULT.json',dict(status='PASS' if all(gates.values()) else 'STOP',gates=gates,weights=weights,scales=cfg['scales'],gradient_squared=squares,normalized_gradient_distance=distance,gradient_cosine=cosine,coverage_by_class=coverage,covered_patient_counts=[len(v) for v in patients],R_foreground_loss_share=share,gradient_groups=groups,combination_gradient_squared=combo,anchor_values=anchor_values,active_R_per_batch=active,parameter_prior_has_no_class_loss_decomposition=True,model_reference_RNG_unchanged=True,probe_discarded=True))

def detach(cfg):
    models=[]
    for variant in ('B','C'):
        c=dict(cfg,group='DETACH_'+variant,variant=variant,weights={});_,m,e,o=setup(c);models.append((m,e,o,Current(c)))
    for step in range(32):
        rng=[]
        for m,e,o,d in models:
            o.zero_grad(set_to_none=True);l,u,_=losses('A',m,d,step);(l+u).backward();o.step();update_ema(e,m);rng.append((torch.get_rng_state(),torch.cuda.get_rng_state_all()))
        a,b=models
        assert runtime.exact(a[0].state_dict(),b[0].state_dict()) and runtime.exact(a[1].state_dict(),b[1].state_dict()) and runtime.exact(a[2].state_dict(),b[2].state_dict()) and runtime.exact(rng[0],rng[1])
    write(Path(cfg['root'])/'DETACH_CHECK.json',dict(status='PASS',updates=64,model_opt_EMA_RNG_exact=True,discarded=True))

def anchor_self_check():
    for device in ('cpu','cuda'):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__();self.w=torch.nn.Parameter(torch.tensor([2.,1.],device=device));self.fixed=torch.nn.Parameter(torch.tensor([3.],device=device),requires_grad=False)
        m=Toy();ref=copy.deepcopy(m).requires_grad_(False);object.__setattr__(m,'reference',ref);object.__setattr__(m,'sensitivity',{'w':torch.tensor([1.,0.],device=device)})
        value=sensitivity_anchor(m);assert float(value)==0 and torch.count_nonzero(torch.autograd.grad(value,m.w)[0])==0
        with torch.no_grad():m.w.add_(torch.tensor([1.,-1.],device=device))
        value=sensitivity_anchor(m);g=torch.autograd.grad(value,m.w)[0];assert torch.allclose(value,torch.tensor(.25,device=device)) and torch.allclose(g,torch.tensor([.5,0.],device=device))
        m.sensitivity={'w':torch.ones(2,device=device)};assert torch.equal(sensitivity_anchor(m),parameter_anchor(m));assert all(p.grad is None for p in ref.parameters()) and m.fixed.grad is None
    return dict(status='PASS',synthetic_reverse=4,zero_at_source=True,weighted_restoring_gradient=True,zero_sensitivity_free=True,uniform_reduces_to_isotropic=True,frozen_excluded=True,reference_detached=True)

def controller(cfg,guard):
    root=Path(cfg['root']);assert json.loads((Path(cfg['compatible_anchor_root'])/'AUDIT.json').read_text())['status']=='PASS' and (Path(cfg['compatible_anchor_root'])/'DELIVERY.json').exists()
    write(root/'WORKER.private.json',dict(pid=os.getpid(),birth=Path('/proc/self/stat').read_text().split()[21]));geometry_check(cfg);write(root/'MARGIN_SELF_CHECK.json',margin_self_check());write(root/'ANCHOR_SELF_CHECK.json',anchor_self_check())
    def stage(mode,c):
        guard(cfg,new_stage=True);assert int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))>=18000
        p=root/('SETTINGS_'+c['group']+'_'+mode+'.private.json');write(p,c)
        with (root/(c['group']+'_'+mode+'.log')).open('x') as log:q=subprocess.Popen([cfg['runtime'],cfg['entry']],env=dict(os.environ,TASK_SETTINGS=str(p),TASK_MODE=mode),stdout=log,stderr=subprocess.STDOUT)
        a=dict(pid=q.pid,birth=Path('/proc/'+str(q.pid)+'/stat').read_text().split()[21],group=c['group'],mode=mode);write(root/'ACTIVE.private.json',a);write(root/('ACTIVE_'+c['group']+'_'+mode+'.private.json'),a);write(root/'STATUS.json',dict(status='RUNNING',**a))
        code=q.wait();write(root/('EXIT_'+c['group']+'_'+mode+'.json'),dict(exit_code=code));assert code==0
    common=dict(cfg,arm='A',preflight_only=False,checkpoint_each_step=True,save_exception_state=True,diagnostic_holdout_excluded=False)
    stage('qualify',dict(common,group='P0_ENGINEERING',variant='B',target_seed=20271390,bucket='p0',phase='engineering',output=str(root/'p0_engineering')))
    c=dict(common,group='PROBE',variant='B',target_seed=20271390,bucket='p0',phase='probe',output=str(root/'probe'))
    stage('probe',c);stage('sensitivity',c);stage('scales',c);stage('measure',c);p0=json.loads((root/'P0_RESULT.json').read_text())
    if p0['status']!='PASS':write(root/'STATUS.json',dict(status='P0_STOP',formal_started=False,resources=guard(cfg)));write(root/'EXIT.json',dict(exit_code=0));return
    common.update(weights=p0['weights'],scales=p0['scales'])
    stage('detach',dict(common,group='DETACH',variant='B',target_seed=SEEDS[0],bucket='detach',phase='detach',output=str(root/'detach')))
    configs=[dict(common,group=f'{s}_{a}',variant=a,target_seed=s) for s in SEEDS for a in METHODS]
    for c in configs:stage('qualify',dict(c,bucket='engineering',phase='engineering',output=str(root/'qualification'/c['group'])))
    for c in configs:
        c=dict(c,bucket='formal',phase='formal',output=str(root/'jobs'/c['group']));stage('train',c);stage('evaluate',dict(c,bucket='evaluation'))
    write(root/'STATUS.json',dict(status='COMPLETE',audit_pending=True,resources=guard(cfg)));write(root/'EXIT.json',dict(exit_code=0))

def main(cfg):
    guard=install(cfg);target.guard=guard;mode=os.environ.get('TASK_MODE','controller');root=Path(cfg['root'])
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    try:
        if mode=='controller':controller(cfg,guard);return
        calls=Calls(cfg,mode);backward=torch.autograd.backward;grad=torch.autograd.grad
        def b(*a,**kw):return calls.call(calls.kind('reverse'),1,lambda:backward(*a,**kw))
        def g(*a,**kw):return calls.call(calls.kind('reverse'),1,lambda:grad(*a,**kw))
        with patch.object(base,'setup',setup),patch.object(base,'losses',losses),patch.object(shared,'CAPS',CAPS),patch.object(core,'build',build),patch.object(core,'target_setup',setup),patch.object(core,'losses',losses),patch.object(core,'update_ema',update_ema),patch.object(target,'Domain',Current),patch.object(target,'preflight',preflight),patch.object(torch.autograd,'backward',b),patch.object(torch.autograd,'grad',g):
            {'qualify':qualify,'probe':lambda c:train(c,256),'train':train,'sensitivity':calibrate_sensitivity,'scales':calibrate_scales,'measure':measure,'detach':detach,'evaluate':evaluate}[mode](cfg)
    except BaseException as exc:
        write(root/('FAILURE_'+cfg.get('group','controller')+'_'+mode+'.json'),dict(error=str(exc),traceback=traceback.format_exc()))
        if mode=='controller':write(root/'STATUS.json',dict(status='FAILED'))
        raise
