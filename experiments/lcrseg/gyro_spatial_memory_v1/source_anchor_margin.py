"""Complete reliable margin with normalized source-parameter anchoring on current L only."""
import copy,json,math,os,subprocess,traceback
from pathlib import Path
from unittest.mock import patch
import torch
from torch.nn import functional as F
from . import source_decision as base
from .source_decision import HEAD,Calls,Current,build,setup,update_ema,train,qualify,evaluate,preflight,geometry_check
from . import core,target,runtime,spectrum_alignment as shared
from .bounded_runtime import install
from ..main_style_ssl_v1.runtime import write,append
from ..siemens_ssl_reproduction_v1.methods import pixel_ce
from ..five_frameworks_v1.recipes import generator
SEEDS=(20271101,20271102,20271103)
METHODS=('B','R','P','C')
CAPS={f'{a}_{b}':n for a,ns in [('p0',(3000,1024)),('detach',(1024,128)),('engineering',(6000,1024)),('formal',(144000,24300)),('evaluation',(34704,0))] for b,n in zip(('native','reverse'),ns)}

from .reliable_margin import weak,margins,references,choose,take,hinge,terms,margin_self_check

def parameter_anchor(m):
    # The frozen reference already exists; no new model, historical data, or buffers.
    source=dict(m.reference.named_parameters())
    pairs=[(p,source[n].detach()) for n,p in m.named_parameters() if p.requires_grad]
    den=sum(q.square().sum() for p,q in pairs).detach()+1e-12
    return sum((p-q).square().sum() for p,q in pairs)/den

def losses(arm,m,data,step):
    l,aux=terms(m,data,step);p=parameter_anchor(m);w=m.cfg.get('weights',{});v=m.cfg['variant']
    u=l*0
    if v in ('R','C'):u=u+w.get('R',0.)*aux['R']
    if v in ('P','C'):u=u+w.get('P',0.)*p
    if m.cfg['bucket']=='formal' and step in (0,999,1999):
        pars=[p for p in m.parameters() if p.requires_grad]
        gs={k:torch.autograd.grad(t,pars,retain_graph=True,allow_unused=True) for k,t in [('CE',l),('R',aux['R']),('P',p)]}
        squared={k:sum(float(g.double().square().sum()) for g in vv if g is not None) for k,vv in gs.items()}
        dot=sum(float((a.double()*b.double()).sum()) for a,b in zip(gs['R'],gs['P']) if a is not None and b is not None)
        append(Path(m.cfg['output'])/'gradient_diagnostics.jsonl',dict(step=step,squared=squared,R_P_dot=dot,parameter_anchor=float(p.detach()),active_R=m.detail['active']['R']))
    stats=dict(supervised=float(l.detach()),unsupervised=float(u.detach()),R=float(aux['R'].detach()),P=float(p.detach()),accepted_class_pixels=[0]*4,U_pixels=0,historical_raw_reads=0)
    m.detail=None;return l,u,stats

def calibrate_scales(cfg):
    source,m,ema,opt=setup(cfg);data=Current(dict(cfg,target_seed=20271191));values=[[] for _ in range(4)]
    for step in range(16):
        x,y,_=data.batch('L',step);_,_,_,a,_,support,_=references(m,x,y,data.cfg['target_seed'],step)
        ix=choose(support,data.cfg['target_seed'],step);labels=take(y[...,None],ix).squeeze(-1);v=take(a,ix)
        for c in range(4):
            v_c=v[labels==c][:,[j for j in range(4) if j!=c]];values[c].append(v_c[v_c>0].cpu())
    scales=[float(torch.cat(v).median()) if sum(len(t) for t in v) else 0. for v in values]
    write(Path(cfg['root'])/'SCALES.json',dict(scales=scales,status='PASS' if all(math.isfinite(s) and s>1e-6 for s in scales) else 'STOP',source_only=True,train_seed=20271191))

def measure(cfg):
    scales=json.loads((Path(cfg['root'])/'SCALES.json').read_text())
    if scales['status']!='PASS':write(Path(cfg['root'])/'P0_RESULT.json',dict(status='STOP',gates={'scales':False}));return
    cfg=dict(cfg,scales=scales['scales']);source,m,ema,opt=setup(cfg);data=Current(dict(cfg,target_seed=20271192))
    saved=torch.load(Path(cfg['root'])/'probe/final.pt',map_location='cpu',weights_only=False);assert saved['step']==256;m.load_state_dict(saved['model'])
    before=copy.deepcopy(m.state_dict());ref=copy.deepcopy(source.state_dict());rng=(torch.get_rng_state(),torch.cuda.get_rng_state_all())
    names=[n for n,p in m.named_parameters() if p.requires_grad];pars=[p for p in m.parameters() if p.requires_grad]
    squares={k:0. for k in ('CE','R','P')};dot=0.;covered=[0]*4;denom=[0]*4;patients=[set() for _ in range(4)];energy=[0.]*4;groups={k:dict(encoder=0.,decoder=0.) for k in ('R','P')};active=[];anchor_values=[]
    for step in range(16):
        l,aux=terms(m,data,step);d=m.detail;p=parameter_anchor(m);anchor_values.append(float(p.detach()))
        gs={k:torch.autograd.grad(t,pars,retain_graph=True,allow_unused=True) for k,t in [('CE',l),('R',aux['R']),('P',p)]}
        for k,gg in gs.items():
            assert all(g is None or torch.isfinite(g).all() for g in gg);squares[k]+=sum(float(g.double().square().sum()) for g in gg if g is not None)
        dot+=sum(float((a.double()*b.double()).sum()) for a,b in zip(gs['R'],gs['P']) if a is not None and b is not None)
        for k in ('R','P'):
            for name,g in zip(names,gs[k]):
                if g is not None:groups[k]['decoder' if 'native.decoder.' in name else 'encoder']+=float(g.double().square().sum())
        for c in range(4):
            mask=d['support']&(d['y']==c);covered[c]+=int(mask.sum());denom[c]+=int((d['y']==c).sum())
            for slot,pid in enumerate(d['ids']):
                if mask[slot].any():patients[c].add(pid)
            chosen=d['values']['R'][d['labels']==c];energy[c]+=float(chosen.mean().detach())/max(1,len(d['labels'].unique())) if len(chosen) else 0.
        active.append(d['active']['R']);m.detail=None
    cosine=dot/math.sqrt(squares['R']*squares['P']) if squares['R']*squares['P']>0 else None;distance=math.sqrt(max(0.,2-2*cosine)) if cosine is not None else 0.
    weights={k:.25*math.sqrt(squares['CE']/squares[k]) if squares[k]>0 else None for k in ('R','P')}
    coverage=[a/b if b else 0. for a,b in zip(covered,denom)];share=sum(energy[1:])/max(sum(energy),1e-30)
    combo=weights['R']**2*squares['R']+weights['P']**2*squares['P']+2*weights['R']*weights['P']*dot if all(v is not None for v in weights.values()) else 0.
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
    from types import SimpleNamespace
    for device in ('cpu','cuda'):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__();self.w=torch.nn.Parameter(torch.tensor([2.,1.],device=device));self.fixed=torch.nn.Parameter(torch.tensor([3.],device=device),requires_grad=False)
        m=Toy();ref=copy.deepcopy(m).requires_grad_(False);object.__setattr__(m,'reference',ref)
        loss=parameter_anchor(m);g=torch.autograd.grad(loss,m.w)[0];assert float(loss)==0 and torch.count_nonzero(g)==0
        with torch.no_grad():m.w.add_(torch.tensor([1.,-1.],device=device))
        loss=parameter_anchor(m);g=torch.autograd.grad(loss,m.w)[0];assert torch.allclose(loss,torch.tensor(.4,device=device)) and torch.allclose(g,torch.tensor([.4,-.4],device=device))
        assert all(p.grad is None for p in ref.parameters()) and m.fixed.grad is None
    return dict(status='PASS',synthetic_reverse=4,zero_at_source=True,restoring_gradient=True,frozen_excluded=True,reference_detached=True)

def controller(cfg,guard):
    root=Path(cfg['root']);assert json.loads((Path(cfg['regional_root'])/'AUDIT.json').read_text())['status']=='PASS' and (Path(cfg['regional_root'])/'DELIVERY.json').exists()
    write(root/'WORKER.private.json',dict(pid=os.getpid(),birth=Path('/proc/self/stat').read_text().split()[21]));geometry_check(cfg);write(root/'MARGIN_SELF_CHECK.json',margin_self_check());write(root/'ANCHOR_SELF_CHECK.json',anchor_self_check())
    def stage(mode,c):
        guard(cfg,new_stage=True);assert int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))>=18000
        p=root/('SETTINGS_'+c['group']+'_'+mode+'.private.json');write(p,c)
        with (root/(c['group']+'_'+mode+'.log')).open('x') as log:q=subprocess.Popen([cfg['runtime'],cfg['entry']],env=dict(os.environ,TASK_SETTINGS=str(p),TASK_MODE=mode),stdout=log,stderr=subprocess.STDOUT)
        a=dict(pid=q.pid,birth=Path('/proc/'+str(q.pid)+'/stat').read_text().split()[21],group=c['group'],mode=mode);write(root/'ACTIVE.private.json',a);write(root/('ACTIVE_'+c['group']+'_'+mode+'.private.json'),a);write(root/'STATUS.json',dict(status='RUNNING',**a))
        code=q.wait();write(root/('EXIT_'+c['group']+'_'+mode+'.json'),dict(exit_code=code));assert code==0
    common=dict(cfg,arm='A',preflight_only=False,checkpoint_each_step=True,save_exception_state=True,diagnostic_holdout_excluded=False)
    stage('qualify',dict(common,group='P0_ENGINEERING',variant='B',target_seed=20271190,bucket='p0',phase='engineering',output=str(root/'p0_engineering')))
    c=dict(common,group='PROBE',variant='B',target_seed=20271190,bucket='p0',phase='probe',output=str(root/'probe'))
    stage('probe',c);stage('scales',c);stage('measure',c);p0=json.loads((root/'P0_RESULT.json').read_text())
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
        with patch.object(base,'losses',losses),patch.object(shared,'CAPS',CAPS),patch.object(core,'build',build),patch.object(core,'target_setup',setup),patch.object(core,'losses',losses),patch.object(core,'update_ema',update_ema),patch.object(target,'Domain',Current),patch.object(target,'preflight',preflight),patch.object(torch.autograd,'backward',b),patch.object(torch.autograd,'grad',g):
            {'qualify':qualify,'probe':lambda c:train(c,256),'train':train,'scales':calibrate_scales,'measure':measure,'detach':detach,'evaluate':evaluate}[mode](cfg)
    except BaseException as exc:
        write(root/('FAILURE_'+cfg.get('group','controller')+'_'+mode+'.json'),dict(error=str(exc),traceback=traceback.format_exc()))
        if mode=='controller':write(root/'STATUS.json',dict(status='FAILED'))
        raise
