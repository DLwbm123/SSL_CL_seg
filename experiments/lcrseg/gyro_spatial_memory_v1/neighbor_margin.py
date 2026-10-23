"""Complete reliable margin with reliable adversarial-neighborhood margin protection on current L only."""
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
SEEDS=(20271401,20271402,20271403)
METHODS=('B','R','W','A','C')
CAPS={f'{a}_{b}':n for a,ns in [('p0',(6000,1500)),('detach',(1024,128)),('engineering',(6000,1200)),('formal',(480000,60200)),('evaluation',(43380,0))] for b,n in zip(('native','reverse'),ns)}

from .reliable_margin import weak,margins,references,choose,take,hinge,aggregate,margin_self_check

def original_terms(m,data,step):
    x,y,ids=data.batch('L',step);z=m(x);h=m.last_h
    z0,h0,a0,a,single,support,r=references(m,x,y,data.cfg['target_seed'],step)
    ix=choose(support,data.cfg['target_seed'],step);im=choose(single,data.cfg['target_seed'],step)
    labels=take(y[...,None],ix).squeeze(-1);lm=take(y[...,None],im).squeeze(-1)
    scales=torch.as_tensor(m.cfg.get('scales',[1.]*4),device=x.device,dtype=z.dtype)
    u,u0=base.patches(h,x.shape[-2:]),base.patches(h0,x.shape[-2:]);u,u0=take(u,ix),take(u0,ix).detach()
    reliability=take(r[...,None],ix).squeeze(-1)
    ds=((u-u0)@m.basis).square().sum(1)/(u0.square().sum(1).mean()+1e-12) if len(u) else u.sum(1)
    delta=z-z0;delta=delta-delta.mean(1,keepdim=True)
    k=take(delta.permute(0,2,3,1),ix).square().mean(-1)/scales[labels].square()
    ms=margins(z,y)
    vals={'S':ds*reliability,'K':k*reliability,'R':hinge(take(ms,ix),take(a,ix),labels,scales)*reliability,'M':hinge(take(ms,im),take(a0,im),lm,scales)}
    aux={key:aggregate(value,lm if key=='M' else labels,z.sum()) for key,value in vals.items()}
    boundary=F.max_pool2d(y.float()[:,None],3,1,1).squeeze(1)!=-F.max_pool2d(-y.float()[:,None],3,1,1).squeeze(1)
    m.detail=dict(x=x,reliability=r,labels=labels,lm=lm,values=vals,support=support,single=single,y=y,ids=ids,boundary=boundary,active={key:float((v>0).float().mean()) if len(v) else 0. for key,v in vals.items()})
    return pixel_ce(z,y).mean(),aux

def terms(m,data,step):
    l,aux=original_terms(m,data,step);d=m.detail;x=d['x'];y=d['y'];rel=d['reliability'];support=d['support']
    ix=choose(support,data.cfg['target_seed'],step);labels=take(y[...,None],ix).squeeze(-1)
    # Only input gradients flow through the frozen reference; its parameters remain detached.
    xg=x.detach().requires_grad_(True);zg=m.reference(xg)
    ce=F.cross_entropy(take(zg.permute(0,2,3,1),ix),labels.long(),reduction='none')*take(rel[...,None],ix).squeeze(-1)
    objective=aggregate(ce,labels,zg.sum());gx=torch.autograd.grad(objective,xg)[0].detach()
    g=generator(data.cfg['target_seed'],1,1,step,'neighbor_random')
    noise=(torch.rand(x.shape,generator=g)>=.5).to(x.device,dtype=x.dtype)*2-1
    views={'W':(x+.05*noise).clamp(-1,1).detach(),'A':(x+.05*gx.sign()).clamp(-1,1).detach()}
    branches={};scales=torch.as_tensor(m.cfg.get('scales',[1.]*4),device=x.device,dtype=x.dtype)
    for key,view in views.items():
        with torch.no_grad():zt=m.reference(view);mask=support&(zt.argmax(1)==y);reference=margins(zt,y)
        zs=m(view);chosen=choose(mask,data.cfg['target_seed'],step);yy=take(y[...,None],chosen).squeeze(-1)
        value=hinge(take(margins(zs,y),chosen),take(reference,chosen),yy,scales)*take(rel[...,None],chosen).squeeze(-1)
        aux[key]=aggregate(value,yy,zs.sum());branches[key]=dict(mask=mask,labels=yy,values=value,active=float((value>0).float().mean()) if len(value) else 0.)
    d['branches']=branches;d['input_gradient_squared']=float(gx.double().square().sum());d['adversarial_max_delta']=float((views['A']-x).abs().max())
    assert all(p.grad is None for p in m.reference.parameters())
    return l,aux

def losses(arm,m,data,step):
    l,aux=terms(m,data,step);w=m.cfg.get('weights',{});v=m.cfg['variant'];u=l*0
    if v in ('R','W','C'):u=u+w.get('R',0.)*aux['R']
    if v=='W':u=u+w.get('W',0.)*aux['W']
    if v in ('A','C'):u=u+w.get('A',0.)*aux['A']
    if m.cfg['bucket']=='formal' and step in (0,999,1999):
        pars=[p for p in m.parameters() if p.requires_grad]
        gs={k:torch.autograd.grad(t,pars,retain_graph=True,allow_unused=True) for k,t in [('CE',l),('R',aux['R']),('W',aux['W']),('A',aux['A'])]}
        squared={k:sum(float(g.double().square().sum()) for g in vv if g is not None) for k,vv in gs.items()}
        append(Path(m.cfg['output'])/'gradient_diagnostics.jsonl',dict(step=step,squared=squared,W_A_dot=sum(float((a.double()*b.double()).sum()) for a,b in zip(gs['W'],gs['A']) if a is not None and b is not None),input_gradient_squared=m.detail['input_gradient_squared'],adversarial_max_delta=m.detail['adversarial_max_delta'],active_R=m.detail['active']['R'],active_neighbor={k:b['active'] for k,b in m.detail['branches'].items()}))
    stats=dict(supervised=float(l.detach()),unsupervised=float(u.detach()),R=float(aux['R'].detach()),W=float(aux['W'].detach()),A=float(aux['A'].detach()),accepted_class_pixels=[0]*4,U_pixels=0,historical_raw_reads=0)
    m.detail=None;return l,u,stats

def calibrate_scales(cfg):
    source,m,ema,opt=setup(cfg);data=Current(dict(cfg,target_seed=20271491));values=[[] for _ in range(4)]
    for step in range(16):
        x,y,_=data.batch('L',step);_,_,_,a,_,support,_=references(m,x,y,data.cfg['target_seed'],step)
        ix=choose(support,data.cfg['target_seed'],step);labels=take(y[...,None],ix).squeeze(-1);v=take(a,ix)
        for c in range(4):
            v_c=v[labels==c][:,[j for j in range(4) if j!=c]];values[c].append(v_c[v_c>0].cpu())
    scales=[float(torch.cat(v).median()) if sum(len(t) for t in v) else 0. for v in values]
    write(Path(cfg['root'])/'SCALES.json',dict(scales=scales,status='PASS' if all(math.isfinite(s) and s>1e-6 for s in scales) else 'STOP',source_only=True,train_seed=20271491))

def measure(cfg):
    scales=json.loads((Path(cfg['root'])/'SCALES.json').read_text())
    if scales['status']!='PASS':write(Path(cfg['root'])/'P0_RESULT.json',dict(status='STOP',gates={'scales':False}));return
    cfg=dict(cfg,scales=scales['scales']);source,m,ema,opt=setup(cfg);data=Current(dict(cfg,target_seed=20271492))
    saved=torch.load(Path(cfg['root'])/'probe/final.pt',map_location='cpu',weights_only=False);assert saved['step']==256;m.load_state_dict(saved['model'])
    before=copy.deepcopy(m.state_dict());ref=copy.deepcopy(source.state_dict());rng=(torch.get_rng_state(),torch.cuda.get_rng_state_all())
    names=[n for n,p in m.named_parameters() if p.requires_grad];pars=[p for p in m.parameters() if p.requires_grad]
    squares={k:0. for k in ('CE','R','W','A')};dot=0.;ra_dot=0.;groups={k:dict(encoder=0.,decoder=0.) for k in ('R','W','A')};covered={k:[0]*4 for k in groups};denom=[0]*4;patients={k:[set() for _ in range(4)] for k in groups};energy={k:[0.]*4 for k in groups};input_energy=[];deltas=[]
    for step in range(16):
        l,aux=terms(m,data,step);d=m.detail
        gs={k:torch.autograd.grad(t,pars,retain_graph=True,allow_unused=True) for k,t in [('CE',l),('R',aux['R']),('W',aux['W']),('A',aux['A'])]}
        for k,gg in gs.items():
            assert all(g is None or torch.isfinite(g).all() for g in gg);squares[k]+=sum(float(g.double().square().sum()) for g in gg if g is not None)
        dot+=sum(float((a.double()*b.double()).sum()) for a,b in zip(gs['W'],gs['A']) if a is not None and b is not None)
        ra_dot+=sum(float((a.double()*b.double()).sum()) for a,b in zip(gs['R'],gs['A']) if a is not None and b is not None)
        for k in groups:
            for name,g in zip(names,gs[k]):
                if g is not None:groups[k]['decoder' if 'native.decoder.' in name else 'encoder']+=float(g.double().square().sum())
        for c in range(4):
            denom[c]+=int((d['y']==c).sum())
            for k in groups:
                mask=(d['support'] if k=='R' else d['branches'][k]['mask'])&(d['y']==c);covered[k][c]+=int(mask.sum())
                for slot,pid in enumerate(d['ids']):
                    if mask[slot].any():patients[k][c].add(pid)
                labels=d['labels'] if k=='R' else d['branches'][k]['labels'];values=d['values']['R'] if k=='R' else d['branches'][k]['values'];chosen=values[labels==c]
                energy[k][c]+=float(chosen.mean().detach())/max(1,len(labels.unique())) if len(chosen) else 0.
        input_energy.append(d['input_gradient_squared']);deltas.append(d['adversarial_max_delta']);m.detail=None
    cosine=dot/math.sqrt(squares['W']*squares['A']) if squares['W']*squares['A']>0 else None;distance=math.sqrt(max(0.,2-2*cosine)) if cosine is not None else 0.
    weights={k:.25*math.sqrt(squares['CE']/squares[k]) if squares[k]>0 else None for k in ('R','W','A')}
    coverage={k:[a/b if b else 0. for a,b in zip(v,denom)] for k,v in covered.items()};shares={k:sum(v[1:])/max(sum(v),1e-30) for k,v in energy.items()}
    combo=weights['R']**2*squares['R']+weights['A']**2*squares['A']+2*weights['R']*weights['A']*ra_dot if all(v is not None for v in weights.values()) else 0.
    gates=dict(scales=True,gradient_calibration=min(squares.values())>=1e-12 and all(v is not None and math.isfinite(v) and 1e-6<=v<=1e4 for v in weights.values()),foreground_coverage=all(coverage[k][c]>=.05 and len(patients[k][c])>=3 for k in groups for c in (1,2,3)),foreground_loss_share=all(v>=.05 for v in shares.values()),gradient_separation=distance>=.05,full_path_gradient=all(v>0 and math.isfinite(v) for gg in groups.values() for v in gg.values()),combination_nonzero=math.isfinite(combo) and combo>=1e-12,input_gradient_nonzero=all(math.isfinite(v) for v in input_energy) and sum(input_energy)>1e-12,perturbation_nonzero=all(math.isfinite(v) and v>0 and v<=.050001 for v in deltas))
    assert runtime.exact(before,m.state_dict()) and runtime.exact(ref,source.state_dict()) and runtime.exact(rng,(torch.get_rng_state(),torch.cuda.get_rng_state_all()))
    assert all(p.grad is None for p in source.parameters()) and m.get_parameter(HEAD).grad is None
    write(Path(cfg['root'])/'P0_RESULT.json',dict(status='PASS' if all(gates.values()) else 'STOP',gates=gates,weights=weights,scales=cfg['scales'],gradient_squared=squares,normalized_gradient_distance=distance,gradient_cosine=cosine,coverage_by_class=coverage,covered_patient_counts={k:[len(v) for v in vv] for k,vv in patients.items()},foreground_loss_share=shares,gradient_groups=groups,combination_gradient_squared=combo,input_gradient_squared=input_energy,adversarial_max_delta=deltas,reference_parameter_gradients_absent=True,model_reference_RNG_unchanged=True,probe_discarded=True))

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

def neighbor_self_check():
    for device in ('cpu','cuda'):
        x=torch.tensor([-.99,.2,.99],device=device,requires_grad=True);w=torch.tensor([1.,-2.,3.],device=device)
        g=torch.autograd.grad((x*w).sum(),x)[0];v=(x+.05*g.sign()).clamp(-1,1).detach()
        assert not v.requires_grad and float((v-x).abs().max())<=.050001 and float((w*(v-x)).sum())>0
        x2=x.detach().clone().requires_grad_(True);zero=torch.autograd.grad(x2.sum()*0,x2)[0];assert torch.equal((x2+.05*zero.sign()).clamp(-1,1),x2)
    return dict(status='PASS',synthetic_reverse=4,input_ascent=True,clamp_bound=True,zero_input_gradient_identity=True,perturbation_detached=True)

def controller(cfg,guard):
    root=Path(cfg['root']);assert json.loads((Path(cfg['sensitivity_root'])/'AUDIT.json').read_text())['status']=='PASS' and (Path(cfg['sensitivity_root'])/'DELIVERY.json').exists()
    write(root/'WORKER.private.json',dict(pid=os.getpid(),birth=Path('/proc/self/stat').read_text().split()[21]));geometry_check(cfg);write(root/'MARGIN_SELF_CHECK.json',margin_self_check());write(root/'NEIGHBOR_SELF_CHECK.json',neighbor_self_check())
    def stage(mode,c):
        guard(cfg,new_stage=True);assert int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))>=18000
        p=root/('SETTINGS_'+c['group']+'_'+mode+'.private.json');write(p,c)
        with (root/(c['group']+'_'+mode+'.log')).open('x') as log:q=subprocess.Popen([cfg['runtime'],cfg['entry']],env=dict(os.environ,TASK_SETTINGS=str(p),TASK_MODE=mode),stdout=log,stderr=subprocess.STDOUT)
        a=dict(pid=q.pid,birth=Path('/proc/'+str(q.pid)+'/stat').read_text().split()[21],group=c['group'],mode=mode);write(root/'ACTIVE.private.json',a);write(root/('ACTIVE_'+c['group']+'_'+mode+'.private.json'),a);write(root/'STATUS.json',dict(status='RUNNING',**a))
        code=q.wait();write(root/('EXIT_'+c['group']+'_'+mode+'.json'),dict(exit_code=code));assert code==0
    common=dict(cfg,arm='A',preflight_only=False,checkpoint_each_step=True,save_exception_state=True,diagnostic_holdout_excluded=False)
    stage('qualify',dict(common,group='P0_ENGINEERING',variant='B',target_seed=20271490,bucket='p0',phase='engineering',output=str(root/'p0_engineering')))
    c=dict(common,group='PROBE',variant='B',target_seed=20271490,bucket='p0',phase='probe',output=str(root/'probe'))
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
