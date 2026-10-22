"""Dense image-class worst-group reliable margin protection on current L only."""
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
SEEDS=(20270801,20270802,20270803)
METHODS=('B','R','E','T')
CAPS={f'{a}_{b}':n for a,ns in [('p0',(3000,1024)),('detach',(1024,128)),('engineering',(6000,1024)),('formal',(144000,24300)),('evaluation',(34704,0))] for b,n in zip(('native','reverse'),ns)}

def weak(x,seed,step):
    v=[]
    for slot in range(len(x)):
        g=generator(seed,1,1,step,'margin_view:'+str(slot));r=torch.rand(2,generator=g)
        v.append((x[slot]*(.9+.2*float(r[0]))+(-.05+.1*float(r[1]))).clamp(-1,1))
    return torch.stack(v)

def margins(z,y):
    zz=z.permute(0,2,3,1);safe=y.clamp(0,3)
    return zz.gather(-1,safe[...,None])-zz

def references(m,x,y,seed,step):
    with torch.no_grad():
        z0=m.reference(x);h0=m.reference.last_h
        z1=m.reference(weak(x,seed,step))
        single=(y!=255)&(z0.argmax(1)==y);support=single&(z1.argmax(1)==y)
        p,q=z0.softmax(1),z1.softmax(1);a=(p+q)/2
        js=(.5*(p*(p.clamp_min(1e-12).log()-a.clamp_min(1e-12).log())+q*(q.clamp_min(1e-12).log()-a.clamp_min(1e-12).log()))).sum(1)
        reliability=(1-js/math.log(2)).clamp(0,1)*support
    return z0,h0,margins(z0,y),torch.minimum(margins(z0,y),margins(z1,y)),single,support,reliability

def choose(mask,seed,step):
    ix=[]
    for slot in range(len(mask)):
        selected=mask[slot].flatten().nonzero().flatten().cpu();g=generator(seed,1,1,step,'margin_support:'+str(slot))
        ix.append(selected[torch.randperm(len(selected),generator=g)[:512]].to(mask.device))
    return ix

def take(v,ix):return torch.cat([a.reshape(-1,*v.shape[3:])[i] for a,i in zip(v,ix)])

def aggregate(values,labels,zero):
    vv=[values[labels==c].mean() for c in range(4) if (labels==c).any()]
    return torch.stack(vv).mean() if vv else zero*0

def hinge(student,source,labels,scales):
    competitor=~F.one_hot(labels.long(),4).bool()
    return ((F.relu(.5*source-student)/scales[labels,None]).square()*competitor).sum(-1)/3

def terms(m,data,step):
    x,y,ids=data.batch('L',step);z=m(x)
    _,_,_,a,_,support,r=references(m,x,y,data.cfg['target_seed'],step)
    scales=torch.as_tensor(m.cfg.get('scales',[1.]*4),device=x.device,dtype=z.dtype)
    ms=margins(z,y);ix=choose(support,data.cfg['target_seed'],step)
    labels=take(y[...,None],ix).squeeze(-1)
    sampled=hinge(take(ms,ix),take(a,ix),labels,scales)*take(r[...,None],ix).squeeze(-1)
    group_values=[];group_labels=[]
    # Only supported GT labels reach hinge (ignore label 255 is never indexed).
    for slot in range(len(x)):
        for c in range(4):
            mask=support[slot]&(y[slot]==c)
            if mask.any():
                values=hinge(ms[slot][mask],a[slot][mask],y[slot][mask],scales)*r[slot][mask]
                group_values.append(values.mean());group_labels.append(c)
    gv=torch.stack(group_values) if group_values else z.sum().reshape(1)[:0]
    gl=torch.tensor(group_labels,device=x.device,dtype=torch.long)
    e=gv.mean() if len(gv) else z.sum()*0
    if len(gv):
        # torch.max(dim) deterministically selects the first group on a tie.
        t,winner=gv.max(dim=0);tv=gv*F.one_hot(winner,len(gv))
    else:t=z.sum()*0;tv=gv
    aux={'R':aggregate(sampled,labels,z.sum()),'E':e,'T':t}
    # Class contributions sum exactly to each aggregation, including T's winner.
    contributions={'R':torch.stack([sampled[labels==c].mean() if (labels==c).any() else z.sum()*0 for c in range(4)])/max(1,len(labels.unique())),
                   'E':torch.stack([gv[gl==c].sum() for c in range(4)])/max(1,len(gv)),
                   'T':torch.stack([tv[gl==c].sum() for c in range(4)])}
    boundary=F.max_pool2d(y.float()[:,None],3,1,1).squeeze(1)!=-F.max_pool2d(-y.float()[:,None],3,1,1).squeeze(1)
    m.detail=dict(support=support,y=y,ids=ids,boundary=boundary,contributions=contributions,
                  active={'R':float((sampled>0).float().mean()) if len(sampled) else 0.,'E':float((gv>0).float().mean()) if len(gv) else 0.,'T':float(t.detach()>0)},
                  groups=len(gv),winner_class=int(gl[winner]) if len(gv) else None)
    return pixel_ce(z,y).mean(),aux

def losses(arm,m,data,step):
    l,aux=terms(m,data,step);variant=m.cfg['variant'];u=m.cfg.get('weights',{}).get(variant,0.)*aux[variant] if variant in aux else l*0
    if m.cfg['bucket']=='formal' and step in (0,999,1999):
        pars=[p for p in m.parameters() if p.requires_grad]
        squared={k:sum(float(g.double().square().sum()) for g in torch.autograd.grad(t,pars,retain_graph=True,allow_unused=True) if g is not None) for k,t in [('CE',l),*aux.items()]}
        append(Path(m.cfg['output'])/'gradient_diagnostics.jsonl',dict(step=step,squared=squared,active=m.detail['active'],boundary_supported=int((m.detail['boundary']&m.detail['support']).sum())))
    stats=dict(supervised=float(l.detach()),unsupervised=float(u.detach()),**{k:float(v.detach()) for k,v in aux.items()},accepted_class_pixels=[0]*4,U_pixels=0,historical_raw_reads=0)
    m.detail=None;return l,u,stats

def calibrate_scales(cfg):
    source,m,ema,opt=setup(cfg);data=Current(dict(cfg,target_seed=20270891));values=[[] for _ in range(4)]
    for step in range(16):
        x,y,_=data.batch('L',step);_,_,_,a,_,support,_=references(m,x,y,data.cfg['target_seed'],step)
        ix=choose(support,data.cfg['target_seed'],step);labels=take(y[...,None],ix).squeeze(-1);v=take(a,ix)
        for c in range(4):
            v_c=v[labels==c][:,[j for j in range(4) if j!=c]];values[c].append(v_c[v_c>0].cpu())
    scales=[float(torch.cat(v).median()) if sum(len(t) for t in v) else 0. for v in values]
    write(Path(cfg['root'])/'SCALES.json',dict(scales=scales,status='PASS' if all(math.isfinite(s) and s>1e-6 for s in scales) else 'STOP',source_only=True,train_seed=20270891))

def measure(cfg):
    scales=json.loads((Path(cfg['root'])/'SCALES.json').read_text())
    if scales['status']!='PASS':write(Path(cfg['root'])/'P0_RESULT.json',dict(status='STOP',gates={'scales':False}));return
    cfg=dict(cfg,scales=scales['scales']);source,m,ema,opt=setup(cfg);data=Current(dict(cfg,target_seed=20270892))
    saved=torch.load(Path(cfg['root'])/'probe/final.pt',map_location='cpu',weights_only=False);assert saved['step']==256;m.load_state_dict(saved['model'])
    before=copy.deepcopy(m.state_dict());ref=copy.deepcopy(source.state_dict());rng=(torch.get_rng_state(),torch.cuda.get_rng_state_all())
    names=[n for n,p in m.named_parameters() if p.requires_grad];pars=[p for p in m.parameters() if p.requires_grad]
    squares={k:0. for k in ('CE','R','E','T')};dot=0.;covered=[0]*4;denom=[0]*4;patients=[set() for _ in range(4)];energy={k:[0.]*4 for k in ('R','E','T')};groups={'encoder':0.,'decoder':0.};active=[]
    for step in range(16):
        l,aux=terms(m,data,step);d=m.detail;gs={k:torch.autograd.grad(t,pars,retain_graph=True,allow_unused=True) for k,t in [('CE',l),*aux.items()]}
        for k,gg in gs.items():
            assert all(g is None or torch.isfinite(g).all() for g in gg);squares[k]+=sum(float(g.double().square().sum()) for g in gg if g is not None)
        dot+=sum(float((a.double()*b.double()).sum()) for a,b in zip(gs['T'],gs['E']) if a is not None and b is not None)
        for name,g in zip(names,gs['T']):
            if g is not None:groups['decoder' if 'native.decoder.' in name else 'encoder']+=float(g.double().square().sum())
        for c in range(4):
            mask=d['support']&(d['y']==c);covered[c]+=int(mask.sum());denom[c]+=int((d['y']==c).sum())
            for slot,pid in enumerate(d['ids']):
                if mask[slot].any():patients[c].add(pid)
            for k,v in d['contributions'].items():energy[k][c]+=float(v[c].detach())
        active.append(d['active']);m.detail=None
    cosine=dot/math.sqrt(squares['T']*squares['E']) if squares['T']*squares['E']>0 else None;distance=math.sqrt(max(0.,2-2*cosine)) if cosine is not None else 0.
    weights={k:.25*math.sqrt(squares['CE']/squares[k]) if squares[k]>0 else None for k in ('R','E','T')}
    coverage=[a/b if b else 0. for a,b in zip(covered,denom)];share={k:sum(v[1:])/max(sum(v),1e-30) for k,v in energy.items()}
    gates=dict(scales=True,gradient_calibration=min(squares.values())>=1e-12 and all(v is not None and math.isfinite(v) and 1e-6<=v<=1e4 for v in weights.values()),foreground_coverage=all(coverage[c]>=.05 and len(patients[c])>=3 for c in (1,2,3)),foreground_loss_share=all(v>=.05 for v in share.values()),gradient_separation=distance>=.05,full_path_gradient=all(v>0 and math.isfinite(v) for v in groups.values()))
    assert runtime.exact(before,m.state_dict()) and runtime.exact(ref,source.state_dict()) and runtime.exact(rng,(torch.get_rng_state(),torch.cuda.get_rng_state_all()))
    assert all(p.grad is None for p in source.parameters()) and m.get_parameter(HEAD).grad is None
    write(Path(cfg['root'])/'P0_RESULT.json',dict(status='PASS' if all(gates.values()) else 'STOP',gates=gates,weights=weights,scales=cfg['scales'],gradient_squared=squares,normalized_gradient_distance=distance,gradient_cosine=cosine,coverage_by_class=coverage,covered_patient_counts=[len(v) for v in patients],foreground_loss_share=share,T_gradient_groups=groups,active_per_batch=active,model_reference_RNG_unchanged=True,probe_discarded=True))

def detach(cfg):
    models=[]
    for variant in ('B','T'):
        c=dict(cfg,group='DETACH_'+variant,variant=variant,weights={});_,m,e,o=setup(c);models.append((m,e,o,Current(c)))
    for step in range(32):
        rng=[]
        for m,e,o,d in models:
            o.zero_grad(set_to_none=True);l,u,_=losses('A',m,d,step);(l+u).backward();o.step();update_ema(e,m);rng.append((torch.get_rng_state(),torch.cuda.get_rng_state_all()))
        a,b=models
        assert runtime.exact(a[0].state_dict(),b[0].state_dict()) and runtime.exact(a[1].state_dict(),b[1].state_dict()) and runtime.exact(a[2].state_dict(),b[2].state_dict()) and runtime.exact(rng[0],rng[1])
    write(Path(cfg['root'])/'DETACH_CHECK.json',dict(status='PASS',updates=64,model_opt_EMA_RNG_exact=True,discarded=True))

def margin_self_check():
    for device in ('cpu','cuda'):
        a=torch.tensor([[0.,2.,4.,6.]],device=device);labels=torch.tensor([0],device=device);scales=torch.ones(4,device=device)
        safe=a.clone().requires_grad_();loss=hinge(safe,a,labels,scales).sum();g=torch.autograd.grad(loss,safe)[0];assert float(loss)==0 and torch.count_nonzero(g)==0
        bad=torch.zeros_like(a,requires_grad=True);loss=hinge(bad,a,labels,scales).sum();g=torch.autograd.grad(loss,bad)[0];assert float(loss)>0 and (g[0,1:]<0).all() and g[0,0]==0
    return dict(status='PASS',synthetic_reverse=4,safe_margin_zero_gradient=True,violated_margin_correct_gradient=True)

def controller(cfg,guard):
    root=Path(cfg['root']);assert json.loads((Path(cfg['reliable_root'])/'AUDIT.json').read_text())['status']=='PASS' and (Path(cfg['reliable_root'])/'DELIVERY.json').exists()
    write(root/'WORKER.private.json',dict(pid=os.getpid(),birth=Path('/proc/self/stat').read_text().split()[21]));geometry_check(cfg);write(root/'MARGIN_SELF_CHECK.json',margin_self_check())
    def stage(mode,c):
        guard(cfg,new_stage=True);assert int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))>=18000
        p=root/('SETTINGS_'+c['group']+'_'+mode+'.private.json');write(p,c)
        with (root/(c['group']+'_'+mode+'.log')).open('x') as log:q=subprocess.Popen([cfg['runtime'],cfg['entry']],env=dict(os.environ,TASK_SETTINGS=str(p),TASK_MODE=mode),stdout=log,stderr=subprocess.STDOUT)
        a=dict(pid=q.pid,birth=Path('/proc/'+str(q.pid)+'/stat').read_text().split()[21],group=c['group'],mode=mode);write(root/'ACTIVE.private.json',a);write(root/('ACTIVE_'+c['group']+'_'+mode+'.private.json'),a);write(root/'STATUS.json',dict(status='RUNNING',**a))
        code=q.wait();write(root/('EXIT_'+c['group']+'_'+mode+'.json'),dict(exit_code=code));assert code==0
    common=dict(cfg,arm='A',preflight_only=False,checkpoint_each_step=True,save_exception_state=True,diagnostic_holdout_excluded=False)
    stage('qualify',dict(common,group='P0_ENGINEERING',variant='B',target_seed=20270890,bucket='p0',phase='engineering',output=str(root/'p0_engineering')))
    c=dict(common,group='PROBE',variant='B',target_seed=20270890,bucket='p0',phase='probe',output=str(root/'probe'))
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
