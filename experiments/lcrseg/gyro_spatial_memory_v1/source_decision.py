"""Frozen native-source decision protection; registers with the existing target engine."""
import copy,json,math,os,subprocess,traceback
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F
from . import core,target,runtime,spectrum_alignment as shared,n_contribution as prior
from .bounded_runtime import install
from ..space_lora_v1.core import source_model
from ..main_style_ssl_v1.runtime import write,append
from ..siemens_ssl_reproduction_v1.methods import pixel_ce
from ..five_frameworks_v1.recipes import generator
SEEDS=(20270601,20270602,20270603)
METHODS=('A','B','C','D')
CAPS={f'{a}_{b}':n for a,ns in [('p0',(2000,512)),('detach',(512,80)),('engineering',(4096,512)),('formal',(96000,24200)),('evaluation',(34704,0))] for b,n in zip(('native','reverse'),ns)}
HEAD='parent.native.decoder.conv_logit.mu.weight'

class Calls(shared.Calls):
    def kind(self,operation):return self.cfg['bucket']+'_'+operation

class Current(prior.Current):
    def __init__(self,cfg,domain='Philips'):
        super().__init__(dict(cfg,group='S'),domain)
    def batch(self,role,step,n=2):
        assert role=='L' and n==2
        return super().batch(role,step,n)

class Native(nn.Module):
    def __init__(self,parent):
        super().__init__();self.parent=parent;self.last_h=None;self.calls=None
    def forward(self,x):
        def run():
            self.last_h=self.parent.features(x)
            return self.parent.native_readout(self.last_h,x.shape[-2:])
        return self.calls.call(self.calls.kind('native'),len(x),run) if self.calls else run()

def build(cfg):
    torch.manual_seed(163);torch.cuda.manual_seed_all(163)
    original=source_model(cfg)
    m=Native(copy.deepcopy(original.parent)).cuda()
    assert runtime.exact(m.state_dict(),original.state_dict())
    del original
    return m

def setup(cfg):
    m=build(cfg);m.parent.configure_stage_training()
    if cfg['variant']!='A':m.get_parameter(HEAD).requires_grad_(False)
    source=Native(copy.deepcopy(m.parent)).eval().requires_grad_(False)
    ema=Native(copy.deepcopy(m.parent)).eval().requires_grad_(False)
    groups=[dict(params=[p for p in m.parameters() if p.requires_grad and (p.ndim>1)==matrix],lr=.001,fixed_lr=.001,name='native_'+str(matrix),weight_decay=.0005 if matrix else 0.) for matrix in (True,False)]
    opt=torch.optim.SGD(groups,lr=.001,momentum=.9,nesterov=True);runtime.attach(opt,cfg,cfg['group'],cfg['phase'])
    m.cfg=cfg;m.calls=source.calls=ema.calls=Calls(cfg,'unused')
    # The reference is intentionally outside persistent model/optimizer state.
    object.__setattr__(m,'reference',source)
    m.basis=torch.load(Path(cfg['root'])/'basis.pt',map_location='cuda',weights_only=False)['V']
    return source,m,ema,opt

def update_ema(ema,m):
    fixed={n for n,p in m.named_parameters() if not p.requires_grad}
    with torch.no_grad():
        state=m.state_dict()
        for k,v in ema.state_dict().items():
            if k in fixed or not v.is_floating_point():v.copy_(state[k])
            else:v.mul_(.999).add_(state[k],alpha=.001)

def patches(h,shape):
    u=F.unfold(h,kernel_size=3,padding=0,stride=1).reshape(h.shape[0],144,h.shape[-2]-2,h.shape[-1]-2)
    return F.interpolate(u,size=shape,mode='bilinear',align_corners=True).permute(0,2,3,1)

def terms(m,data,step):
    x,y,ids=data.batch('L',step)
    z=m(x);h=m.last_h
    with torch.no_grad():z0=m.reference(x);h0=m.reference.last_h
    support=(z0.argmax(1)==y)&(y!=255)
    ix=[];ys=[]
    for slot in range(len(x)):
        selected=support[slot].flatten().nonzero().flatten().cpu()
        g=generator(data.cfg['target_seed'],1,1,step,'decision_support:'+str(slot))
        selected=selected[torch.randperm(len(selected),generator=g)[:512]]
        ix.append(selected.to(x.device));ys.append(y[slot].flatten()[ix[-1]])
    u=patches(h,x.shape[-2:]);u0=patches(h0,x.shape[-2:])
    a=torch.cat([v.reshape(-1,144)[i] for v,i in zip(u,ix)])
    b=torch.cat([v.reshape(-1,144)[i] for v,i in zip(u0,ix)]).detach()
    labels=torch.cat(ys);delta=a-b
    if len(a):
        den=b.square().sum(1).mean()+1e-12
        full=delta.square().sum(1)/den;ds=(delta@m.basis).square().sum(1)/den
        f,d=full.mean(),ds.mean()
    else:full=ds=delta.sum(1)*0;f=d=h.sum()*0
    m.detail=dict(support=support,labels=labels,ids=ids,y=y,full=full,ds=ds,z=z,z0=z0,u=u,u0=u0)
    return pixel_ce(z,y).mean(),d,f

def losses(arm,m,data,step):
    l,d,f=terms(m,data,step);v=m.cfg['variant']
    weights=m.cfg.get('weights',{'DS':0.,'FULL':0.})
    u=weights['DS']*d if v=='D' else weights['FULL']*f if v=='C' else l*0
    if m.cfg['bucket']=='formal' and step in (0,999,1999):
        pars=[p for p in m.parameters() if p.requires_grad]
        gs=[torch.autograd.grad(t,pars,retain_graph=True,allow_unused=True) for t in (l,d,f)]
        append(Path(m.cfg['output'])/'gradient_diagnostics.jsonl',dict(step=step,squared={k:sum(float(g.double().square().sum()) for g in vv if g is not None) for k,vv in zip(('CE','DS','FULL'),gs)}))
    stats=dict(supervised=float(l.detach()),unsupervised=float(u.detach()),DS=float(d.detach()),FULL=float(f.detach()),accepted_class_pixels=[0]*4,U_pixels=0,historical_raw_reads=0)
    m.detail=None
    return l,u,stats

def preflight(cfg,out,source,m,ema,opt,data):
    initial=copy.deepcopy(target.engine.snapshot(m,ema,opt,data,0));runtime.save(cfg,initial,out/'initial.pt')
    assert not opt.state and runtime.exact(source.state_dict(),m.state_dict()) and runtime.exact(ema.state_dict(),m.state_dict())
    if cfg['bucket']=='formal':
        # Qualifications already exercised this graph; formal forward cap is exactly 4/update.
        runtime.save(cfg,initial,out/'latest.pt');return
    x,_,_=data.labeled(0)
    with torch.no_grad():
        a=source(x);b=m(x);c=ema(x)
        assert torch.equal(a,b) and torch.equal(b,c)
        original=m.parent.native_readout(m.last_h,x.shape[-2:])
        assert torch.equal(original,b)
        w=m.get_parameter(HEAD).flatten(1)
        recon=torch.einsum('bhwd,cd->bchw',patches(m.last_h,x.shape[-2:]),w)
        error=float((recon-b).abs().max());assert error<=1e-5*max(1.,float(b.abs().max()))
    target.engine.restore(initial,m,ema,opt,data)
    l,u,_=losses('A',m,data,0);(l+u).backward();value=float(l+u)
    grads=[None if p.grad is None else p.grad.clone() for p in m.parameters()]
    target.engine.restore(initial,m,ema,opt,data)
    l,u,_=losses('A',m,data,0);(l+u).backward()
    assert value==float(l+u) and all((g is None and p.grad is None) or (g is not None and p.grad is not None and torch.equal(g,p.grad)) for p,g in zip(m.parameters(),grads))
    assert all(p.grad is None for p in source.parameters()) and all(p.grad is None for p in m.parameters() if not p.requires_grad)
    target.engine.restore(initial,m,ema,opt,data);runtime.save(cfg,initial,out/'latest.pt')
    write(out/'PREFLIGHT.json',dict(status='PASS',original_source_exact=True,golden_exact=True,readout_error=error,empty_optimizer=True))

def train(cfg,steps=2000):
    original=target.engine.main
    def bounded(c):
        with patch.object(target.engine,'STEPS',steps):return original(c)
    with patch.object(target.engine,'main',bounded):target.main(cfg)

def qualify(cfg):
    train(dict(cfg,preflight_only=True))
    out=Path(cfg['output']);source,m,ema,opt=setup(cfg);data=Current(cfg)
    initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=False);target.engine.restore(initial,m,ema,opt,data)
    x,y,_=data.labeled(0)
    def step():
        opt.zero_grad(set_to_none=True);v=pixel_ce(m(x),y).mean();v.backward();opt.step();update_ema(ema,m);return float(v)
    first=float(pixel_ce(m(x),y).mean())
    for i in range(30):
        before=copy.deepcopy(target.engine.snapshot(m,ema,opt,data,i)) if i==15 else None
        v=step()
        if i==15:
            expected=copy.deepcopy(target.engine.snapshot(m,ema,opt,data,i+1));target.engine.restore(before,m,ema,opt,data)
            assert step()==v and runtime.exact(expected,target.engine.snapshot(m,ema,opt,data,i+1))
    last=float(pixel_ce(m(x),y).mean());assert first-last>=min(1e-4,.01*first) and first>=1e-6
    target.engine.restore(initial,m,ema,opt,data);opt.zero_grad(set_to_none=True);l,u,_=losses('A',m,data,0);(l+u).backward();opt.step();update_ema(ema,m)
    assert all(torch.equal(p.cpu(),initial['model'][n]) for n,p in m.named_parameters() if not p.requires_grad)
    runtime.save(cfg,target.engine.snapshot(m,ema,opt,data,1),out/'engineering_terminal.pt')
    write(out/'ENGINEERING.json',dict(status='PASS',actual_calls=32,first=first,final=last,discarded=True,nonempty_restore_exact=True))

def measure(cfg):
    source,m,ema,opt=setup(cfg);data=Current(dict(cfg,target_seed=20270691))
    s=torch.load(Path(cfg['root'])/'probe/final.pt',map_location='cpu',weights_only=False);assert s['step']==64
    m.load_state_dict(s['model']);before=copy.deepcopy(m.state_dict());ref=copy.deepcopy(source.state_dict());rng=(torch.get_rng_state(),torch.cuda.get_rng_state_all())
    names=[n for n,p in m.named_parameters() if p.requires_grad];pars=[p for p in m.parameters() if p.requires_grad]
    squares={'CE':0.,'DS':0.,'FULL':0.};dot=0.;coverage=[0]*4;denom=[0]*4;patients=[set() for _ in range(4)];energy={k:[0.]*4 for k in ('DS','FULL')};group_energy={'encoder':0.,'decoder':0.};max_error=0.
    for step in range(16):
        l,d,f=terms(m,data,step);detail=m.detail
        gs={k:torch.autograd.grad(t,pars,retain_graph=True,allow_unused=True) for k,t in [('CE',l),('DS',d),('FULL',f)]}
        for k,values in gs.items():
            assert all(v is None or torch.isfinite(v).all() for v in values)
            squares[k]+=sum(float(v.double().square().sum()) for v in values if v is not None)
        dot+=sum(float((a.double()*b.double()).sum()) for a,b in zip(gs['DS'],gs['FULL']) if a is not None and b is not None)
        for n,g in zip(names,gs['DS']):
            if g is not None:group_energy['decoder' if 'native.decoder.' in n else 'encoder']+=float(g.double().square().sum())
        for c in range(4):
            mask=detail['support']&(detail['y']==c);coverage[c]+=int(mask.sum());denom[c]+=int((detail['y']==c).sum())
            for slot,pid in enumerate(detail['ids']):
                if mask[slot].any():patients[c].add(pid)
            for k,key in [('DS','ds'),('FULL','full')]:energy[k][c]+=float(detail[key][detail['labels']==c].sum().detach())/max(1,len(detail['labels']))
        w=source.get_parameter(HEAD).flatten(1);b=w-w.mean(0,keepdim=True)
        lhs=detail['z']-detail['z0'];lhs=lhs-lhs.mean(1,keepdim=True)
        rhs=torch.einsum('bhwd,cd->bchw',detail['u']-detail['u0'],b)
        error=float((lhs-rhs).abs().max());max_error=max(max_error,error)
        assert error<=1e-5*max(1.,float(detail['z'].abs().max()),float(detail['z0'].abs().max()))
        assert all(p.grad is None for p in source.parameters()) and m.get_parameter(HEAD).grad is None
        m.detail=None
    cosine=dot/math.sqrt(squares['DS']*squares['FULL']) if min(squares['DS'],squares['FULL'])>0 else None
    distance=math.sqrt(max(0.,2-2*cosine)) if cosine is not None else 0.
    weights={k:.25*math.sqrt(squares['CE']/squares[k]) if squares[k]>0 else None for k in ('DS','FULL')}
    cov=[a/b if b else 0. for a,b in zip(coverage,denom)];share={k:sum(v[1:])/max(sum(v),1e-30) for k,v in energy.items()}
    gates=dict(gradient_calibration=min(squares.values())>=1e-12 and all(v is not None and math.isfinite(v) and 1e-6<=v<=1e4 for v in weights.values()),foreground_coverage=all(cov[c]>=.05 and len(patients[c])>=3 for c in (1,2,3)),foreground_loss_share=all(v>=.05 for v in share.values()),gradient_separation=distance>=.05,full_path_gradient=all(v>0 and math.isfinite(v) for v in group_energy.values()))
    assert runtime.exact(before,m.state_dict()) and runtime.exact(ref,source.state_dict()) and runtime.exact(rng,(torch.get_rng_state(),torch.cuda.get_rng_state_all()))
    write(Path(cfg['root'])/'P0_RESULT.json',dict(status='PASS' if all(gates.values()) else 'STOP',gates=gates,weights=weights,gradient_squared=squares,gradient_cosine=cosine,normalized_gradient_distance=distance,coverage_by_class=cov,covered_patient_counts=[len(v) for v in patients],foreground_loss_share=share,DS_gradient_groups=group_energy,centered_readout_error=max_error,model_reference_RNG_unchanged=True,probe_discarded=True))

def detach(cfg):
    models=[]
    for variant in ('B','D'):
        c=dict(cfg,group='DETACH_'+variant,variant=variant,weights={'DS':0.,'FULL':0.});source,m,e,o=setup(c);models.append((m,e,o,Current(c)))
    for step in range(32):
        rngs=[]
        for m,e,o,d in models:
            o.zero_grad(set_to_none=True);l,u,_=losses('A',m,d,step);(l+u).backward();o.step();update_ema(e,m);rngs.append((torch.get_rng_state(),torch.cuda.get_rng_state_all()))
        a,b=models
        assert runtime.exact(a[0].state_dict(),b[0].state_dict()) and runtime.exact(a[1].state_dict(),b[1].state_dict()) and runtime.exact(a[2].state_dict(),b[2].state_dict()) and runtime.exact(rngs[0],rngs[1])
    write(Path(cfg['root'])/'DETACH_CHECK.json',dict(status='PASS',updates=64,model_opt_EMA_RNG_exact=True,discarded=True))

def evaluate(cfg):
    from ..gyro_transfer_v1.run import finalize
    m=build(cfg);e=build(cfg);m.calls=e.calls=Calls(cfg,'evaluate');out=Path(cfg['output'])
    s=torch.load(out/'final.pt',map_location='cpu',weights_only=False);assert s['step']==s['physical_updates']==2000
    m.load_state_dict(s['model']);e.load_state_dict(s['ema'])
    with patch.object(target.engine,'subset',lambda access,patients:access):finalize(cfg,out,m,e,2000,2000)
    result=json.loads((out/'RESULTS.json').read_text());result['primary']='fixed final EMA .999';result['historical_entry_comparisons_auxiliary']=True;write(out/'RESULTS.json',result)
    write(out/'EVALUATION_EXIT.json',dict(exit_code=0))

def geometry_check(cfg):
    m=build(cfg);w=m.get_parameter(HEAD).detach().flatten(1).double();assert tuple(w.shape)==(4,144)
    b=w-w.mean(0,keepdim=True);_,s,vh=torch.linalg.svd(b,full_matrices=False);rank=int((s>s[0]*1e-10).sum());assert rank==3
    v=vh[:rank].T;p=v@v.T;eye=torch.eye(144,device=w.device,dtype=w.dtype)
    assert torch.allclose(p,p.T,atol=1e-6) and torch.allclose(p@p,p,atol=1e-6) and float((b@(eye-p)).norm()/b.norm())<=1e-6
    torch.save(dict(V=v.float(),W=w.float(),rank=rank),Path(cfg['root'])/'basis.pt')
    checks={}
    for device in ('cpu','cuda'):
        g=torch.Generator(device=device).manual_seed(123);h=torch.randn(2,16,9,11,device=device,generator=g,requires_grad=True);ww=torch.randn(4,16,3,3,device=device,generator=g)
        z=F.interpolate(F.conv2d(h,ww),size=(17,19),mode='bilinear',align_corners=True)
        reconstructed=torch.einsum('bhwd,cd->bchw',patches(h,(17,19)),ww.flatten(1))
        assert torch.allclose(z,reconstructed,atol=2e-5,rtol=1e-5)
        aa=torch.autograd.grad(z.square().sum(),h,retain_graph=True)[0];bb=torch.autograd.grad(reconstructed.square().sum(),h)[0];assert torch.allclose(aa,bb,atol=.01,rtol=2e-5)
        bc=ww.flatten(1).double();bc-=bc.mean(0,keepdim=True);_,ss,vv=torch.linalg.svd(bc,full_matrices=False);pp=vv[:3].T@vv[:3];ii=torch.eye(144,device=device,dtype=torch.float64)
        assert torch.allclose(pp@pp,pp,atol=1e-6) and float((bc@(ii-pp)).norm()/bc.norm())<1e-6
        checks[device]=dict(patch_forward_gradient=True,projection=True)
    write(Path(cfg['root'])/'SELF_CHECK.json',dict(status='PASS',source_SVD=1,synthetic_SVD=2,synthetic_reverse=4,medical_forwards=0,rank=rank,checks=checks))

def controller(cfg,guard):
    root=Path(cfg['root']);assert json.loads((Path(cfg['ema_root'])/'AUDIT.json').read_text())['status']=='PASS' and (Path(cfg['ema_root'])/'DELIVERY.json').exists()
    write(root/'WORKER.private.json',dict(pid=os.getpid(),birth=Path('/proc/self/stat').read_text().split()[21]))
    geometry_check(cfg)
    def stage(mode,c):
        guard(cfg,new_stage=True);assert int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))>=18000
        p=root/('SETTINGS_'+c['group']+'_'+mode+'.private.json');write(p,c)
        with (root/(c['group']+'_'+mode+'.log')).open('x') as log:q=subprocess.Popen([cfg['runtime'],cfg['entry']],env=dict(os.environ,TASK_SETTINGS=str(p),TASK_MODE=mode),stdout=log,stderr=subprocess.STDOUT)
        a=dict(pid=q.pid,birth=Path('/proc/'+str(q.pid)+'/stat').read_text().split()[21],group=c['group'],mode=mode);write(root/'ACTIVE.private.json',a);write(root/('ACTIVE_'+c['group']+'_'+mode+'.private.json'),a);write(root/'STATUS.json',dict(status='RUNNING',**a))
        code=q.wait();write(root/('EXIT_'+c['group']+'_'+mode+'.json'),dict(exit_code=code));assert code==0
    base=dict(cfg,arm='A',preflight_only=False,checkpoint_each_step=True,save_exception_state=True,diagnostic_holdout_excluded=False)
    stage('qualify',dict(base,group='P0_ENGINEERING',variant='B',target_seed=20270690,bucket='p0',phase='engineering',output=str(root/'p0_engineering')))
    c=dict(base,group='PROBE',variant='B',target_seed=20270690,bucket='p0',phase='probe',output=str(root/'probe'))
    stage('probe',c);stage('measure',c)
    p0=json.loads((root/'P0_RESULT.json').read_text())
    if p0['status']!='PASS':
        write(root/'STATUS.json',dict(status='P0_STOP',formal_started=False,resources=guard(cfg)));write(root/'EXIT.json',dict(exit_code=0));return
    base['weights']=p0['weights']
    stage('detach',dict(base,group='DETACH',variant='B',target_seed=SEEDS[0],bucket='detach',phase='detach',output=str(root/'detach')))
    configs=[dict(base,group=f'{s}_{a}',variant=a,target_seed=s) for s in SEEDS for a in METHODS]
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
        with patch.object(shared,'CAPS',CAPS),patch.object(core,'build',build),patch.object(core,'target_setup',setup),patch.object(core,'losses',losses),patch.object(core,'update_ema',update_ema),patch.object(target,'Domain',Current),patch.object(target,'preflight',preflight),patch.object(torch.autograd,'backward',b),patch.object(torch.autograd,'grad',g):
            {'qualify':qualify,'probe':lambda c:train(c,64),'train':train,'measure':measure,'detach':detach,'evaluate':evaluate}[mode](cfg)
    except BaseException as exc:
        write(root/('FAILURE_'+cfg.get('group','controller')+'_'+mode+'.json'),dict(error=str(exc),traceback=traceback.format_exc()))
        if mode=='controller':write(root/'STATUS.json',dict(status='FAILED'))
        raise
