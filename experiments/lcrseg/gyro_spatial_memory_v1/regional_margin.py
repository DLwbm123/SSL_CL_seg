"""Anatomical boundary/interior stratified reliable margin protection on current L only."""
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
SEEDS=(20271001,20271002,20271003)
METHODS=('B','R','W','T')
CAPS={f'{a}_{b}':n for a,ns in [('p0',(3000,1024)),('detach',(1024,128)),('engineering',(6000,1024)),('formal',(144000,24300)),('evaluation',(34704,0))] for b,n in zip(('native','reverse'),ns)}

from .reliable_margin import weak,margins,references,choose,take,aggregate,hinge,margin_self_check

def regions(y):
    # Replicate padding avoids inventing an image-edge anatomical boundary.
    t=F.pad(y.float()[:,None],(1,1,1,1),mode='replicate')
    hi=F.max_pool2d(t,3,1);lo=-F.max_pool2d(-t,3,1)
    valid=(hi.squeeze(1)<4)&(lo.squeeze(1)>=0)
    return (hi!=lo).squeeze(1)&valid,valid

def regional_aggregate(means,counts,classes,region_ids,zero):
    w=[];t=[];wc=[];tc=[];boundary_parts=[]
    for c in range(4):
        ix=classes==c
        if not ix.any():
            wc.append(zero*0);tc.append(zero*0);boundary_parts.append(zero*0);continue
        cw=(means[ix]*counts[ix]).sum()/counts[ix].sum();rr=[];br=zero*0
        for region in (0,1):
            ii=ix&(region_ids==region)
            if ii.any():
                v=(means[ii]*counts[ii]).sum()/counts[ii].sum();rr.append(v)
                if region==1:br=v
        ct=torch.stack(rr).mean();w.append(cw);t.append(ct);wc.append(cw);tc.append(ct);boundary_parts.append(br/len(rr))
    n=max(1,len(w))
    return (torch.stack(w).mean() if w else zero*0,torch.stack(t).mean() if t else zero*0,
            torch.stack(wc)/n,torch.stack(tc)/n,torch.stack(boundary_parts)/n)

def terms(m,data,step):
    x,y,ids=data.batch('L',step);z=m(x)
    _,_,_,a,_,support,r=references(m,x,y,data.cfg['target_seed'],step)
    scales=torch.as_tensor(m.cfg.get('scales',[1.]*4),device=x.device,dtype=z.dtype)
    ms=margins(z,y);ix=choose(support,data.cfg['target_seed'],step)
    labels=take(y[...,None],ix).squeeze(-1)
    sampled=hinge(take(ms,ix),take(a,ix),labels,scales)*take(r[...,None],ix).squeeze(-1)
    boundary,valid=regions(y);support_region=support&valid
    means=[];counts=[];classes=[];region_ids=[];sample_counts=[]
    for slot in range(len(x)):
        for c in range(4):
            for region in (0,1):
                mask=support_region[slot]&(y[slot]==c)&(boundary[slot]==bool(region))
                idx=mask.flatten().nonzero().flatten().cpu();n=len(idx)
                if not n:continue
                g=generator(data.cfg['target_seed'],1,1,step,f'regional_support:{slot}:{c}:{region}')
                selected=idx[torch.randperm(n,generator=g)[:64]].to(x.device)
                yy=y[slot].flatten()[selected]
                vv=hinge(ms[slot].reshape(-1,4)[selected],a[slot].reshape(-1,4)[selected],yy,scales)*r[slot].flatten()[selected]
                means.append(vv.mean());counts.append(n);classes.append(c);region_ids.append(region);sample_counts.append(len(selected))
    gv=torch.stack(means) if means else z.sum().reshape(1)[:0]
    counts=torch.tensor(counts,device=x.device,dtype=z.dtype);classes=torch.tensor(classes,device=x.device);region_ids=torch.tensor(region_ids,device=x.device)
    w,t,wc,tc,bc=regional_aggregate(gv,counts,classes,region_ids,z.sum())
    aux={'R':aggregate(sampled,labels,z.sum()),'W':w,'T':t}
    rc=torch.stack([sampled[labels==c].mean() if (labels==c).any() else z.sum()*0 for c in range(4)])/max(1,len(labels.unique()))
    m.detail=dict(support=support,y=y,ids=ids,boundary=boundary,valid=valid,contributions={'R':rc,'W':wc,'T':tc},boundary_contribution=bc,
                  active={'R':float((sampled>0).float().mean()) if len(sampled) else 0.,'W':float((gv>0).float().mean()) if len(gv) else 0.,'T':float(t.detach()>0)},
                  sampled_regional=sum(sample_counts),sampled_uniform=len(labels),region_diagnostics={str(region):dict(total=int((valid&(boundary==bool(region))).sum()),supported=int((support_region&(boundary==bool(region))).sum()),student_errors=int((valid&(boundary==bool(region))&(z.argmax(1)!=y)).sum())) for region in (0,1)})
    return pixel_ce(z,y).mean(),aux

def losses(arm,m,data,step):
    l,aux=terms(m,data,step);variant=m.cfg['variant'];u=m.cfg.get('weights',{}).get(variant,0.)*aux[variant] if variant in aux else l*0
    if m.cfg['bucket']=='formal' and step in (0,999,1999):
        pars=[p for p in m.parameters() if p.requires_grad]
        squared={k:sum(float(g.double().square().sum()) for g in torch.autograd.grad(t,pars,retain_graph=True,allow_unused=True) if g is not None) for k,t in [('CE',l),*aux.items()]}
        append(Path(m.cfg['output'])/'gradient_diagnostics.jsonl',dict(step=step,squared=squared,active=m.detail['active'],boundary_supported=int((m.detail['boundary']&m.detail['support']).sum()),regions=m.detail['region_diagnostics'],sampled_uniform=m.detail['sampled_uniform'],sampled_regional=m.detail['sampled_regional'],boundary_loss_contribution=float(m.detail['boundary_contribution'].sum().detach())))
    stats=dict(supervised=float(l.detach()),unsupervised=float(u.detach()),**{k:float(v.detach()) for k,v in aux.items()},accepted_class_pixels=[0]*4,U_pixels=0,historical_raw_reads=0)
    m.detail=None;return l,u,stats

def calibrate_scales(cfg):
    source,m,ema,opt=setup(cfg);data=Current(dict(cfg,target_seed=20271091));values=[[] for _ in range(4)]
    for step in range(16):
        x,y,_=data.batch('L',step);_,_,_,a,_,support,_=references(m,x,y,data.cfg['target_seed'],step)
        ix=choose(support,data.cfg['target_seed'],step);labels=take(y[...,None],ix).squeeze(-1);v=take(a,ix)
        for c in range(4):
            v_c=v[labels==c][:,[j for j in range(4) if j!=c]];values[c].append(v_c[v_c>0].cpu())
    scales=[float(torch.cat(v).median()) if sum(len(t) for t in v) else 0. for v in values]
    write(Path(cfg['root'])/'SCALES.json',dict(scales=scales,status='PASS' if all(math.isfinite(s) and s>1e-6 for s in scales) else 'STOP',source_only=True,train_seed=20271091))

def measure(cfg):
    scales=json.loads((Path(cfg['root'])/'SCALES.json').read_text())
    if scales['status']!='PASS':write(Path(cfg['root'])/'P0_RESULT.json',dict(status='STOP',gates={'scales':False}));return
    cfg=dict(cfg,scales=scales['scales']);source,m,ema,opt=setup(cfg);data=Current(dict(cfg,target_seed=20271092))
    saved=torch.load(Path(cfg['root'])/'probe/final.pt',map_location='cpu',weights_only=False);assert saved['step']==256;m.load_state_dict(saved['model'])
    before=copy.deepcopy(m.state_dict());ref=copy.deepcopy(source.state_dict());rng=(torch.get_rng_state(),torch.cuda.get_rng_state_all())
    names=[n for n,p in m.named_parameters() if p.requires_grad];pars=[p for p in m.parameters() if p.requires_grad]
    squares={k:0. for k in ('CE','R','W','T')};dot=0.;covered=[0]*4;denom=[0]*4;patients=[set() for _ in range(4)];energy={k:[0.]*4 for k in ('R','W','T')};groups={'encoder':0.,'decoder':0.};active=[];bcov=[0]*4;bden=[0]*4;bpatients=[set() for _ in range(4)];boundary_energy=0.;sample_counts=[]
    for step in range(16):
        l,aux=terms(m,data,step);d=m.detail;gs={k:torch.autograd.grad(t,pars,retain_graph=True,allow_unused=True) for k,t in [('CE',l),*aux.items()]}
        for k,gg in gs.items():
            assert all(g is None or torch.isfinite(g).all() for g in gg);squares[k]+=sum(float(g.double().square().sum()) for g in gg if g is not None)
        dot+=sum(float((a.double()*b.double()).sum()) for a,b in zip(gs['T'],gs['W']) if a is not None and b is not None)
        for name,g in zip(names,gs['T']):
            if g is not None:groups['decoder' if 'native.decoder.' in name else 'encoder']+=float(g.double().square().sum())
        for c in range(4):
            bm=d['boundary']&(d['y']==c);bcov[c]+=int((bm&d['support']).sum());bden[c]+=int(bm.sum())
            for slot,pid in enumerate(d['ids']):
                if (bm[slot]&d['support'][slot]).any():bpatients[c].add(pid)
            mask=d['support']&(d['y']==c);covered[c]+=int(mask.sum());denom[c]+=int((d['y']==c).sum())
            for slot,pid in enumerate(d['ids']):
                if mask[slot].any():patients[c].add(pid)
            for k,v in d['contributions'].items():energy[k][c]+=float(v[c].detach())
        boundary_energy+=float(d['boundary_contribution'].sum().detach());sample_counts.append([d['sampled_uniform'],d['sampled_regional']]);active.append(d['active']);m.detail=None
    cosine=dot/math.sqrt(squares['T']*squares['W']) if squares['T']*squares['W']>0 else None;distance=math.sqrt(max(0.,2-2*cosine)) if cosine is not None else 0.
    weights={k:.25*math.sqrt(squares['CE']/squares[k]) if squares[k]>0 else None for k in ('R','W','T')}
    coverage=[a/b if b else 0. for a,b in zip(covered,denom)];share={k:sum(v[1:])/max(sum(v),1e-30) for k,v in energy.items()}
    boundary_coverage=[a/b if b else 0. for a,b in zip(bcov,bden)];boundary_share=boundary_energy/max(sum(energy['T']),1e-30)
    gates=dict(boundary_coverage=all(boundary_coverage[c]>=.05 and len(bpatients[c])>=3 for c in (1,2,3)),boundary_loss_share=boundary_share>=.05,scales=True,gradient_calibration=min(squares.values())>=1e-12 and all(v is not None and math.isfinite(v) and 1e-6<=v<=1e4 for v in weights.values()),foreground_coverage=all(coverage[c]>=.05 and len(patients[c])>=3 for c in (1,2,3)),foreground_loss_share=all(v>=.05 for v in share.values()),gradient_separation=distance>=.05,full_path_gradient=all(v>0 and math.isfinite(v) for v in groups.values()))
    assert runtime.exact(before,m.state_dict()) and runtime.exact(ref,source.state_dict()) and runtime.exact(rng,(torch.get_rng_state(),torch.cuda.get_rng_state_all()))
    assert all(p.grad is None for p in source.parameters()) and m.get_parameter(HEAD).grad is None
    write(Path(cfg['root'])/'P0_RESULT.json',dict(status='PASS' if all(gates.values()) else 'STOP',gates=gates,weights=weights,scales=cfg['scales'],gradient_squared=squares,normalized_gradient_distance=distance,gradient_cosine=cosine,coverage_by_class=coverage,covered_patient_counts=[len(v) for v in patients],foreground_loss_share=share,T_gradient_groups=groups,boundary_coverage_by_class=boundary_coverage,boundary_patient_counts=[len(v) for v in bpatients],boundary_loss_share=boundary_share,sampled_counts=sample_counts,active_per_batch=active,model_reference_RNG_unchanged=True,probe_discarded=True))

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

def regional_self_check():
    for device in ('cpu','cuda'):
        y=torch.zeros((1,5,5),device=device,dtype=torch.long);y[:,2:,2:]=1
        boundary,valid=regions(y);assert valid.all() and not boundary[0,0,0] and boundary[0,1,1]
        yy=torch.full((1,4,4),255,device=device);bb,vv=regions(yy);assert not vv.any() and not bb.any()
        m=torch.tensor([1.,3.],device=device,requires_grad=True);counts=torch.tensor([9.,1.],device=device);c=torch.zeros(2,device=device);r=torch.tensor([0,1],device=device)
        w,t,wc,tc,bc=regional_aggregate(m,counts,c,r,m.sum());assert torch.allclose(w,torch.tensor(1.2,device=device)) and float(t)==2.
        assert torch.allclose(torch.autograd.grad(t,m)[0],torch.tensor([.5,.5],device=device));assert float(tc.sum())==float(t) and float(bc.sum())==1.5
        one=regional_aggregate(m[:1],counts[:1],c[:1],r[:1],m.sum());assert float(one[0])==float(one[1])==1.
        empty=regional_aggregate(m[:0],counts[:0],c[:0],r[:0],m.sum());assert float(empty[0])==float(empty[1])==0.
    return dict(status='PASS',synthetic_reverse=2,weighted_and_equal_region=True,empty_and_single_region=True,ignore_and_padding=True)

def controller(cfg,guard):
    root=Path(cfg['root']);assert json.loads((Path(cfg['compatible_root'])/'AUDIT.json').read_text())['status']=='PASS' and (Path(cfg['compatible_root'])/'DELIVERY.json').exists()
    write(root/'WORKER.private.json',dict(pid=os.getpid(),birth=Path('/proc/self/stat').read_text().split()[21]))
    geometry_check(cfg);write(root/'MARGIN_SELF_CHECK.json',margin_self_check());write(root/'REGIONAL_SELF_CHECK.json',regional_self_check())
    def stage(mode,c):
        guard(cfg,new_stage=True);assert int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))>=18000
        p=root/('SETTINGS_'+c['group']+'_'+mode+'.private.json');write(p,c)
        with (root/(c['group']+'_'+mode+'.log')).open('x') as log:q=subprocess.Popen([cfg['runtime'],cfg['entry']],env=dict(os.environ,TASK_SETTINGS=str(p),TASK_MODE=mode),stdout=log,stderr=subprocess.STDOUT)
        a=dict(pid=q.pid,birth=Path('/proc/'+str(q.pid)+'/stat').read_text().split()[21],group=c['group'],mode=mode);write(root/'ACTIVE.private.json',a);write(root/('ACTIVE_'+c['group']+'_'+mode+'.private.json'),a);write(root/'STATUS.json',dict(status='RUNNING',**a))
        code=q.wait();write(root/('EXIT_'+c['group']+'_'+mode+'.json'),dict(exit_code=code));assert code==0
    common=dict(cfg,arm='A',preflight_only=False,checkpoint_each_step=True,save_exception_state=True,diagnostic_holdout_excluded=False)
    stage('qualify',dict(common,group='P0_ENGINEERING',variant='B',target_seed=20271090,bucket='p0',phase='engineering',output=str(root/'p0_engineering')))
    c=dict(common,group='PROBE',variant='B',target_seed=20271090,bucket='p0',phase='probe',output=str(root/'probe'))
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
