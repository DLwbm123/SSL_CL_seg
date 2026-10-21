"""Equal-energy polar displacement intervention on eight encoder matrices."""
import json,math,os,subprocess,traceback
from pathlib import Path
from unittest.mock import patch
import torch
from . import core,target,runtime,spectrum_alignment as shared,n_contribution as prior,projection_orientation as orientation
from .bounded_runtime import install
from ..main_style_ssl_v1.runtime import write,append

SEEDS=(20261111,20261112)
METHODS=('S','SP','GR','GRP')
CAPS=dict(detach_native=160,detach_encoder=1632,detach_reverse=64,engineering_native=768,engineering_encoder=1536,engineering_reverse=320,formal_native=32400,formal_encoder=770000,formal_reverse=16032,diagnostic_reverse=80,evaluation_native=23136)
POLAR_STEPS=8192
SHAPES=((16,1,3,3),(16,16,3,3),(32,16,3,3),(32,32,3,3),(64,32,3,3),(64,64,3,3),(128,64,3,3),(128,128,3,3))


def balanced(delta):
    """Preserve Frobenius energy and numerical row/column support, flatten modes."""
    a=delta.reshape(delta.shape[0],-1).float()
    u,s,vh=torch.linalg.svd(a,full_matrices=False,**({'driver':'gesvd'} if a.is_cuda else {}))
    assert torch.isfinite(s).all()
    norm=torch.linalg.vector_norm(s);keep=s>s[0]*max(a.shape)*torch.finfo(a.dtype).eps;r=int(keep.sum())
    out=torch.zeros_like(a) if r==0 else (u[:,keep]@vh[keep,:])*(norm/math.sqrt(r))
    assert torch.isfinite(out).all() and torch.allclose(out.norm(),a.norm(),rtol=2e-5,atol=1e-12)
    metrics=dict(rank=r,proposal_norm=float(a.norm()),balanced_norm=float(out.norm()),stable_rank=float(norm.square()/s[0].square()) if r else 0.,cosine=float((a*out).sum()/norm.square()) if r else 1.)
    return out.reshape_as(delta),metrics


def self_check():
    records={}
    for device in ['cpu','cuda']:
        a=torch.diag(torch.tensor([4.,2.,1.,0.],device=device));b,m=balanced(a)
        expected=torch.diag(torch.tensor([math.sqrt(7)]*3+[0.],device=device))
        assert torch.allclose(b,expected,rtol=1e-5,atol=1e-6) and m['rank']==3
        assert torch.equal(b,balanced(a)[0]) and torch.equal(balanced(torch.zeros_like(a))[0],torch.zeros_like(a))
        assert torch.allclose(balanced(a.T)[0],b.T,atol=1e-6)
        # Rectangular/conv layout and nonzero rank support; no model or optimizer call.
        q=torch.arange(1,97,dtype=torch.float32,device=device).sin().reshape(4,3,2,4)
        x,_=balanced(q);assert x.shape==q.shape and torch.allclose(x.norm(),q.norm(),rtol=2e-5)
        records[device]=dict(energy_preserved=True,rank_support_preserved=True,zero_exact=True,repeat_exact=True)
    return dict(status='PASS',medical_model_forwards=0,optimizer_updates=0,checks=records)


def make_setup(cfg,calls):
    build_base,setup_base=prior.make_setup(core.build)
    value=torch.load(cfg['projection_control'],map_location='cpu',weights_only=False)['rotated_weight']
    def attach(m):
        m.register_buffer('rotated_projection',value.to(next(m.parameters()).device));return m
    def build(c):return attach(build_base(c))
    def setup(c):
        source,m,ema,opt=setup_base(c);source,m,ema=map(attach,(source,m,ema))
        m.collect_stats=False;m.pending_diagnostic=None;m.calls=calls
        selected=[(n,p) for n,p in m.parent.named_parameters() if p.requires_grad and p.ndim>1 and n.startswith(('native.enc','native.bottleneck'))]
        assert tuple(tuple(p.shape) for _,p in selected)==SHAPES
        root=Path(c['root']);ledger=root/'polar_counter.json'
        def before(*_):
            m.proposal_before={n:p.detach().clone() for n,p in selected} if c['polar'] else {}
            if m.pending_diagnostic is not None:
                m.diagnostic_before={n:p.detach().cpu().clone() for n,p in m.parent.named_parameters() if p.requires_grad}
                m.pending_diagnostic['observed_clipped_global_gradient_norm']=sum(float(p.grad.detach().double().square().sum()) for p in m.parent.parameters() if p.requires_grad and p.grad is not None)**.5
            if c['polar']:
                v=json.loads(ledger.read_text()) if ledger.exists() else dict(executed_steps=0,executed_SVD=0,pending=None,cap_steps=POLAR_STEPS,cap_SVD=8*POLAR_STEPS)
                assert v['pending'] is None and v['executed_steps']<POLAR_STEPS
                v['pending']=dict(group=c['group'],phase=c['phase'],matrices=8);write(ledger,v)
        def after(*_):
            if c['polar']:
                stats=[]
                with torch.no_grad():
                    for n,p in selected:
                        old=m.proposal_before[n];delta=p-old;new,z=balanced(delta);p.copy_(old+new)
                        z['module']=n;z['actual_norm']=float((p-old).norm())
                        assert abs(z['actual_norm']-z['proposal_norm'])<=1e-7+.005*z['proposal_norm']
                        stats.append(z)
                v=json.loads(ledger.read_text());assert v['pending']==dict(group=c['group'],phase=c['phase'],matrices=8)
                v['executed_steps']+=1;v['executed_SVD']+=8;v['pending']=None
                append(root/'polar_events.jsonl',dict(group=c['group'],phase=c['phase'],ordinal=v['executed_steps'],matrices=stats));write(ledger,v)
            m.proposal_before=None
            if m.pending_diagnostic is not None:
                named=[(n,p) for n,p in m.parent.named_parameters() if p.requires_grad]
                delta=[p.detach().cpu().double()-m.diagnostic_before[n].double() for n,p in named]
                m.pending_diagnostic['actual_SGD_displacement']=shared.group_norms([n for n,p in named],delta)
                m.pending_diagnostic['displacement_includes_polar']=c['polar']
                append(Path(c['output'])/'gradient_diagnostics.jsonl',m.pending_diagnostic)
                m.pending_diagnostic=None;m.diagnostic_before=None
        opt.register_step_pre_hook(before);opt.register_step_post_hook(after)
        return source,m,ema,opt
    return build,setup


def controller(cfg,guard):
    root=Path(cfg['root']);(root/'preparation').symlink_to(Path(cfg['comparison_root'])/'preparation',target_is_directory=True)
    assert json.loads((Path(cfg['orientation_root'])/'AUDIT.json').read_text())['status']=='PASS'
    assert cfg['lambda_R']==json.loads((Path(cfg['orientation_root'])/'CALIBRATION.json').read_text())['lambda_R']
    def stage(mode,c):
        guard(cfg,new_stage=True);p=root/('SETTINGS_'+c['group']+'_'+mode+'.private.json');write(p,c)
        free=int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True));assert free>=18000
        with (root/(c['group']+'_'+mode+'.log')).open('x') as log:
            child=subprocess.Popen([cfg['runtime'],cfg['entry']],env=dict(os.environ,TASK_SETTINGS=str(p),TASK_MODE=mode),stdout=log,stderr=subprocess.STDOUT)
        active=dict(pid=child.pid,birth=Path('/proc/'+str(child.pid)+'/stat').read_text().split()[21],group=c['group'],mode=mode)
        write(root/'ACTIVE.private.json',active);write(root/('ACTIVE_'+c['group']+'_'+mode+'.private.json'),active);write(root/'STATUS.json',dict(status='RUNNING',**active))
        code=child.wait();write(root/('EXIT_'+c['group']+'_'+mode+'.json'),dict(exit_code=code));assert code==0,c['group']+' '+mode+' failed'
    detach=root/'detach';detach.mkdir();stage('detach',dict(cfg,group='DETACH',method='S',polar=True,phase='engineering',target_seed=SEEDS[0],output=str(detach)))
    configs=[dict(cfg,group=f'{seed}_{name}',method='GR' if name.startswith('GR') else 'S',polar=name.endswith('P'),variant=name,target_seed=seed,arm='A',stage=f'{seed}_{name}',preflight_only=False,checkpoint_each_step=True,save_exception_state=True,audit_source_training=True,diagnostic_holdout_excluded=False) for seed in SEEDS for name in METHODS]
    for c in configs:stage('qualify',dict(c,phase='engineering',output=str(root/'qualification'/c['group'])))
    for c in configs:
        c=dict(c,phase='formal',output=str(root/'jobs'/c['group']));stage('train',c);stage('evaluate',c)
    write(root/'STATUS.json',dict(status='COMPLETE',audit_pending=True,resources=guard(cfg)));write(root/'EXIT.json',dict(exit_code=0))


def main(cfg):
    guard=install(cfg);target.guard=guard;mode=os.environ.get('TASK_MODE','controller');root=Path(cfg['root'])
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    try:
        if mode=='controller':
            write(root/'WORKER.private.json',dict(pid=os.getpid(),birth=Path('/proc/self/stat').read_text().split()[21]));write(root/'SELF_CHECK.json',self_check());controller(cfg,guard);return
        with patch.object(shared,'CAPS',CAPS),patch.object(orientation,'SEEDS',SEEDS):
            calls=shared.Calls(cfg,mode);build,setup=make_setup(cfg,calls);forward=core.Model.forward;backward=torch.autograd.backward;grad=torch.autograd.grad
            def f(m,x):return calls.call(calls.kind('native'),len(x),lambda:forward(m,x))
            def e(m,x):return calls.call(calls.kind('encoder'),len(x),lambda:orientation.native_encode(m,x))
            def b(*a,**kw):return calls.call(calls.kind('reverse'),1,lambda:backward(*a,**kw))
            def g(*a,**kw):return calls.call(calls.kind('reverse'),1,lambda:grad(*a,**kw))
            with patch.object(core,'build',build),patch.object(core,'target_setup',setup),patch.object(core,'losses',orientation.losses),patch.object(target,'Domain',shared.Current),patch.object(target,'preflight',shared.preflight),patch.object(prior,'losses',orientation.losses),patch.object(shared,'losses',orientation.losses),patch.object(core.Model,'forward',f),patch.object(core.Model,'encode',e),patch.object(torch.autograd,'backward',b),patch.object(torch.autograd,'grad',g):
                {'detach':orientation.detach_check,'qualify':shared.qualify,'train':target.main,'evaluate':target.evaluate}[mode](cfg)
    except BaseException as exc:
        write(root/('FAILURE_'+cfg.get('group','controller')+'_'+mode+'.json'),dict(error=str(exc),traceback=traceback.format_exc()))
        if mode=='controller':write(root/'STATUS.json',dict(status='FAILED'))
        raise
