"""One new target stream, five frozen current-data/frame contribution groups."""
import copy,json,math,os,subprocess,time,traceback
from pathlib import Path
from unittest.mock import patch
import h5py,numpy as np,torch
from PIL import Image,ImageOps,ImageEnhance
from . import core,target,runtime
from .bounded_runtime import install
from .module import rows,centroid,anchored_loss
from ..main_style_ssl_v1.data import Accessor
from ..di_dmpa_gate1.binding import safe_asset
from ..five_frameworks_v1.recipes import generator
from ..main_style_ssl_v1.runtime import write,append
from ..siemens_ssl_reproduction_v1.methods import pixel_ce

GROUPS=('S','G00','G10','G01','G11')


def positions(seed,step,n,device):
    result=[]
    for slot in range(n):
        g=generator(seed,1,1,step,'positions:'+str(slot))
        delta=torch.randint(0,4,(4,2),generator=g)
        result.append((delta[:,0]+torch.tensor([0,0,4,4]))*8+delta[:,1]+torch.tensor([0,4,0,4]))
    return torch.stack(result).to(device)


def strong(images,seed,step,view):
    output=[]
    for slot,image in enumerate(images.detach().cpu()):
        g=generator(seed,1,1,step,'strong:'+str(view)+':'+str(slot))
        rng=np.random.default_rng(int(torch.randint(2**62,(),generator=g)))
        im=Image.fromarray(((image[0].numpy()+1)*127.5).clip(0,255).round().astype(np.uint8))
        for _ in range(2):
            op=int(rng.integers(8));level=float(rng.uniform(0,1));sign=int(rng.choice([-1,1]))
            if op==1:im=ImageOps.autocontrast(im)
            elif op==2:im=ImageOps.equalize(im)
            elif op in (3,4,5):im={3:ImageEnhance.Brightness,4:ImageEnhance.Contrast,5:ImageEnhance.Sharpness}[op](im).enhance(1+sign*.9*level)
            elif op==6:im=ImageOps.posterize(im,max(1,8-round(4*level)))
            elif op==7:im=ImageOps.solarize(im,round(256*(1-level)))
        array=np.array(im,copy=True);h,w=array.shape;side=min(h,w)//2;cy,cx=int(rng.integers(h)),int(rng.integers(w))
        array[max(0,cy-side//2):min(h,cy+side//2),max(0,cx-side//2):min(w,cx+side//2)]=128
        output.append(torch.from_numpy(array.astype(np.float32)/127.5-1)[None])
    return torch.stack(output).to(images.device)


class Current:
    def __init__(self,cfg,domain='Philips'):
        assert domain=='Philips';self.cfg=cfg;self.group=cfg['group'];self.arm='A';self.l_reads=self.u_reads=0
        self._l=Accessor(cfg['data'],'mnms','Philips','train_labeled')
        self._u=Accessor(cfg['data'],'mnms','Philips','train_unlabeled') if self.group in ('G10','G11') else None
        self._images=copy.copy(self._l);self._images.role='train_unlabeled';self._images.checked=set()
        assert len(self._l.patients)==13
        allowed={str(safe_asset(Path(cfg['data']),r[k]).resolve()) for a in (self._l,self._u) if a is not None for r in a.rows for k in ('image_h5_relpath','label_h5_relpath') if k in r}
        real=h5py.File
        def restricted(name,*a,**kw):
            if str(Path(name).resolve()) not in allowed:raise PermissionError('outside current Philips training whitelist')
            return real(name,*a,**kw)
        h5py.File=restricted;self.seen_steps=set();self.current_keys=()
    def batch(self,role,step,n=2):
        access=self._l if role=='L' else self._u if role=='U' else self._images
        assert access is not None and (role!='L' or n==2)
        xs=[];ys=[];ids=[];keys=[]
        for block in range(n//2):
            g=generator(self.cfg['target_seed'],1,1,step,role+str(block))
            for j in torch.randperm(len(access.patients),generator=g)[:2].tolist():
                patient=access.patients[j];ii=access.by_patient[patient];i=ii[int(torch.randint(len(ii),(),generator=g))];z=int(torch.randint(access.depths[i],(),generator=g))
                item=access.read(i,z);xs.append(item['image']);ids.append(patient);keys.append((access.rows[i]['case_id'],z))
                if role=='L':ys.append(item['label'])
                else:assert 'label' not in item
            if role=='L':self.l_reads+=1
            else:self.u_reads+=1
        if role=='L':self.current_keys=tuple(keys)
        elif step not in self.seen_steps:
            allkeys=self.current_keys+tuple(keys);self.seen_steps.add(step)
            append(Path(self.cfg['output'])/'sampling_aggregates.jsonl',dict(step=step,images=len(allkeys),distinct_images=len(set(allkeys)),repeat_fraction=1-len(set(allkeys))/len(allkeys),fill_pool='U' if role=='U' else 'L_images_only',fill_labels_read=0))
        return torch.stack(xs).cuda(),torch.stack(ys).cuda() if ys else None,tuple(ids)
    def labeled(self,step):return self.batch('L',step)
    def unlabeled(self,step):return self.batch('U',step)


def make_setup(original_build):
    def build(cfg):
        m=original_build(cfg);m.generator=None;m.code=None;return m
    def setup(cfg):
        m=build(cfg);m.projection.load_state_dict(torch.load(cfg['shared_projection'],map_location='cpu',weights_only=False))
        m.parent.configure_stage_training();m.projection.requires_grad_(False).eval();m.cfg=cfg
        groups=[dict(params=[p for p in m.parent.parameters() if p.requires_grad and (p.ndim>1)==matrix],lr=.001,fixed_lr=.001,name='native_'+str(matrix),weight_decay=.0005 if matrix else 0.) for matrix in (True,False)]
        opt=torch.optim.SGD(groups,lr=.001,momentum=.9,nesterov=True);runtime.attach(opt,cfg,cfg['group'],cfg['phase'])
        return m.teacher(),m,m.teacher(),opt
    return build,setup


def losses(arm,model,data,step):
    x,y,_=data.batch('L',step);ce=pixel_ce(model(x),y).mean();group=data.group
    if group=='S':geo=ce*0;images=0;nr=0
    else:
        u,_,_=data.batch('U' if group in ('G10','G11') else 'F',step,14);allx=torch.cat([x,u]);seed=data.cfg['target_seed']
        pos=positions(seed,step,16,x.device);fields=model.encode(torch.cat([allx,strong(allx,seed,step,0),strong(allx,seed,step,1)]))
        center=centroid(*[rows(v,pos) for v in fields.chunk(3)])
        if group in ('G01','G11'):geo,_=anchored_loss(center,model.frames,torch.ones(64,device=x.device,dtype=torch.long))
        else:geo=1-torch.linalg.svdvals((2/3)*center).sum()/math.sqrt(64*32)
        images=48;nr=64
    return ce,.1*geo,dict(supervised=float(ce.detach()),unsupervised=float(.1*geo.detach()),accepted_class_pixels=[0]*4,U_pixels=0,geometry_rows=nr,geometry_images=images,historical_raw_reads=0)


def preflight(cfg,out,source,model,ema,opt,data):
    engine=target.engine;initial=copy.deepcopy(engine.snapshot(model,ema,opt,data,0));runtime.save(cfg,initial,out/'initial.pt')
    assert not opt.state and model.generator is None and model.code is None
    fixed={'native.decoder.conv_logit.sigma.weight','native.decoder.conv_logit.grad_update'}
    assert {n for n,p in model.parent.named_parameters() if not p.requires_grad}==fixed
    x,_,_=data.labeled(0)
    with torch.no_grad():assert torch.equal(source(x),model(x)) and torch.equal(ema(x),model(x))
    l,u,_=losses('A',model,data,0);ep=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith(('parent.native.enc','parent.native.bottleneck'))]
    gg=torch.autograd.grad(u,ep,retain_graph=True,allow_unused=True);norm=sum(float(g.square().sum()) for g in gg if g is not None)**.5
    assert all(g is None or torch.isfinite(g).all() for g in gg)
    assert (norm==0) if cfg['group']=='S' else (norm>0)
    (l+u).backward();value=float(l+u);pars=[p for p in model.parameters() if p.requires_grad];gold=[p.grad.clone() for p in pars]
    engine.restore(initial,model,ema,opt,data);l,u,_=losses('A',model,data,0);(l+u).backward()
    assert value==float(l+u) and all(torch.equal(p.grad,g) for p,g in zip(pars,gold))
    engine.restore(initial,model,ema,opt,data);runtime.save(cfg,engine.snapshot(model,ema,opt,data,0),out/'latest.pt')
    write(out/'PREFLIGHT.json',dict(status='PASS',common_native_entry=True,empty_optimizer=True,full_native_trainable=True,frozen_memory=True,encoder_geometry_gradient=norm,golden_exact=True,target_only_HDF5=True,group=cfg['group'],target_seed=cfg['target_seed']))


def main(cfg):
    guard=install(cfg);target.guard=guard;mode=os.environ.get('TASK_MODE','controller');root=Path(cfg['root'])
    if mode=='controller':
        try:
            stages=[(g,'qualify','engineering') for g in GROUPS]+[(g,m,'formal') for g in GROUPS for m in ('train','evaluate')]
            for group,mode,phase in stages:
                guard(cfg,new_stage=True);out=root/('qualification' if phase=='engineering' else 'jobs')/group
                c=dict(cfg,group=group,arm='A',stage=group,phase=phase,output=str(out),preflight_only=False,checkpoint_each_step=True,save_exception_state=True,audit_source_training=True,diagnostic_holdout_excluded=False)
                path=root/('SETTINGS_'+group+'_'+mode+'.private.json');write(path,c)
                free=int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>=14000
                with (root/(group+'_'+mode+'.log')).open('x') as f:p=subprocess.Popen([cfg['runtime'],cfg['entry']],env=dict(os.environ,TASK_SETTINGS=str(path),TASK_MODE=mode),stdout=f,stderr=subprocess.STDOUT)
                active=dict(pid=p.pid,birth=Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],group=group,mode=mode)
                write(root/'ACTIVE.private.json',active);write(root/('ACTIVE_'+group+'_'+mode+'.private.json'),active);write(root/'STATUS.json',dict(status='RUNNING',group=group,mode=mode))
                code=p.wait();write(root/('EXIT_'+group+'_'+mode+'.json'),dict(exit_code=code))
                if code:raise RuntimeError(group+' '+mode+' failed')
            write(root/'STATUS.json',dict(status='COMPLETE',audit_pending=True,resources=guard(cfg)));write(root/'EXIT.json',dict(exit_code=0))
        except BaseException as e:
            write(root/'FAILURE.json',dict(error=str(e),traceback=traceback.format_exc()));write(root/'STATUS.json',dict(status='FAILED'));raise
        return
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    build,setup=make_setup(core.build)
    with patch.object(core,'build',build),patch.object(core,'target_setup',setup),patch.object(core,'losses',losses),patch.object(target,'Domain',Current),patch.object(target,'preflight',preflight):
        {'qualify':target.qualify,'train':target.main,'evaluate':target.evaluate}[mode](cfg)
