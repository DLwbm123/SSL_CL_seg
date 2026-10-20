"""Frozen inference-only EMA/readout diagnostic; preserves parent round endpoints."""
import copy,json,os,time,subprocess
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from . import core
from .module import unit
from ..philips_transfer_v1.run import evaluate
from ..siemens_lonly_v1.run import compare


def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def eligible(k,v):return k.startswith('parent.') and v.is_floating_point() and not k.endswith(('sigma.weight','grad_update'))


def mixed(student,ema,entry,kind):
    enc=('parent.native.enc1.','parent.native.enc2.','parent.native.enc3.','parent.native.bottleneck.')
    dec='parent.native.decoder.'
    native=[k for k in student if k.startswith('parent.native.')]
    assert native and all(k.startswith(enc) or k.startswith(dec) for k in native),'unknown native block'
    result={}
    for k,v in student.items():
        if kind=='studentE_emaD':value=ema[k] if k.startswith(dec) else v
        elif kind=='emaE_studentD':value=ema[k] if k.startswith(enc) else v
        elif kind=='entry_mix':value=entry[k]*(.999**2000)+v*(1-.999**2000) if eligible(k,v) else v
        else:raise ValueError(kind)
        result[k]=value.clone()
    assert all(torch.isfinite(v).all() for v in result.values())
    return result


class Ledger:
    caps={'real_validation':7224,'synthetic_segmentation':192,'synthetic_encoder':192}
    def __init__(self,cfg):self.cfg=cfg;self.root=Path(cfg['diagnostic_root']);self.counts={k:0 for k in self.caps};self.calls=0
    def guard(self,new=False):
        elapsed=time.time()-self.cfg['diagnostic_started_epoch']
        if elapsed>=6300 or (new and elapsed>=5400):raise TimeoutError('diagnostic original clock limit')
        used=int(subprocess.check_output(['du','-s','-B1',str(self.root)],text=True).split()[0])
        if used+100_000_000>=500_000_000:raise OSError('diagnostic storage hard limit with exit reserve')
        if used>=100_000_000:raise OSError('diagnostic storage warning: stop before growth')
        return dict(elapsed_seconds=elapsed,allocated_bytes=used,reserve_bytes=100_000_000)
    def call(self,kind,x,fn):
        n=len(x);assert self.counts[kind]+n<=self.caps[kind]
        if self.calls%50==0:self.guard()
        with (self.root/'forward_events.jsonl').open('a') as f:f.write(json.dumps(dict(kind='reserved',role=kind,n=n,call=self.calls+1))+'\n')
        y=fn(x);assert torch.isfinite(y).all()
        self.counts[kind]+=n;self.calls+=1
        with (self.root/'forward_events.jsonl').open('a') as f:f.write(json.dumps(dict(kind='executed',role=kind,n=n,call=self.calls))+'\n')
        write(self.root/'FORWARD_COUNTER.json',dict(counts=self.counts,calls=self.calls,optimizer=0,functional=0))
        return y


class Counted(nn.Module):
    def __init__(self,m,ledger):super().__init__();self.m=m;self.ledger=ledger
    def forward(self,x):return self.ledger.call('real_validation',x,self.m)


def paired(a,b):
    out=compare(a,b)
    def patient(v):
        d={}
        for row in v:d.setdefault(row['patient'],[]).append(np.mean(row['dice']))
        assert all(len(x)==2 for x in d.values());return {k:np.mean(x) for k,x in d.items()}
    aa,bb=patient(a),patient(b);assert aa.keys()==bb.keys()
    delta=np.array([100*(bb[k]-aa[k]) for k in sorted(aa)])
    rng=np.random.default_rng(163);means=delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(1)
    out['patient_bootstrap95_pp']=np.quantile(means,[.025,.975]).tolist();return out


@torch.inference_mode()
def main(cfg):
    root=Path(cfg['diagnostic_root']);parent=Path(cfg['root']);ledger=Ledger(cfg)
    assert read(parent/'AUDIT.json')['status']=='PASS'
    assert not (root/'RESULTS.json').exists()
    torch.set_num_threads(4);torch.manual_seed(163);torch.cuda.manual_seed_all(163)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    result={'status':'RUNNING','primary_round_unchanged':True,'ema_alpha':.999,'entry_weight':.999**2000,'full_trajectory_reconstructed':False,'arms':{}}
    try:
        for arm in ['A','C']:
            ledger.guard(new=True);out=parent/'jobs'/arm
            initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=False)
            final=torch.load(out/'final.pt',map_location='cpu',weights_only=False)
            assert initial['step']==0 and final['step']==final['physical_updates']==2000
            entry,student,ema=initial['model'],final['model'],final['ema']
            assert all(torch.equal(v,initial['ema'][k]) for k,v in entry.items())
            assert all(torch.equal(v,ema[k]) for k,v in student.items() if not eligible(k,v))
            model=core.build(cfg).eval().requires_grad_(False)
            native_keys=[k for k in student if k.startswith('parent.native.')]
            assert set(k.split('.')[2] for k in native_keys)=={'enc1','enc2','enc3','bottleneck','decoder'}
            # A self-mix must preserve every tensor, without spending model forwards.
            noop=mixed(student,student,student,'studentE_emaD')
            assert all(torch.equal(v,noop[k]) for k,v in student.items())
            del noop
            ar={'blocks':{'encoder':['enc1','enc2','enc3','bottleneck'],'decoder':['decoder']},'models':{},'synthetic':{}}
            for kind in ['studentE_emaD','emaE_studentD','entry_mix']:
                ledger.guard(new=True);model.load_state_dict(mixed(student,ema,entry,kind))
                values={}
                for domain in ['Siemens','Philips']:
                    dest=root/('private_'+arm+'_'+kind+'_'+domain+'.jsonl');assert not dest.exists()
                    metrics,newrows=evaluate(Counted(model,ledger),cfg,domain,'val',dest)
                    expected=rows(out/('private_'+domain+'_val.jsonl'))
                    emarows=rows(out/('private_ema_'+domain+'_val.jsonl'))
                    assert {(r['patient'],r['phase']) for r in newrows}=={(r['patient'],r['phase']) for r in expected}
                    values[domain]={'metrics':metrics,'versus_student':paired(expected,newrows),'versus_EMA':paired(emarows,newrows)}
                ar['models'][kind]=values
                write(root/'PARTIAL.json',dict(arm=arm,completed_kind=kind,results=ar))
            # Fixed 32 codes from this arm, same generated soft masks for each model state.
            model.load_state_dict(student)
            z=unit(model.code).clone();image_shape=(384,384)
            totals={name:[] for name in ['entry','student','EMA']}
            for start in range(0,32,2):
                images,mask=model.generator(z[start:start+2],image_shape)
                assert torch.isfinite(images).all() and torch.isfinite(mask).all()
                for name,state in [('entry',entry),('student',student),('EMA',ema)]:
                    # Entry has random pre-memory extras: only replace native entry tensors.
                    model.load_state_dict({k:state[k] if k.startswith('parent.') else student[k] for k in student})
                    logits=ledger.call('synthetic_segmentation',images,model);pred=logits.softmax(1)
                    feature=ledger.call('synthetic_encoder',images,model.encode)
                    ce=float(-(mask*logits.log_softmax(1)).sum(1).mean())
                    dice=float(((2*(pred[:,1:]*mask[:,1:]).sum((-2,-1))+1e-6)/(pred[:,1:].sum((-2,-1))+mask[:,1:].sum((-2,-1))+1e-6)).mean())
                    zz=z[start:start+2];mse=float((feature-zz).square().mean());cos=float(F.cosine_similarity(feature,zz,dim=1).mean())
                    totals[name].append(dict(soft_CE=ce,foreground_soft_Dice=dice,code_MSE=mse,code_cosine=cos))
            ar['synthetic']={name:{k:float(np.mean([v[k] for v in values])) for k in values[0]} for name,values in totals.items()}
            result['arms'][arm]=ar;write(root/'PARTIAL.json',result)
            del model,initial,final;torch.cuda.empty_cache()
        assert ledger.counts==ledger.caps
        result.update(status='COMPLETE',forward_counts=ledger.counts,optimizer_calls=0,functional_updates=0,peak_cuda_bytes=torch.cuda.max_memory_allocated(),resources=ledger.guard(),limitations=['exploratory single seed','hybrids measure compatibility, not additive causality','synthetic soft masks are not independent ground truth'])
        write(root/'RESULTS.json',result);write(root/'STATUS.json',{'status':'COMPLETE'});write(root/'EXIT.json',{'exit_code':0})
    except BaseException as e:
        import traceback
        write(root/'FAILURE.json',dict(error=str(e),traceback=traceback.format_exc(),counts=ledger.counts));write(root/'STATUS.json',{'status':'FAILED'});raise


@torch.inference_mode()
def mask_check(cfg):
    root=Path(cfg['diagnostic_root']);assert read(root/'STATUS.json')['status']=='COMPLETE'
    assert not (root/'MASK_COMPOSITION.json').exists();ledger=Ledger(cfg);ledger.guard(new=True)
    results={}
    for arm in ['A','C']:
        state=torch.load(Path(cfg['root'])/'jobs'/arm/'final.pt',map_location='cpu',weights_only=False)['model']
        model=core.build(cfg).eval().requires_grad_(False);model.load_state_dict(state)
        soft=torch.zeros(4,dtype=torch.float64);hard=torch.zeros(4,dtype=torch.float64);nonempty=torch.zeros(4,dtype=torch.long);image_min=1e99;image_max=-1e99
        z=unit(model.code)
        for i in range(0,32,2):
            x,mask=model.generator(z[i:i+2],(384,384));assert torch.isfinite(x).all() and torch.isfinite(mask).all()
            soft+=mask.double().sum((0,2,3)).cpu();labels=mask.argmax(1)
            for c in range(4):
                hard[c]+=(labels==c).sum().cpu();nonempty[c]+=(labels==c).flatten(1).any(1).sum().cpu()
            image_min=min(image_min,float(x.min()));image_max=max(image_max,float(x.max()))
        results[arm]=dict(soft_class_mass_fraction=(soft/soft.sum()).tolist(),hard_class_pixel_fraction=(hard/hard.sum()).tolist(),codes_with_nonempty_hard_class=nonempty.tolist(),codes=32,image_min=image_min,image_max=image_max)
        del model;torch.cuda.empty_cache()
    write(root/'MASK_COMPOSITION.json',dict(status='COMPLETE',class_order=['background','LV','MYO','RV'],metrics=results,extra_generator_samples=64,additional_native_forwards=0,additional_optimizer_updates=0,resources=ledger.guard()))
