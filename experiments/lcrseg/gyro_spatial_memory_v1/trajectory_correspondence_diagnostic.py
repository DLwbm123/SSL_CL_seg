"""Frozen post-N EMA-init removal and image/mask correspondence; zero updates."""
import json,time,subprocess
from pathlib import Path
import numpy as np
import torch
from torch import nn
from . import core
from .module import JointGenerator,unit
from .single_target_audit import paired,rows
from ..philips_transfer_v1.run import evaluate


def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def eligible(k,v):return k.startswith('parent.') and v.is_floating_point() and not k.endswith(('sigma.weight','grad_update'))


class Ledger:
    caps={'real_validation':3612,'synthetic_native':128,'generator':64}
    def __init__(self,cfg):self.cfg=cfg;self.root=Path(cfg['diagnostic_root']);self.count={k:0 for k in self.caps};self.calls=0
    def guard(self,new=False):
        elapsed=time.time()-self.cfg['diagnostic_started_epoch']
        if elapsed>=6300 or (new and elapsed>=5400):raise TimeoutError('diagnostic original clock')
        used=int(subprocess.check_output(['du','-s','-B1',str(self.root)],text=True).split()[0])
        if used>=100_000_000 or used+100_000_000>=500_000_000:raise OSError('diagnostic storage protection')
        return dict(elapsed_seconds=elapsed,allocated_bytes=used,reserve_bytes=100_000_000)
    def call(self,kind,n,fn):
        assert self.count[kind]+n<=self.caps[kind]
        if self.calls%50==0:self.guard()
        with (self.root/'forward_events.jsonl').open('a') as f:f.write(json.dumps(dict(kind='reserved',role=kind,n=n,call=self.calls+1))+'\n')
        result=fn()
        for v in result if isinstance(result,tuple) else (result,):assert torch.isfinite(v).all()
        self.count[kind]+=n;self.calls+=1
        with (self.root/'forward_events.jsonl').open('a') as f:f.write(json.dumps(dict(kind='executed',role=kind,n=n,call=self.calls))+'\n')
        write(self.root/'COUNTER.json',dict(count=self.count,calls=self.calls,optimizer=0,functional=0,pending=0))
        return result


class Counted(nn.Module):
    def __init__(self,model,ledger):super().__init__();self.model=model;self.ledger=ledger
    def forward(self,x):return self.ledger.call('real_validation',len(x),lambda:self.model(x))


def model_for(cfg,state):
    m=core.build(cfg)
    if 'code' not in state:m.generator=None;m.code=None
    m.load_state_dict(state);return m.eval().requires_grad_(False)


class Correspondence:
    def __init__(self):
        self.pixels=0;self.ce=0.;self.confusion=torch.zeros(4,4,dtype=torch.int64)
        self.conditional_sum=torch.zeros(4,dtype=torch.float64);self.conditional_count=torch.zeros(4,dtype=torch.int64)
        self.soft_intersection=torch.zeros(4,dtype=torch.float64);self.soft_prediction=torch.zeros(4,dtype=torch.float64);self.soft_target=torch.zeros(4,dtype=torch.float64)
    def add(self,logits,mask):
        pred=logits.softmax(1);label=mask.argmax(1);hard=logits.argmax(1);loss=-(mask*logits.log_softmax(1)).sum(1)
        self.ce+=float(loss.double().sum());self.pixels+=label.numel();self.confusion+=torch.bincount((label*4+hard).flatten().cpu(),minlength=16).reshape(4,4)
        for c in range(4):
            use=label==c;self.conditional_count[c]+=int(use.sum());self.conditional_sum[c]+=float(loss[use].double().sum())
        self.soft_intersection+=(pred.double()*mask.double()).sum((0,2,3)).cpu();self.soft_prediction+=pred.double().sum((0,2,3)).cpu();self.soft_target+=mask.double().sum((0,2,3)).cpu()
    def result(self):
        cm=self.confusion;den=cm.sum(0)+cm.sum(1);softden=self.soft_prediction+self.soft_target
        div=lambda a,b:None if float(b)==0 else float(a/b)
        return dict(images=32,pixels=self.pixels,full_soft_CE=self.ce/self.pixels,foreground_conditional_soft_CE=div(self.conditional_sum[1:].sum(),self.conditional_count[1:].sum()),class_conditional_soft_CE=[div(self.conditional_sum[c],self.conditional_count[c]) for c in range(4)],hard_Dice=[div(2*cm[c,c],den[c]) for c in range(4)],soft_Dice=[div(2*self.soft_intersection[c],softden[c]) for c in range(4)],generated_hard_pixel_counts=cm.sum(1).tolist(),native_hard_pixel_counts=cm.sum(0).tolist(),confusion_generated_rows_native_columns=cm.tolist(),generated_soft_mass_fraction=(self.soft_target/self.pixels).tolist(),native_soft_mass_fraction=(self.soft_prediction/self.pixels).tolist(),generated_foreground_wrong_class_fraction=div(cm[1:].sum()-sum(cm[c,c] for c in (1,2,3)),cm[1:].sum()),zero_denominator='NA represented by null; generated mask is not ground truth')


def self_check():
    m=Correspondence();logits=torch.tensor([[[[10.]],[[0.]],[[0.]],[[0.]]]]);mask=torch.zeros_like(logits);mask[:,0]=1;m.add(logits,mask);r=m.result()
    assert r['foreground_conditional_soft_CE'] is None and r['hard_Dice'][1:]==[None,None,None]
    assert r['confusion_generated_rows_native_columns'][0][0]==1 and sum(map(sum,r['confusion_generated_rows_native_columns']))==1
    n=Correspondence();mask.zero_();mask[:,1]=1;n.add(logits,mask);r=n.result();assert r['hard_Dice'][1]==0 and r['generated_foreground_wrong_class_fraction']==1


@torch.inference_mode()
def main(cfg):
    root=Path(cfg['diagnostic_root']);ledger=Ledger(cfg);self_check()
    assert not (root/'RESULTS.json').exists()
    torch.set_num_threads(4);torch.manual_seed(163);torch.cuda.manual_seed_all(163);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    result={'status':'RUNNING','class_order':['background','LV','MYO','RV'],'trajectory_models':{},'correspondence':{},'alpha':.999,'T':2000,'explicit_initial_coefficient':.999**2000,'limitations':['float32 EMA accumulation cannot be reversed exactly','early near-entry trajectory remains','generated mask and entry model are not ground truth','single seed development validation'],'frozen_student_endpoints_unchanged':True}
    try:
        for name in ['A','M','N']:
            ledger.guard(new=True);source=Path(cfg['arm_roots'][name]);assert read(source/'AUDIT.json')['status']=='PASS'
            initial=load(source/'jobs/A/initial.pt');final=load(source/'jobs/A/final.pt');assert final['step']==final['physical_updates']==2000
            assert all(torch.equal(v,initial['ema'][k]) for k,v in initial['model'].items())
            w=.999**2000;state={}
            for k,v in final['ema'].items():
                state[k]=((v.double()-w*initial['model'][k].double())/(1-w)).to(v.dtype) if eligible(k,v) else v
                assert torch.isfinite(state[k]).all()
            model=model_for(cfg,state);metrics={}
            for domain in ['Siemens','Philips']:
                dest=root/('private_trajectory_'+name+'_'+domain+'.jsonl');assert not dest.exists()
                score,newrows=evaluate(Counted(model,ledger),cfg,domain,'val',dest)
                student_rows=rows(source/'jobs/A'/('private_'+domain+'_val.jsonl'));ema_rows=rows(source/'jobs/A'/('private_ema_'+domain+'_val.jsonl'))
                metrics[domain]=dict(metrics=score,versus_student=paired(student_rows,newrows),versus_EMA=paired(ema_rows,newrows))
            result['trajectory_models'][name]=metrics;write(root/'PARTIAL.json',result);del model,initial,final,state;torch.cuda.empty_cache()
        for name in ['M','A']:
            ledger.guard(new=True);source=Path(cfg['arm_roots'][name]);initial=load(source/'jobs/A/initial.pt');final=load(source/'jobs/A/final.pt');state=final['model']
            generator=JointGenerator(1).cuda().eval().requires_grad_(False);generator.load_state_dict({k[len('generator.'):]:v for k,v in state.items() if k.startswith('generator.')});z=unit(state['code']).cuda()
            states={'entry':{k:initial['model'][k] if k.startswith('parent.') else v for k,v in state.items()}}
            if name=='M':states.update(student=state,EMA=final['ema'])
            models={key:model_for(cfg,v) for key,v in states.items()};stats={key:Correspondence() for key in states}
            for i in range(0,32,2):
                image,mask=ledger.call('generator',2,lambda:generator(z[i:i+2],(384,384)))
                for key,model in models.items():
                    logits=ledger.call('synthetic_native',2,lambda:model(image));stats[key].add(logits,mask)
            result['correspondence'][name]={key:v.result() for key,v in stats.items()};write(root/'PARTIAL.json',result)
            del generator,models,states,state,initial,final,z;torch.cuda.empty_cache()
        assert ledger.count==ledger.caps
        result.update(status='COMPLETE',forward_counts=ledger.count,optimizer_updates=0,functional_updates=0,peak_cuda_bytes=torch.cuda.max_memory_allocated(),resources=ledger.guard())
        write(root/'RESULTS.json',result);write(root/'STATUS.json',{'status':'COMPLETE'});write(root/'EXIT.json',{'exit_code':0})
    except BaseException as e:
        import traceback
        write(root/'FAILURE.json',dict(error=str(e),traceback=traceback.format_exc(),counts=ledger.count));write(root/'STATUS.json',{'status':'FAILED'});raise
