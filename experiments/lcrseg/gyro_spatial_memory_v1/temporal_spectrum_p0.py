"""Bounded, zero-update P0 using sealed endpoints and shared evaluators."""
import copy,json,time,subprocess
from pathlib import Path
import numpy as np
import torch
from torch import nn
from . import core
from .module import rows as spatial_rows,centroid
from . import n_contribution as prior
from .trajectory_correspondence_diagnostic import model_for
from .single_target_audit import rows,paired
from ..philips_transfer_v1.run import evaluate


def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def load(p):return torch.load(p,map_location='cpu',weights_only=False)

class Ledger:
    caps={'native_validation':9632,'encoder_training':5376}
    def __init__(self,cfg):self.cfg=cfg;self.root=Path(cfg['root']);self.count={k:0 for k in self.caps};self.pending=None;self.save()
    def guard(self,new=False):
        elapsed=time.time()-self.cfg['started_epoch'];used=int(subprocess.check_output(['du','-s','-B1',str(self.root)]).split()[0])+2_000_000
        if elapsed>=6300 or new and elapsed>=5400:raise TimeoutError('original P0 clock')
        if used>=400_000_000 or new and used>=100_000_000:raise OSError('P0 storage protection')
        return {'elapsed_seconds':elapsed,'allocated_bytes_including_metadata':used}
    def save(self):write(self.root/'COUNTER.json',dict(executed=self.count,caps=self.caps,pending=self.pending,optimizer=0,functional=0,reverse=0))
    def call(self,kind,n,fn):
        self.guard();assert self.pending is None and self.count[kind]+n<=self.caps[kind]
        self.pending={'kind':kind,'n':n};self.save()
        with (self.root/'events.jsonl').open('a') as f:f.write(json.dumps({'action':'reserved',**self.pending})+'\n')
        v=fn();assert torch.isfinite(v).all();self.count[kind]+=n
        with (self.root/'events.jsonl').open('a') as f:f.write(json.dumps({'action':'executed',**self.pending})+'\n')
        self.pending=None;self.save();return v

class Counted(nn.Module):
    def __init__(self,m,b):super().__init__();self.m=m;self.b=b
    def forward(self,x):return self.b.call('native_validation',len(x),lambda:self.m(x))

def patient_map(rr):
    d={}
    for r in rr:d.setdefault(r['patient'],[]).append(float(np.mean(r['dice'])))
    assert all(len(v)==2 for v in d.values());return {k:float(np.mean(v)) for k,v in d.items()}

def damage(a,b):
    aa,bb=patient_map(a),patient_map(b);assert aa.keys()==bb.keys();v=np.array([(bb[k]-aa[k])*100 for k in sorted(aa)])
    return dict(mean_pp=float(v.mean()),median_pp=float(np.median(v)),p10_pp=float(np.quantile(v,.1)),increased=int((v>0).sum()),decreased=int((v<0).sum()),decline_over_pp={str(t):int((v < -t).sum()) for t in [.5,1,2]},leave_one_mean_range_pp=[float((v.sum()-v).min()/(len(v)-1)),float((v.sum()-v).max()/(len(v)-1))],paired=paired(a,b))

def rank(v):
    out=np.empty(len(v),float);order=np.argsort(v);start=0
    while start<len(v):
        end=start+1
        while end<len(v) and v[order[end]]==v[order[start]]:end+=1
        out[order[start:end]]=(start+end-1)/2;start=end
    return out

def correlation(a,b):
    def c(x,y):return None if np.std(x)==0 or np.std(y)==0 else float(np.corrcoef(x,y)[0,1])
    return dict(Pearson=c(a,b),Spearman=c(rank(a),rank(b)),both_improve=int(((a>0)&(b>0)).sum()),both_decline=int(((a<0)&(b<0)).sum()),opposite_sign=int((a*b<0).sum()))

def patient_analysis(cfg):
    old=Path(cfg['spectrum_root']);base=Path(cfg['comparison_root']);out={};cross={}
    original=rows(Path(cfg['source_checkpoint']).parent/'private_val.jsonl')
    assert abs(np.mean([np.mean(r['dice']) for r in original])-.8036740663770262)<1e-8
    for seed in ['20261021','20261022']:
        out[seed]={}
        for domain in ['Siemens','Philips']:
            initial=rows(cfg['source_val_rows']) if domain=='Siemens' else rows(base/'preparation/private_Philips_val.jsonl')
            records={}
            for arm in ['S','G00']:
                for mode,prefix in [('student','private_'),('EMA','private_ema_')]:
                    records[arm+'_'+mode]=rows(old/'jobs'/(seed+'_'+arm)/(prefix+domain+'_val.jsonl'))
            comparisons={}
            for mode in ['student','EMA']:
                comparisons['G00_minus_S_'+mode]=damage(records['S_'+mode],records['G00_'+mode])
            for arm in ['S','G00']:comparisons[arm+'_EMA_minus_student']=damage(records[arm+'_student'],records[arm+'_EMA'])
            for key,v in records.items():
                comparisons[key+'_minus_entry']=damage(initial,v)
                if domain=='Siemens':comparisons[key+'_minus_original_source']=damage(original,v)
            out[seed][domain]=comparisons
            for mode in ['student','EMA']:
                aa,bb=patient_map(records['S_'+mode]),patient_map(records['G00_'+mode]);cross.setdefault(domain+'_'+mode,[]).append(np.array([(bb[k]-aa[k])*100 for k in sorted(aa)]))
    return {'comparisons':out,'cross_stream_difference_consistency':{k:correlation(*v) for k,v in cross.items()},'clinical_thresholds_validated':False}

def feature_stats(fields,pos):
    zz=[spatial_rows(v,pos).double() for v in fields.chunk(3)];c=centroid(*zz);sv=torch.linalg.svdvals(c);energy=float(c.square().sum());centered=c-c.mean(0);cov=centered.T@centered/(len(c)-1);cs=torch.linalg.svdvals(centered)
    return {'centroid_mean_squared_norm':energy/len(c),'nuclear_norm':float(sv.sum()),'singular_values':sv.cpu().tolist(),'r_star_uncentered':None if energy==0 else float(sv.sum().square())/energy,'view_squared_distances':{f'{i}_{j}':float((zz[i]-zz[j]).square().sum(1).mean()) for i,j in [(0,1),(0,2),(1,2)]},'centered_covariance_trace':float(cov.trace()),'centered_covariance_offdiag_squared':float((cov-torch.diag(cov.diag())).square().sum()),'centered_covariance_eigenvalues':torch.linalg.eigvalsh(cov).cpu().tolist(),'r_star_centered':None if float(centered.square().sum())==0 else float(cs.sum().square()/centered.square().sum())}

def self_check():
    fields=torch.ones(3,32,8,8)/32**.5;pos=torch.tensor([[0,1,2,3]])
    q=feature_stats(fields,pos);assert abs(q['r_star_uncentered']-1)<1e-6 and q['r_star_centered'] is None
    a=np.array([1.,2.,3.]);assert abs(correlation(a,a)['Spearman']-1)<1e-12

@torch.inference_mode()
def main(cfg):
    torch.set_num_threads(4);torch.manual_seed(163);torch.cuda.manual_seed_all(163);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=Path(cfg['root']);assert not (root/'RESULTS.json').exists();ledger=Ledger(cfg);old=Path(cfg['spectrum_root']);assert read(old/'AUDIT.json')['status']=='PASS';self_check()
    write(root/'WORKER.private.json',dict(pid=__import__('os').getpid(),birth=Path('/proc/self/stat').read_text().split()[21]))
    result={'status':'RUNNING','patient_analysis':patient_analysis(cfg),'exchange':{},'features':{},'old_endpoints_unchanged':True,'optimizer':0,'functional':0,'reverse':0}
    write(root/'PARTIAL.json',result)
    try:
        enc={'enc1','enc2','enc3','bottleneck'};dec={'decoder'}
        for seed in ['20261021','20261022']:
            for arm in ['S','G00']:
                final=load(old/'jobs'/(seed+'_'+arm)/'final.pt');s,e=final['model'],final['ema'];assert final['step']==final['physical_updates']==2000
                assert {k.split('.')[2] for k in s if k.startswith('parent.native.')}==enc|dec
                assert all(k.startswith('parent.native.') for k in s if k.startswith('parent.'))
                for swap in ['EMA_encoder_student_decoder','student_encoder_EMA_decoder']:
                    ledger.guard(new=True);state={k:(e[k] if k.startswith('parent.native.') and ((k.split('.')[2] in enc)==(swap=='EMA_encoder_student_decoder')) else v) for k,v in s.items()}
                    m=model_for(cfg,state);before={k:v.clone() for k,v in m.state_dict().items()};metrics={}
                    for domain in ['Siemens','Philips']:
                        dest=root/f'private_{seed}_{arm}_{swap}_{domain}.jsonl';assert not dest.exists();score,rr=evaluate(Counted(m,ledger),cfg,domain,'val',dest)
                        original=rows(old/'jobs'/(seed+'_'+arm)/('private_'+domain+'_val.jsonl'));ema=rows(old/'jobs'/(seed+'_'+arm)/('private_ema_'+domain+'_val.jsonl'))
                        assert {(r['patient'],r['phase']) for r in rr}=={(r['patient'],r['phase']) for r in original}
                        metrics[domain]={'metrics':score,'versus_student':paired(original,rr),'versus_EMA':paired(ema,rr)}
                    assert all(torch.equal(v,before[k]) for k,v in m.state_dict().items());result['exchange'][seed+'_'+arm+'_'+swap]=metrics;write(root/'PARTIAL.json',result);del m,before,state;torch.cuda.empty_cache()
        # Complete independent validation before Current installs its training-only HDF5 guard.
        feature_cfg=dict(cfg,target_seed=20261028,group='G00',output=str(root));data=prior.Current(feature_cfg)
        states=['entry']+[seed+'_'+arm for seed in ['20261021','20261022'] for arm in ['S','G00','Aalign']]
        for key in states:
            ledger.guard(new=True);cp=old/'jobs/20261021_S/initial.pt' if key=='entry' else old/'jobs'/key/'final.pt';state=load(cp)['model'];m=model_for(cfg,state);before={k:v.clone() for k,v in m.state_dict().items()};stats=[]
            for step in range(16):
                x,_,_=data.batch('L',step);fill,_,_=data.batch('F',step,14);xx=torch.cat([x,fill]);pos=prior.positions(20261028,step,16,x.device);images=torch.cat([xx,prior.strong(xx,20261028,step,0),prior.strong(xx,20261028,step,1)])
                fields=ledger.call('encoder_training',48,lambda:m.encode(images));stats.append(feature_stats(fields,pos))
            assert all(torch.equal(v,before[k]) for k,v in m.state_dict().items());result['features'][key]=stats;write(root/'PARTIAL.json',result);del m,before,state;torch.cuda.empty_cache()
        assert ledger.count==ledger.caps and ledger.pending is None
        result.update(status='COMPLETE',counts=ledger.count,resources=ledger.guard(),peak_cuda_bytes=torch.cuda.max_memory_allocated(),read_only_exact=True,decision='engineering_complete; analyze P0 before freezing four-group execution; no validation-dependent tuning')
        write(root/'RESULTS.json',result);write(root/'STATUS.json',{'status':'COMPLETE','audit_pending':True});write(root/'EXIT.json',{'exit_code':0})
    except BaseException as exc:
        import traceback
        write(root/'FAILURE.json',{'error':str(exc),'traceback':traceback.format_exc(),'counter':read(root/'COUNTER.json')});write(root/'STATUS.json',{'status':'FAILED'});raise
