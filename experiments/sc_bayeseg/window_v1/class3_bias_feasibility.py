"""Exhaustive common-class3 bias feasibility on sealed oracle probabilities."""
import json
import os
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset,collate_labeled
from sc_bayeseg.window_v1.r_start import write
from sc_bayeseg.window_v1.fullview import flip
from sc_bayeseg.window_v1.common import storage


def intervals(mask,low,high):
    out=[];i=0
    while i<len(mask):
        if not mask[i]:i+=1;continue
        j=i
        while j+1<len(mask) and mask[j+1]:j+=1
        out.append((float(low[i]),float(high[j])));i=j+1
    return out


def intersect(a,b):
    out=[];i=j=0
    while i<len(a) and j<len(b):
        lo=max(a[i][0],b[j][0]);hi=min(a[i][1],b[j][1])
        if lo<hi:out.append((lo,hi))
        if a[i][1]<b[j][1]:i+=1
        else:j+=1
    return out


def encode(intervals_):
    return [[None if not np.isfinite(lo) else lo,None if not np.isfinite(hi) else hi] for lo,hi in intervals_]


def witness(lo,hi):return float(hi if np.isfinite(hi) else lo+1 if np.isfinite(lo) else 0.)


def scores(p):
    p=np.asarray(p,dtype=np.float64)
    assert np.isfinite(p).all() and (p>=0).all() and np.max(abs(p.sum(1)-1))<2e-6
    base=p[:,:3].argmax(1);other=p[:,:3].max(1)
    with np.errstate(divide='ignore',invalid='raise'):margin=np.log(p[:,3])-np.log(other)
    assert not np.isnan(margin).any()
    return margin,base


def direct(m,base,y,b):
    pred=np.where(m+b>0,3,base);dice=[]
    for v in range(len(y)):
        for c in (1,2,3):
            target=y[v]==c;chosen=pred[v]==c
            dice.append(2*np.count_nonzero(target&chosen)/(target.sum()+chosen.sum()))
    return float(min(dice)),int(sum(v>=.8 for v in dice))


def scan(p,y):
    m,base=scores(p);view=np.broadcast_to(np.arange(len(y))[:,None,None],y.shape)
    initial=np.where(np.isposinf(m),3,base)
    finite=np.flatnonzero(np.isfinite(m.flatten()))
    order=finite[np.argsort(-m.flatten()[finite],kind='stable')]
    sm=m.flatten()[order];n=len(order)
    starts=np.r_[0,np.flatnonzero(sm[1:]!=sm[:-1])+1] if n else np.array([],dtype=int)
    ends=np.r_[starts[1:]-1,n-1] if n else np.array([],dtype=int)
    at=np.r_[0,ends+1].astype(int)
    low=np.r_[-np.inf,-sm[starts]];high=np.r_[-sm[starts],np.inf]
    minimum=np.ones(n+1);passed=np.zeros(n+1,dtype=np.int16)
    sv=view.flatten()[order];sy=y.flatten()[order];sb=base.flatten()[order]
    for v in range(len(y)):
        for c in (1,2,3):
            gt=int((y[v]==c).sum());assert gt>0
            chosen=initial[v]==c;pred0=int(chosen.sum());tp0=int((chosen&(y[v]==c)).sum())
            if c==3:dp=(sv==v).astype(np.int32);dt=((sv==v)&(sy==3)).astype(np.int32)
            else:
                dp=-((sv==v)&(sb==c)).astype(np.int32)
                dt=-((sv==v)&(sb==c)&(sy==c)).astype(np.int32)
            pred=pred0+np.r_[0,np.cumsum(dp,dtype=np.int32)]
            tp=tp0+np.r_[0,np.cumsum(dt,dtype=np.int32)]
            d=2*tp/(gt+pred);minimum=np.minimum(minimum,d);passed+=(d>=.8)
    minimum=minimum[at];passed=passed[at]
    feasible=intervals(passed==len(y)*3,low,high);best=int(minimum.argmax())
    checks=[]
    for i in np.unique(np.linspace(0,len(low)-1,min(8,len(low)),dtype=int)):
        b=witness(low[i],high[i]);value,count=direct(m,base,y,b)
        assert abs(value-minimum[i])<1e-12 and count==int(passed[i]);checks.append(int(i))
    original=direct(m,base,y,0.)
    summary=dict(candidate_intervals=len(low),finite_pixel_events=n,distinct_events=len(starts),
      feasible_intervals=encode(feasible),any_bias_qualified=bool(feasible),
      best_minimum_foreground_Dice=float(minimum[best]),best_interval=encode([(low[best],high[best])])[0],
      original_bias0_minimum_Dice=original[0],original_bias0_passed24=original[1],
      direct_curve_checks=checks)
    return summary,feasible,dict(low=low,high=high,minimum=minimum,passed=passed)


def synthetic():
    y=(np.arange(32)%4).reshape(2,4,4);rng=np.random.default_rng(163)
    for p in (rng.uniform(.01,1,(2,4,4,4)),np.eye(4)[y].transpose(0,3,1,2)*.9+.025,np.eye(4)[y].transpose(0,3,1,2),np.full((2,4,4,4),.25)):
        p=p/p.sum(1,keepdims=True);s,_,curve=scan(p,y);m,b=scores(p)
        for i,(lo,hi) in enumerate(zip(curve['low'],curve['high'])):
            d,n=direct(m,b,y,witness(lo,hi))
            assert abs(d-curve['minimum'][i])<1e-12 and n==curve['passed'][i]
    assert intersect([(-np.inf,0),(1,3)],[(-1,2)])==[(-1,0),(1,2)]
    return dict(status='PASS',all_synthetic_events_bruteforce=True,intersection_checked=True)


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);started=time.time()
    try:
        torch.set_num_threads(4);write(root/'CONTEXT.json',s);storage(s,reserve=2_000_000_000)
        test=synthetic();write(root/'SYNTHETIC.json',test)
        risk=Path(s['risk_source']);control=Path(s['allocation_source'])
        assert json.loads((risk/'AUDIT.json').read_text())['status']=='PASS'
        sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text())
        ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        lb=collate_labeled([ds[i] for i in sel['train_indices']])
        y=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long();y=torch.cat([flip(y,k) for k in range(4)]).numpy()
        records={};rows=[];metadata={}
        for arm in ('ZERO','R1_WARM'):
            for objective,folder in (('PIXEL_CE',control/(arm+'_T2L2')),('CLASS_BALANCED',risk/(arm+'_CLASS_BALANCED'))):
                file=folder/'PRED_0500.pt';metadata[file]=(file.stat().st_size,file.stat().st_mtime_ns)
                p=torch.load(file,map_location='cpu',weights_only=False)['generator_flows']
                flows=[torch.stack(p).mean(0)]+list(p);common=[(-np.inf,np.inf)];name=arm+'_'+objective
                for i,pred in enumerate(flows):
                    assert time.time()-s['original_started_unix']<85500
                    mode='MIXTURE' if i==0 else 'FLOW'+str(i)
                    summary,feasible,curve=scan(pred.numpy(),y)
                    np.savez_compressed(root/(name+'_'+mode+'.npz'),**curve)
                    common=intersect(common,feasible)
                    rows.append(dict(candidate=name,mode=mode,**summary))
                    write(root/'STATUS.json',dict(status='RUNNING',candidate=name,mode=mode,scans=len(rows)))
                records[name]=dict(robust_common_bias_exists=bool(common),common_feasible_intervals=encode(common))
        assert len(rows)==20 and all((p.stat().st_size,p.stat().st_mtime_ns)==st for p,st in metadata.items())
        write(root/'RESULTS.json',dict(status='COMPLETE',rows=rows,candidates=records,synthetic=test,
          costs=dict(scans=20,optimizer=0,reverse=0,generator_forward=0,parameter_applications=0),source_unchanged=True,
          phase_elapsed_seconds=time.time()-started,original_elapsed_seconds=time.time()-s['original_started_unix'],space=storage(s),
          interpretation='Finite saved scores and one shared class3 log-score bias only; not G-family unreachability or deployment evidence'))
        write(root/'STATUS.json',dict(status='COMPLETE',scans=20));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        write(root/'FAILURE.json',dict(traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise

if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
