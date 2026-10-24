"""Private scalar train-L reference risk; source cache inaccessible in target."""
import json
from pathlib import Path
import torch
from ..main_style_ssl_v1.runtime import write

def load_cache(cfg):
    p=Path(cfg['cache']).resolve();base=(Path(cfg['root'])/'cache').resolve();assert p.parent==base
    expected='source.private.json' if cfg['stage']=='source' else 'target_'+('B' if cfg['arm']=='B' else 'F')+'.private.json'
    assert p.name==expected,'cache domain boundary';return json.loads(p.read_text())

def values(lp,y):
    p=lp.exp();rr=[];present=[]
    for c in (1,2,3):
        mask=(y==c);mass=mask.sum((1,2));ce=-(lp[:,c]*mask).sum((1,2))/mass.clamp_min(1);dc=(2*(p[:,c]*mask).sum((1,2))+1e-5)/(p[:,c].sum((1,2))+mass+1e-5);rr.append(ce+1-dc);present.append(mass>0)
    return torch.stack(rr,1),torch.stack(present,1)

def penalty(m,lp,y,keys,cache):
    if m.calls:m.calls.extra('risk_class_samples',len(y)*3)
    rr,seen=values(lp,y);G=len(cache['groups']);terms=[];positive=[];update=torch.zeros_like(m.dual)
    for j,key in enumerate(keys):
        item=cache['samples'][key];assert item['present']==seen[j].tolist()
        for c in range(3):
            gi=item['groups'][c]
            if gi<0:continue
            g=cache['groups'][gi];res=rr[j,c]-g['bound'];weight=1/(len(keys)*g['probability']);terms.append(m.dual[gi].detach()*res*weight/G);positive.append(res.relu()*weight/G);update[gi]+=res.detach()*weight
    zero=lp.sum()*0;reg=sum(terms,zero);positive=sum(positive,zero);m.last_dual_update=update
    return reg,positive

@torch.no_grad()
def calibrate(cfg):
    from .model import make
    from ..main_style_ssl_v1.data import Accessor
    stage=cfg['stage'];domain='Siemens' if stage=='source' else 'Philips';access=Accessor(cfg['data'],'mnms',domain,'train_labeled');assert len(access.patients)==(10 if stage=='source' else 13)
    m=make(cfg).eval().requires_grad_(False);items={};group_values={};native=0
    for i,meta in enumerate(access.rows):
        for z in range(0,access.depths[i],2):
            parts=[access.read(i,j) for j in range(z,min(z+2,access.depths[i]))];x=torch.stack([a['image'] for a in parts]).cuda();y=torch.stack([a['label'] for a in parts]).cuda();lp=m(x);rr,seen=values(lp,y);m.calls.extra('risk_class_samples',len(x)*3);native+=len(x)
            for j in range(len(x)):
                sample=dict(present=seen[j].tolist(),risks=rr[j].tolist(),group_keys=[])
                for c in range(3):
                    key=str(meta['patient_id'])+'|'+str(meta['phase'])+'|'+str(c+1);sample['group_keys'].append(key if sample['present'][c] else None)
                    if sample['present'][c]:group_values.setdefault(key,[]).append(sample['risks'][c])
                items[str(i)+':'+str(z+j)]=sample
    assert native==(230 if stage=='source' else 254);ids={k:i for i,k in enumerate(sorted(group_values))};groups=[]
    for key in sorted(group_values):
        patient,phase,_=key.split('|');indices=[i for i,r in enumerate(access.rows) if str(r['patient_id'])==patient and str(r['phase'])==phase];assert len(indices)==1
        i=indices[0];prob=(1/len(access.patients))*(1/len(access.by_patient[patient]))*len(group_values[key])/access.depths[i];v=group_values[key];risk=sum(v)/len(v);groups.append(dict(key=key,baseline=risk,bound=risk+.05*max(risk,.1),probability=prob,count=len(v)))
    for item in items.values():item['groups']=[ids[k] if k is not None else -1 for k in item.pop('group_keys')]
    if cfg['calibration']=='baseline':
        result=dict(status='COMPLETE',domain=domain,native_samples=native,groups=groups,samples=items);write(cfg['calibration_output'],result)
    else:
        baseline=load_cache(cfg);assert [g['key'] for g in baseline['groups']]==[g['key'] for g in groups]
        residual=[g['baseline']-b['bound'] for g,b in zip(groups,baseline['groups'])];result=dict(status='PASS' if max(residual)<=1e-6 else 'FAIL',native_samples=native,groups=len(groups),max_residual=max(residual),positive_violation_groups=sum(v>1e-6 for v in residual),tolerance=1e-6);write(cfg['calibration_output'],result)
