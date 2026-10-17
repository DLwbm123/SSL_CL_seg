"""Within-image, fixed-grid graph diagnosis; labels are read only for scoring."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .data import subset
from ..main_style_ssl_v1.data import Accessor
from ..main_style_ssl_v1.runtime import write, append
from ..siemens_ssl_reproduction_v1.methods import Segmenter
from ..di_dmpa_jascl.data import stable_seed


def graph(features):
    f=F.normalize(features.float(),dim=1)
    similarity=f@f.T
    similarity.fill_diagonal_(-float('inf'))
    # Stable ordering resolves identical similarities by spatial-node index.
    indices=torch.argsort(similarity,dim=1,descending=True,stable=True)[:,:8]
    weights=(similarity.gather(1,indices)/.1).softmax(1)
    return torch.zeros_like(similarity).scatter(1,indices,weights)


def propagate(g,p):
    seed=p.max(1).values>=.95
    seed_values=F.one_hot(p.argmax(1),4).to(p)
    initial=torch.where(seed[:,None],seed_values,p)
    q=initial.clone()
    for _ in range(20):
        q=.8*(g@q)+.2*initial
        q=torch.where(seed[:,None],seed_values,q)
    return q


def counts(pred,base,truth,mask):
    return dict(n=int(mask.sum()),correct=int(((pred==truth)&mask).sum()),
                corrected=int(((pred==truth)&(base!=truth)&mask).sum()),
                broken=int(((pred!=truth)&(base==truth)&mask).sum()))


@torch.inference_mode()
def run(cfg,current,out):
    current.eval().requires_grad_(False); current.role='eval'
    source=Segmenter(cfg['reference']).cuda()
    state=torch.load(cfg['source_checkpoint'],map_location='cpu',weights_only=False)
    source.load_state_dict(state['ema']); source.eval().requires_grad_(False)
    access=Accessor(cfg['data'],'mnms','Philips','train_labeled')
    subset(access,set(json.loads(Path(cfg['split']).read_text())['holdout']))
    rows=[]
    for index,meta in enumerate(access.rows):
        item=access.read(index); images=item['image']; truths=item['label']
        for z in range(len(images)):
            x=images[z:z+1,None].cuda(); truth=truths[z].cuda()
            h=current.parts(x)[2]; old=source.parent.features(x)
            logits=current.parent.native_readout(h,x.shape[-2:],'eval')
            yy=torch.linspace(0,x.shape[-2]-1,24,device=x.device).round().long()
            xx=torch.linspace(0,x.shape[-1]-1,24,device=x.device).round().long()
            def nodes(value):
                return value[0,:,yy[:,None],xx[None,:]].flatten(1).T
            def aligned(value):
                return F.interpolate(value[...,1:-1,1:-1],size=x.shape[-2:],mode='bilinear',align_corners=True)
            p=nodes(logits.softmax(1)); g=graph(nodes(aligned(h))); historical=graph(nodes(aligned(old)))
            generator=torch.Generator(device='cpu').manual_seed(stable_seed(163,'graph_permutation',meta['case_id'],z))
            permutation=torch.randperm(len(p),generator=generator).cuda()
            shuffled=historical[permutation][:,permutation]
            predictions=[p,propagate(g,p),propagate(historical,p),propagate(shuffled,p)]
            labels=[v.argmax(1) for v in predictions]; conf=[v.max(1).values for v in predictions]
            y=truth[yy[:,None],xx[None,:]].flatten()
            boundary=torch.zeros_like(truth,dtype=torch.bool)
            boundary[1:] |= truth[1:]!=truth[:-1]; boundary[:-1] |= truth[:-1]!=truth[1:]
            boundary[:,1:] |= truth[:,1:]!=truth[:,:-1]; boundary[:,:-1] |= truth[:,:-1]!=truth[:,1:]
            boundary=boundary[yy[:,None],xx[None,:]].flatten()
            nonseed=conf[0]<.95
            # Same query support, and a separate common-confidence support for D1/D2/D3.
            common=nonseed & (conf[1]>=.95) & (conf[2]>=.95) & (conf[3]>=.95)
            for d in range(4):
                for region,mask in [('all',torch.ones_like(nonseed)),('nonseed',nonseed),
                                    ('common_accepted',common),('boundary',nonseed&boundary),
                                    ('interior',nonseed&~boundary)]:
                    for c in range(4):
                        m=mask&(y==c)
                        row=dict(patient=meta['patient_id'],phase=meta['phase'],slice=z,variant='D'+str(d),
                                 region=region,truth_class=c,**counts(labels[d],labels[0],y,m),
                                 accepted=int((m&(conf[d]>=.95)).sum()),
                                 predicted_class_counts=[int((mask&(labels[d]==k)).sum()) for k in range(4)])
                        rows.append(row)
                        append(out/'private_relation_nodes.jsonl',row)
    # Equal-patient aggregation; per-class denominators and absent classes stay explicit.
    patient_totals={}
    for r in rows:
        key=(r['patient'],r['variant'],r['region'],r['truth_class'])
        total=patient_totals.setdefault(key,dict(n=0,correct=0,corrected=0,broken=0,accepted=0))
        for k in total: total[k]+=r[k]
    prediction_counts={}
    for row in rows:
        if row['truth_class']==0:
            key=row['variant']+'/'+row['region']
            prediction_counts.setdefault(key,[0]*4)
            prediction_counts[key]=[a+b for a,b in zip(prediction_counts[key],row['predicted_class_counts'])]
    summaries=[]
    for d in range(4):
        for region in ('all','nonseed','common_accepted','boundary','interior'):
            for c in range(4):
                ts=[v for (patient,variant,r,k),v in patient_totals.items() if variant=='D'+str(d) and r==region and k==c]
                active=[v for v in ts if v['n']]
                summaries.append(dict(variant='D'+str(d),region=region,truth_class=c,patients_with_support=len(active),
                    **{k:sum(v[k] for v in ts) for k in ('n','correct','corrected','broken','accepted')},
                    patient_mean_accuracy=None if not active else float(np.mean([v['correct']/v['n'] for v in active])),
                    patient_mean_net_correction=None if not active else float(np.mean([(v['corrected']-v['broken'])/v['n'] for v in active])),
                    patient_mean_coverage=None if not active else float(np.mean([v['accepted']/v['n'] for v in active]))))
    write(out/'RELATION_DIAGNOSTIC.json',dict(status='COMPLETE',optimizer_updates=0,patients=len(access.patients),
        volumes=len(access.rows),grid=[24,24],knn=8,temperature=.1,iterations=20,alpha=.8,confidence=.95,
        source='C final student weak predictions and actual post-F5 pre-head features; old source EMA features',
        D0='current weak',D1='current graph',D2='old graph',D3='simultaneous row/column permutation of old graph',
        labels='reserved TRAIN patients, scoring only; no GT seeds or ROI selection',results=summaries,
        predicted_class_counts=prediction_counts,
        interpretation='Node diagnosis, not dense Dice; three held-out patients do not establish statistical or continual-learning efficacy.'))
