"""No-update raw-coordinate class-risk and semantic-objective derivatives."""
import json
import os
import time
import traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset, collate_labeled
from sc_bayeseg.run import same
from sc_bayeseg.window_v1.r_start import cpu, write, save, append, draw_g, local_sample, local_kl
from sc_bayeseg.window_v1.same_family_oracle import posterior, ARMS
from sc_bayeseg.window_v1.spatial_residual import SpatialResidualModel
from sc_bayeseg.window_v1.fullview import flip
from sc_bayeseg.window_v1.common import storage

TRAIN=(1020163,1030163,1040163,1050163)
EVAL=(1120163,1130163,1140163,1150163)
BLOCKS=('c_mean','c_rawsd','R_mean','R_rawsd')


def flat(v):return torch.cat([x.detach().flatten().double().cpu() for x in v])

def summarize(gd,gk,ga,gq,ge):
    vectors={'D':gd,'K':gk,'A':ga,'Q':gq}
    out=dict(norm={k:float(v.norm()) for k,v in vectors.items()},eval_Q_norm=float(ge.norm()))
    out['cos_D_K']=float(torch.dot(gd,gk)/(gd.norm()*gk.norm()).clamp_min(1e-30))
    for label,risk in [('same_flow',gq),('independent_flow',ge)]:
        out[label]={k:float(-torch.dot(risk,v)/v.norm().clamp_min(1e-30)) for k,v in vectors.items()}
    out['independent_unnormalized']={k:float(-torch.dot(ge,v)) for k,v in vectors.items()}
    return out


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);started=time.time()
    costs=dict(reverse=0,optimizer=0,parameter_applications=0,generator_forward=0,generator_image_forward=0,theta=0)
    try:
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        write(root/'CONTEXT.json',s);storage(s,reserve=2_000_000_000)
        old=Path(s['oracle_source']);assert json.loads((old/'AUDIT.json').read_text())['status']=='PASS'
        files=[Path(s['checkpoint'])]+[old/a/'S1000.pt' for a in ARMS]
        metadata={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in files}
        payload=torch.load(files[0],map_location='cpu',weights_only=False);assert payload['cursor']==8675
        model=SpatialResidualModel().cuda().requires_grad_(False);model.load_state_dict(payload['model']);model.eval()
        source=cpu(model.state_dict())
        sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text())
        ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        lb=collate_labeled([ds[i] for i in sel['train_indices']])
        y=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
        y=torch.cat([flip(y,k) for k in range(4)]).cuda();positive=y[6]==3
        assert positive.any() and (~positive).any()
        rows=[];saved={};kl_errors=[]
        def grad(loss,raw,arm,seed,name,retain=False):
            value=torch.autograd.grad(loss,raw,retain_graph=retain)
            costs['reverse']+=1
            assert all(torch.isfinite(v).all() for v in value)
            append(root/'reverse.jsonl',dict(index=costs['reverse'],arm=arm,seed=seed,term=name))
            return flat(value)
        def losses(raw,seed):
            q=posterior(raw);tg=torch.Generator(device='cuda').manual_seed(seed)
            lg=torch.Generator(device='cuda').manual_seed(seed+1000000);data=[];risk=[]
            for _ in range(2):
                draw_g(model,tg);costs['theta']+=1
                for _ in range(2):
                    c,r=local_sample(q,lg);logits=model.spatial_logits(c,r)
                    data.append(F.cross_entropy(logits,y,reduction='none').mean())
                    z=logits[6];den=z.logsumexp(0)
                    pos=z[3]-den;neg=z[[0,1,2]].logsumexp(0)-den
                    risk.append(-.5*(pos[positive].mean()+neg[~positive].mean()))
                    costs['generator_forward']+=1;costs['generator_image_forward']+=8
            return torch.stack(data).mean(),torch.stack(risk).mean()
        for arm in ARMS:
            state=torch.load(old/arm/'S1000.pt',map_location='cpu',weights_only=False)
            original=state['raw'];raw=tuple(v.cuda().detach().requires_grad_(True) for v in original)
            q=posterior(raw);kc,kr=local_kl(q);kl=(kc+kr).mean()/16384
            gk=grad(kl,raw,arm,None,'K')
            analytic=[]
            for i,v in enumerate(raw):
                if i%2==0:analytic.append(v/(8*16384))
                else:
                    ls=-4+3*torch.tanh(v/3)
                    analytic.append(((2*ls).exp()-1)*(1-torch.tanh(v/3).square())/(8*16384))
            err=float((gk-flat(analytic)).abs().max());assert err<1e-9;kl_errors.append(err)
            sizes=[v.numel() for v in raw];saved[arm]=dict(K=gk,raw_shapes=[list(v.shape) for v in raw],flows=[])
            for seed,evseed in zip(TRAIN,EVAL):
                assert time.time()-s['original_started_unix']<85500
                data,risk=losses(raw,seed)
                gd=grad(data,raw,arm,seed,'D',True)
                gq=grad(risk,raw,arm,seed,'Q',True)
                kc,kr=local_kl(posterior(raw));a=data+(kc+kr).mean()/16384
                ga=grad(a,raw,arm,seed,'A')
                assert torch.allclose(ga,gd+gk,atol=2e-9,rtol=2e-5)
                _,evrisk=losses(raw,evseed);ge=grad(evrisk,raw,arm,evseed,'Q_eval')
                stats=summarize(gd,gk,ga,gq,ge)
                blocks={};offset=0
                for name,n in zip(BLOCKS,sizes):
                    blocks[name]=summarize(*(v[offset:offset+n] for v in (gd,gk,ga,gq,ge)));offset+=n
                row=dict(arm=arm,seed=seed,eval_seed=evseed,data=float(data.detach()),risk=float(risk.detach()),
                         eval_risk=float(evrisk.detach()),KL=float(kl.detach()),summary=stats,blocks=blocks,
                         gradient_sum_max_error=float((ga-gd-gk).abs().max()))
                rows.append(row);saved[arm]['flows'].append(dict(D=gd,Q=gq,A=ga,Q_eval=ge))
                write(root/'STATUS.json',dict(status='RUNNING',arm=arm,seed=seed,costs=costs))
                model.clear_draw()
            assert same(cpu(raw),original)
        assert costs==dict(reverse=34,optimizer=0,parameter_applications=0,generator_forward=64,generator_image_forward=512,theta=32)
        assert same(source,model.state_dict()) and all(v.grad is None for v in model.parameters())
        assert all((p.stat().st_size,p.stat().st_mtime_ns)==st for p,st in metadata.items())
        save(root/'GRADIENTS.pt',saved)
        write(root/'RESULTS.json',dict(status='COMPLETE',costs=costs,rows=rows,closed_form_KL_max_errors=kl_errors,
          risk=dict(view=3,image=0,category=3,positive_pixels=int(positive.sum()),negative_pixels=int((~positive).sum()),definition='half conditional binary CE positive plus half negative'),
          source_frozen=True,raw_unchanged=True,phase_elapsed_seconds=time.time()-started,
          original_elapsed_seconds=time.time()-s['original_started_unix'],space=storage(s)))
        write(root/'STATUS.json',dict(status='COMPLETE',costs=costs));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()))
        write(root/'EXIT.json',dict(exit_code=1));raise

if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
