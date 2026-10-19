"""Finite toy evidence and labeled discrete-model no-update audit."""
import json,os,itertools,traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset,collate_labeled
from lcrseg.engine.trainer import seed_everything
from care_seg_study_v1.run import write,append
from sc_bayeseg.run import same
from sc_bayeseg.split_v1.model import semantic
from sc_bayeseg.window_v1.common import storage
from sc_bayeseg.window_v1.fullview import flip
from sc_bayeseg.window_v1.discrete_labeled import DiscreteLabeledModel


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);costs=dict(reverse=0,bound=0,optimizer=0,parameter_applications=0,local_updates=0,toy_states=16)
    def grad(label,*a,**kw):
        costs['reverse']+=1;assert costs['reverse']<=7;append(root/'reverse_calls.jsonl',dict(index=costs['reverse'],label=label));return torch.autograd.grad(*a,**kw)
    def forbid(*a,**kw):raise AssertionError('optimizer forbidden')
    torch.optim.Optimizer.__init__=forbid;torch.optim.Adam.step=forbid
    try:
        storage(s);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        zz=torch.tensor(list(itertools.product(range(4),repeat=2)));j=torch.arange(2)[None]
        pl=torch.tensor([[.2,-.4,.6,.1],[-.2,.3,.1,-.5]],dtype=torch.float64).log_softmax(1)
        lpz=pl[j,zz].sum(1);assert abs(float(lpz.exp().sum())-1)<1e-12
        features=torch.tensor([-.6,.2,.7,1.1],dtype=torch.float64)
        mean=features[zz[:,0]]-.8*features[zz[:,1]]+.2*(zz[:,0]==zz[:,1])
        lp=lpz+torch.distributions.Normal(mean,.7).log_prob(torch.tensor(.4,dtype=torch.float64))
        raw=torch.tensor([[.4,-.1,.2,-.3],[.1,.2,-.2,.4]],dtype=torch.float64,requires_grad=True)
        lq=raw.log_softmax(1)[j,zz].sum(1);q=lq.exp();elbo=(q*(lp-lq)).sum();loge=torch.logsumexp(lp,0)
        kl=(q*(lq-(lp-loge))).sum();assert float(elbo)<=float(loge)+1e-12 and abs(float(loge-elbo-kl))<1e-12
        direct=grad('toy_enumerated_ELBO',elbo,raw,retain_graph=True)[0]
        score=grad('toy_score_function',(q.detach()*(lp-lq).detach()*lq).sum(),raw)[0]
        assert torch.allclose(direct,score,atol=1e-12,rtol=0)
        posterior=(lp-loge).exp();tight=(posterior*(lp-(lp-loge))).sum();assert abs(float(tight-loge))<1e-12
        toy=dict(logevidence=float(loge),ELBO=float(elbo.detach()),KL=float(kl.detach()),gap_identity_error=abs(float(loge-elbo-kl)),score_gradient_max_error=float((direct-score).abs().max()),posterior_ELBO_tight_error=abs(float(tight-loge)),prior_normalized=True)
        source=torch.load(s['readout_states']['G'],map_location='cpu',weights_only=False);model=DiscreteLabeledModel().cuda();model.load_state_dict(source['model'])
        sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text());ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        lb=collate_labeled([ds[i] for i in sel['train_indices']]).to('cuda');im=F.interpolate(lb.image,(128,128),mode='bilinear',align_corners=False);yy=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
        image=torch.cat([flip(im,k) for k in range(4)]);y=torch.cat([flip(yy,k) for k in range(4)]);named=list(model.named_parameters());params=[p for n,p in named]
        rows=[]
        for draw in range(8):
            seed_everything(80163+draw);model.draw()
            try:
                with torch.set_grad_enabled(draw==0):
                    b,t,diag=model.bound(image,y);costs['bound']+=1;append(root/'bound_calls.jsonl',dict(index=costs['bound'],draw=draw))
                    loss=-b.mean()/128**2+model.kl()/(2*128**2)
                    assert set(t)=={'code','appearance','image','shape','categories'} and torch.isfinite(b).all() and diag['categorical_normalization_error']<1e-6
                    row=dict(joint=float(loss.detach()),terms={k:float(-v.mean().detach()/128**2) for k,v in t.items()},logq=float(diag['logq'].mean().detach()/128**2),kl=float(model.kl().detach()/(2*128**2)),normalization_error=diag['categorical_normalization_error'])
                    assert torch.equal(b,sum(t.values())-diag['logq'])
                    if draw==0:
                        gs={}
                        for index,(key,value) in enumerate([('full',loss),('categories',-t['categories'].mean()/128**2),('shape',-t['shape'].mean()/128**2),('image',-t['image'].mean()/128**2)]):
                            g=grad('real_'+key,value,params,allow_unused=True,retain_graph=index<3)
                            assert all(v is None or torch.isfinite(v).all() for v in g)
                            assert all(v is None for (n,p),v in zip(named,g) if semantic(n))
                            gs[key]={scope:sum(float(v.double().square().sum()) for (n,p),v in zip(named,g) if v is not None and (scope=='all' or n.startswith(('anatomy.','structure.'))))**.5 for scope in ['all','generator']}
                        recognition=F.cross_entropy(model.recognition_logits(image),y)
                        rg=grad('recognition_CE_separate',recognition,params,allow_unused=True)
                        assert any(v is not None and float(v.abs().sum())>0 for (n,p),v in zip(named,rg) if semantic(n))
                        assert all(v is None for (n,p),v in zip(named,rg) if not semantic(n))
                        row.update(gradient_norms=gs,L_ELBO_semantic_grad_all_None=True,separate_recognition_CE=float(recognition),recognition_semantic_gradient_norm=sum(float(v.double().square().sum()) for v in rg if v is not None)**.5)
                rows.append(row)
            finally:model.clear_draw()
        assert costs==dict(reverse=7,bound=8,optimizer=0,parameter_applications=0,local_updates=0,toy_states=16)
        assert same(source['model'],model.state_dict()) and all(p.grad is None for p in params)
        assert len((Path(s['view_guard'])/'physical_updates.jsonl').read_text().splitlines())==104
        result=dict(status='COMPLETE',costs=costs,toy=toy,draws=rows,model_unchanged=True,grad_buffers_none=True,heldout_or_realU_accessed=False,source_physical_ledger_unchanged=True,space=storage(s))
        write(root/'RESULTS.json',result);write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
