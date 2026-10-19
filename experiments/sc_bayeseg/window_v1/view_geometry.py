"""Read-only factor gradients and semantic descent certificates after calibration."""
import copy,json,os,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset,collate_labeled
from lcrseg.engine.trainer import seed_everything
from lcrseg.engine.checkpoint import capture_rng_state,restore_rng_state
from care_seg_study_v1.run import write,append
from sc_bayeseg.split_v1.model import Model,semantic
from sc_bayeseg.jointwrite_v1.run import Adapter
from sc_bayeseg.run import same
from sc_bayeseg.adam_audit_v1.run import candidate
from sc_bayeseg.geometry_v1.run import flat
from sc_bayeseg.geometry_v1.geometry import solve,selfcheck
from sc_bayeseg.window_v1.fullview import flip
from sc_bayeseg.window_v1.common import storage


def constraints(model,image,y):
    rng=capture_rng_state()
    try:
        h,_=model.proposal(image,sample=False)
        z=lambda h:torch.cat([torch.zeros_like(h['ell'][:,:1]),h['ell']],1)
        probs=[z(h).softmax(1)];seed_everything(50163);samples=[]
        for _ in range(4):
            h,_=model.proposal(image);samples.append(z(h).softmax(1))
        probs.append(torch.stack(samples).mean(0));values=[];names=[]
        for label,p in zip(['mean','stochastic'],probs):
            loss=-p.clamp_min(1e-30).log().gather(1,y[:,None])[:,0]
            for i in range(8):
                names.append(f'{label}_view{i//2}_image{i%2}_all');values.append(loss[i].mean())
                for k in (1,2,3):
                    assert (y[i]==k).any()
                    names.append(f'{label}_view{i//2}_image{i%2}_C{k}');values.append(loss[i][y[i]==k].mean())
        return names,values
    finally:restore_rng_state(rng)


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);costs=dict(reverse=0,pure_adam=0,optimizer=0,candidate_forwards=0,parameter_applications=0)
    def forbid(*a,**kw):raise AssertionError('optimizer forbidden')
    torch.optim.Optimizer.__init__=forbid;torch.optim.Adam.step=forbid
    def grad(label,*a,**kw):
        assert costs['reverse']<150;costs['reverse']+=1
        append(root/'reverse_calls.jsonl',dict(index=costs['reverse'],label=label));return torch.autograd.grad(*a,**kw)
    try:
        storage(s);write(root/'SOLVER_CHECK.json',selfcheck());torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        old=Path(s['view_joint']);assert json.loads((old/'EXIT.json').read_text())['exit_code']==0
        assert len((old/'physical_updates.jsonl').read_text().splitlines())==812
        history=[json.loads(v) for v in (old/'J/steps.jsonl').read_text().splitlines()]
        sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text())
        ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        lb=collate_labeled([ds[i] for i in sel['train_indices']]).to('cuda')
        im=F.interpolate(lb.image,(128,128),mode='bilinear',align_corners=False)
        yy=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
        image=torch.cat([flip(im,k) for k in range(4)]);y=torch.cat([flip(yy,k) for k in range(4)])
        model=Model().cuda();named=list(model.named_parameters());params=[p for n,p in named]
        ix=[i for i,(n,p) in enumerate(named) if semantic(n)];sp=[params[i] for i in ix]
        assert len(params)==100 and len(sp)==40
        method=Adapter(model,'joint');method.nl=2;method.begin_site('Siemens',None,7475)
        stash={};bound=model.bound
        def capture(*a,**kw):
            v=bound(*a,**kw);stash.update(bound=v[0],terms=v[1]);return v
        model.bound=capture
        # Adapter receives already resized tensors; metadata is unchanged and never public.
        from dataclasses import replace
        batch=replace(lb,image=image,label=y,valid_mask=torch.ones_like(image,dtype=torch.bool),
                      case_id=lb.case_id*4,patient_id=lb.patient_id*4,site=lb.site*4,slice_index=lb.slice_index*4)
        results={};private={}
        for step in (0,25):
            payload=torch.load(old/'J'/f'S{step:04}.pt',map_location='cpu',weights_only=False);original=copy.deepcopy(payload)
            model.load_state_dict(payload['model']);restore_rng_state(payload['rng'])
            names,values=constraints(model,image,y);assert len(values)==64
            C=np.stack([flat(grad(f'S{step}_{name}',v,sp,allow_unused=True,retain_graph=i<63),sp) for i,(name,v) in enumerate(zip(names,values))])
            baseline=[float(v.detach()) for v in values];objectives=[];direction_rows={};term_rows={};factor_vectors=[]
            for noise in ('original',50163,80163):
                if noise=='original':restore_rng_state(payload['rng'])
                else:seed_everything(noise)
                r=method.training_step(batch,None,payload['cursor'],payload['cursor'])
                full=grad(f'S{step}_joint_{noise}',r.total_loss,params,allow_unused=True,retain_graph=noise=='original')
                assert all(g is not None and torch.isfinite(g).all() for g in full)
                j=flat([full[i] for i in ix],sp);objectives.append(j)
                if noise=='original':
                    assert float(r.total_loss)==history[step]['loss']
                    nn=torch.linalg.vector_norm(torch.stack(torch._foreach_norm(full,2)),2)
                    clipped=torch._foreach_mul(full,(1/(nn+1e-6)).clamp(max=1.))
                    op=payload['trainer']['optimizer_state'];group=op['param_groups'][0]
                    with torch.no_grad():delta,_,_,_,info=candidate(params,clipped,[op['state'][i] for i in group['params']],group)
                    costs['pure_adam']+=1;append(root/'adam_candidates.jsonl',dict(index=costs['pure_adam'],state=step,applied=False,forward_evaluated=False))
                    ad=flat([delta[i] for i in ix],sp)
                    for name,d in [('negative_joint',-j),('adam',ad)]:
                        dots=C@d;direction_rows[name]=dict(norm=float(np.linalg.norm(d)),constraint_derivatives=dots.tolist(),worsening_constraints=int((dots>1e-10).sum()),improving_constraints=int((dots < -1e-10).sum()))
                    terms={k:-v.mean()/128**2 for k,v in stash['terms'].items()}
                    terms['logq']=(sum(stash['terms'].values())-stash['bound']).mean()/128**2
                    terms['kl']=model.kl()/(2*128**2)
                    for i,(key,value) in enumerate(terms.items()):
                        gg=grad(f'S{step}_{key}',value,params,allow_unused=True,retain_graph=i<len(terms)-1)
                        vector=flat(gg,params);factor_vectors.append(vector);sem=flat([gg[i] for i in ix],sp)
                        dots=C@(-sem)
                        groups={}
                        for prefix in {n.split('.')[0] for n,p in named}:
                            ids=[i for i,(n,p) in enumerate(named) if n.split('.')[0]==prefix]
                            groups[prefix]=float(np.linalg.norm(flat([gg[i] for i in ids],[params[i] for i in ids])))
                        term_rows[key]=dict(loss=float(value.detach()),semantic_norm=float(np.linalg.norm(sem)),group_norms=groups,
                                            negative_factor_constraint_derivatives=dots.tolist(),worsening_constraints=int((dots>1e-10).sum()))
                    total=flat(full,params);residual=float(np.linalg.norm(np.stack(factor_vectors).sum(0)-total)/(np.linalg.norm(total)+1e-30))
                    assert residual<1e-5
                    first=dict(loss=float(r.total_loss),original_J_next_loss_exact=True,gradient_norm=float(nn),gradient_sum_relative_residual=residual,adam=info)
                    private[f'S{step}_adam_semantic']=ad
            J=np.stack(objectives);geometry,dirs=solve(C,J)
            for row in direction_rows.values():
                d=-objectives[0] if row is direction_rows['negative_joint'] else private[f'S{step}_adam_semantic']
                row['joint_derivatives']=(J@d).tolist()
            results[str(step)]=dict(baseline=baseline,first=first,terms=term_rows,directions=direction_rows,geometry=geometry)
            private[f'S{step}_C']=C;private[f'S{step}_J']=J
            for n,d in dirs.items():private[f'S{step}_{n}']=d
            restore_rng_state(payload['rng'])
            assert same(payload,original) and same(model.state_dict(),payload['model']) and same(capture_rng_state(),payload['rng'])
            assert all(p.grad is None for p in params)
            write(root/'STATUS.json',dict(status='RUNNING',state=step,costs=costs))
        storage(s,reserve=sum(v.nbytes for v in private.values())+100_000_000)
        np.savez(root/'PRIVATE_GEOMETRY.npz',**private)
        assert costs==dict(reverse=150,pure_adam=2,optimizer=0,candidate_forwards=0,parameter_applications=0)
        assert len((old/'physical_updates.jsonl').read_text().splitlines())==812
        result=dict(status='COMPLETE',costs=costs,constraints=names,states=results,model_payload_rng_unchanged=True,
                    grad_buffers_none=True,original_physical_ledger_unchanged=True,space=storage(s))
        write(root/'RESULTS.json',result);write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
