"""Diagnose fresh-gradient versus gate objectives and coupled block motion."""
import json,os,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scipy.optimize import lsq_linear
from lcrseg.data import H5LabeledDataset,collate_labeled
from lcrseg.engine.checkpoint import restore_rng_state
from care_seg_study_v1.run import write,append
from sc_bayeseg.split_v1.model import Model,semantic
from sc_bayeseg.jointwrite_v1.run import Adapter
from sc_bayeseg.geometry_v1.run import losses,flat
from sc_bayeseg.geometry_v1.geometry import solve,unit,row_units
from sc_bayeseg.adam_audit_v1.run import candidate
from sc_bayeseg.guard_v1.run import frame,accepts
from sc_bayeseg.window_v1.common import storage,objective,joint_pass


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);costs=dict(reverse=0,adam=0,optimizer=0,candidates=0)
    try:
        storage(s);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        def forbid(*a,**kw):raise AssertionError('optimizer forbidden')
        torch.optim.Optimizer.__init__=forbid;torch.optim.Adam.step=forbid
        p=torch.load(s['checkpoint'],map_location='cpu',weights_only=False);assert p['cursor']==3125
        model=Model().cuda();model.load_state_dict(p['model']);named=list(model.named_parameters());params=[v for _,v in named];ix=[i for i,(n,v) in enumerate(named) if semantic(n)];sp=[params[i] for i in ix]
        ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None);sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text());lb=collate_labeled([ds[i] for i in sel['train_indices']]).to('cuda')
        image=F.interpolate(lb.image,(128,128),mode='bilinear',align_corners=False);y=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
        method=Adapter(model,'joint');method.nl=2;method.begin_site('Siemens',None,3600);f0=frame(model,image,y);j0=objective(model,image,y,s['replay'])
        def grad(*a,**kw):
            assert costs['reverse']<19;costs['reverse']+=1;append(root/'reverse_calls.jsonl',dict(index=costs['reverse']));return torch.autograd.grad(*a,**kw)
        vals=losses(model,image,y);cg=[]
        for i,v in enumerate(vals):cg.append(flat(grad(v,sp,allow_unused=True,retain_graph=i<15),sp))
        C=np.stack(cg);joint=[];semjoint=[];delta=None
        for noise in ['current',1,25]:
            restore_rng_state(p['rng'] if noise=='current' else torch.load(Path(s['replay'])/f'S{noise-1:02}.pt',map_location='cpu',weights_only=False)['rng'])
            r=method.training_step(lb,None,3125,3125);full=grad(r.total_loss,params,allow_unused=True);assert all(g is not None and torch.isfinite(g).all() for g in full)
            joint.append(flat(full,params));semjoint.append(flat([full[i] for i in ix],sp))
            if noise=='current':
                nn=torch.linalg.vector_norm(torch.stack(torch._foreach_norm(full,2)),2);gg=torch._foreach_mul(full,(1/(nn+1e-6)).clamp(max=1.));op=p['trainer']['optimizer_state'];group=op['param_groups'][0]
                with torch.no_grad():delta,_,_,_,_=candidate(params,gg,[op['state'][i] for i in group['params']],group)
                costs['adam']=1
        J=np.stack(joint);SJ=np.stack(semjoint)
        single,sd=solve(C,SJ[:1]);robust,rd=solve(C,SJ)
        radius=json.loads((Path(s['geometry'])/'RESULTS.json').read_text())['states']['0']['joint'][0]['adam_semantic_norm']/16
        H=np.zeros((16,J.shape[1]));pos=semp=0
        for i,(n,v) in enumerate(named):
            if semantic(n):H[:,pos:pos+v.numel()]=C[:,semp:semp+v.numel()];semp+=v.numel()
            pos+=v.numel()
        A,_=row_units(np.concatenate([H,J]));ad=flat(delta,params);adu,_=unit(ad)
        opt=lsq_linear(A.T,adu,bounds=(0,np.inf),method='bvls',tol=1e-12,max_iter=10000)
        pd=adu-A.T@opt.x;pu,pn=unit(pd)
        projection=dict(norm=pn,multipliers=opt.x.tolist(),max_normalized_derivative=float((A@pu).max()),primal_violation=max(0.,float((A@pd).max())),solver_success=bool(opt.success),joint_derivatives=(J@pu).tolist())
        dirs={'single_semantic':sd['strict'],'robust_semantic':rd['strict']};rows=[];passing={}
        for name,d in dirs.items():
            for beta in [1.,.25,0.]:
                key=name+'_beta'+str(beta);passing[key]=[]
                for alpha in [1.,.25,.0625,.015625,.00390625,.0009765625,.000244140625,0.]:
                    storage(s);model.load_state_dict(p['model']);pos=0
                    with torch.no_grad():
                        for i,(n,v) in enumerate(named):
                            base=p['model'][n].to(v.device)
                            if semantic(n):
                                dd=torch.as_tensor(d[pos:pos+v.numel()].reshape(v.shape),dtype=v.dtype,device=v.device);v.copy_(base+radius*alpha*dd);pos+=v.numel()
                            else:v.copy_(base+beta*delta[i])
                    f=f0 if alpha==0 else frame(model,image,y);jj=j0 if alpha==0 and beta==0 else objective(model,image,y,s['replay'])
                    ok=accepts(f['vector'],f0['vector']) and joint_pass(jj,j0);costs['candidates']+=1
                    row=dict(index=costs['candidates'],direction=name,beta=beta,alpha=alpha,semantic_pass=accepts(f['vector'],f0['vector']),joint_pass=joint_pass(jj,j0),pass_gate=ok,joint=jj,vector=f['vector']);append(root/'candidates.jsonl',row);rows.append(row)
                    if ok:passing[key].append(alpha)
        passing['full_projection']=[]
        for alpha in [1.,.25,.0625,.015625,.00390625,.0009765625,.000244140625,0.]:
            storage(s);model.load_state_dict(p['model']);pos=0
            with torch.no_grad():
                for n,v in named:
                    dd=torch.as_tensor(pu[pos:pos+v.numel()].reshape(v.shape),dtype=v.dtype,device=v.device);v.add_(dd,alpha=radius*alpha);pos+=v.numel()
            f=f0 if alpha==0 else frame(model,image,y);jj=j0 if alpha==0 else objective(model,image,y,s['replay']);ok=accepts(f['vector'],f0['vector']) and joint_pass(jj,j0);costs['candidates']+=1
            row=dict(index=costs['candidates'],direction='full_projection',beta=None,alpha=alpha,semantic_pass=accepts(f['vector'],f0['vector']),joint_pass=joint_pass(jj,j0),pass_gate=ok,joint=jj,vector=f['vector']);append(root/'candidates.jsonl',row);rows.append(row)
            if ok:passing['full_projection'].append(alpha)
        storage(s);np.savez(root/'PRIVATE_GRADIENTS.npz',constraints=C,joint=J,semantic_joint=SJ,adam=ad,single=sd['strict'],robust=rd['strict'],projection=pu)
        assert costs==dict(reverse=19,adam=1,optimizer=0,candidates=56)
        write(root/'RESULTS.json',dict(status='COMPLETE',costs=costs,baseline=dict(vector=f0['vector'],joint=j0),single_geometry=single,robust_geometry=robust,semantic_direction_joint_cosines={k:(row_units(SJ)[0]@v).tolist() for k,v in dirs.items()},projection=projection,passing=passing,rows=rows,space=storage(s)))
        write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
