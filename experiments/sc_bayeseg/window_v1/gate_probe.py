"""Finite source of plateau: semantic radius vs nonsemantic Adam proposal."""
import json,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset,collate_labeled
from lcrseg.engine.checkpoint import restore_rng_state
from care_seg_study_v1.run import write,append
from sc_bayeseg.split_v1.model import Model,semantic
from sc_bayeseg.jointwrite_v1.run import Adapter
from sc_bayeseg.geometry_v1.run import losses,flat
from sc_bayeseg.geometry_v1.geometry import solve
from sc_bayeseg.adam_audit_v1.run import candidate
from sc_bayeseg.guard_v1.run import frame,accepts
from sc_bayeseg.window_v1.common import storage,objective,joint_pass


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);costs=dict(reverse=0,adam_candidates=0,optimizer=0,candidates=0)
    try:
        storage(s);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        def forbid(*a,**kw):raise AssertionError('optimizer forbidden')
        torch.optim.Optimizer.__init__=forbid;torch.optim.Adam.step=forbid
        p=torch.load(s['checkpoint'],weights_only=False,map_location='cpu');assert p['cursor']==3025
        model=Model().cuda();model.load_state_dict(p['model']);named=list(model.named_parameters());params=[v for _,v in named];ix=[i for i,(n,v) in enumerate(named) if semantic(n)];sp=[params[i] for i in ix]
        ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text());lb=collate_labeled([ds[i] for i in sel['train_indices']]).to('cuda');image=F.interpolate(lb.image,(128,128),mode='bilinear',align_corners=False);y=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
        method=Adapter(model,'joint');method.nl=2;method.begin_site('Siemens',None,3600)
        restore_rng_state(p['rng']);f0=frame(model,image,y);j0=objective(model,image,y,s['replay']);vals=losses(model,image,y);gs=[]
        def grad(*a,**kw):
            assert costs['reverse']<17;costs['reverse']+=1;append(root/'reverse_calls.jsonl',dict(index=costs['reverse']));return torch.autograd.grad(*a,**kw)
        for i,v in enumerate(vals):gs.append(flat(grad(v,sp,allow_unused=True,retain_graph=i<15),sp))
        restore_rng_state(p['rng']);r=method.training_step(lb,None,3025,3025);full=grad(r.total_loss,params,allow_unused=True)
        totalnorm=torch.linalg.vector_norm(torch.stack(torch._foreach_norm(full,2)),2);gg=torch._foreach_mul(full,(1/(totalnorm+1e-6)).clamp(max=1.))
        opt=p['trainer']['optimizer_state'];group=opt['param_groups'][0]
        with torch.no_grad():delta,_,_,_,_=candidate(params,gg,[opt['state'][i] for i in group['params']],group)
        costs['adam_candidates']=1
        geometry,dirs=solve(np.stack(gs),flat([full[i] for i in ix],sp)[None]);direction=dirs['strict'];valid=geometry['strict']['strict_witness']
        radius=json.loads((Path(s['geometry'])/'RESULTS.json').read_text())['states']['0']['joint'][0]['adam_semantic_norm']/16
        rows=[];passing={}
        for mode in ['frozen_nonsemantic','adam_nonsemantic']:
            passing[mode]=[]
            for alpha in [1.,.25,.0625,.015625,.00390625,.0009765625,.000244140625,0.]:
                storage(s);model.load_state_dict(p['model']);pos=0
                with torch.no_grad():
                    for i,(n,v) in enumerate(named):
                        base=p['model'][n].to(v.device)
                        if semantic(n):
                            dd=torch.as_tensor(direction[pos:pos+v.numel()].reshape(v.shape),dtype=v.dtype,device=v.device);v.copy_(base+radius*alpha*dd);pos+=v.numel()
                        elif mode=='adam_nonsemantic':v.copy_(base+delta[i])
                costs['candidates']+=1
                f=f0 if alpha==0 else frame(model,image,y);j=j0 if mode=='frozen_nonsemantic' and alpha==0 else objective(model,image,y,s['replay'])
                ok=valid and accepts(f['vector'],f0['vector']) and joint_pass(j,j0)
                row=dict(index=costs['candidates'],mode=mode,alpha=alpha,semantic_pass=accepts(f['vector'],f0['vector']),joint_pass=joint_pass(j,j0),pass_gate=ok,vector=f['vector'],joint=j,joint_changes=[a-b for a,b in zip(j,j0)])
                append(root/'candidates.jsonl',row);rows.append(row)
                if ok:passing[mode].append(alpha)
        assert costs==dict(reverse=17,adam_candidates=1,optimizer=0,candidates=16)
        write(root/'RESULTS.json',dict(status='COMPLETE',costs=costs,baseline=dict(vector=f0['vector'],joint=j0),geometry=geometry,passing=passing,rows=rows,radius=radius,space=storage(s)))
        write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
