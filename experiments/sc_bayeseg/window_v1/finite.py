"""Read-only finite probes of previously computed common-descent directions."""
import json,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset,collate_labeled
from care_seg_study_v1.run import write,append
from sc_bayeseg.split_v1.model import Model,semantic
from sc_bayeseg.guard_v1.run import frame,accepts
from sc_bayeseg.window_v1.common import storage,objective,joint_pass


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);start=time.time();count=0
    try:
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        storage(s)
        def forbid(*a,**kw):raise AssertionError('optimizer prohibited')
        torch.optim.Optimizer.__init__=forbid;torch.optim.Adam.step=forbid
        ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        selection=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text());lb=collate_labeled([ds[i] for i in selection['train_indices']]).to('cuda')
        image=F.interpolate(lb.image,(128,128),mode='bilinear',align_corners=False);y=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
        vectors=np.load(Path(s['geometry'])/'PRIVATE_GRADIENTS_AND_DIRECTIONS.npz');geo=json.loads((Path(s['geometry'])/'RESULTS.json').read_text())
        model=Model().cuda();rows=[];bases={};qualified={}
        for sk in [0,24]:
            payload=torch.load(Path(s['guard'])/'G'/f'S{sk:02}.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(payload['model']);base=frame(model,image,y);bj=objective(model,image,y,s['replay']);bases[str(sk)]=dict(vector=base['vector'],metrics=base['metrics'],joint=bj)
            radius=geo['states'][str(sk)]['joint'][0 if sk==0 else 1]['adam_semantic_norm']/16
            for name in ['weak','strict']:
                qualified[f'{sk}_{name}']=[]
                d=vectors[f'S{sk}_{name}'];assert abs(np.linalg.norm(d)-1)<1e-7
                for scale in [1.,.25,.0625]:
                    storage(s);model.load_state_dict(payload['model']);pos=0
                    with torch.no_grad():
                        for n,p in model.named_parameters():
                            if semantic(n):
                                dd=torch.as_tensor(d[pos:pos+p.numel()].reshape(p.shape),device=p.device,dtype=p.dtype);p.add_(dd,alpha=radius*scale);pos+=p.numel()
                    assert pos==len(d) and count<12;count+=1
                    f=frame(model,image,y);j=objective(model,image,y,s['replay']);ok=accepts(f['vector'],base['vector']) and joint_pass(j,bj)
                    row=dict(candidate=count,state=sk,direction=name,scale=scale,radius=radius,semantic_pass=accepts(f['vector'],base['vector']),joint_pass=joint_pass(j,bj),pass_gate=ok,vector=f['vector'],metrics=f['metrics'],joint=j,joint_changes=[a-b for a,b in zip(j,bj)])
                    append(root/'candidates.jsonl',row);rows.append(row)
                    if ok:qualified[f'{sk}_{name}'].append(scale)
                    write(root/'STATUS.json',dict(phase='FINITE',candidate=count,space=storage(s)))
        assert count==12
        r=dict(status='COMPLETE',optimizer_updates=0,reverse_calls=0,candidate_evaluations=count,baseline=bases,qualification=qualified,training_qualified=bool(qualified['0_weak'] and qualified['0_strict']),rows=rows,started=start,finished=time.time(),space=storage(s))
        write(root/'RESULTS.json',r);write(root/'STATUS.json',dict(phase='COMPLETE',training_qualified=r['training_qualified'],space=storage(s)));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        write(root/'FAILURE.json',dict(candidate_evaluations=count,traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
