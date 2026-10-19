"""Pure Adam reconstruction for the current four-view supervised objective."""
import json,os,traceback,copy
from pathlib import Path
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset,collate_labeled
from lcrseg.engine.checkpoint import restore_rng_state,capture_rng_state
from care_seg_study_v1.run import write,save,append
from sc_bayeseg.split_v1.model import Model,semantic
from sc_bayeseg.adam_audit_v1.run import candidate,dot,norm
from sc_bayeseg.run import same
from sc_bayeseg.window_v1.augment_pair import flip
from sc_bayeseg.window_v1.common import storage


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);costs=dict(optimizer=0,reverse=0,pure_Adam=0,candidate_forward=0)
    def forbid(*a,**kw):raise AssertionError('Optimizer forbidden')
    torch.optim.Optimizer.__init__=forbid;torch.optim.Adam.step=forbid
    try:
        storage(s);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        payload=torch.load(s['checkpoint'],map_location='cpu',weights_only=False);assert payload['cursor']==4375
        model=Model().cuda();model.load_state_dict(payload['model']);named=list(model.named_parameters())
        for n,p in named:p.requires_grad_(semantic(n))
        selected=[(i,n,p) for i,(n,p) in enumerate(named) if semantic(n)];params=[p for _,_,p in selected];assert len(params)==40
        sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text());ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        lb=collate_labeled([ds[i] for i in sel['train_indices']]).to('cuda');im=F.interpolate(lb.image,(128,128),mode='bilinear',align_corners=False);yy=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
        image=torch.cat([flip(im,k) for k in range(4)]);y=torch.cat([flip(yy,k) for k in range(4)])
        restore_rng_state(payload['rng']);h,_=model.proposal(image,sample=False);z=torch.cat([torch.zeros_like(h['ell'][:,:1]),h['ell']],1);field=F.cross_entropy(z,y,reduction='none');loss=field.mean()
        def grad(v,kind):
            costs['reverse']+=1;append(root/'reverse_calls.jsonl',dict(index=costs['reverse'],kind=kind));return list(torch.autograd.grad(v,params,retain_graph=True))
        full=grad(loss,'mean_CE');total=torch.linalg.vector_norm(torch.stack(torch._foreach_norm(full,2)),2);coef=(1/(total+1e-6)).clamp(max=1.);clipped=torch._foreach_mul(full,coef)
        opt=payload['trainer']['optimizer_state'];states=[opt['state'][i] for i,_,_ in selected]
        delta,history,current,raw,meta=candidate(params,clipped,states,opt['param_groups'][0]);costs['pure_Adam']=1
        rows=[];private=[]
        for i in range(8):
            for label in ['all',1,2,3]:
                v=field[i].mean() if label=='all' else field[i][y[i]==label].mean();g=grad(v,f'view{i//2}_image{i%2}_{label}')
                rows.append(dict(view=i//2,image_index=i%2,category=label,CE=float(v.detach()),adam_derivative=dot(g,delta),negative_mean_gradient_derivative=-dot(g,full)));private.append([x.detach().cpu() for x in g])
        deriv=dot(full,delta);qualified=bool(torch.isfinite(loss) and norm(full)>0 and deriv<0)
        assert costs==dict(optimizer=0,reverse=33,pure_Adam=1,candidate_forward=0)
        assert all(torch.equal(v.cpu(),payload['model'][k]) for k,v in model.state_dict().items()) and same(capture_rng_state(),payload['rng'])
        storage(s);save(root/'PRIVATE_GRADIENTS.pt',dict(gradient=[x.cpu() for x in full],delta=[x.cpu() for x in delta],constraints=private))
        write(root/'RESULTS.json',dict(status='COMPLETE',costs=costs,qualified=qualified,mean_CE=float(loss.detach()),gradient_norm=norm(full),clip_coefficient=float(coef),adam_mean_CE_derivative=deriv,negative_gradient_mean_CE_derivative=-dot(full,full),directions=rows,meta=meta,model_and_rng_unchanged=True,space=storage(s)));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        write(root/'FAILURE.json',dict(traceback=traceback.format_exc(),costs=costs));write(root/'EXIT.json',dict(exit_code=1));raise


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
