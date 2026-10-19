"""No-update change-of-variables and paired ELBO audit of a label proposal."""
import json,os,traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset,collate_labeled
from lcrseg.engine.trainer import seed_everything
from care_seg_study_v1.run import write,append
from sc_bayeseg.split_v1.model import Model,semantic
from sc_bayeseg.run import same
from sc_bayeseg.window_v1.label_chart import LabelChartModel,transform,inverse
from sc_bayeseg.window_v1.fullview import flip
from sc_bayeseg.window_v1.common import storage


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);costs=dict(reverse=0,bound=0,optimizer=0,parameter_applications=0,local_updates=0,synthetic_transform=0)
    def reverse(label,*a,**kw):
        costs['reverse']+=1;assert costs['reverse']<=20
        append(root/'reverse_calls.jsonl',dict(index=costs['reverse'],label=label));return torch.autograd.grad(*a,**kw)
    def bound(model,image,y,**metadata):
        costs['bound']+=1;append(root/'bound_calls.jsonl',dict(index=costs['bound'],**metadata));return model.bound(image,y)
    def forbid(*a,**kw):raise AssertionError('optimizer forbidden')
    torch.optim.Optimizer.__init__=forbid;torch.optim.Adam.step=forbid
    try:
        storage(s);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        math=[]
        for k in range(4):
            u=torch.tensor([[[[-1.2]],[[.7]],[[1.6]]]],dtype=torch.float64,requires_grad=True);y=torch.full((1,1,1),k,dtype=torch.long)
            ell,ld=transform(u,y);costs['synthetic_transform']+=1
            jac=torch.stack([reverse(f'jacobian_Y{k}_row{j}',ell.flatten()[j],u,retain_graph=j<2)[0].flatten() for j in range(3)])
            actual=torch.linalg.slogdet(jac).logabsdet
            assert torch.allclose(actual,ld[0],atol=1e-12,rtol=0)
            back=inverse(ell.detach(),y);assert torch.allclose(back,u.detach(),atol=1e-12,rtol=0)
            pred=torch.cat([torch.zeros_like(ell[:,:1]),ell],1).argmax(1);assert (pred==y).all()
            normal=torch.distributions.Normal(torch.zeros_like(u),torch.ones_like(u))
            density=normal.log_prob(back).sum()-ld.sum();direct=normal.log_prob(u.detach()).sum()-actual
            assert torch.allclose(density,direct,atol=1e-12,rtol=0)
            math.append(dict(label=k,jacobian_logdet=float(actual),analytic_logdet=float(ld[0]),roundtrip_max_error=float((back-u.detach()).abs().max()),density_error=float((density-direct).abs())))
        sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text());ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        lb=collate_labeled([ds[i] for i in sel['train_indices']]).to('cuda')
        im=F.interpolate(lb.image,(128,128),mode='bilinear',align_corners=False);yy=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
        image=torch.cat([flip(im,k) for k in range(4)]);y=torch.cat([flip(yy,k) for k in range(4)])
        results={}
        for state,path in s['readout_states'].items():
            payload=torch.load(path,map_location='cpu',weights_only=False);model=LabelChartModel().cuda();model.load_state_dict(payload['model'])
            params=list(model.parameters());named=list(model.named_parameters());out={}
            for family in ('U','L'):
                model.chart_enabled=family=='L';rows=[]
                for draw in range(8):
                    seed_everything(80163+draw);model.draw()
                    try:
                        with torch.set_grad_enabled(draw==0):
                            b,terms,_=bound(model,image,y,state=state,family=family,draw=draw);kl=model.kl()/(2*128**2);loss=-b.mean()/128**2+kl
                            row=dict(joint=float(loss.detach()),kl=float(kl.detach()),logq=float((sum(terms.values())-b).mean().detach()/128**2),
                                     terms={k:float(-v.mean().detach()/128**2) for k,v in terms.items()},chart_stats=model.chart_stats.copy() if family=='L' else None)
                            assert all(torch.isfinite(v).all() for v in [b,*terms.values(),kl])
                            if draw==0:
                                g=reverse(f'{state}_{family}_joint',loss,params,retain_graph=True,allow_unused=True)
                                label_loss=-terms['labels'].mean()/128**2
                                gl=reverse(f'{state}_{family}_labels',label_loss,params,allow_unused=True)
                                row['gradient']={scope:dict(joint_norm=sum(float(v.detach().double().square().sum()) for (n,p),v in zip(named,g) if v is not None and (scope=='all' or semantic(n)))**.5,
                                        label_norm=sum(float(v.detach().double().square().sum()) for (n,p),v in zip(named,gl) if v is not None and (scope=='all' or semantic(n)))**.5) for scope in ('all','semantic')}
                                assert all(v is None or torch.isfinite(v).all() for v in [*g,*gl])
                        if family=='U' and draw==0:
                            reference=Model().cuda();reference.load_state_dict(payload['model']);seed_everything(80163);reference.draw()
                            with torch.no_grad():bb,tt,_=bound(reference,image,y,state=state,family='original_control',draw=draw)
                            assert torch.equal(bb,b.detach()) and all(torch.equal(tt[k],v.detach()) for k,v in terms.items())
                            reference.clear_draw();row['original_U_exact']=True
                        rows.append(row)
                    finally:model.clear_draw()
                out[family]=rows
            assert same(payload['model'],model.state_dict()) and all(p.grad is None for p in params)
            results[state]=out;write(root/'STATUS.json',dict(status='RUNNING',state=state,costs=costs))
        assert costs==dict(reverse=20,bound=34,optimizer=0,parameter_applications=0,local_updates=0,synthetic_transform=4)
        result=dict(status='COMPLETE',costs=costs,math=math,states=results,all_model_tensors_unchanged=True,grad_buffers_none=True,heldout_accessed=False,space=storage(s))
        write(root/'RESULTS.json',result);write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
