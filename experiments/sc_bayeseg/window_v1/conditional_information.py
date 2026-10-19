"""Frozen conditional semantic preference and softmax Jacobian audit."""
import json,os,traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset,collate_labeled
from lcrseg.engine.trainer import seed_everything
from care_seg_study_v1.run import write,save,append
from sc_bayeseg.split_v1.model import Model
from sc_bayeseg.model import student_log_prob,sar
from sc_bayeseg.window_v1.common import storage


def summarize(v):
    return dict(mean=float(v.mean()),per_image=[float(x) for x in v.mean(0)],train=float(v[:,:2].mean()),heldout=float(v[:,2:].mean()))


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);costs=dict(optimizer=0,local_updates=0,reverse=0,density_scores=0,shape_forwards=0,pure_tensor_directions=0)
    try:
        storage(s);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text());assert len(set(sel['identities']))==4
        ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        lb=collate_labeled([ds[i] for i in sel['train_indices']+sel['heldout_indices']]).to('cuda')
        image=F.interpolate(lb.image,(128,128),mode='bilinear',align_corners=False);y=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
        local=torch.load(s['states']['terminal'],map_location='cpu',weights_only=False);groups={};private={}
        for post in ['terminal','prior']:
            sd=local['model'].copy()
            if post=='prior':
                source=torch.load(s['anchor'],map_location='cpu',weights_only=False)['model']
                for key in sd:
                    if key.startswith(('anatomy.','structure.')):
                        prefix,leaf=key.rsplit('.',1)
                        sd[key]=source[prefix+'.prior_'+leaf] if leaf in ['mean_w','mean_b','logsd_w','logsd_b'] else source[key]
            model=Model().cuda();model.load_state_dict(sd);model.requires_grad_(False)
            before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            with torch.no_grad():
                h,_=model.proposal(image,sample=False);z=torch.cat([torch.zeros_like(h['ell'][:,:1]),h['ell']],1);p=z.softmax(1);pred=z.argmax(1)
                true_value=z.gather(1,y[:,None]);top=z.gather(1,pred[:,None])
                oracle=z.scatter(1,y[:,None],top).scatter(1,pred[:,None],true_value);oracle_ell=oracle[:,1:]-oracle[:,:1]
                oracle_prob=torch.cat([torch.zeros_like(oracle_ell[:,:1]),oracle_ell],1).softmax(1)
                assert torch.allclose(p.sort(1).values,oracle_prob.sort(1).values,atol=1e-6,rtol=0)
                oracle_correct=float((oracle_prob.argmax(1)==y).float().mean())
            score_groups={};raw={}
            for code in ['fixed','reencoded']:
                for template,ell in [('actual',h['ell']),('oracle',oracle_ell)]:
                    rows=[]
                    for draw in range(8):
                        seed_everything(80163+draw);model.draw()
                        try:
                            with torch.no_grad():
                                hh=h.copy();hh['ell']=ell
                                if code=='reencoded':hh['c']=model.gaussian(model.code_encoder(ell),sample=False)[0]
                                _,parts,_=model.log_joint(image,hh);costs['density_scores']+=1
                                row={k:v.cpu()/128**2 for k,v in parts.items() if k!='labels'};row['total']=sum(row.values());rows.append(row)
                        finally:model.clear_draw()
                    score={k:torch.stack([v[k] for v in rows]) for k in rows[0]};raw[code+'__'+template]=score
                    score_groups[code+'__'+template]={k:summarize(v) for k,v in score.items()}
            contrasts={code:{k:summarize(raw[code+'__oracle'][k]-raw[code+'__actual'][k]) for k in raw[code+'__actual']} for code in ['fixed','reencoded']}
            def shape_loss(prob):
                values=[]
                for draw in range(8):
                    seed_everything(80163+draw);model.draw()
                    try:
                        value=student_log_prob(sar(h['x']-model.structure_mean(prob)),.2)+model.logdet
                        values.append(-value.sum()/128**2);costs['shape_forwards']+=1
                    finally:model.clear_draw()
                return torch.stack(values).mean()
            pp=p.detach().requires_grad_();loss=shape_loss(pp);costs['reverse']+=1;append(root/'reverse_calls.jsonl',dict(index=costs['reverse'],posterior=post,coordinate='probability'))
            gp=torch.autograd.grad(loss,pp)[0]
            ell=h['ell'].detach().requires_grad_();prob=torch.cat([torch.zeros_like(ell[:,:1]),ell],1).softmax(1);loss2=shape_loss(prob)
            costs['reverse']+=1;append(root/'reverse_calls.jsonl',dict(index=costs['reverse'],posterior=post,coordinate='logits'))
            ge=torch.autograd.grad(loss2,ell)[0];analytic=(p*(gp-(p*gp).sum(1,keepdim=True)))[:,1:]
            assert torch.allclose(ge,analytic,atol=1e-7,rtol=1e-4)
            tangent=gp-gp.mean(1,keepdim=True);natural=-(gp[:,1:]-gp[:,:1]);costs['pure_tensor_directions']+=1
            assert float((ge*natural).sum())<=1e-8
            masks=dict(all=torch.ones_like(y,dtype=torch.bool),wrong=pred!=y,foreground=y>0,class2=y==2)
            statistics={}
            for part,ix in [('train',slice(0,2)),('heldout',slice(2,4))]:
                statistics[part]={}
                for maskname,mask in masks.items():
                    mm=mask[ix];n=int(mm.sum())
                    if not n:statistics[part][maskname]=dict(pixels=0);continue
                    pr=float(tangent[ix].square().sum(1)[mm].mean().sqrt());er=float(ge[ix].square().sum(1)[mm].mean().sqrt())
                    statistics[part][maskname]=dict(pixels=n,probability_tangent_gradient_rms=pr,logit_gradient_rms=er,transmission_ratio=er/max(pr,1e-30),max_probability_mean=float(p[ix].max(1).values[mm].mean()),fraction_max_probability_gt_999999=float((p[ix].max(1).values[mm]>.999999).float().mean()))
            assert all(torch.equal(v.cpu(),before[k]) for k,v in model.state_dict().items())
            groups[post]=dict(scores=score_groups,oracle_minus_actual=contrasts,gradient_statistics=statistics,oracle_argmax_correct_fraction=oracle_correct,probability_spectrum_preserved=True,jacobian_verified=True,shape_natural_first_order_derivative=float((ge*natural).sum()),model_unchanged=True)
            private[post]=dict(scores=raw,g_probability=gp.cpu(),g_logits=ge.cpu(),natural_direction=natural.cpu())
        assert costs==dict(optimizer=0,local_updates=0,reverse=4,density_scores=64,shape_forwards=32,pure_tensor_directions=2)
        storage(s);save(root/'PRIVATE_DIAGNOSTIC.pt',private);write(root/'RESULTS.json',dict(status='COMPLETE',costs=costs,groups=groups,space=storage(s)));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        write(root/'FAILURE.json',dict(traceback=traceback.format_exc(),costs=costs));write(root/'EXIT.json',dict(exit_code=1));raise


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
