"""Bounded two-stage analytic pilot, shared feature extraction and Dice engine."""
import copy
import json
import os
import time
import traceback
import unittest
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from . import core, test_contract
from ..space_lora_v1.core import source_model
from ..main_style_ssl_v1.data import Accessor
from ..main_style_ssl_v1.runtime import write, append
from ..five_frameworks_v1.checkpoint import atomic_save
from ..siemens_lonly_v1.run import aggregate, compare, dice


class Readout(nn.Module):
    def __init__(self, source, basis):
        super().__init__()
        self.source=source
        for name,value in basis.items(): self.register_buffer(name,value)
        self.register_buffer('w',torch.zeros(129,4,dtype=torch.float64,device='cuda'))

    def patches(self,x):
        h=self.source.parent.features(x)
        if h.shape[1]!=16: raise ValueError('native feature contract')
        return F.unfold(h,3).transpose(1,2), (h.shape[-2]-2,h.shape[-1]-2)

    def response(self,patches):
        z=(patches-self.mean)/self.std
        d=(z.square().sum(-1,keepdim=True)+self.centers.square().sum(1)-2*z@self.centers.T).clamp_min(0)/144
        phi=torch.exp(-d/(2*self.bandwidth))
        return torch.cat([phi,torch.ones_like(phi[...,:1])],-1)

    def forward(self,x):
        patches,shape=self.patches(x)
        outputs=[]
        for start in range(0,patches.shape[1],4096):
            outputs.append(self.response(patches[:,start:start+4096]).double()@self.w)
        logits=torch.cat(outputs,1).transpose(1,2).reshape(len(x),4,*shape).float()
        return F.interpolate(logits,size=x.shape[-2:],mode='bilinear',align_corners=True)


def samples(model,access,ledger,kind):
    for index in range(len(access.rows)):
        item=access.read(index)
        patient=access.rows[index]['patient_id']
        depth=sum(access.depths[i] for i in access.by_patient[patient])
        for z in range(len(item['image'])):
            x=item['image'][z:z+1,None].cuda()
            patches,shape=model.patches(x)
            seed=163+index*100000+z
            positions=torch.randperm(patches.shape[1],generator=torch.Generator().manual_seed(seed))[:1024].cuda()
            p=patches[0,positions]
            y=None
            if access.role=='train_labeled':
                labels=F.interpolate(item['label'][z:z+1,None].float().cuda(),size=shape,mode='nearest').long().flatten()
                y=labels[positions]
            elif 'label' in item: raise PermissionError('U label access')
            append(ledger,dict(kind=kind,domain=access.rows[index].get('domain',model.domain),role=access.role,
                               volume_index=index,slice=z,forward_slices=1,sampled_pixels=len(positions)))
            yield p,y,1/(len(access.patients)*depth*len(positions))
        print(json.dumps(dict(kind=kind,domain=model.domain,volume=index+1,total_volumes=len(access.rows))),flush=True)


def labeled_state(model,access,out):
    gl=torch.zeros(129,129,dtype=torch.float64,device='cuda'); cl=torch.zeros(129,4,dtype=torch.float64,device='cuda')
    for p,y,weight in samples(model,access,out/'exposure.jsonl','statistics_L'):
        phi=model.response(p).double()
        gl+=weight*(phi.T@phi)
        cl+=weight*(phi.T@F.one_hot(y,4).double())
    return gl,cl


def unlabeled_state(model,access,out):
    values=[]; weights=[]
    for p,y,weight in samples(model,access,out/'exposure.jsonl','statistics_U'):
        if y is not None: raise PermissionError('U truth')
        phi=model.response(p)
        values.append(phi); weights.append(torch.full((len(phi),),weight,dtype=torch.float64,device='cuda'))
    x=torch.cat(values); weights=torch.cat(weights); del values
    result=core.summarize(x,weights)
    del x,weights
    return result


def evaluate(model,cfg,domain,role,path,out):
    access=Accessor(cfg['data'],'mnms',domain,role)
    expected={('Siemens','val'):23,('Philips','val'):31,('Siemens','train_labeled'):10,('Philips','train_labeled'):13}
    if len(access.patients)!=expected[domain,role]: raise ValueError('evaluation population mismatch')
    rows=[]; slices=0
    for i,meta in enumerate(access.rows):
        item=access.read(i); x=item['image']; truth=item['label'].numpy()
        predictions=[]
        for z in range(0,len(x),2):
            predictions.append(model(x[z:z+2,None].cuda()).argmax(1).cpu().numpy())
        row=dict(patient=meta['patient_id'],phase=meta['phase'],dice=dice(np.concatenate(predictions),truth).tolist())
        append(path,row); rows.append(row); slices+=len(x)
    append(out/'exposure.jsonl',dict(kind='evaluation',domain=domain,role=role,forward_slices=slices))
    return aggregate(rows),rows


def engineering(model,access,out):
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_contract))
    if not result.wasSuccessful(): raise AssertionError('analytic contract failed')
    item=access.read(0); x=item['image'][:1,None].cuda(); p,shape=model.patches(x)
    append(out/'exposure.jsonl',dict(kind='engineering',domain='Siemens',role='train_labeled',forward_slices=1))
    phi=model.response(p[0,:128]).double()
    y=F.interpolate(item['label'][:1,None].float().cuda(),size=shape,mode='nearest').long().flatten()[:128]
    w=core.solve(phi.T@phi/len(phi),phi.T@F.one_hot(y,4).double()/len(phi))
    before=copy.deepcopy(model.state_dict()); model.w.copy_(w)
    prediction=model(x); append(out/'exposure.jsonl',dict(kind='engineering_golden',domain='Siemens',role='train_labeled',forward_slices=1))
    atomic_save(model.state_dict(),out/'engineering.pt')
    state=torch.load(out/'engineering.pt',map_location='cuda',weights_only=False)
    model.w.zero_(); model.load_state_dict(state)
    restored=model(x); append(out/'exposure.jsonl',dict(kind='engineering_resume',domain='Siemens',role='train_labeled',forward_slices=1))
    if not torch.equal(prediction,restored): raise AssertionError('checkpoint prediction mismatch')
    if any(p.grad is not None or p.requires_grad for p in model.parameters()): raise AssertionError('frozen feature violation')
    model.load_state_dict(before)
    write(out/'ENGINEERING.json',dict(status='PASS',unit_tests=result.testsRun,real_slice_solve=True,
        golden_checkpoint_prediction_exact=True,two_case_interpolation=True,optimizer_updates=0,
        backward='not applicable: inference-only fixed features, analytic head; all parameters frozen'))


@torch.inference_mode()
def main(cfg):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='3': raise PermissionError('GPU3 only')
    out=Path(cfg['output']); out.mkdir(exist_ok=False)
    started=time.time(); counts=dict(engineering=0,formal=0); category='engineering'
    def record():
        cap=16 if category=='engineering' else 82
        if counts[category]>=cap: raise PermissionError('analytic solve cap')
        counts[category]+=1
        append(out/'solve_calls.jsonl',dict(category=category,call=counts[category],time=time.time()))
    core.record_solve=record
    try:
        torch.set_num_threads(4); torch.manual_seed(163); torch.cuda.manual_seed_all(163)
        torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True; torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        source=source_model(cfg)
        # Source-only basis fit: all source L slices, uniformly sampled 1024 positions each.
        basis=dict(mean=torch.zeros(144,device='cuda'),std=torch.ones(144,device='cuda'),
                   centers=torch.zeros(128,144,device='cuda'),bandwidth=torch.ones((),device='cuda'))
        model=Readout(source,basis).eval().requires_grad_(False); model.domain='Siemens'
        access=Accessor(cfg['data'],'mnms','Siemens','train_labeled')
        if len(access.patients)!=10 or sum(access.depths)!=230: raise ValueError('source L population')
        write(out/'STATUS.json',dict(status='SOURCE_BASIS',formal_solves=0))
        pool=[]
        for p,y,weight in samples(model,access,out/'exposure.jsonl','source_basis'):
            pool.append(p)
        pool=torch.cat(pool)
        model.mean.copy_(pool.mean(0)); model.std.copy_(pool.std(0).clamp_min(.001))
        selection=torch.randperm(len(pool),device='cpu')[:128].cuda()
        model.centers.copy_((pool[selection]-model.mean)/model.std)
        distances=torch.pdist(model.centers).square()/144
        model.bandwidth.copy_(distances.median().clamp_min(1e-4))
        del pool
        engineering(model,access,out)
        atomic_save({k:v.cpu() for k,v in model.state_dict().items() if not k.startswith('source.')},out/'basis.pt')
        category='formal'
        gl,cl=labeled_state(model,access,out)
        common=core.solve(gl,cl)
        heads={'l_only':common.clone()}; source_q=None
        stats=[]; stage_results={}; private={}
        for stage,domain in enumerate(('Siemens','Philips')):
            model.domain=domain
            write(out/'STATUS.json',dict(status='BUILDING_STATE',stage=stage,domain=domain,formal_solves=counts['formal']))
            la=Accessor(cfg['data'],'mnms',domain,'train_labeled'); ua=Accessor(cfg['data'],'mnms',domain,'train_unlabeled')
            if stage:
                if len(la.patients)!=13: raise ValueError('Philips L population')
                dg,dc=labeled_state(model,la,out); gl+=dg; cl+=dc
                heads['l_only']=core.solve(gl,cl)
            if set(la.patients)&set(ua.patients): raise ValueError('L/U patient overlap')
            gu,memory=unlabeled_state(model,ua,out); stats.append((gu,memory))
            total_gu=sum(s[0] for s in stats); memories=core.join([s[1] for s in stats])
            if not stage:
                heads.update(fixed_q=common.clone(),revisable_q=common.clone())
                source_q=core.responsibility(memory['mu'],common)
                fixed_q=source_q
            else:
                fixed_q=torch.cat([source_q,core.responsibility(memory['mu'],heads['fixed_q'])])
            atomic_save(dict(gl=gl,cl=cl,gu=total_gu,memory=memories,fixed_q=fixed_q),out/f'stage{stage}_state.pt')
            for arm in core.ARMS:
                if arm!='l_only':
                    q_before=core.responsibility(memories['mu'],heads[arm])
                    def report(iteration,before,after,delta):
                        append(out/'iterations.jsonl',dict(stage=stage,arm=arm,iteration=iteration,
                            objective_before=before,objective_after=after,head_change=delta))
                    heads[arm]=core.optimize(gl,cl,total_gu,memories,heads[arm],
                        fixed_q=fixed_q if arm=='fixed_q' else None,report=report)
                    q_after=core.responsibility(memories['mu'],heads[arm])
                    append(out/'state_diagnostics.jsonl',dict(stage=stage,arm=arm,
                        total_responsibility_change=float((q_after-q_before).abs().mean()),
                        old_responsibility_change=float((q_after[:len(stats[0][1]['n'])]-q_before[:len(stats[0][1]['n'])]).abs().mean()),
                        compression_cross_moment_bound=core.compression_bound(memories,heads[arm]),
                        summaries=len(memories['n']),summary_mass=float(memories['n'].sum())))
                model.w.copy_(heads[arm]); atomic_save(dict(w=heads[arm],stage=stage,arm=arm),out/f'stage{stage}_{arm}.pt')
                write(out/'STATUS.json',dict(status='EVALUATING',stage=stage,arm=arm,formal_solves=counts['formal']))
                result={}; priv={}
                domains=('Siemens',) if not stage else ('Siemens','Philips')
                for evaluated in domains:
                    for role in ('train_labeled','val'):
                        name=evaluated+'_'+role
                        result[name],priv[name]=evaluate(model,cfg,evaluated,role,out/f'private_s{stage}_{arm}_{name}.jsonl',out)
                stage_results[f'{stage}_{arm}']=result; private[f'{stage}_{arm}']=priv
        comparisons={}
        for stage in range(2):
            for first,second in (('l_only','fixed_q'),('l_only','revisable_q'),('fixed_q','revisable_q')):
                a=private[f'{stage}_{first}']; b=private[f'{stage}_{second}']
                comparisons[f'{stage}_{second}_minus_{first}']={name:compare(a[name],b[name]) for name in a}
        retention={arm:compare(private[f'0_{arm}']['Siemens_val'],private[f'1_{arm}']['Siemens_val']) for arm in core.ARMS}
        for arm in core.ARMS:
            r=stage_results[f'1_{arm}']; r['two_domain_mean']=(r['Siemens_val']['primary_dice']+r['Philips_val']['primary_dice'])/2
        if counts['formal']!=82 or any(p.grad is not None or p.requires_grad for p in model.parameters()):
            raise AssertionError('budget/freeze violation')
        result=dict(status='COMPLETE',stages=stage_results,paired=comparisons,retention=retention,
            costs=dict(solve_calls=counts,optimizer_updates=0,seconds=time.time()-started),
            test_access=False,U_GT_access=False,source_feature_prior='DenseFixMatch43600 EMA; NOT a whole-pipeline L-only comparison',
            selection='fixed final 20 alternating iterations, no validation selection')
        write(out/'RESULTS.json',result)
        write(out/'STATUS.json',dict(status='COMPLETE',formal_solves=counts['formal']))
        write(out/'EXIT.json',dict(exit_code=0))
    except BaseException as exc:
        failure=dict(status='FAILED',error=str(exc),traceback=traceback.format_exc(),solve_calls=counts)
        write(out/'FAILURE.json',failure); write(out/'STATUS.json',failure)
        write(out/'EXIT.json',dict(exit_code=1)); raise
