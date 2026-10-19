"""Symmetric nuisance variational optimization for fixed GT/BG hypotheses."""
import json,os,traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset,collate_labeled
from lcrseg.engine.trainer import seed_everything
from lcrseg.engine.checkpoint import capture_rng_state,restore_rng_state
from care_seg_study_v1.run import write,append,save
from sc_bayeseg.run import same
from sc_bayeseg.diagnostic_v1.run import dice
from sc_bayeseg.window_v1.discrete_labeled import DiscreteLabeledModel
from sc_bayeseg.window_v1.common import storage


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);costs=dict(reverse=0,candidates=0,bound=0,accepted=0,decoder_diagnostics=0,optimizer=0,parameter_applications=0);active={}
    def forbid(*a,**kw):raise AssertionError('optimizer forbidden')
    torch.optim.Optimizer.__init__=forbid;torch.optim.Adam.step=forbid
    try:
        storage(s);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        origin=Path(s['discrete_train']);source={str(k):torch.load(origin/f'S{k:04}.pt',map_location='cpu',weights_only=False)['model'] for k in (0,600)}
        assert len((origin/'physical_updates.jsonl').read_text().splitlines())==604
        sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text());ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        lb=collate_labeled([ds[i] for i in sel['train_indices']]).to('cuda');image=F.interpolate(lb.image,(128,128),mode='bilinear',align_corners=False);gt=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
        model=DiscreteLabeledModel().cuda();model.requires_grad_(False);results={}
        def saa(z,shifts,seed,requires_grad=False):
            rng=capture_rng_state();values=[];parts=[]
            try:
                seed_everything(seed)
                with torch.set_grad_enabled(requires_grad):
                    for _ in range(2):
                        model.draw()
                        try:
                            for _ in range(4):
                                b,t,d=model.bound(image,z,shifts=shifts);values.append(b/128**2)
                                parts.append({**{k:v.detach()/128**2 for k,v in t.items()},'negative_logq':-d['logq'].detach()/128**2})
                                costs['bound']+=1;assert costs['bound']<=7360
                                append(root/'bound_calls.jsonl',dict(index=costs['bound'],seed=seed,gradient=requires_grad))
                        finally:model.clear_draw()
                    score=torch.stack(values).mean(0)
                    details={k:torch.stack([v[k] for v in parts]).mean(0).cpu().tolist() for k in parts[0]}
                return score,details
            finally:restore_rng_state(rng)
        def decoder(z,shifts):
            model.clear_draw()
            for layer in model.bayes_layers():layer.sampled=(layer.mean_w,layer.mean_b)
            try:
                with torch.no_grad():
                    h,_=model.labeled_proposal(image,z,sample=False,shifts=shifts);a=model.anatomy_mean(h['c']);p=torch.cat([torch.zeros_like(a[:,:1]),a],1).argmax(1)
                    costs['decoder_diagnostics']+=1
                    return dict(foreground_fraction=float((p>0).float().mean()),GT_dice_given_candidate_labels=dice(p,gt))
            finally:model.clear_draw()
        for theta in ('0','600'):
            sd=source['600'].copy()
            for k in sd:
                if k.startswith(('anatomy.','structure.')):sd[k]=source[theta][k]
            model.load_state_dict(sd)
            for label,z in [('GT',gt),('BG',torch.zeros_like(gt))]:
                name=f'theta{theta}_{label}';out=root/name;out.mkdir(exist_ok=False)
                shapes={'x':(2,1,128,128),'m':(2,1,128,128),'c':(2,16)}
                shifts={k:tuple(torch.zeros(shape,device='cuda',requires_grad=True) for _ in range(2)) for k,shape in shapes.items()}
                before,firstparts=saa(z,shifts,50163);parity,_=saa(z,None,50163);assert torch.equal(before,parity)
                initial_other,initial_otherparts=saa(z,shifts,80163);initial_decoder=decoder(z,shifts)
                start=dict(primary=before.detach().cpu().tolist(),other=initial_other.cpu().tolist(),primary_parts=firstparts,other_parts=initial_otherparts,decoder=initial_decoder)
                storage(s);save(out/'S00.pt',{k:tuple(v.detach().cpu() for v in pair) for k,pair in shifts.items()})
                active.update(out=out,shifts=shifts,step=0);accepted=0
                for step in range(1,26):
                    current,_=saa(z,shifts,50163,True);flat=[v for pair in shifts.values() for v in pair]
                    costs['reverse']+=1;assert costs['reverse']<=100
                    append(root/'reverse_calls.jsonl',dict(index=costs['reverse'],cell=name,step=step))
                    g=torch.autograd.grad(-current.sum(),flat);assert all(torch.isfinite(v).all() for v in g)
                    chosen=None;trial=0
                    for alpha in (1.,.25,.0625,.015625,.00390625,.0009765625,.000244140625,0.):
                        candidate={};index=0
                        for k,pair in shifts.items():
                            rate=128**2 if k in ('x','m') else 1.
                            candidate[k]=tuple((v-.05*alpha*rate*g[index+j]).detach() for j,v in enumerate(pair));index+=2
                        v,_=saa(z,candidate,50163);finite=bool(torch.isfinite(v).all())
                        ok=finite and bool((v>=current.detach()-1e-6).all()) and float(v.mean())>float(current.detach().mean())+1e-5
                        costs['candidates']+=1;assert costs['candidates']<=800;trial+=1
                        append(root/'candidates.jsonl',dict(index=costs['candidates'],cell=name,step=step,alpha=alpha,accepted=ok,value=v.cpu().tolist() if finite else None))
                        if ok:chosen=(alpha,candidate,v);break
                    if chosen is not None:
                        alpha,shifts,after=chosen;shifts={k:tuple(v.requires_grad_() for v in pair) for k,pair in shifts.items()};accepted+=1;costs['accepted']+=1
                        append(root/'local_updates.jsonl',dict(index=costs['accepted'],cell=name,step=step,alpha=alpha))
                    else:alpha=0.;after=current.detach()
                    append(out/'steps.jsonl',dict(step=step,alpha=alpha,accepted=chosen is not None,before=current.detach().cpu().tolist(),after=after.detach().cpu().tolist(),trials=trial))
                    active.update(shifts=shifts,step=step);storage(s);save(out/f'S{step:02}.pt',{k:tuple(v.detach().cpu() for v in pair) for k,pair in shifts.items()})
                    write(root/'STATUS.json',dict(status='RUNNING',cell=name,step=step,costs=costs,space=storage(s)))
                final,finalparts=saa(z,shifts,50163);other,otherparts=saa(z,shifts,80163)
                assert same(sd,model.state_dict())
                results[name]=dict(initial=start,final=dict(primary=final.cpu().tolist(),other=other.cpu().tolist(),primary_parts=finalparts,other_parts=otherparts,decoder=decoder(z,shifts)),accepted=accepted,model_unchanged=True,zero_shift_parity=True)
                write(out/'RESULTS.json',results[name]);active.clear()
        assert costs['reverse']==100 and costs['decoder_diagnostics']==8 and costs['optimizer']==0
        assert len((origin/'physical_updates.jsonl').read_text().splitlines())==604
        write(root/'RESULTS.json',dict(status='COMPLETE',costs=costs,cells=results,heldout_or_realU_accessed=False,source_physical_ledger_unchanged=True,space=storage(s)));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        if active:storage(s);save(active['out']/'failure_partial.pt',dict(step=active['step'],shifts={k:tuple(v.detach().cpu() for v in pair) for k,pair in active['shifts'].items()},costs=costs))
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
