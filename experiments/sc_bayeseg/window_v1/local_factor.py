"""Frozen local-shift factorial attribution with exact common random streams."""
import json,os,traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset,collate_labeled
from lcrseg.engine.trainer import seed_everything
from care_seg_study_v1.run import write,save
from sc_bayeseg.split_v1.model import Model
from sc_bayeseg.window_v1.common import storage
from sc_bayeseg.window_v1.local_gate import metrics


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);costs=dict(optimizer=0,reverse=0,local_updates=0,density_scores=0,prediction_sets=0)
    try:
        storage(s);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text());assert len(set(sel['identities']))==4
        ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        lb=collate_labeled([ds[i] for i in sel['train_indices']+sel['heldout_indices']]).to('cuda')
        image=F.interpolate(lb.image,(128,128),mode='bilinear',align_corners=False);y=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long().cpu()
        local=torch.load(s['states']['terminal'],map_location='cpu',weights_only=False);all_results={};private={}
        for post in ['terminal','prior']:
            sd=local['model'].copy()
            if post=='prior':
                source=torch.load(s['anchor'],map_location='cpu',weights_only=False)['model']
                for key in sd:
                    if key.startswith(('anatomy.','structure.')):
                        prefix,leaf=key.rsplit('.',1)
                        sd[key]=source[prefix+'.prior_'+leaf] if leaf in ['mean_w','mean_b','logsd_w','logsd_b'] else source[key]
            model=Model().cuda();model.load_state_dict(sd);model.requires_grad_(False)
            initial=torch.load(Path(s['local_gate'])/(post+'_S00.pt'),map_location='cuda',weights_only=False)
            final=torch.load(Path(s['local_gate'])/(post+'_S25.pt'),map_location='cuda',weights_only=False)
            assert all(torch.count_nonzero(v)==0 for pair in initial.values() for v in pair)
            groups={};predictions={};raw={}
            for variant in ['zero','physical','semantic','all','mean_only','variance_only']:
                shifts={k:tuple((final if variant=='all' or variant=='physical' and k in ['x','m'] or variant=='semantic' and k in ['ell','c'] or variant=='mean_only' and i==0 or variant=='variance_only' and i==1 else initial)[k][i] for i in range(2)) for k in initial}
                streams={};preds={}
                for seed in [50163,80163]:
                    seed_everything(seed);vals=[];probs=[];weights=[]
                    with torch.no_grad():
                        try:
                            for _ in range(2):
                                model.draw()
                                for _ in range(4):
                                    q=[];original=model.gaussian
                                    def gaussian(*args,**kwargs):
                                        r=original(*args,**kwargs);q.append(r[1]);return r
                                    model.gaussian=gaussian
                                    try:h,logq=model.proposal(image,shifts)
                                    finally:model.gaussian=original
                                    assert len(q)==4
                                    joint,terms,p=model.log_joint(image,h);costs['density_scores']+=1
                                    parts={k:v/128**2 for k,v in terms.items() if k!='labels'}
                                    parts.update({'entropy_'+k:-v/128**2 for k,v in zip(['x','m','ell','c'],q)})
                                    elbo=(joint-logq)/128**2;assert torch.isfinite(elbo).all()
                                    assert torch.allclose(sum(parts.values()),elbo,atol=1e-5,rtol=0)
                                    parts['total']=elbo;vals.append({k:v.cpu() for k,v in parts.items()})
                                    if seed==50163:probs.append(p.cpu());weights.append((joint-logq).cpu())
                        finally:model.clear_draw()
                    scores={k:torch.stack([v[k] for v in vals]) for k in vals[0]};raw[variant+'__'+str(seed)]=scores
                    streams[str(seed)]={k:dict(mean=float(v.mean()),per_image=[float(x) for x in v.mean(0)]) for k,v in scores.items()}
                    if seed==50163:
                        w=torch.stack(weights).softmax(0);sn=(w[:,:,None,None,None]*torch.stack(probs)).sum(0)
                        with torch.no_grad():
                            h,_=model.proposal(image,shifts,sample=False);pmean=torch.cat([torch.zeros_like(h['ell'][:,:1]),h['ell']],1).softmax(1).cpu()
                        preds=dict(mean=pmean,SNIS=sn);diag=dict(ess_per_image=[float(v) for v in 1/w.square().sum(0)],max_weight_per_image=[float(v) for v in w.max(0).values])
                costs['prediction_sets']+=1;groups[variant]=dict(streams=streams,diagnostics=diag);predictions[variant]=preds
            for variant,g in groups.items():g['metrics']={k:metrics(v,y) for k,v in predictions[variant].items()}
            all_results[post]=groups;private[post]=dict(predictions=predictions,scores=raw)
        assert costs['density_scores']==192 and costs['prediction_sets']==12
        reference=json.loads((Path(s['local_gate'])/'RESULTS.json').read_text())
        for post,gg in all_results.items():
            for variant,state in [('zero','initial'),('all','final')]:
                for seed,field in [('50163',state+'_elbo'),('80163','cross_'+state+'_elbo')]:
                    assert all(abs(a-b)<1e-5 for a,b in zip(gg[variant]['streams'][seed]['total']['per_image'],reference['groups'][post][field]))
        storage(s);save(root/'PRIVATE_READOUT.pt',private)
        write(root/'RESULTS.json',dict(status='COMPLETE',costs=costs,groups=all_results,endpoint_ELBO_reproduced=True,space=storage(s)));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        write(root/'FAILURE.json',dict(traceback=traceback.format_exc(),costs=costs));write(root/'EXIT.json',dict(exit_code=1));raise


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
