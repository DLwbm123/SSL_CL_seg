"""Frozen model inference ablation with paired final sampling streams."""
import json,os,traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset,collate_labeled
from lcrseg.engine.trainer import seed_everything
from lcrseg.engine.checkpoint import capture_rng_state,restore_rng_state
from care_seg_study_v1.run import write,save,append
from sc_bayeseg.window_v1.mesh_model import Model
from sc_bayeseg.guard_v1.run import dice
from sc_bayeseg.window_v1.common import storage


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);costs=dict(model_optimizer=0,model_parameter_updates=0,local_reverse=0,local_updates=0,prediction_calls=0)
    try:
        storage(s);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text());assert len(set(sel['identities']))==4 and not set(sel['train_indices'])&set(sel['heldout_indices'])
        ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        lb=collate_labeled([ds[i] for i in sel['train_indices']+sel['heldout_indices']]).to('cuda')
        image=F.interpolate(lb.image,(128,128),mode='bilinear',align_corners=False);y=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
        states={'terminal':s['states']['terminal']};results={};private={}
        for local,path in states.items():
            payload=torch.load(path,map_location='cpu',weights_only=False)
            for post,qpath in {**states,'prior':s['anchor']}.items():
                q=torch.load(qpath,map_location='cpu',weights_only=False);sd=payload['model'].copy()
                for key in sd:
                    if key.startswith(('anatomy.','structure.')):
                        prefix,leaf=key.rsplit('.',1)
                        sd[key]=q['model'][prefix+'.prior_'+leaf] if post=='prior' and leaf in ['mean_w','mean_b','logsd_w','logsd_b'] else q['model'][key]
                model=Model().cuda();model.load_state_dict(sd);before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};key=local+'__'+post;results[key]={};private[key]={}
                for mode,corrected,refine in [('SNIS',True,0),('mesh3',True,3),('mesh25',True,25)]:
                    seed_everything(50163);rng=capture_rng_state();original=model.draw;draws=[0]
                    def draw():
                        if refine and draws[0]==refine:restore_rng_state(rng)
                        draws[0]+=1;original()
                    def audit_callback(kind,row):
                        if kind=='reverse':
                            costs['local_reverse']+=1
                            append(root/'reverse_calls.jsonl',dict(index=costs['local_reverse'],group=key,mode=mode,**row))
                        else:
                            costs['local_updates']+=1
                            append(root/'local_updates.jsonl',dict(index=costs['local_updates'],group=key,mode=mode,**row))
                    model.audit_callback=audit_callback
                    model.draw=draw;calls=model.calls.copy()
                    try:p,diag=model.prediction(image,corrected=corrected,refine=refine,parameter_samples=2,local_samples=4)
                    finally:model.draw=original
                    assert torch.isfinite(p).all() and all(torch.equal(v.cpu(),before[k]) for k,v in model.state_dict().items())
                    n=model.calls['refinement_derivatives']-calls['refinement_derivatives'];assert n==refine
                    costs['prediction_calls']+=1
                    assert draws[0]==(refine+2 if corrected else 0)
                    private[key][mode]=p.detach().cpu();metrics={}
                    for group,ix in [('train',slice(0,2)),('heldout',slice(2,4))]:
                        pp,yy=p[ix],y[ix];pred=pp.argmax(1);ce=-pp.clamp_min(1e-30).log().gather(1,yy[:,None])[:,0]
                        metrics[group]=dict(dice=dice(pred,yy),ce=float(ce.mean()),class2_ce=float(ce[yy==2].mean()),class2_confusion=[int(((yy==2)&(pred==c)).sum()) for c in range(4)])
                    results[key][mode]=dict(metrics=metrics,diagnostics=diag,draws=draws[0],local_reverse=n,model_unchanged=True,mesh=model.mesh_diagnostics)
                    write(root/'STATUS.json',dict(phase='INFERENCE',local=local,posterior=post,mode=mode,costs=costs))
        assert costs['prediction_calls']==6 and costs['local_reverse']==costs['local_updates']==56
        storage(s);save(root/'PRIVATE_PREDICTIONS.pt',private)
        write(root/'RESULTS.json',dict(status='COMPLETE',costs=costs,groups=results,space=storage(s),no_label_in_prediction=True,all_models_unchanged=True,paired_final_sampling=True));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        write(root/'FAILURE.json',dict(traceback=traceback.format_exc(),costs=costs));write(root/'EXIT.json',dict(exit_code=1));raise


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
