"""Crossed inference/generator readout of full discrete labeled local bounds."""
import json,os,traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset,collate_labeled
from lcrseg.engine.trainer import seed_everything
from lcrseg.engine.checkpoint import capture_rng_state,restore_rng_state
from care_seg_study_v1.run import write,append,save
from sc_bayeseg.run import same
from sc_bayeseg.window_v1.discrete_labeled import DiscreteLabeledModel
from sc_bayeseg.window_v1.common import storage
from sc_bayeseg.window_v1.view_posterior import summarize


def margins(v):
    return dict(GT_spatial=v[:,:,0]-v[:,:,[1,2,3]].mean(-1),GT_identity=v[:,:,0]-v[:,:,[4,5]].mean(-1),
                own_spatial=v[:,:,6]-v[:,:,[7,8,9]].mean(-1),own_identity=v[:,:,6]-v[:,:,[10,11]].mean(-1),
                GT_minus_own=v[:,:,0]-v[:,:,6],GT_minus_background=v[:,:,0]-v[:,:,12],own_minus_background=v[:,:,6]-v[:,:,12])


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);costs=dict(bound=0,generator_draws=0,optimizer=0,reverse=0,local_updates=0)
    try:
        storage(s);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        origin=Path(s['discrete_train']);assert json.loads((origin/'EXIT.json').read_text())['exit_code']==0
        assert len((origin/'physical_updates.jsonl').read_text().splitlines())==604
        states={str(k):torch.load(origin/f'S{k:04}.pt',map_location='cpu',weights_only=False)['model'] for k in (0,600)}
        sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text());assert len(set(sel['identities']))==4
        ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        lb=collate_labeled([ds[i] for i in sel['train_indices']+sel['heldout_indices']]).to('cuda')
        image=F.interpolate(lb.image,(128,128),mode='bilinear',align_corners=False);y=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
        model=DiscreteLabeledModel().cuda();model.requires_grad_(False);model.load_state_dict(states['0'])
        with torch.no_grad():own=model.recognition_logits(image).argmax(1)
        templates=[];names=[]
        for name,z in [('GT',y),('own',own)]:
            templates.extend([z,torch.roll(z,(0,32),(-2,-1)),torch.roll(z,(32,0),(-2,-1)),torch.roll(z,(64,64),(-2,-1)),torch.tensor([0,2,3,1],device=z.device)[z],torch.tensor([0,3,1,2],device=z.device)[z]])
            names.extend([name+'_'+v for v in ['base','roll0_32','roll32_0','roll64_64','cycle123','cycle132']])
        templates.append(torch.zeros_like(y));names.append('all_background')
        components=['local_bound','categories','shape','image','code','appearance','negative_logq'];raw={};groups={};kl={}
        for psi,ps in states.items():
            for theta,ts in states.items():
                sd=ps.copy()
                for key in sd:
                    if key.startswith(('anatomy.','structure.')):sd[key]=ts[key]
                model.load_state_dict(sd);key=f'psi{psi}_theta{theta}';rows=[]
                with torch.no_grad():
                    assert torch.equal(model.recognition_logits(image).argmax(1),own);kl[key]=float(model.kl())
                for draw in range(8):
                    seed_everything(80163+draw);model.draw();costs['generator_draws']+=1;noise=capture_rng_state();parts=[];end_noise=None
                    try:
                        with torch.no_grad():
                            for index,z in enumerate(templates):
                                restore_rng_state(noise);b,t,d=model.bound(image,z)
                                if end_noise is None:end_noise=capture_rng_state()
                                else:assert same(end_noise,capture_rng_state())
                                v=torch.stack([b,t['categories'],t['shape'],t['image'],t['code'],t['appearance'],-d['logq']],-1)/128**2
                                assert torch.isfinite(v).all();parts.append(v.cpu());costs['bound']+=1
                                append(root/'bound_calls.jsonl',dict(index=costs['bound'],psi=psi,theta=theta,draw=draw,template=index))
                                if costs['bound']%32==0:write(root/'STATUS.json',dict(status='RUNNING',costs=costs,space=storage(s)))
                    finally:model.clear_draw()
                    rows.append(torch.stack(parts,-1))
                score=torch.stack(rows);raw[key]=score
                groups[key]={c:{name:summarize(v) for name,v in margins(score[:,:,i]).items()} for i,c in enumerate(components)}
                assert same(sd,model.state_dict())
        effects={}
        pairs=[('theta_at_psi0','psi0_theta600','psi0_theta0'),('theta_at_psi600','psi600_theta600','psi600_theta0'),
               ('psi_at_theta0','psi600_theta0','psi0_theta0'),('psi_at_theta600','psi600_theta600','psi0_theta600')]
        for name,a,b in pairs:
            diff=raw[a]-raw[b];effects[name]={c:{m:summarize(v) for m,v in margins(diff[:,:,i]).items()} for i,c in enumerate(components)}
        diff=raw['psi600_theta600']-raw['psi600_theta0']-raw['psi0_theta600']+raw['psi0_theta0']
        effects['interaction']={c:{m:summarize(v) for m,v in margins(diff[:,:,i]).items()} for i,c in enumerate(components)}
        assert costs==dict(bound=416,generator_draws=32,optimizer=0,reverse=0,local_updates=0)
        assert len((origin/'physical_updates.jsonl').read_text().splitlines())==604
        storage(s);save(root/'PRIVATE_SCORES.pt',raw)
        write(root/'RESULTS.json',dict(status='COMPLETE',costs=costs,templates=names,groups=groups,effects=effects,global_KL_separate=kl,
              all_model_tensors_unchanged=True,source_physical_ledger_unchanged=True,template_noises_paired=True,space=storage(s)))
        write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
