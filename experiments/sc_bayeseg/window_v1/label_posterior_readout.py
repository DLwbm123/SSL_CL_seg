"""Single fixed inference anchor, three paired generator posteriors."""
import json,os,traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset,collate_labeled
from lcrseg.engine.trainer import seed_everything
from care_seg_study_v1.run import write,append,save
from sc_bayeseg.split_v1.model import Model
from sc_bayeseg.run import same
from sc_bayeseg.window_v1.common import storage
from sc_bayeseg.window_v1.view_posterior import summarize


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);costs=dict(density=0,optimizer=0,reverse=0,local_updates=0)
    try:
        storage(s);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        pair=Path(s['label_pair']);assert json.loads((pair/'EXIT.json').read_text())['exit_code']==0
        assert len((pair/'physical_updates.jsonl').read_text().splitlines())==1208
        states={name:torch.load(path,map_location='cpu',weights_only=False)['model'] for name,path in s['posterior_states'].items()}
        anchor=torch.load(s['readout_states']['G'],map_location='cpu',weights_only=False)['model']
        sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text());assert len(set(sel['identities']))==4
        ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        lb=collate_labeled([ds[i] for i in sel['train_indices']+sel['heldout_indices']]).to('cuda')
        image=F.interpolate(lb.image,(128,128),mode='bilinear',align_corners=False);y=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
        model=Model().cuda();model.requires_grad_(False);model.load_state_dict(anchor)
        with torch.no_grad():
            h,_=model.proposal(image,sample=False);h={k:v.detach() for k,v in h.items()}
            z=torch.cat([torch.zeros_like(h['ell'][:,:1]),h['ell']],1).double();own=z.log_softmax(1)
            pred=z.argmax(1);oracle=z.scatter(1,y[:,None],z.gather(1,pred[:,None])).scatter(1,pred[:,None],z.gather(1,y[:,None]))
            oracle_ell=(oracle[:,1:]-oracle[:,:1]);oracle_p=torch.cat([torch.zeros_like(oracle_ell[:,:1]),oracle_ell],1).softmax(1)
            err=float((z.softmax(1).sort(1).values-oracle_p.sort(1).values).abs().max());assert err<1e-12
            oracle_ell=oracle_ell.float();gt=(F.one_hot(y,4).permute(0,3,1,2).float()*.99+.01/4).log()
        raw={};groups={};components=['shape','logits','structural_total','code','semantic_joint']
        for post,sd in states.items():
            assert all(torch.equal(v,anchor[k]) for k,v in sd.items() if not k.startswith(('anatomy.','structure.')))
            model.load_state_dict(sd)
            for template,base in [('GT',gt),('own',own)]:
                pp=[base,torch.roll(base,(0,32),(-2,-1)),torch.roll(base,(32,0),(-2,-1)),torch.roll(base,(64,64),(-2,-1)),base[:,[0,2,3,1]],base[:,[0,3,1,2]]]
                ell=[(v[:,1:]-v[:,:1]).float() for v in pp]
                if template=='own':ell.append(oracle_ell)
                rows=[]
                for draw in range(8):
                    seed_everything(80163+draw);model.draw();parts=[]
                    try:
                        with torch.no_grad():
                            for index,e in enumerate(ell):
                                hh=h.copy();hh['ell']=e;hh['c']=model.gaussian(model.code_encoder(e),sample=False)[0]
                                _,t,_=model.log_joint(image,hh)
                                v=torch.stack([t['shape'],t['logits'],t['shape']+t['logits'],t['code'],t['code']+t['shape']+t['logits']],-1)/128**2
                                assert torch.isfinite(v).all();parts.append(v.cpu());costs['density']+=1
                                append(root/'density_calls.jsonl',dict(index=costs['density'],posterior=post,template=template,draw=draw,pattern=index))
                                if costs['density']%32==0:write(root/'STATUS.json',dict(status='RUNNING',costs=costs,space=storage(s)))
                    finally:model.clear_draw()
                    rows.append(torch.stack(parts,-1))
                scores=torch.stack(rows);raw[post+'__'+template]=scores
                for i,c in enumerate(components):
                    a=scores[:,:,i];d={'spatial':a[:,:,0]-a[:,:,[1,2,3]].mean(-1),'identity':a[:,:,0]-a[:,:,[4,5]].mean(-1)}
                    if template=='own':d['oracle_minus_actual']=a[:,:,6]-a[:,:,0]
                    groups.setdefault(post+'__'+template,{})[c]={f:summarize(v) for f,v in d.items()}
            assert same(model.state_dict(),sd)
        contrasts={}
        for a,b in [('L','U'),('U','G'),('L','G')]:
            for template in ('GT','own'):
                score=raw[a+'__'+template]-raw[b+'__'+template];out={}
                for i,c in enumerate(components):
                    v=score[:,:,i];d={'spatial':v[:,:,0]-v[:,:,[1,2,3]].mean(-1),'identity':v[:,:,0]-v[:,:,[4,5]].mean(-1)}
                    if template=='own':d['oracle_minus_actual']=v[:,:,6]-v[:,:,0]
                    out[c]={f:summarize(z) for f,z in d.items()}
                contrasts[a+'_minus_'+b+'__'+template]=out
        assert costs==dict(density=312,optimizer=0,reverse=0,local_updates=0)
        assert len((pair/'physical_updates.jsonl').read_text().splitlines())==1208
        storage(s);save(root/'PRIVATE_SCORES.pt',raw)
        write(root/'RESULTS.json',dict(status='COMPLETE',costs=costs,groups=groups,contrasts=contrasts,
              all_models_unchanged=True,fixed_inference=True,source_physical_ledger_unchanged=True,FP64_spectrum_error=err,space=storage(s)))
        write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
