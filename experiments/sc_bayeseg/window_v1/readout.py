"""Fixed-initial-inference generative posterior readout; no updates."""
import json,os,time,traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset,collate_labeled
from lcrseg.engine.trainer import seed_everything
from care_seg_study_v1.run import write,save
from sc_bayeseg.split_v1.model import Model
from sc_bayeseg.window_v1.common import storage


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False)
    try:
        storage(s);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text());lb=collate_labeled([ds[i] for i in sel['train_indices']]).to('cuda')
        image=F.interpolate(lb.image,(128,128),mode='bilinear',align_corners=False);y=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
        initial=torch.load(s['anchor'],map_location='cpu',weights_only=False);model=Model().cuda();model.load_state_dict(initial['model'])
        with torch.no_grad():anchor,_=model.proposal(image,sample=False)
        anchor={k:v.detach() for k,v in anchor.items()};p=F.one_hot(y,4).permute(0,3,1,2).float()*.99+.01/4
        patterns=[p,torch.roll(p,(0,32),(-2,-1)),torch.roll(p,(32,0),(-2,-1)),torch.roll(p,(64,64),(-2,-1)),p[:,[0,2,3,1]],p[:,[0,3,1,2]]]
        scores={};states={'initial':s['anchor'],**s['states']}
        for name,path in states.items():
            q=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(initial['model']);sd=model.state_dict()
            for key in sd:
                if key.startswith(('anatomy.','structure.')):sd[key]=q['model'][key]
            model.load_state_dict(sd)
            assert all(torch.equal(v.cpu(),initial['model'][k]) for k,v in model.state_dict().items() if not k.startswith(('anatomy.','structure.')))
            rows=[]
            for draw in range(8):
                seed_everything(80163+draw);model.draw();parts=[]
                try:
                    with torch.no_grad():
                        for pp in patterns:
                            h=anchor.copy();h['ell']=(pp[:,1:]/pp[:,:1]).log();_,terms,_=model.log_joint(image,h)
                            parts.append(torch.stack([terms['shape'],terms['logits'],terms['shape']+terms['logits']],-1)/128**2)
                    rows.append(torch.stack(parts,-1).cpu())
                finally:model.clear_draw()
            scores[name]=torch.stack(rows)
        out={}
        for name,score in scores.items():
            out[name]={}
            for i,component in enumerate(['shape','logits','total']):
                out[name][component]={}
                for family,ix in [('spatial',[1,2,3]),('identity',[4,5])]:
                    m=score[:,:,i,0]-score[:,:,i,ix].mean(-1)
                    out[name][component][family]=dict(mean=float(m.mean()),per_image=[float(x) for x in m.mean(0)],positive_fraction=float((m>0).float().mean()))
        storage(s);save(root/'PRIVATE_SCORES.pt',scores)
        write(root/'RESULTS.json',dict(status='COMPLETE',optimizer_updates=0,reverse_calls=0,fixed_inference='J initial',fixed_local='initial mean x/m/c',label_likelihood=False,groups=out,calls=model.calls,space=storage(s)))
        write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        write(root/'FAILURE.json',dict(traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
