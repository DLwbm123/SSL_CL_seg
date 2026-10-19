"""Cross inference and generator posteriors without updating any state."""
import json,os,traceback
from pathlib import Path
import torch
from torch.nn import functional as F
from lcrseg.data import H5LabeledDataset,collate_labeled
from lcrseg.engine.trainer import seed_everything
from care_seg_study_v1.run import write,save,append
from sc_bayeseg.split_v1.model import Model
from sc_bayeseg.run import same
from sc_bayeseg.window_v1.common import storage


def summarize(v):
    return {name:dict(mean=float(v[:,ix].mean()),per_image=v[:,ix].mean(0).tolist(),positive_fraction=float((v[:,ix]>0).float().mean())) for name,ix in [('train',slice(0,2)),('heldout',slice(2,4))]}


def run(s):
    root=Path(s['output']);root.mkdir(exist_ok=False);costs=dict(density=0,optimizer=0,reverse=0,local_updates=0)
    try:
        storage(s);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        assert json.loads((Path(s['view_guard'])/'EXIT.json').read_text())['exit_code']==0
        assert len((Path(s['view_guard'])/'physical_updates.jsonl').read_text().splitlines())==104
        sources={name:torch.load(path,map_location='cpu',weights_only=False) for name,path in s['readout_states'].items()};assert set(sources)=={'K','G'}
        sel=json.loads((Path(s['origin'])/'PRIVATE_SELECTION.json').read_text());assert len(set(sel['identities']))==4
        ds=H5LabeledDataset(s['data'],seed=0,dataset='mnms',sites=['Siemens'],transform=None)
        lb=collate_labeled([ds[i] for i in sel['train_indices']+sel['heldout_indices']]).to('cuda')
        image=F.interpolate(lb.image,(128,128),mode='bilinear',align_corners=False);y=F.interpolate(lb.label[:,None].float(),(128,128),mode='nearest')[:,0].long()
        model=Model().cuda();model.requires_grad_(False);private={};groups={};spectrum={}
        components=['shape','logits','structural_total','code','semantic_joint']
        for local,source in sources.items():
            model.load_state_dict(source['model'])
            with torch.no_grad():
                h,_=model.proposal(image,sample=False);h={k:v.detach() for k,v in h.items()}
                z=torch.cat([torch.zeros_like(h['ell'][:,:1]),h['ell']],1).double();own=z.log_softmax(1)
                pred=z.argmax(1);top=z.gather(1,pred[:,None]);truth=z.gather(1,y[:,None]);oracle=z.scatter(1,y[:,None],top).scatter(1,pred[:,None],truth)
                oe=oracle[:,1:]-oracle[:,:1];op=torch.cat([torch.zeros_like(oe[:,:1]),oe],1).softmax(1)
                err=float((z.softmax(1).sort(1).values-op.sort(1).values).abs().max());assert err<1e-12
                oe=oe.float();actualp=torch.cat([torch.zeros_like(h['ell'][:,:1]),h['ell']],1).softmax(1)
                fp=torch.cat([torch.zeros_like(oe[:,:1]),oe],1).softmax(1)
                spectrum[local]=dict(FP64_max_error=err,FP32_max_error=float((actualp.sort(1).values-fp.sort(1).values).abs().max()),oracle_accuracy=float((fp.argmax(1)==y).float().mean()))
                gt=(F.one_hot(y,4).permute(0,3,1,2).float()*.99+.01/4).log()
            for post in ('K','G','prior'):
                sd=source['model'].copy();q=sources['K' if post=='prior' else post]['model']
                for key in sd:
                    if key.startswith(('anatomy.','structure.')):
                        prefix,leaf=key.rsplit('.',1)
                        if post=='prior' and leaf in ('mean_w','mean_b','logsd_w','logsd_b'):
                            assert torch.equal(q[prefix+'.prior_'+leaf],q[prefix+'.initial_'+leaf]);sd[key]=q[prefix+'.prior_'+leaf]
                        else:sd[key]=q[key]
                model.load_state_dict(sd);before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                assert all(torch.equal(v.cpu(),source['model'][k]) for k,v in model.state_dict().items() if not k.startswith(('anatomy.','structure.')))
                for template,base in [('GT',gt),('own',own)]:
                    patterns=[base,torch.roll(base,(0,32),(-2,-1)),torch.roll(base,(32,0),(-2,-1)),torch.roll(base,(64,64),(-2,-1)),base[:,[0,2,3,1]],base[:,[0,3,1,2]]]
                    ells=[(v[:,1:]-v[:,:1]).float() for v in patterns]
                    if template=='own':ells.append(oe)
                    for code in ('fixed','reencoded'):
                        name=f'{local}__{post}__{template}__{code}';rows=[]
                        for draw in range(8):
                            seed_everything(80163+draw);model.draw();row=[]
                            try:
                                with torch.no_grad():
                                    for pattern,ell in enumerate(ells):
                                        hh=h.copy();hh['ell']=ell
                                        if code=='reencoded':hh['c']=model.gaussian(model.code_encoder(ell),sample=False)[0]
                                        _,terms,_=model.log_joint(image,hh)
                                        score=torch.stack([terms['shape'],terms['logits'],terms['shape']+terms['logits'],terms['code'],terms['code']+terms['shape']+terms['logits']],-1)/128**2
                                        assert torch.isfinite(score).all();row.append(score.cpu())
                                        costs['density']+=1;append(root/'density_calls.jsonl',dict(index=costs['density'],local=local,post=post,template=template,code=code,draw=draw,pattern=pattern))
                                        if costs['density']%64==0:write(root/'STATUS.json',dict(status='RUNNING',costs=costs,space=storage(s)))
                            finally:model.clear_draw()
                            rows.append(torch.stack(row,-1))
                        score=torch.stack(rows);private[name]=score;out={}
                        for i,c in enumerate(components):
                            a=score[:,:,i];out[c]={}
                            for family,ix in [('spatial',[1,2,3]),('identity',[4,5])]:out[c][family]=summarize(a[:,:,0]-a[:,:,ix].mean(-1))
                            if template=='own':out[c]['oracle_minus_actual']=summarize(a[:,:,6]-a[:,:,0])
                        groups[name]=out
                assert same(before,model.state_dict())
        assert costs==dict(density=1248,optimizer=0,reverse=0,local_updates=0)
        assert len((Path(s['view_guard'])/'physical_updates.jsonl').read_text().splitlines())==104
        storage(s);save(root/'PRIVATE_SCORES.pt',private)
        write(root/'RESULTS.json',dict(status='COMPLETE',costs=costs,groups=groups,spectrum=spectrum,all_model_tensors_unchanged=True,
              source_physical_ledger_unchanged=True,label_likelihood=False,space=storage(s)));write(root/'EXIT.json',dict(exit_code=0))
    except BaseException:
        write(root/'FAILURE.json',dict(costs=costs,traceback=traceback.format_exc()));write(root/'EXIT.json',dict(exit_code=1));raise


if __name__=='__main__':run(json.loads(Path(os.environ['TASK_SETTINGS']).read_text()))
