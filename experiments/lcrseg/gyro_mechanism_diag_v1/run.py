"""Bounded inference-only GYRO mechanism diagnostic. No optimizer is constructed."""
import copy,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ..gyro_native_v1.core import NativeGyro
from ..space_lora_v1.core import source_model
from ..main_style_ssl_v1.data import Accessor
from ..main_style_ssl_v1.runtime import write,append
from ..philips_transfer_v1.run import evaluate
from ..siemens_lonly_v1.run import aggregate,compare,dice
from ..siemens_ssl_reproduction_v1.methods import strong_view
from ..gyro_autonomy_12h.space_guard import SpaceGuard

class Ledger:
    def __init__(self,root,guard):self.root=root;self.guard=guard;self.counts={};self.calls=0
    def call(self,kind,n,fn):
        cap={'mixed_segmentation':2176,'diagnostic_segmentation':108,'diagnostic_encoder_only':108,'synthetic_segmentation':9}[kind]
        assert self.counts.get(kind,0)+n<=cap
        if self.calls%50==0:self.guard.check(20_000_000)
        token=self.calls+1;append(self.root/'forward_events.jsonl',dict(kind='reserved',token=token,role=kind,slices=n))
        result=fn();self.calls+=1;self.counts[kind]=self.counts.get(kind,0)+n
        append(self.root/'forward_events.jsonl',dict(kind='executed',token=token,role=kind,slices=n))
        write(self.root/'FORWARD_COUNTER.json',dict(model_slices=self.counts,calls=self.calls,optimizer_calls=0))
        return result

class Counted(nn.Module):
    def __init__(self,model,ledger):super().__init__();self.model=model;self.ledger=ledger
    def forward(self,x):
        def call():
            y=self.model(x);assert torch.isfinite(y).all();return y
        return self.ledger.call('mixed_segmentation',len(x),call)

def encoder(model,x):
    m=model.parent.native;e1=m.enc1(x);e2=m.enc2(m.pool(e1));e3=m.enc3(m.pool(e2));b=m.bottleneck(m.pool(e3))
    return [e1,e2,e3,b]

def encoded(model,x,segmentation):
    h=encoder(model,x);z=model.projector(h[-1].mean((-2,-1)))
    assert torch.isfinite(z).all() and all(torch.isfinite(v).all() for v in h)
    if not segmentation:return h,z,None,None
    d=model.parent.native.decoder;local=d.dec1(d.dec2(d.dec3(h[3],h[2]),h[1]),h[0])
    logits=model.parent.native_readout(local,x.shape[-2:]);assert torch.isfinite(logits).all()
    return h,z,local,logits

def hybrid(enc,readout):
    model=copy.deepcopy(enc)
    model.parent.native.decoder.load_state_dict(readout.parent.native.decoder.state_dict())
    expected=enc.state_dict();right=readout.state_dict()
    assert all(torch.equal(v,(right[k] if k.startswith('parent.native.decoder.') else expected[k])) for k,v in model.state_dict().items())
    return model.eval().requires_grad_(False)

def cosine(a,b,dim):return float((1-F.cosine_similarity(a,b,dim=dim,eps=1e-12)).mean())
def relative(a,b):return float((b-a).double().norm()/a.double().norm().clamp_min(1e-12))
def plainrows(p):return [json.loads(l) for l in Path(p).read_text().splitlines()]
def summary(rows,keys):
    return {k:dict(mean=float(np.mean([v[k] for v in rows])),median=float(np.median([v[k] for v in rows])),max=float(max(v[k] for v in rows))) for k in keys}

def prepare_models(cfg):
    template=NativeGyro(source_model(cfg)).cuda()
    entry_state=torch.load(cfg['initial'],map_location='cpu',weights_only=False)
    assert entry_state['step']==0
    template.load_state_dict(entry_state['model']);entry=template.eval().requires_grad_(False)
    models={}
    for arm in ['G_U','M_U']:
        state=torch.load(cfg['finals'][arm],map_location='cpu',weights_only=False);assert state['step']==state['physical_updates']==2000
        model=copy.deepcopy(entry);model.load_state_dict(state['model']);model.eval().requires_grad_(False)
        assert all(torch.equal(v,entry.state_dict()[k]) for k,v in model.state_dict().items() if k.startswith('projector.') or k=='frames')
        assert all(torch.isfinite(v).all() for v in model.state_dict().values())
        models[arm]=model
    return entry,models

@torch.inference_mode()
def main(cfg):
    root=Path(cfg['root']);guard=SpaceGuard(cfg['storage_authorization']);guard.check(100_000_000)
    torch.manual_seed(163);torch.cuda.manual_seed_all(163);torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ledger=Ledger(root,guard)
    try:
        write(root/'STATUS.json',dict(status='CONTRACTS',optimizer_calls=0))
        entry,models=prepare_models(cfg)
        snapshots=[{k:v.cpu().clone() for k,v in m.state_dict().items()} for m in [entry,*models.values()]]
        x=torch.linspace(-1,1,32*32,device='cuda').reshape(1,1,32,32)
        for m in [entry,*models.values()]:
            noop=hybrid(m,m)
            y=ledger.call('synthetic_segmentation',1,lambda:m(x));z=ledger.call('synthetic_segmentation',1,lambda:noop(x))
            manual=ledger.call('synthetic_segmentation',1,lambda:encoded(m,x,True)[3])
            assert torch.equal(y,z) and torch.allclose(y,manual,atol=1e-6,rtol=1e-6)
        write(root/'CONTRACTS.json',dict(status='PASS',self_swap_exact=True,manual_encoder_decoder_matches_native=True,all_params_frozen=True,optimizer_calls=0))
        baseline=plainrows(cfg['source_val_rows']);assert aggregate(baseline)['patients']==23
        valkeys={(r['patient_id'],r['phase']) for r in Accessor(cfg['data'],'mnms','Siemens','val').rows}
        assert len(valkeys)==46 and {(r['patient'],r['phase']) for r in baseline}==valkeys
        grid={};comparisons={}
        for arm,final in models.items():
            write(root/'STATUS.json',dict(status='CROSS_EVALUATION',arm=arm,optimizer_calls=0))
            prior=plainrows(Path(cfg['finals'][arm]).parent/'private_Siemens_val.jsonl')
            cells={'E0_D0':aggregate(baseline),'E1_D1':aggregate(prior)};cellrows={'E0_D0':baseline,'E1_D1':prior}
            for name,e,d in [('E1_D0',final,entry),('E0_D1',entry,final)]:
                model=Counted(hybrid(e,d),ledger)
                result,rows=evaluate(model,cfg,'Siemens','val',root/('private_'+arm+'_'+name+'.jsonl'))
                assert result['forward_slices']==544 and {(r['patient'],r['phase']) for r in rows}==valkeys
                cells[name]=result;cellrows[name]=rows;del model
            grid[arm]=cells;comparisons[arm]={k:compare(baseline,rows) for k,rows in cellrows.items() if k!='E0_D0'}
        assert ledger.counts['mixed_segmentation']==2176
        # Diagnostic sample selection depends only on frozen volume order/depth and bank order.
        access=Accessor(cfg['data'],'mnms','Siemens','train_labeled')
        assert len(access.rows)==20 and len(access.patients)==10
        samples=[]
        for i,meta in enumerate(access.rows):
            z=access.depths[i]//2;item=access.read(i,z)
            samples.append(dict(role='source_L_center',image=item['image'],truth=item['label'],patient=meta['patient_id'],phase=meta['phase']))
        bank=torch.load(cfg['bank'],map_location='cpu',weights_only=False);assert bank['size']==256 and bank['label_fields']==[]
        selected=[i for i,(a,v,z) in enumerate(bank['ids']) if a==1][:16];assert len(selected)==16
        for i in selected:samples.append(dict(role='source_U_bank',image=bank['images'][i],truth=None,bank_index=i))
        records={a:[] for a in models};l_dice={a:[] for a in ['entry',*models]}
        for index,sample in enumerate(samples):
            write(root/'STATUS.json',dict(status='GEOMETRY_FUNCTION',sample=index,total=36,optimizer_calls=0))
            x=sample['image'][None].cuda();strong=strong_view(x,10000+index)
            observed={}
            for name,m in [('entry',entry),*models.items()]:
                weak=ledger.call('diagnostic_segmentation',1,lambda:encoded(m,x,True))
                augmented=ledger.call('diagnostic_encoder_only',1,lambda:encoded(m,strong,False))
                h,z,local,logits=weak;c=(F.normalize(z,dim=-1)+F.normalize(augmented[1],dim=-1))/2
                anchor=int((c@m.frames[0].T).argmax(1));observed[name]=(h,z,local,logits,c,anchor)
            ref=observed['entry'];pred0=ref[3].argmax(1)[0]
            for arm in models:
                h,z,local,logits,c,anchor=observed[arm];pred=logits.argmax(1)[0];union=(pred0!=0)|(pred!=0)
                row=dict(index=index,role=sample['role'],GAP_relative_change=relative(ref[0][-1].mean((-2,-1)),h[-1].mean((-2,-1))),
                    projection_centroid_cosine_drift=cosine(ref[4],c,-1),anchor_changed=anchor!=ref[5],
                    bottleneck_token_cosine_drift=cosine(ref[0][-1],h[-1],1),decoder_token_cosine_drift=cosine(ref[2],local,1),
                    prediction_disagreement=float((pred!=pred0).float().mean()),
                    foreground_union_disagreement=float((pred[union]!=pred0[union]).float().mean()) if union.any() else 0.)
                for j,key in enumerate(['enc1','enc2','enc3']):row[key+'_token_cosine_drift']=cosine(ref[0][j],h[j],1)
                row['geometry_stable']=row['projection_centroid_cosine_drift']<=.01 and not row['anchor_changed']
                row['function_drift']=row['foreground_union_disagreement']>=.1
                if sample['truth'] is not None:
                    truth=sample['truth'].cuda();fg=truth!=0
                    other=logits[0].clone();other.scatter_(0,truth[None],-float('inf'))
                    margin=logits[0].gather(0,truth[None])[0]-other.max(0).values
                    row.update(L_foreground_accuracy=float((pred[fg]==truth[fg]).float().mean()) if fg.any() else None,
                        L_foreground_accuracy_change=float(((pred[fg]==truth[fg]).float()-(pred0[fg]==truth[fg]).float()).mean()) if fg.any() else None,
                        normalized_L_foreground_margin=float((margin/logits[0].std(0).clamp_min(1e-6))[fg].mean()) if fg.any() else None)
                    l_dice[arm].append(dict(patient=sample['patient'],phase=sample['phase'],dice=dice(pred.cpu().numpy(),sample['truth'].numpy()).tolist()))
                records[arm].append(row);append(root/('private_geometry_'+arm+'.jsonl'),row)
            if sample['truth'] is not None:l_dice['entry'].append(dict(patient=sample['patient'],phase=sample['phase'],dice=dice(pred0.cpu().numpy(),sample['truth'].numpy()).tolist()))
            del observed,ref
        geometry={}
        keys=['GAP_relative_change','projection_centroid_cosine_drift','bottleneck_token_cosine_drift','decoder_token_cosine_drift','prediction_disagreement','foreground_union_disagreement']
        for arm,rows in records.items():
            geometry[arm]={}
            for role in ['source_L_center','source_U_bank']:
                selected=[r for r in rows if r['role']==role]
                flags=sum(r['geometry_stable'] and r['function_drift'] for r in selected)
                a=np.array([r['projection_centroid_cosine_drift'] for r in selected]);b=np.array([r['foreground_union_disagreement'] for r in selected])
                geometry[arm][role]=dict(samples=len(selected),metrics=summary(selected,keys),anchor_changes=sum(r['anchor_changed'] for r in selected),
                    geometry_stable_count=sum(r['geometry_stable'] for r in selected),stable_geometry_but_function_drift_count=flags,
                    descriptive_Pearson=float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else None)
        for arm,rows in records.items():
            labeled=[r for r in rows if r['role']=='source_L_center' and r['L_foreground_accuracy'] is not None]
            geometry[arm]['source_L_center'].update(foreground_accuracy_mean=float(np.mean([r['L_foreground_accuracy'] for r in labeled])) if labeled else None,
                foreground_accuracy_change_mean=float(np.mean([r['L_foreground_accuracy_change'] for r in labeled])) if labeled else None,
                normalized_foreground_margin_mean=float(np.mean([r['normalized_L_foreground_margin'] for r in labeled])) if labeled else None,labeled_slices_with_foreground=len(labeled))
        for m,before in zip([entry,*models.values()],snapshots):
            assert all(not p.requires_grad and p.grad is None for p in m.parameters())
            assert all(torch.equal(v.cpu(),before[k]) for k,v in m.state_dict().items())
        assert ledger.counts=={'synthetic_segmentation':9,'mixed_segmentation':2176,'diagnostic_segmentation':108,'diagnostic_encoder_only':108}
        events=plainrows(root/'forward_events.jsonl');r=[x for x in events if x['kind']=='reserved'];e=[x for x in events if x['kind']=='executed']
        assert r==[dict(x,kind='reserved') for x in e]
        report=dict(status='COMPLETE',optimizer_calls=0,old_model_states_exact=True,frames_projectors_exact=True,all_params_frozen=True,
            cross_grid=grid,cross_paired=comparisons,geometry=geometry,center_slice_Dice={a:aggregate(rows) for a,rows in l_dice.items()},
            forward_model_slices=ledger.counts,real_segmentation_model_slices=2284,real_encoder_only_model_slices=108,synthetic_model_slices=9,
            no_U_GT_read=True,no_test_access=True,source_L_center_slices=20,source_U_bank_images=16,
            validation_only=True,additional_training_started=False,CARE_independent=True,BayeSeg_independent=True,
            limits=['Hybrid mismatch is not an additive causal decomposition','Center-slice Dice is not full-volume source training Dice','Stable projected centroid does not prove all geometry stable','Descriptive sample correlations are not independent patient/seed evidence'],storage=guard.check())
        write(root/'PUBLIC_SUMMARY.json',report);write(root/'STATUS.json',dict(status='COMPLETE',optimizer_calls=0));write(root/'EXIT.json',dict(exit_code=0));print(json.dumps(report))
    except BaseException as exc:
        write(root/'FAILURE.json',dict(error=str(exc),optimizer_calls=0,forward_model_slices=ledger.counts));write(root/'EXIT.json',dict(exit_code=1));raise
