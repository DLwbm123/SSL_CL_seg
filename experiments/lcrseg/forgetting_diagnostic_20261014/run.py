"""Finite checkpoint/seen-val forensics; never constructs or steps an optimizer."""
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from experiments.lcrseg.five_frameworks_v1.native_parent import build, NativeLRParent
from experiments.lcrseg.five_frameworks_v1.model import Deployment
from experiments.lcrseg.lctx_weight_memory_v0_1 import core as lr
from experiments.lcrseg.main_style_ssl_v1.model import Parent, MainModel, style, weighted_kl
from experiments.lcrseg.main_style_ssl_v1.protocol import MAPS, SHAPES, ORDERS
from experiments.lcrseg.main_style_ssl_v1.data import Accessor
from experiments.lcrseg.five_frameworks_v1.reliability import CurrentPrototypes
from experiments.lcrseg.f5_class_reference_v13.model import class_reference

ROOT = Path(os.environ['RUN_ROOT'])
OUT = Path(os.environ['DIAG_ROOT'])
DEVICE = torch.device('cuda:0')
REFERENCE = str(ROOT.parent.parent/'reference/JASCL_REFERENCE')
DATA = os.environ.get('DATA_ROOT', str(ROOT.parent.parent/'data'))
COUNTS = dict(forwards=0, backward=0, optimizer=0, test=0)


def native(node, state, prefix):
    model = build(REFERENCE, DEVICE, node['seed'], input_channels=SHAPES[node['dataset']][0],
                  num_classes=len(MAPS[node['dataset']]), domain=ORDERS[node['dataset']][0][0])
    dense = {k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)
             and not k.endswith(('.A', '.B', '.right', '.left'))}
    model.load_state_dict(dense, strict=True)
    return model


def checkpoint_model(node, state):
    # Restore saved tensors directly, without recomputing a stage-entry SVD or bypassing training restore guards.
    model = native(node, state, 'parent.native.')
    parent = Parent.__new__(Parent)
    parent.source_dense = False
    NativeLRParent.__init__(parent, model, node['seed'], {}, adapt=False)
    if any(k.endswith('.A') for k in state if k.startswith('parent.native.')):
        def tensors(name):
            p = 'parent.native.'+name[:-7]
            a, v = state[p+'.A'], state[p+'.right']
            return dict(W0=state[p+'.weight'].cpu(), G=a.cpu().double(), V=v.cpu().double(),
                        U=torch.zeros(state[p+'.weight'].shape[0], v.shape[1], dtype=torch.float64))
        lr.configure(model, 'LR_SRC_A', tensors)
        for i in range(len(lr.LAYERS)):
            parent.register_buffer('V64_'+str(i), state['parent.V64_'+str(i)].to(DEVICE))
    result = MainModel(parent, MAPS[node['dataset']], previous=state['sidecar.previous'].to(DEVICE),
                       initialize_basis=False).to(DEVICE)
    result.load_state_dict(state, strict=True)
    result.role = 'eval'
    return result.eval().requires_grad_(False)


def sealed_model(node, saved):
    model = native(node, saved['native'], '')
    parent = NativeLRParent(model, node['seed'], {}, adapt=False).eval().requires_grad_(False)
    return Deployment(parent, saved['transform'].to(DEVICE)).eval().requires_grad_(False)


def forward(model, images, features=False):
    COUNTS['forwards'] += 1
    if COUNTS['forwards'] > 3500:
        raise RuntimeError('finite diagnostic forward cap exceeded')
    if features:
        _, _, h = model.parts(images, mode='eval')
        logits = model.parent.native_readout(h, images.shape[-2:], 'eval')
        return logits.softmax(1), h
    return model(images).softmax(1)


def metrics(q, y):
    pred = q.argmax(1).cpu().numpy()
    y = np.asarray(y)
    output = []
    for c in range(1,q.shape[1]):
        p, t = pred == c, y == c
        tp, fp, fn = int((p&t).sum()), int((p&~t).sum()), int((~p&t).sum())
        output.append(dict(class_index=c, Dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
                           recall=tp/max(1,tp+fn), precision=tp/max(1,tp+fp),
                           predicted_pixels=int(p.sum()), truth_pixels=int(t.sum())))
    return output


def probe_gradient(node, model, ema, bank, accessor):
    # Fixed first two validation patients and their middle slices; labels score a graph diagnostic only.
    x, y = [], []
    for i in range(2):
        z = None if node['dataset']=='fundus' else accessor.depths[i]//2
        item = accessor.read(i,z);x.append(item['image']);y.append(item['label'])
    x = torch.stack(x).to(DEVICE);y = torch.stack(y).to(DEVICE)
    model.configure_training()
    with torch.no_grad():
        tq, h = forward(ema,x,True);anchor=forward(model.entry_anchor,x)
        geom=torch.ones_like(tq[:,0],dtype=torch.bool)
        pas,_=bank.admission(ema,tq,h,geom,.7,.5)
        valid=model.parent_map and geom&(tq.max(1).values>.7)&((tq.argmax(1)>0)|pas)
    logits=model(style(x,x.flip(0),(-1,1) if node['dataset']=='prostate' else (0,1)),
                 detach_parent=model.detach_u_parent())
    COUNTS['forwards'] += 1
    _,kept,_=weighted_kl(logits,tq,valid,anchor)
    support=kept&(anchor.max(1).values>=.7)&(anchor.argmax(1)==tq.argmax(1))
    ref, pixels=class_reference(logits,anchor,support)
    params=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
    grads=torch.autograd.grad(ref,[p for _,p in params],allow_unused=True)
    COUNTS['backward'] += 1
    norms={'A':0.,'B':0.,'R':0.};nonzero={'A':0,'B':0,'R':0}
    for (name,p),g in zip(params,grads):
        group='A' if name.endswith('.A') else 'B' if name.endswith('.B') else 'R'
        if g is not None:
            assert torch.isfinite(g).all();norms[group]+=float(g.double().square().sum());nonzero[group]+=int(g.norm()>0)
    model.eval().requires_grad_(False)
    return dict(reference_class_pixels=pixels,reference_loss=float(ref.detach()),
                gradient_norms={k:v**.5 for k,v in norms.items()},nonzero_tensors=nonzero,
                styled_U_parent_detached=model.detach_u_parent(),clean_SWD_parent_detached=model.detach_u_parent(clean=True),
                optimizer_calls=0,scope='fixed validation graph probe; not a recorded training minibatch')


def main():
    torch.set_num_threads(2)
    plan=json.loads((ROOT/'PLAN.json').read_text())['nodes']
    keys=['prostate_s163_o1_t2','prostate_s164_o1_t2','prostate_s163_o1_t4','prostate_s164_o1_t4',
          'fundus_s163_o2_t2','fundus_s164_o2_t2']
    if os.environ.get('DIAG_KEYS'): keys=json.loads(os.environ['DIAG_KEYS'])
    started=time.time();results=[]
    for key in keys:
        node=next(n for n in plan if n['id']==key);folder=ROOT/'jobs'/key
        assert json.loads((folder/'EXIT.json').read_text())['exit_code']==0
        cp=torch.load(folder/'final.pt',map_location=DEVICE,weights_only=False)
        sealed=torch.load(folder/'sealed.pt',map_location=DEVICE,weights_only=False)
        student=checkpoint_model(node,cp['student']);ema=checkpoint_model(node,cp['ema']);deploy=sealed_model(node,sealed)
        bank=CurrentPrototypes(len(MAPS[node['dataset']]),16);bank.values=cp['prototypes'];bank.support=cp['support']
        domains=(['RUNMC','BMC','I2CVB'] if node['stage']==2 else ['I2CVB','HK','UCL']) if node['dataset']=='prostate' else json.loads(os.environ.get('FUNDUS_DOMAINS', '["REFUGE","Drishti_GS","RIM_ONE_r3"]'))
        cells=[];accessors={}
        for domain in domains:
            access=Accessor(DATA,node['dataset'],domain,'val');accessors[domain]=access
            indices=list(range(len(access.rows))) if node['dataset']=='prostate' or os.environ.get('FULL_FUNDUS_VAL')=='1' else np.linspace(0,len(access.rows)-1,min(8,len(access.rows)),dtype=int).tolist()
            patients=[];totals=dict(student_fg=0,entry_fg=0,ema_fg=0,truth_fg=0,accepted_fg=0,
                                   accepted_fg_wrong=0,confident_bg_on_truth_fg=0,truth_fg_pixels=0,
                                   reference_fg=0,reference_bg=0,kept_pixels=0,confident_bg_pixels=0,
                                   confident_bg_wrong=0,total_pixels=0)
            seal_error=0.;seal_disagreement=0
            with torch.inference_mode():
                for i in indices:
                    item=access.read(i);im=item['image'];truth=item['label'].numpy()
                    batches=[im[None]] if node['dataset']=='fundus' else [im[z:z+2,None] for z in range(0,len(im),2)]
                    outputs={k:[] for k in ['student','ema','entry']}
                    for b in batches:
                        b=b.to(DEVICE);q,h=forward(ema,b,True);aq=forward(student.entry_anchor,b);sq=forward(deploy,b)
                        # Check both saved deployment and pre-seal checkpoint on every validation batch.
                        pq=forward(student,b);seal_error=max(seal_error,float((pq-sq).abs().max()));seal_disagreement+=int((pq.argmax(1)!=sq.argmax(1)).sum())
                        geom=torch.ones_like(q[:,0],dtype=torch.bool);pas,_=bank.admission(ema,q,h,geom,.7,.5)
                        valid=geom&(q.max(1).values>.7)&((q.argmax(1)>0)|pas)
                        _,kept,stats=weighted_kl(torch.log(sq.clamp_min(1e-30)),q,valid,aq)
                        ref=kept&(aq.max(1).values>=.7)&(aq.argmax(1)==q.argmax(1))
                        outputs['student'].append(sq.cpu());outputs['ema'].append(q.cpu());outputs['entry'].append(aq.cpu())
                        size=q.shape[0];offset=sum(x.shape[0] for x in outputs['ema'][:-1])
                        y=torch.as_tensor(truth if node['dataset']=='fundus' else truth[offset:offset+size],device=DEVICE)
                        if y.ndim==2:y=y[None]
                        pred=q.argmax(1);fg=(pred>0)&kept;bg=(pred==0)&(q.max(1).values>.7)
                        totals['accepted_fg']+=int(fg.sum());totals['accepted_fg_wrong']+=int((fg&(pred!=y)).sum())
                        totals['confident_bg_on_truth_fg']+=int((bg&(y>0)).sum());totals['truth_fg_pixels']+=int((y>0).sum())
                        totals['reference_fg']+=int((ref&(aq.argmax(1)>0)).sum());totals['reference_bg']+=int((ref&(aq.argmax(1)==0)).sum());totals['kept_pixels']+=int(kept.sum())
                        totals['confident_bg_pixels']+=int(bg.sum());totals['confident_bg_wrong']+=int((bg&(y>0)).sum());totals['total_pixels']+=y.numel()
                    patient={role:metrics(torch.cat(chunks),truth if node['dataset']=='prostate' else truth[None]) for role,chunks in outputs.items()}
                    patients.append(patient)
                    for role,chunks in outputs.items():totals[role+'_fg']+=int((torch.cat(chunks).argmax(1)>0).sum())
                    totals['truth_fg']+=int((truth>0).sum())
            classes={role:{str(c):{metric:float(np.mean([p[role][c-1][metric] for p in patients])) for metric in ['Dice','recall','precision']} for c in range(1,len(MAPS[node['dataset']]))} for role in ['student','ema','entry']}
            cell=dict(domain=domain,val_cases=len(indices),class_metrics=classes,counts=totals,
                      teacher_accepted_fg_precision=1-totals['accepted_fg_wrong']/max(1,totals['accepted_fg']),
                      teacher_confident_bg_fraction_of_GT_foreground=totals['confident_bg_on_truth_fg']/max(1,totals['truth_fg_pixels']),
                      seal_probability_max_difference=seal_error,seal_argmax_disagreements=seal_disagreement,
                      full_val=len(indices)==len(access.rows))
            recorded=json.loads((folder/'VALIDATION.json').read_text())[domain]
            if cell['full_val']:
                fresh=float(np.mean([classes['student'][str(c)]['Dice'] for c in range(1,len(MAPS[node['dataset']]))]))
                assert abs(fresh-recorded['primary_dice'])<2e-5,(key,domain,fresh,recorded['primary_dice'])
            cells.append(cell)
        probe=probe_gradient(node,student,ema,bank,accessors[node['domain']])
        result=dict(key=key,cells=cells,gradient_probe=probe,counts=dict(COUNTS));results.append(result)
        (OUT/(key+'.json')).write_text(json.dumps(result,indent=2));print(key,'PASS',json.dumps(probe),flush=True)
        del student,ema,deploy,cp,sealed
        torch.cuda.empty_cache()
    assert COUNTS['optimizer']==COUNTS['test']==0
    (OUT/'INFERENCE.json').write_text(json.dumps(dict(status='PASS',results=results,counts=COUNTS,seconds=time.time()-started),indent=2))


if __name__=='__main__':
    try: main()
    except BaseException as exc:
        import traceback
        (OUT/'FAILURE.json').write_text(json.dumps(dict(error=str(exc),traceback=traceback.format_exc(),counts=COUNTS),indent=2))
        raise
