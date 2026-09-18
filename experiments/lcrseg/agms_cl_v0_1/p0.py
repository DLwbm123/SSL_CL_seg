"""Future read-only L opportunity audit; confidence-only, not stage PAS."""
from pathlib import Path
import torch
from .protocol import read,write,digest,canonical_plan,execution_plan
from .authority import preflight,baseline_environment
from .execution import owned_root,choose_gpu,accept_prefix
from ..f5_confirmation_v1.assurance import environment,bind_environment,cost_session,session_totals
from ..five_frameworks_v1.native_parent import build,NativeLRParent
from ..five_frameworks_v1.model import Model
from ..five_frameworks_v1.native_data import NativeCurrentDomain
from ..five_frameworks_v1.semantics import tensor_fingerprint


def opportunities(q,geometry):
    return (geometry & (q.detach().max(1).values < .6) & (q.detach()[:,1:3].sum(1) >= .9)).detach()


def count(q,labels,selected,geometry):
    from scipy.ndimage import distance_transform_edt
    valid=geometry & (labels!=255);sel=selected & valid;disc=(labels==1)|(labels==2);cup=labels==2
    pred=q.argmax(1);out=dict(valid=int(valid.sum()),selected=int(sel.sum()),
        parent_correct=int((sel&disc).sum()),fine_correct=int((sel&(pred==labels)).sum()),
        rim_selected=int((sel&(labels==1)).sum()),cup_selected=int((sel&cup).sum()))
    for name,mask in [('outer',disc),('inner',cup)]:
        # Euclidean output-grid distance to either side of the binary boundary.
        band=torch.zeros_like(mask)
        for b in range(len(mask)):
            a=mask[b].cpu().numpy()
            if a.any() and (~a).any():
                band[b]=torch.from_numpy((distance_transform_edt(a)+distance_transform_edt(~a))<=5).to(mask.device)
        out[name+'_band_selected']=int((sel&band).sum())
    return out


def validate_report(r,config,plan):
    expected={1:min(16,10),2:min(16,16)}
    if (r.get('status')!='PASS' or r.get('execution_commit')!=config['execution_commit']
            or r.get('plan_sha256')!=plan['plan_sha256'] or r.get('optimizer_calls')!=0
            or {int(x['order']):x['images'] for x in r['orders']}!=expected or r['images']!=26
            or r.get('selector')!='confidence_only_not_PAS'):
        raise ValueError('incomplete P0 evidence')
    return r


def run(config):
    plan,permit=preflight(config);choose_gpu()
    from ..single_teacher_scd_v0_1.engine import precision
    precision();device=torch.device('cuda:0');torch.cuda.set_device(device)
    with owned_root(config,plan) as root:
        bind_environment(root,baseline_environment(environment()));out=root/'P0_REPORT.json'
        if out.exists():
            validate_report(read(out),config,plan);session_totals(root/'costs'/'P0');return read(out)
        if list((root/'costs'/'P0').glob('*/session.json')):raise RuntimeError('partial P0; no automatic retry')
        rows=[]
        with cost_session(root/'costs'/'P0','P0_read_only'):
            for order in (1,2):
                n=next(n for n in plan['nodes'] if n['order']==order)
                old,receipt=accept_prefix(n,config,plan,permit,root,device)
                value=torch.load(old/'student.pt',map_location=device,weights_only=False)
                if value['identity']!=receipt['identity'] or tensor_fingerprint(value['student'])!=receipt['student_hash']:
                    raise ValueError('prefix changed after acceptance')
                native=build(config['reference'],device,163);native.load_state_dict(value['student']);del value
                from .execution import source_identity
                sid=source_identity(n,receipt)
                model=Model(NativeLRParent(native,163,sid,adapt=False),'B2_PARENT_PAS_KL').eval().requires_grad_(False);model.role='eval'
                provider=NativeCurrentDomain(config['data'],163,order,2,sid,device,permit,allow_u=False)
                # Canonical accessor sorts manifest cases; fixed independent audit seed.
                g=torch.Generator().manual_seed(163000+order)
                indices=torch.randperm(len(provider._l),generator=g).tolist()[:16];total={};private=[]
                for ordinal,i in enumerate(indices):
                    item=provider._l[i];x=item['image'][None].to(device);geometry=item['geometry'][None].to(device)
                    with torch.no_grad():q=model(x,mode='eval').softmax(1)
                    mask=opportunities(q,geometry)  # labels not passed to selector
                    counts=count(q,item['label'][None].to(device),mask,geometry)
                    private.append(dict(ordinal=ordinal,**counts))
                    for k,v in counts.items():total[k]=total.get(k,0)+v
                write(root/f'private_P0_O{order}.json',private)
                rows.append(dict(order=order,domain=n['domain'],images=len(indices),counts=total,
                                 parent_accuracy=total['parent_correct']/total['selected'] if total['selected'] else None,
                                 fine_accuracy=total['fine_correct']/total['selected'] if total['selected'] else None))
                del model,provider
        result=dict(status='PASS',execution_commit=config['execution_commit'],plan_sha256=plan['plan_sha256'],
                    selector='confidence_only_not_PAS',optimizer_calls=0,images=sum(r['images'] for r in rows),orders=rows)
        validate_report(result,config,plan);write(out,result);return result
