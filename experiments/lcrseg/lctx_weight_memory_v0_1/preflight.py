"""Weight-only basis preparation; zero image/GT access."""
import argparse,json,os,subprocess
from pathlib import Path
import torch
from . import core as c
from .contract import verify,read,protocol,nas,PARENT_SOURCE
from experiments.lcrseg.di_dmpa_gate1.binding import sha256,safe_asset

def prepare(base,data,parent):
    source=verify();b=nas(base);b.mkdir();parent=Path(parent)
    if os.statvfs(b).f_bavail*os.statvfs(b).f_frsize<10*1024**3:raise RuntimeError('10GiB free required')
    allrows=c.metadata(data);counts={}
    for d in c.DOMAINS:
        counts[d]={}
        for role,n in zip(('train_labeled','train_unlabeled','val'),c.COUNTS[d]):
            rr=[r for r in allrows if r['site_or_vendor']==d and r['primary_20pct_split']==role]
            assert len(rr)==n and len({r['patient_id'] for r in rr})==n
            counts[d][role]=n
            if role!='train_unlabeled':
                for r in rr:
                    for k in ['image','label']:assert safe_asset(data,r[k+'_h5_relpath']).is_file()
    entries={}
    for src in protocol()['sources']:
        tid=src['task_id'];root=parent/'tasks'/tid;r=read(root/'receipt.json')
        assert r['task']['task_id']==tid and r['task']['arm']=='SRC_CE' and r['source']==PARENT_SOURCE and r['status']=='TRAINING_COMPLETE'
        payload=torch.load(root/'deploy_student.pt',map_location='cpu',weights_only=False)
        assert set(payload)=={'student','student_hash','task','source','head_mode','complete'}
        assert payload['complete'] and payload['source']==PARENT_SOURCE and payload['task']==r['task'] and payload['head_mode']==c.LINEAR
        assert c.hash_state(payload['student'])==payload['student_hash']==r['student_hash']==src['student_hash']
        out=b/'bases'/tid;out.mkdir(parents=True);layers=[]
        for name in c.LAYERS:
            t,m=c.basis(payload['student'][name],src['seed'],src['domain'],name)
            m['adapter_initialization']={a:dict(A0_hash=c.hash_state({'A':c.initial_A(t,a)}),B0_hash=c.hash_state({'B':torch.zeros(t['W0'].shape[0],8,dtype=t['W0'].dtype)}),A0_norm=float(c.initial_A(t,a).double().norm()),protected_right_FP32_hash=c.hash_state({'right':t['Q' if a=='LR_RAND' else 'V'].float()}) if a!='LR_FREE' else None,protected_left_FP32_hash=c.hash_state({'left':t['U'].float()}) if a=='LR_SRC_AB' else None) for a in c.ARMS if a.startswith('LR_')}
            torch.save(t,out/(name+'.pt'));layers.append(m)
        full=read(parent/'tasks'/f"{src['order']}_s{src['seed']}_T_LCTX"/'receipt.json')
        entries[tid]=dict(receipt=r,receipt_sha256=sha256(root/'receipt.json'),student_hash=src['student_hash'],layers=layers,parent_F_FULL_reference={k:full[k] for k in ['student_hash','EMA_hash','optimizer_hash','label_order_hash']})
        del payload,t
    result=dict(status='PASS',source=source,source_recovery_updates=0,sources=entries,counts=counts,manifest_sha=c.MANIFEST_SHA,split_sha=c.SPLIT_SHA,GT_reads=0,image_reads=0,rank=8,scaling=1,source_information='weights only')
    c.write_json(b/'SOURCE_BINDING_AND_LAYER_MANIFEST.json',result)
    c.write_json(b/'private_inputs.json',dict(data=str(Path(data).resolve()),parent=str(parent.resolve())))
    print(json.dumps(dict(status='PASS',sources=len(entries),layers_per_source=14,GT_reads=0,source_recovery_updates=0)))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['base','data','parent']:p.add_argument('--'+k,required=True)
    prepare(**vars(p.parse_args()))
