"""Bind six immutable sources, derive channel bases once, reuse approved score files."""
import json
from pathlib import Path
import torch
from experiments.lcrseg.di_dmpa_gate1.binding import sha256,safe_asset
from . import core as c,contract as ct

def prepare(base,data,parent,baseline_original,baseline_recovery):
    source=ct.verify();b=ct.nas(base);b.mkdir(exist_ok=True)
    if (b/'SOURCE_BINDING.json').exists():raise FileExistsError('create-only source binding')
    parent,old,recovery=map(Path,(parent,baseline_original,baseline_recovery))
    terminal=ct.read(recovery/'TERMINAL.json');assert terminal['engineering']=='ENGINEERING_COMPLETE' and terminal['scientific_state']=='RETENTION_VALUE_NOT_ESTABLISHED'
    metadata=c.metadata(data);counts={}
    for d in c.DOMAINS:
        counts[d]={}
        for role,number in zip(('train_labeled','train_unlabeled','val'),c.COUNTS[d]):
            rows=[r for r in metadata if r['site_or_vendor']==d and r['primary_20pct_split']==role]
            assert len(rows)==number and len({r['patient_id'] for r in rows})==number;counts[d][role]=number
            if role in ('train_labeled','train_unlabeled'):
                for r in rows:
                    assert safe_asset(data,r['image_h5_relpath']).is_file()
                    if role=='train_labeled':assert safe_asset(data,r['label_h5_relpath']).is_file()
                    else:assert not r['label_h5_relpath'] and not r['label_sha256']
    (b/'bases').mkdir();sources={}
    for src in ct.protocol()['sources']:
        tid=src['task_id'];root=parent/'tasks'/tid;r=ct.read(root/'receipt.json')
        p=torch.load(root/'deploy_student.pt',map_location='cpu',weights_only=False)
        assert r['source']==p['source']==ct.PARENT_SOURCE and r['status']=='TRAINING_COMPLETE' and p['complete']
        assert p['task']==r['task'] and p['task']['task_id']==tid and p['head_mode']==c.LINEAR
        assert c.hash_state(p['student'])==p['student_hash']==r['student_hash']==src['student_hash']
        tensors,meta=c.readout_basis(p['student']['decoder.conv_logit.mu.weight'],src['seed'],src['domain'])
        torch.save(tensors,b/'bases'/(tid+'.pt'))
        sources[tid]=dict(student_hash=src['student_hash'],receipt_sha256=sha256(root/'receipt.json'),deploy_sha256=sha256(root/'deploy_student.pt'),training_source=ct.PARENT_SOURCE,basis=meta)
        del p,tensors
    manifest=ct.read(recovery/'RECOVERY_MANIFEST.json');seal=ct.read(recovery/'TARGET_WEIGHT_SEAL.json');baselines={}
    for src in ct.protocol()['sources']:
        prefix=src['task_id'].removesuffix('SRC_CE')
        for arm in ('F_FULL','F_CONV','STATIC_SOURCE'):
            tid=prefix+arm
            root=(old if tid in manifest['reused_task_ids'] else recovery)/'tasks'/tid
            ev=ct.read(root/'evaluation/receipt.json');scores=root/'evaluation/private_patient_metrics.csv'
            assert ev['status']=='COMPLETE' and ev['models']==1 and scores.is_file()
            if arm!='STATIC_SOURCE':
                r=ct.read(root/'receipt.json')
                assert r['status']=='TRAINING_COMPLETE' and r['student_hash']==ev['student_hash']==seal['students'][tid]
                assert r['boundary']['source_student_hash']==src['student_hash']
            else:assert ev['student_hash']==src['student_hash'] and ev['optimizer_updates']==0
            baselines[tid]=dict(arm=arm,student_hash=ev['student_hash'],evaluation_source=ev['source'],
                               score_sha256=sha256(scores),source_student_hash=src['student_hash'],
                               storage='original' if tid in manifest['reused_task_ids'] else 'recovery')
        source_scores=parent/'tasks'/src['task_id']/'evaluation/private_patient_metrics.csv'
        sources[src['task_id']]['source_scores_sha256']=sha256(source_scores)
    c.write_json(b/'SOURCE_BINDING.json',dict(status='PASS',source=source,sources=sources,counts=counts,GT_reads=0,image_reads=0,source_training_updates=0))
    c.write_json(b/'BASELINE_BINDING.json',dict(status='PASS',source=source,baselines=baselines,baseline_retraining_updates=0,old_terminal_unchanged=True))
    c.write_json(b/'private_inputs.json',dict(data=str(Path(data).resolve()),parent=str(parent),baseline_original=str(old),baseline_recovery=str(recovery)))
    print(json.dumps(dict(status='PASS',sources=6,readout_bases=6,baselines=18,GT_reads=0)),flush=True)
