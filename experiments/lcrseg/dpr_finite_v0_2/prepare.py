"""Bind real source students and immutable F_CONV/F_FULL records; no training data opens."""
from pathlib import Path
import torch
from . import core as c,contract as ct

def prepare(base,previous):
    source=ct.verify();b=ct.nas(base);prev=Path(previous);inputs=ct.read(prev/'private_inputs.json')
    assert not (b/'SOURCE_BINDING.json').exists()
    assert ct.read(prev/'TERMINAL.json')['scientific_state']=='SMALL_POSITIVE_PRIMARY_SIGNAL'
    oldsb=ct.read(prev/'SOURCE_BINDING.json');oldbb=ct.read(prev/'BASELINE_BINDING.json')
    assert oldsb==ct.read('experiments/lcrseg/docs/dpr_v0_1/results/SOURCE_BINDING.json')
    assert oldbb==ct.read('experiments/lcrseg/docs/dpr_v0_1/results/BASELINE_BINDING.json')
    inputs=dict(inputs,baseline_dpr=str(prev))
    sb=dict(status='PASS',source=source,sources={},source_training_updates=0);bb=dict(status='PASS',source=source,baselines={},baseline_retraining_updates=0)
    for src in ct.protocol()['sources']:
        tid=src['task_id'];entry={k:v for k,v in oldsb['sources'][tid].items() if k!='basis'};root=Path(inputs['parent'])/'tasks'/tid
        ct.check_hash(root/'receipt.json',entry['receipt_sha256']);ct.check_hash(root/'deploy_student.pt',entry['deploy_sha256'])
        payload=torch.load(root/'deploy_student.pt',map_location='cpu',weights_only=False)
        assert c.hash_state(payload['student'])==entry['student_hash']==src['student_hash'];del payload
        ct.check_hash(root/'evaluation/private_patient_metrics.csv',entry['source_scores_sha256']);sb['sources'][tid]=entry
        for arm in ('F_FULL','F_CONV','STATIC_SOURCE'):
            bid=tid.replace('SRC_CE',arm);bound=dict(oldbb['baselines'][bid]);br=Path(inputs['baseline_'+bound['storage']])/'tasks'/bid
            ct.check_hash(br/'evaluation/private_patient_metrics.csv',bound['score_sha256'])
            if arm!='STATIC_SOURCE':
                receipt=ct.read(br/'receipt.json');evaluation=ct.read(br/'evaluation/receipt.json')
                assert receipt['student_hash']==evaluation['student_hash']==bound['student_hash']
                assert receipt['source_student_hash']==src['student_hash'] if 'source_student_hash' in receipt else receipt['boundary']['source_student_hash']==src['student_hash']
                bound['training_receipt_sha256']=ct.sha256(br/'receipt.json');bound['deploy_sha256']=ct.sha256(br/'deploy_student.pt')
                bound['training_config_source']=receipt['source'];bound['label_order_hash']=receipt['label_order_hash']
                if arm=='F_CONV':bound['warmup']=ct.read(br/'warmup.json')
            bb['baselines'][bid]=bound
    seal=ct.read(prev/'TARGET_WEIGHT_SEAL.json')
    assert seal['source']==ct.protocol()['prior_execution_source'] and len(seal['students'])==18
    for src in ct.protocol()['sources']:
        for arm in ('DPR_U','RESPONSE_U','DPR_L'):
            bid=src['task_id'].replace('SRC_CE',arm);root=prev/'tasks'/bid
            r=ct.read(root/'receipt.json');ev=ct.read(root/'evaluation/receipt.json')
            assert r['source']==ev['source']==seal['source'] and r['student_hash']==ev['student_hash']==seal['students'][bid]['student_hash']
            assert r['boundary']['source_student_hash']==src['student_hash']
            assert r['label_order_hash']==bb['baselines'][src['task_id'].replace('SRC_CE','F_CONV')]['label_order_hash']
            bb['baselines'][bid]=dict(storage='dpr',student_hash=r['student_hash'],source_student_hash=src['student_hash'],
                score_sha256=ct.sha256(root/'evaluation/private_patient_metrics.csv'),training_receipt_sha256=ct.sha256(root/'receipt.json'),
                training_config_source=r['source'],label_order_hash=r['label_order_hash'])
    counts={}
    for d in c.DOMAINS:
        rows=c.metadata(inputs['data']);counts[d]={role:sum(r['site_or_vendor']==d and r['primary_20pct_split']==role for r in rows) for role in ('train_labeled','train_unlabeled','val')}
        assert tuple(counts[d].values())==c.COUNTS[d]
    for name,obj in [('SOURCE_BINDING.json',sb),('BASELINE_BINDING.json',bb),('private_inputs.json',inputs),('SOURCE_AND_BASELINE_BINDING.json',dict(sources=sb,baselines=bb))]:c.write_json(b/name,obj)
    c.write_json(b/'INPUT_AUDIT.json',dict(status='PASS',source=source,counts=counts,sources=6,baselines=30,static_references=6,GT_opens=0,image_opens=0,old_negative_terminals_preserved=2))
