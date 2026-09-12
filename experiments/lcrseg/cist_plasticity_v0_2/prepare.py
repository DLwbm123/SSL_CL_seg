"""Reuse bound sources, cached source bases and five matched references; no image/GT opens."""
from pathlib import Path
import torch
from . import core as c,contract as ct

def prepare(base,previous):
    source=ct.verify();b=ct.nas(base);prev=Path(previous);inputs=ct.read(prev/'private_inputs.json')
    assert not (b/'SOURCE_BINDING.json').exists()
    assert ct.read(prev/'TERMINAL.json')['scientific_state']=='NO_PRIMARY_ACCURACY_GAIN'
    oldsb=ct.read(prev/'SOURCE_BINDING.json');sb=ct.read('experiments/lcrseg/docs/cist_v0_1/results/SOURCE_BINDING.json');assert oldsb==sb
    bb=ct.read(prev/'BASELINE_BINDING.json');assert bb==ct.read('experiments/lcrseg/docs/cist_v0_1/results/BASELINE_BINDING.json')
    (b/'bases').mkdir();paramlists={}
    for src in ct.protocol()['sources']:
        tid=src['task_id'];entry=sb['sources'][tid];root=Path(inputs['parent'])/'tasks'/tid
        ct.check_hash(root/'receipt.json',entry['receipt_sha256']);ct.check_hash(root/'deploy_student.pt',entry['deploy_sha256'])
        payload=torch.load(root/'deploy_student.pt',map_location='cpu',weights_only=False);assert c.hash_state(payload['student'])==entry['student_hash']==src['student_hash'];del payload
        tensors,entry=ct.load_basis(prev,dict(source_task_id=tid));torch.save(tensors,b/'bases'/(tid+'.pt'))
        ct.check_hash(root/'evaluation/private_patient_metrics.csv',entry['source_scores_sha256'])
        for arm in ('ISO_COND_L','ISO_COND_LU'):
            bid=tid.removesuffix('SRC_CE')+arm;oldroot=prev/'tasks'/bid;r=ct.read(oldroot/'receipt.json');ev=ct.read(oldroot/'evaluation/receipt.json')
            assert r['source']==ev['source']==oldsb['source'] and r['boundary']['source_student_hash']==src['student_hash']
            assert r['student_hash']==ev['student_hash']==ct.read(prev/'TARGET_WEIGHT_SEAL.json')['students'][bid]['student_hash']
            bb['baselines'][bid]=dict(arm=arm,student_hash=r['student_hash'],evaluation_source=ev['source'],source_student_hash=src['student_hash'],storage='cist',score_sha256=ct.sha256(oldroot/'evaluation/private_patient_metrics.csv'))
    for tid,entry in bb['baselines'].items():
        root=prev if entry['storage']=='cist' else Path(inputs['baseline_'+entry['storage']])
        ct.check_hash(root/'tasks'/tid/'evaluation/private_patient_metrics.csv',entry['score_sha256'])
    metadata=c.metadata(inputs['data']);counts={}
    for domain in c.DOMAINS:
        counts[domain]={role:sum(r['site_or_vendor']==domain and r['primary_20pct_split']==role for r in metadata) for role in ('train_labeled','train_unlabeled','val')}
        assert tuple(counts[domain].values())==c.COUNTS[domain]
    sb.update(source=source,inherited_binding_source=oldsb['source'],source_training_updates=0,bases_recomputed=False)
    bb.update(source=source,inherited_binding_source=oldsb['source'],baseline_retraining_updates=0)
    c.write_json(b/'SOURCE_BINDING.json',sb);c.write_json(b/'BASELINE_BINDING.json',bb)
    c.write_json(b/'private_inputs.json',dict(**inputs,baseline_cist=str(prev)))
    c.write_json(b/'INPUT_AUDIT.json',dict(status='PASS',source=source,counts=counts,sources=6,baselines=30,cached_bases=6,GT_opens=0,image_opens=0,U_metadata_only=True,existing_equivalent_GN_tasks=0))
    print('PASS:6 immutable sources,6 reused bases,30 cached baselines; no image or GT opens',flush=True)
