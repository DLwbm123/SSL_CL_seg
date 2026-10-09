"""Once-only audit of the frozen zero-update diagnostic."""
import fcntl
import importlib.util
import json
import math
import os
from collections import Counter
from pathlib import Path


def read(p):return json.loads(p.read_text())


def main(root):
    lock=(root/'COORDINATOR.lock').open();fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert read(root/'FINAL.json')['status']=='COMPLETE' and read(root/'EXIT.json')['exit_code']==0 and not (root/'FAILURE.json').exists()
    events=[json.loads(s) for s in (root/'FORWARD_LEDGER.jsonl').read_text().splitlines()]
    key=lambda r:tuple(r[k] for k in ('ordinal','context','image_index','kind'))
    aa=Counter(key(r) for r in events if r['event']=='attempt');ss=Counter(key(r) for r in events if r['event']=='success')
    assert aa==ss and len(aa)==160 and all(v==1 for v in aa.values()) and len(events)==320
    costs=read(root/'COSTS.json');assert costs['model_image_forwards']==160 and costs['current_labeled_diagnostic_images']==40
    assert all(costs[k]==0 for k in ('native_optimizer_updates','actor_optimizer_updates','linear_solves','Q_train_query_calls','Q_dev_query_calls','U_images_read','new_annotation_cases'))
    details=read(root/'DETAILS.private.json');rows=read(root/'RESULTS.json');assert len(details)==40 and len(rows)==40
    assert len({(r['context'],r['image_index']) for r in details})==40
    spec=importlib.util.spec_from_file_location('metrics',root/'metrics.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    support=[]
    for i in range(8):
        d=[r for r in details if r['context']==i];n=len(d);assert n in (2,8) and set(v['image_index'] for v in d)==set(range(n))
        assert {tuple(v[1:]) for v in aa if v[1]==i}=={(i,j,k) for j in range(n) for k in ('ema','student','memory','memory_flip')}
        for v in d:assert v['support_indices']==[(v['image_index']+k)%n for k in range(1,min(2,n-1)+1)] and v['image_index'] not in v['support_indices']
        for row in (r for r in rows if r['context']==i):
            group=[r for v in d for r in v['metrics'] if r['scope']==row['scope']];assert len(group)==n
            for k in group[0]:
                if k not in ('context','source_step','labeled_images','scope'):assert row[k]==sum(r[k] for r in group)
            assert all(row[k]==v for k,v in m.rates(row).items())
            assert row['corrected_correct']-row['base_correct']==row['repaired']-row['harmed']
            assert all(math.isfinite(v) for v in row.values() if isinstance(v,(int,float)))
        support.append(dict(context=i,labeled_images=n,support_images_per_target=min(2,n-1),missing_class_targets=[sum(v['support_pixel_counts'][k]==0 for v in d) for k in range(3)],zero_norm_class_targets=[sum(v['prototype_norms'][k]<=1e-8 for v in d) for k in range(3)],support_pixel_min=[min(v['support_pixel_counts'][k] for v in d) for k in range(3)]))
    summary=read(root/'CONTROL_SUMMARY.json');assert len(summary)==8
    for row in summary:
        group=[v['control_counts'] for v in details if v['context']==row['context']]
        for k in group[0]:assert row[k]==sum(r[k] for r in group)
        assert row['EMA_HALF_changed']==row['RANDOM_HALF_changed']==row['PROTO_HALF_changed']==row['half_budget']
        assert row['EMA_HALF_changed_weight']==row['RANDOM_HALF_changed_weight']==row['PROTO_HALF_changed_weight']
        assert row['conflict_1_0']+row['conflict_2_1']==row['conflict_total']==row['EMA_ALL_changed']
        assert row['quota_1_0']+row['quota_2_1']==row['half_budget']
    assert read(root/'REFERENCE_CHECK.json')['BASE_matches_V121']
    assert m.analyze(rows)==read(root/'DECISION.json') and read(root/'QUALIFICATION.json')['status']=='PASS'
    for name,value in [('SUPPORT_SUMMARY.json',support),('COMPLETION_AUDIT.json',dict(status='PASS',forward_pairs=160,held_out_images=40,contexts=8,aggregate_rows=40,all_support_exclusions=True,all_statistics_recomputed=True,no_training_or_query_updates=True))]:
        with (root/name).open('x') as f:json.dump(value,f,indent=2)
    print(json.dumps(read(root/'COMPLETION_AUDIT.json')))


if __name__=='__main__':main(Path(os.environ['EXEC_RUN']))
