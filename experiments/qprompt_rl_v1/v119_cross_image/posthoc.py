"""Zero-model arithmetic linking published V118/V119 context aggregates."""
import json
from pathlib import Path


def analyze(root):
    read=lambda p:json.loads(p.read_text())
    current=read(root/'RESULTS.json');old=read(root.parent/'v118_semantic_target/CONTEXT_SUMMARY.json');lookup={(r['context'],r['method']):r for r in old}
    selected=[r for r in current if r['scope']=='selected'];assert len(selected)==8
    paired=[]
    for r in selected:
        i=r['context'];a,b=lookup[i,'SEMANTIC'],lookup[i,'BASE']
        paired.append(dict(context=i,diagnostic_weighted_net_accuracy=r['weighted_net_accuracy'],diagnostic_weighted_nll_delta=r['weighted_nll_delta'],training_final_new_delta=a['final_new_gain']-b['final_new_gain'],training_old_delta=a['old_gain']-b['old_gain'],training_reward_delta=a['reward']-b['reward']))
    counts={k:sum(r[k] for r in selected) for k in ('pixels','changed','repaired','harmed')}
    classes=[dict(class_id=k,net_corrected=sum(r['repaired']-r['harmed'] for r in current if r['scope']==f'true_class_{k}')) for k in range(3)]
    assert sum(r['net_corrected'] for r in classes)==counts['repaired']-counts['harmed']
    assert next(r['context'] for r in paired if r['diagnostic_weighted_net_accuracy']>0 and r['diagnostic_weighted_nll_delta']<0)==5
    assert min(paired,key=lambda r:r['training_final_new_delta'])['context']==5
    low=[r for r in current if r['scope']=='confidence_0.7_0.8']
    return dict(scope='posthoc published-scalar arithmetic; no model/image/query/fit/optimization',changed_exposure_fraction=counts['changed']/counts['pixels'],pooled_repair_fraction=counts['repaired']/counts['changed'],class_net_corrected=classes,paired_contexts=paired,low_confidence_joint_positive_contexts=[r['context'] for r in low if r['weighted_net_accuracy']>0 and r['weighted_nll_delta']<0],low_confidence_joint_negative_contexts=[r['context'] for r in low if r['weighted_net_accuracy']<0 and r['weighted_nll_delta']>0],diagnostic_pass_does_not_establish_training_gain=True,original_gate_unchanged=True)


if __name__=='__main__':
    root=Path(__file__).resolve().parent
    with (root/'POSTHOC_ANALYSIS.json').open('x') as f:json.dump(analyze(root),f,indent=2)
