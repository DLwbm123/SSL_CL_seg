"""Published-scalar arithmetic only; no models, images, labels, fitting or queries."""
import json
from pathlib import Path


def analyze(root):
    summary=json.loads((root/'SUMMARY.json').read_text());decision=json.loads((root/'DECISION.json').read_text());rows=json.loads((root/'RESULTS.json').read_text());controls=json.loads((root/'CONTROL_SUMMARY.json').read_text())
    methods={r['arm']:r for r in summary['methods']};counts=summary['controls'];comparisons=decision['comparisons'][2]['contexts']
    assert methods['PROTO']==dict(methods['EMA'],arm='PROTO')
    assert counts['EMA_agrees_prototype']==counts['candidate']==3858
    assert counts['shuffle_locations_different']==2*(counts['candidate']-counts['retained_original_changed_locations'])
    return dict(scope='posthoc scalar arithmetic and separate code-derived mechanism explanation; original gate unchanged',candidate_fraction=counts['candidate']/counts['selected'],prototype_repair_fraction=methods['PROTO']['repaired']/methods['PROTO']['changed'],shuffle_repair_fraction=methods['SHUFFLE']['repaired']/methods['SHUFFLE']['changed'],shuffle_retained_accuracy_effect=methods['SHUFFLE']['weighted_accuracy_delta']/methods['PROTO']['weighted_accuracy_delta'],shuffle_retained_nll_improvement=methods['SHUFFLE']['weighted_nll_delta']/methods['PROTO']['weighted_nll_delta'],shuffle_vs_base_joint_positive_contexts=sum(r['weighted_net_accuracy']>0 and r['weighted_nll_delta']<0 for r in rows if r['scope']=='SHUFFLE'),prototype_vs_shuffle_accuracy_positive_contexts=sum(r['weighted_accuracy_delta']>0 for r in comparisons),prototype_vs_shuffle_nll_improved_contexts=sum(r['weighted_nll_delta']<0 for r in comparisons),prototype_vs_shuffle_nll_worse_contexts=sum(r['weighted_nll_delta']>0 for r in comparisons),prototype_vs_shuffle_nll_delta_without_context5=sum(r['weighted_nll_delta'] for r in comparisons if r['context']!=5)/7,shuffle_L1_relative_extra=counts['SHUFFLE_weighted_target_L1_change']/counts['PROTO_weighted_target_L1_change']-1,shuffle_L1_larger_contexts=sum(r['SHUFFLE_weighted_target_L1_change']>r['PROTO_weighted_target_L1_change'] for r in controls),shuffle_overlap_fraction=counts['retained_original_changed_locations']/counts['candidate'],shuffle_new_changed_locations=counts['candidate']-counts['retained_original_changed_locations'],prototype_dependent_shuffle_budgets=True,model_forwards=0,image_reads=0,query_calls=0,optimizer_updates=0)


if __name__=='__main__':
    root=Path(__file__).resolve().parent;result=analyze(root)
    with (root/'POSTHOC_ANALYSIS.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
