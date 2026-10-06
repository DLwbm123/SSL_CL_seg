"""All four pairs and preregistered gates, expressed explicitly in pp."""
import csv,json,time
from collections import Counter
from pathlib import Path
from statistics import mean,median,pstdev
import torch
from experiments.qprompt_rl_v1.v6_low_label_grpo import run as base
from . import protocol as p
read,write=base.read,base.write

def csvwrite(path,rows):
    with path.open('w') as f:
        out=csv.DictWriter(f,fieldnames=list(rows[0]));out.writeheader();out.writerows(rows)

def comparisons(rows,old):
    pairs=[]
    for r in rows:
        if r['method']!='GRPO_SAMPLE':continue
        d,s=r['domain'],r['controller'];hist=[a for a in old if a['domain']==d]
        uniform=next(a for a in rows if (a['domain'],a['controller'],a['method'])==(d,s,'UNIFORM_MATCHED'))
        greedy=next(a for a in hist if (a['method'],a['controller'])==('GRPO',s))
        random=next(a for a in hist if (a['method'],a['controller'])==('RANDOM',s));allu=next(a for a in hist if a['method']=='ALL_U')
        best=max([a for a in hist if a['method'] in p.old.FIXED]+[random,uniform],key=lambda a:a['macro_Dice'])
        pairs.append(dict(domain=d,controller=s,deploy_new_pp=100*(r['macro_Dice']-greedy['macro_Dice']),deploy_old_pp=100*(r['old_REFUGE']-greedy['old_REFUGE']),learn_new_pp=100*(r['macro_Dice']-uniform['macro_Dice']),learn_old_pp=100*(r['old_REFUGE']-uniform['old_REFUGE']),best_nonRL_method=best['method'],best_nonRL_new=best['macro_Dice'],vs_best_nonRL_pp=100*(r['macro_Dice']-best['macro_Dice']),vs_random_pp=100*(r['macro_Dice']-random['macro_Dice']),old_vs_allU_pp=100*(r['old_REFUGE']-allu['old_REFUGE'])))
    assert len(pairs)==4
    av=lambda key:mean(a[key] for a in pairs)
    seed=lambda key:all(mean(a[key] for a in pairs if a['controller']==s)>0 for s in p.old.SEEDS)
    d2=dict(mean_new_at_least_0_5pp=av('learn_new_pp')>=.5,positive_new_at_least_3_of_4=sum(a['learn_new_pp']>0 for a in pairs)>=3,each_seed_new_mean_positive=seed('learn_new_pp'),mean_old_at_least_minus0_5pp=av('learn_old_pp')>=-.5,min_old_at_least_minus1pp=min(a['learn_old_pp'] for a in pairs)>=-1)
    retention=dict(mean_old_allU_at_least_minus0_5pp=av('old_vs_allU_pp')>=-.5,min_old_allU_at_least_minus1pp=min(a['old_vs_allU_pp'] for a in pairs)>=-1)
    d3=dict(mean_best_at_least_0_5pp=av('vs_best_nonRL_pp')>=.5,mean_random_at_least_0_5pp=av('vs_random_pp')>=.5,min_best_at_least_minus0_25pp=min(a['vs_best_nonRL_pp'] for a in pairs)>=-.25,each_seed_best_mean_positive=seed('vs_best_nonRL_pp'),**retention)
    learning=all(d2.values());kept=all(retention.values());value=all(d3.values());recovery=av('deploy_new_pp')>0
    state=('DEPLOYMENT_RECOVERY_ONLY' if recovery else 'NO_POSITIVE_EVIDENCE') if not learning else 'LEARNED_SIGNAL_RETENTION_FAIL' if not kept else 'LEARNED_SIGNAL_NOT_COMPETITIVE' if not value else 'SCREENING_CANDIDATE'
    keys=[k for k in pairs[0] if k.endswith('_pp')]
    summary=lambda subset:{k:mean(r[k] for r in subset) for k in keys}
    decision=dict(status=state,D1_descriptive_mean_new_positive=recovery,D1_positive_new_pairs=sum(a['deploy_new_pp']>0 for a in pairs),D2=d2,D2_pass=learning,ALL_U_retention=retention,ALL_U_retention_pass=kept,D3=d3,D3_pass=value,overall_equal_pair_mean_pp=summary(pairs),domain_means_pp={d:summary([a for a in pairs if a['domain']==d]) for d in p.old.DOMAINS},controller_means_pp={str(s):summary([a for a in pairs if a['controller']==s]) for s in p.old.SEEDS},paired=pairs,statistical_significance_claim=False,independent_source_repeats=1,independent_patient_confirmation=False,V7_decision_unchanged=True,automatic_followup=False)
    return pairs,decision

def exposure_summary(a):
    total=sum(a);return dict(exposure_sorted=sorted(a),total=total,covered=sum(v>0 for v in a),coverage=sum(v>0 for v in a)/len(a),unused_fraction=sum(v==0 for v in a)/len(a),minimum=min(a),median=median(a),maximum=max(a),cv=pstdev(a)/mean(a) if total else None,N_eff=total**2/sum(v*v for v in a) if total else 0.)

def report(root,c,jobs,evals):
    rows=[];audits=[];totals=Counter();measured=Counter();per_job=[];distribution=[];reward=[];exposures=[];evaluator=[]
    old=read(Path(c['old_campaign'])/'RESULTS.json')
    for j in jobs+evals:
        path=base.jobroot(root,j);audit=read(path/'COMPLETION_AUDIT.json');assert audit['status']=='PASS'
        audits.append(dict(job=j['id'],**audit));totals.update(audit['physical_calls']);cost=read(path/'MEASURED_COSTS.json');measured.update(cost)
        per_job.append(dict(job=j['id'],kind=j['job'],method=j.get('method'),domain=j['domain'],controller=j.get('controller'),**cost))
        for r in (path/'evaluation_costs').glob('*/RESOURCE.json'):evaluator.append(dict(job=j['id'],checkpoint=r.parent.name,**read(r)))
        if j['job']=='diagnosis':
            distribution+=read(path/'POLICY_DISTRIBUTION_DIAGNOSTICS.json');reward+=read(path/'REWARD_COMPONENT_DIAGNOSTICS.json');exposures+=read(path/'EXPOSURE_DIAGNOSTICS.json')
        if j['job']=='evaluate':
            rows+=read(path/'RESULTS.json');source=read(Path(j['target'])/'entry.scores.json')['scores']['REFUGE']['macro_Dice']
            expected=next(a['source_old'] for a in read(root/'PROVENANCE_AUDIT.json')['baselines'] if a['domain']==j['domain'])
            assert abs(source-expected)<1e-12,'common source score mismatch'
        if j['job']=='endpoint':
            saved=torch.load(path/'student_latest.private.pt',map_location='cpu',weights_only=False)
            windows=[json.loads(line) for line in (path/'SELECTION_SUMMARY.jsonl').read_text().splitlines()]
            counts=saved['state']['exposure'];run=[0]*len(counts);longest=[0]*len(counts)
            selected=[json.loads(line) for line in (path/'SELECTIONS.private.jsonl').read_text().splitlines()]
            for window in selected:
                active=set(window['indices'])
                for i in range(len(counts)):run[i]=run[i]+1 if i in active else 0;longest[i]=max(longest[i],run[i])
            exposures.append(dict(domain=j['domain'],method=j['method'],controller=j['controller'],global_actual=exposure_summary(counts),windows_actual=[dict(block=a['block'],jaccard_previous=a['previous_jaccard'],**exposure_summary(a['actual_exposure_sorted'])) for a in windows],max_consecutive_windows_sorted=sorted(longest)))
    assert len(rows)==8 and dict(totals)==dict(smoke=32,endpoint=21200)
    for r in rows:
        allu=next(a for a in old if a['domain']==r['domain'] and a['method']=='ALL_U')
        r.update(absolute_forgetting_pp=-100*r['old_change'],old_vs_allU_pp=100*(r['old_REFUGE']-allu['old_REFUGE']))
        exp=next(a['global_actual'] for a in exposures if (a['domain'],a['method'],a['controller'])==(r['domain'],r['method'],r['controller']))
        r.update(U_coverage=exp['coverage'],U_exposure_min=exp['minimum'],U_exposure_median=exp['median'],U_exposure_max=exp['maximum'],U_exposure_cv=exp['cv'],U_N_eff=exp['N_eff'])
    pairs,decision=comparisons(rows,old);write(root/'RESULTS.json',rows);csvwrite(root/'RESULTS.csv',rows);csvwrite(root/'PAIRED_COMPARISONS.csv',pairs);csvwrite(root/'HISTORICAL_V7_RESULTS.csv',old)
    write(root/'DECISION.json',decision);write(root/'POLICY_DISTRIBUTION_DIAGNOSTICS.json',distribution);write(root/'REWARD_COMPONENT_DIAGNOSTICS.json',reward);write(root/'EXPOSURE_DIAGNOSTICS.json',exposures)
    historical=read(Path(c['old_campaign'])/'COSTS.json')
    cost=dict(physical_student_updates=sum(totals.values()),physical_calls=dict(totals),controller_updates=0,diagnostic_optimizer_updates=0,virtual_updates=0,measured_new=dict(measured),per_job=per_job,evaluators=evaluator,prior_policy_learning_student_updates=42400,prior_policy_learning_controller_updates=1696,historical_V7_total_costs=historical,notes=['CUDA intervals contain host gaps, not exact GPU busy time.','Deployment timings include feature/selection/checkpoint work; separate component clocks are also reported.','Prior source training 8000 is common to all methods. Existing V7 policy learning is additional method cost.','Same 25% subsets and fixed updates do not establish training speedup; shared hardware and scheduling can affect wall time.'])
    write(root/'COSTS.json',cost)
    write(root/'COMPLETION_AUDIT.json',dict(status='PASS',endpoints=8,expected=8,student_updates=21232,controller_updates=0,all_training_locked_before_validation=True,all_common_source_scores_match=True,all_policy_hashes_unchanged=True,source_repeats=1,jobs=audits))
    mechanism='# Mechanism audit\n\nZero student/controller updates and no virtual steps. Features and all restored native snapshot fields were checked exactly. Physical operation counts are outside restored model state.\n\n'
    mechanism+='## Policy distribution\n\nThe policy is set-conditioned; each greedy-conditioned selection position records legal count, entropy, uniform entropy, KL, finite-logit SD, maximum probability and top-two gap. Eight private-RNG sampled sets per existing state supply pairwise Jaccards. States: original entry, greedy endpoint, last-group entry and branch 0. No intermediate checkpoint availability is assumed.\n\n'
    for a in distribution:
        g=a['gradient'];vals=[v for s in a['states'] for v in s['greedy_conditioned_positions']]
        mechanism+=f"- {a['domain']} / {a['controller']}: mean KL to uniform {mean(v['KL_to_uniform'] for v in vals):.6g}; mean max probability {mean(v['max_probability'] for v in vals):.6g}; policy gradient norm {g['policy_grad_norm']:.6g}, weighted entropy gradient norm {g['weighted_entropy_grad_norm']:.6g}, angle {g['angle_degrees']}.\n"
    mechanism+='\nGradient diagnostics average the four saved group samples at the saved behavior policy, before any Adam step or clipping. They do not reconstruct later policy-update epochs or justify changing entropy regularization. High entropy alone is not evidence of excessive entropy regularization; greedy repetition alone is not stochastic distribution collapse.\n\n## Exposure\n\nHistorical global counts are actual saved training exposures. Historical per-window counts are reconstructed from the unchanged loader indices and checked against those saved totals. New windows record actual exposure deltas. Anonymous sorted distributions preserve every count without publishing image mappings. Coverage does not imply within-window balance.\n\n## Reward\n\nOnly the last saved group branch 0 has matched before/after states. Its total delta was reproduced within 1e-6, using the saved original anchors, coefficient 0.1, threshold >0.7 and all-pixel denominator. Quality is negative balanced NLL plus rim/cup soft Dice loss. Source KL channel terms may individually be negative; their sum is KL. Other branches/earlier groups are unavailable for decomposition; no training was replayed. Source predictions on target U do not guarantee REFUGE retention.\n'
    (root/'MECHANISM_AUDIT.md').write_text(mechanism)
    text='# V7.1 Frozen-policy deployment diagnosis\n\nStatus: **'+decision['status']+'**. Eight of eight final endpoints; 21,232 student updates, zero controller updates. Two independent source-to-target adaptations, not sequential three-domain training. NATIVE_LR_SRC_A reference, not a claim of exact original KI reproduction.\n\n## All new endpoints\n\n| Domain | Method | Controller | Macro % | Rim % | Cup % | Disc union % | REFUGE % | Absolute forgetting pp | Old vs ALL_U pp |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for r in rows:text+=f"| {r['domain']} | {r['method']} | {r['controller']} | {100*r['macro_Dice']:.4f} | {100*r['rim']:.4f} | {100*r['cup']:.4f} | {100*r['disc_union']:.4f} | {100*r['old_REFUGE']:.4f} | {r['absolute_forgetting_pp']:.4f} | {r['old_vs_allU_pp']:.4f} |\n"
    text+='\n## All four paired comparisons (pp)\n\n| Domain | Seed | Sample − uniform new | Sample − uniform old | Sample − greedy new | Sample − greedy old | Sample − best non-RL |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for a in pairs:text+=f"| {a['domain']} | {a['controller']} | {a['learn_new_pp']:+.4f} | {a['learn_old_pp']:+.4f} | {a['deploy_new_pp']:+.4f} | {a['deploy_old_pp']:+.4f} | {a['vs_best_nonRL_pp']:+.4f} |\n"
    text+='\nOverall equal-pair means: '+json.dumps(decision['overall_equal_pair_mean_pp'])+'. Domain and controller means and every gate are in DECISION.json. 0.005 Dice equals 0.5 pp.\n\n## Gates\n\n'
    for section in ('D2','D3'):
        for k,value in decision[section].items():text+=f'- {section} / {k}: **{value}**.\n'
    text+='\n## Interpretation and limits\n\n'
    text+=('The prespecified matched-uniform learning screen passed.' if decision['D2_pass'] else 'No practically meaningful learning advantage was established under the prespecified matched-uniform screen; this is not proof of equivalence.')+'\n\n'
    text+=('Stochastic deployment improved mean new-domain Dice over greedy deployment descriptively.' if decision['D1_descriptive_mean_new_positive'] else 'Mean new-domain Dice did not improve over greedy deployment.')+' All pairwise new/old outcomes remain visible. A positive deployment comparison alone supports only a deployment-recovery explanation. Frozen final policies still differ from evolving training behavior policies; not all train/deploy mismatch has been removed.\n\n'
    text+='Relative ALL_U retention and absolute forgetting from the source are distinct. One fixed REFUGE source, two targets and two controller seeds are not four independent source or clinical repeats. Historical development patients cannot confirm independent-patient generalization. No statistical significance claim is made. No sealed test accessed.\n\n## Mechanism and costs\n\nSee MECHANISM_AUDIT.md and all three diagnostic JSON files for policy/exposure/reward results and missing-state limitations. COSTS.json separates new work, per-job deployment overhead and prior learning cost (42,400 student and 1,696 controller updates). No speedup claim follows from selecting 25% U at fixed student steps.\n\n## Historical controls and stop\n\nAll 18 V7 controls are included in HISTORICAL_V7_RESULTS.csv, with patient-role, native-code, source and evaluation provenance in PROVENANCE_AUDIT.json. Historical training commit '+p.TRAINING+'; public report baseline '+p.BASELINE+'. V7 original decision remains unchanged. This round ends here; no next-round experiments launched.\n'
    (root/'FINAL_INTERPRETATION.md').write_text(text);(root/'RESULT_REPORT.md').write_text(text)
    write(root/'FINAL.json',dict(status='COMPLETE_PRIVATE',time=time.time(),endpoints=8,publication='pending'))

def self_check():
    # Exercise boundary units and adverse cells without any student/model update.
    old=[];rows=[]
    for d in p.old.DOMAINS:
        old.extend(dict(domain=d,method=m,controller=None,macro_Dice=.6,old_REFUGE=.8) for m in p.old.FIXED)
        for s in p.old.SEEDS:
            old.extend(dict(domain=d,method=m,controller=s,macro_Dice=.6,old_REFUGE=.8) for m in ('RANDOM','GRPO'))
            rows.extend(dict(domain=d,method=m,controller=s,macro_Dice=.61 if m=='GRPO_SAMPLE' else .6,old_REFUGE=.8) for m in p.METHODS)
    pairs,dec=comparisons(rows,old);assert dec['status']=='SCREENING_CANDIDATE' and len(pairs)==4 and abs(pairs[0]['learn_new_pp']-1)<1e-10
    rows[0]['old_REFUGE']=.78;_,dec=comparisons(rows,old);assert not dec['D2']['min_old_at_least_minus1pp'] and not dec['ALL_U_retention_pass']
    assert exposure_summary([0,2,2,0])['N_eff']==2
    return dict(status='PASS',optimizer_updates=0)

if __name__=='__main__':print(self_check())
