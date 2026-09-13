"""Paired saved-score statistics; never loads models or image/GT arrays."""
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
from experiments.lcrseg.ams_seq_transfer_v0_1.results import csv_read,trajectory,effect_summary,MEASURES
from experiments.lcrseg.ssl_foundation_v0_1.diagnostics import csv_write,METRICS
from . import core as c,contract as ct

def dump(path,rows):csv_write(path,rows,fields=list(dict.fromkeys(k for row in rows for k in row)))

def scores(path,domains):
    rows=csv_read(path);result={}
    for d in domains:
        rr=[r for r in rows if r['domain']==d]
        if [int(x['patient_index']) for x in rr]!=list(range(c.COUNTS[d][2])):raise ValueError('missing or unpaired patient scores: '+str(path))
        a=np.array([[float(r[k]) for k in METRICS] for r in rr]);assert np.isfinite(a).all();result[d]=a
    return result

def paired(out,private,source_scores,seeds,arms,comparisons,analysis_seed,scope):
    out=Path(out);out.mkdir();site=[]
    for (o,seed,a,d),v in private.items():
        src=c.DOMAINS[0 if o=='O1' else 1]
        site.append(dict(order=o,seed=seed,arm=a,domain=d,role='old' if d==src else 'current',**dict(zip(METRICS,map(float,v.mean(0))))))
    def measurements(weights=None):
        def mean(x,d):return float(x[:,3].mean() if weights is None else x[:,3]@weights[d])
        values={a:np.empty((2,len(seeds),4)) for a in arms}
        for oi,o in enumerate(('O1','O2')):
            src,dst=(c.DOMAINS if o=='O1' else c.DOMAINS[::-1])
            for si,seed in enumerate(seeds):
                for a in arms:values[a][oi,si]=trajectory(mean(source_scores[o,seed],src),mean(private[o,seed,a,src],src),mean(private[o,seed,a,dst],dst))
        return values
    vv=measurements();tra=[];summary=[];effects=[];cost=[];tails=[]
    for a in arms:
        mean,sd=effect_summary(vv[a]);summary.append(dict(arm=a,**dict(zip(MEASURES,map(float,mean))),**{k+'_seed_SD':float(sd[i]) for i,k in enumerate(MEASURES)}))
        for oi,o in enumerate(('O1','O2')):
            for si,seed in enumerate(seeds):tra.append(dict(arm=a,order=o,seed=seed,**dict(zip(MEASURES,map(float,vv[a][oi,si])))))
    for a,ref in comparisons:
        delta=vv[a]-vv[ref]
        assert np.allclose(delta[:,:,0],(delta[:,:,1]+delta[:,:,2])/2) and np.allclose(delta[:,:,3],-delta[:,:,2])
        mean,sd=effect_summary(delta)
        for k,m in enumerate(MEASURES):
            effects.append(dict(arm=a,baseline=ref,scope='overall',order='both',seed=None,metric=m,delta=float(mean[k]),SD=float(sd[k])))
            for oi,o in enumerate(('O1','O2')):
                effects.append(dict(arm=a,baseline=ref,scope='order',order=o,seed=None,metric=m,delta=float(delta[oi,:,k].mean()),SD=None))
                for si,seed in enumerate(seeds):effects.append(dict(arm=a,baseline=ref,scope='seed_order',order=o,seed=seed,metric=m,delta=float(delta[oi,si,k]),SD=None))
            for si,seed in enumerate(seeds):effects.append(dict(arm=a,baseline=ref,scope='seed_order_average',order='both',seed=seed,metric=m,delta=float(delta[:,si,k].mean()),SD=None))
        for o in ('O1','O2'):
            src=c.DOMAINS[0 if o=='O1' else 1]
            for seed in seeds:
                for d in c.DOMAINS:
                    x=private[o,seed,a,d]-private[o,seed,ref,d]
                    for mi,m in enumerate(METRICS):cost.append(dict(arm=a,baseline=ref,order=o,seed=seed,domain=d,role='old' if d==src else 'current',metric=m,delta=float(x[:,mi].mean())))
                    y=x[:,3];tails.append(dict(arm=a,baseline=ref,order=o,seed=seed,domain=d,role='old' if d==src else 'current',n=len(y),mean=float(y.mean()),**{f'q{q:02d}':float(np.percentile(y,q)) for q in (0,5,25,50,75,95,100)},harm_below_minus005=int((y<-.05).sum()),harm_below_minus001=int((y<-.01).sum())))
    rng=np.random.default_rng(analysis_seed);draws=defaultdict(list)
    for _ in range(2000):
        weights={d:rng.multinomial(c.COUNTS[d][2],np.full(c.COUNTS[d][2],1/c.COUNTS[d][2]))/c.COUNTS[d][2] for d in c.DOMAINS};v=measurements(weights)
        for a,ref in comparisons:
            for oi,o in ((None,'both'),(0,'O1'),(1,'O2')):draws[a,ref,o].append((v[a]-v[ref]).mean((0,1)) if oi is None else (v[a]-v[ref])[oi].mean(0))
    intervals=[]
    for (a,ref,o),d in draws.items():
        lo,hi=np.percentile(d,[2.5,97.5],axis=0)
        for i,m in enumerate(MEASURES):intervals.append(dict(arm=a,baseline=ref,order=o,metric=m,lower_95=float(lo[i]),upper_95=float(hi[i]),crosses_zero=bool(lo[i]<=0<=hi[i]),analysis_seed=analysis_seed,replicates=2000))
    for n,rows in [('ARM_SUMMARY',summary),('TRAJECTORIES',tra),('PAIRED_EFFECTS',effects),('DOMAIN_CLASS_COSTS',cost),('PATIENT_TAILS',tails),('PATIENT_INTERVALS',intervals),('SITE_CLASS_METRICS',site)]:dump(out/(n+'.csv'),rows)
    result=dict(scope=scope,seeds=seeds,arms=arms,comparisons=comparisons,analysis_seed=analysis_seed,paired_physical_domain_weights_shared=True,condition='trained models and repeatedly exposed development patients; not independent patient confirmation',optimizer_updates=0,new_model_forwards=0,new_image_GT_array_access=0)
    c.write_json(out/'ANALYSIS_RECEIPT.json',result);return vv,intervals

FIELDS=('candidate_accepted','guard_rejected','d0_norm','eta_norm','correction_ratio','b_norm','raw_response_norm','delta_norm','raw_non_descent')

def mechanisms(base,tasks,out):
    per=[];diag=[];missing=[]
    for t in tasks:
        rows=[];path=Path(base)/'tasks'/t['task_id']/'steps.jsonl'
        if not path.exists():missing.append(t['task_id']);continue
        with path.open() as f:
            for line in f:
                z=json.loads(line)
                if z.get('lambda_response',0)>0:rows.append(z)
        if not rows:continue
        def collect(key):return [float(z[key]) for z in rows if isinstance(z.get(key),(int,float))]
        rr=dict(task_id=t['task_id'],arm=t['arm'],order=t['order'],seed=t['seed'],active_steps=len(rows))
        for key in FIELDS:
            v=collect(key);rr[key+'_mean']=float(np.mean(v)) if len(v)==len(rows) else 'NOT_RECORDED'
        # Joint moment from original steps. Explicitly algebra eta, not unavailable FP32 vector norm.
        v=[z['candidate_accepted']*z['correction_ratio'] for z in rows if z.get('correction_ratio') is not None and 'candidate_accepted' in z]
        rr['accepted_algebra_eta_over_d0_mean']=float(np.mean(v)) if len(v)==len(rows) else 'NOT_RECORDED'
        rr['actual_FP32_applied_correction_norm']='NOT_RECORDED'
        for kind in ('candidate','applied'):
            v=[z['pooled_'+kind+'_RMS']/z['pooled_raw_RMS'] for z in rows if z.get('pooled_raw_RMS',0)>1e-30 and z.get('pooled_'+kind+'_RMS') is not None]
            rr['pooled_'+kind+'_ratio_mean']=float(np.mean(v)) if v else 'NOT_RECORDED';rr['pooled_'+kind+'_ratio_count']=len(v)
        per.append(rr)
        for pos in t['actual_diagnostic_positions']:
            path=Path(base)/'tasks'/t['task_id']/('actual_'+str(pos)+'.json')
            if path.exists():diag.append(dict(task_id=t['task_id'],**ct.read(path)))
    summary=[]
    for arm in sorted(set(x['arm'] for x in per)):
        part=[x for x in per if x['arm']==arm];row=dict(arm=arm,tasks=len(part),weighting='equal tasks = equal orders and seeds')
        for key in per[0]:
            if key.endswith('_mean'):
                vals=[x[key] for x in part];row[key]=float(np.mean(vals)) if all(isinstance(x,(int,float)) for x in vals) else 'NOT_RECORDED'
        summary.append(row)
    c.write_json(Path(out)/'MISSING_MECHANISM_LOGS.json',dict(status='NOT_RECORDED' if missing else 'COMPLETE',tasks=missing,not_a_B_gate=True))
    dump(Path(out)/'MECHANISM_BY_TASK.csv',per);dump(Path(out)/'MECHANISM_EQUAL_TASK.csv',summary)
    if diag:dump(Path(out)/'RECORDED_DIAGNOSTICS.csv',diag)
    return summary

def post_hoc(base):
    b=Path(base);source=ct.verify();inputs=ct.read(b/'private_inputs.json');old=Path(inputs['old_finite']);binding=ct.read(old/'BASELINE_BINDING.json');sb=ct.read(old/'SOURCE_BINDING.json');oi=ct.read(old/'private_inputs.json')
    assert ct.read(old/'TERMINAL.json')['scientific_state']=='NO_PRIMARY_ACCURACY_GAIN'
    private={};sources={};arms=[c.MAIN,'F_CONV','DPR_U'];reads=0;missing=[]
    for seed in (61,62,63):
        for o in ('O1','O2'):
            src=c.DOMAINS[0 if o=='O1' else 1];sid=f'{o}_s{seed}_SRC_CE';path=Path(oi['parent'])/'tasks'/sid/'evaluation/private_patient_metrics.csv'
            try:
                ct.check_hash(path,sb['sources'][sid]['source_scores_sha256']);sources[o,seed]=scores(path,[src])[src];reads+=1
                for a in arms:
                    tid=f'{o}_s{seed}_{a}'
                    if a==c.MAIN:root=old
                    else:
                        bound=binding['baselines'][tid];root=Path(oi['baseline_'+bound['storage']]);ct.check_hash(root/'tasks'/tid/'evaluation/private_patient_metrics.csv',bound['score_sha256'])
                    data=scores(root/'tasks'/tid/'evaluation/private_patient_metrics.csv',c.DOMAINS);reads+=1
                    for d,v in data.items():private[o,seed,a,d]=v
            except FileNotFoundError:missing.append(sid)
    out=b/'post_hoc_analysis'
    if missing:
        out.mkdir();c.write_json(out/'ANALYSIS_RECEIPT.json',dict(status='MISSING_SAVED_SCORES',missing=missing,model_forwards=0,does_not_gate_B=True))
    else:paired(out,private,sources,[61,62,63],arms,ct.protocol()['A']['comparisons'],2026091301,'POST_HOC_CONTROL_ANALYSIS')
    oldp=ct.read(old/'r0/experiments/lcrseg/docs/dpr_finite_v0_2/PROTOCOL.json');summary=mechanisms(old,oldp['tasks'],out)
    c.write_json(out/'PROVENANCE.json',dict(source=source,old_execution_source=ct.protocol()['prior_execution_source'],saved_score_files_read=reads,optimizer_updates=0,model_forwards=0,image_GT_array_access=0,missing=missing,not_a_gate_for_B=True))
    (out/'POST_HOC_SIGN_ANALYSIS.md').write_text('# Post-hoc SIGN analysis\n\nOld seeds61/62/63, never the old primary. Intervals are computed directly from saved paired patient scores with2000 shared physical-domain draws (2026091301), not by subtracting CI endpoints. No training, model forwards or image/GT arrays.\n\nProbe absolute coefficients match at fixed delta, but q_sign has RMS/q_align=1/sqrt(n_eff). R and Rnorm also change, so this does not prove eta is smaller. SIGN is not a strictly amplitude-matched direction ablation. See per-task b/Rnorm/eta and acceptance. accepted_algebra_eta_over_d0 is computed jointly from steps; the unrecorded exact FP32 applied norm is NOT_RECORDED, not imputed. Guard nonincrease is imposed, not independent memory evidence.\n')
