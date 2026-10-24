"""Patient ED/ES paired aggregates only; no model or image access."""
import json
from pathlib import Path
import numpy as np
from ..main_style_ssl_v1.runtime import write
from ..siemens_lonly_v1.run import aggregate

def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(v) for v in Path(p).read_text().splitlines()]
def patients(rr):
    out={}
    for r in rr:
        p=out.setdefault(r['patient'],{});assert r['phase'] not in p;p[r['phase']]=float(np.mean(r['dice']))
    assert all(set(v)=={'ED','ES'} for v in out.values())
    return {k:float(np.mean(list(v.values()))) for k,v in out.items()}
def compare(a,b):
    aa,bb=patients(a),patients(b);assert aa.keys()==bb.keys();v=np.array([(bb[k]-aa[k])*100 for k in sorted(aa)])
    idx=np.random.default_rng(163).integers(0,len(v),(10000,len(v)))
    return dict(mean_pp=float(v.mean()),CI95_pp=np.quantile(v[idx].mean(1),[.025,.975]).tolist(),p10_pp=float(np.quantile(v,.1)),over1=int((v < -1).sum()),over2=int((v < -2).sum()),patients=len(v))
def tail(v):return v['over1']<=2 and v['over2']==0 and v['p10_pp']>=-.5
def ni(v):return v['mean_pp']>=-.2 and v['CI95_pp'][0]>=-.2

def source_gate(cfg):
    root=Path(cfg['root']);baseline=rows(cfg['original_source_rows']);assert len(patients(baseline))==23
    score=aggregate(baseline);assert abs(score['primary_dice']*100-80.36740663770262)<1e-9
    n=root/'source/N';n.mkdir(parents=True,exist_ok=False)
    (n/'private_Siemens_val.jsonl').write_text(Path(cfg['original_source_rows']).read_text());write(n/'RESULTS.json',dict(status='REUSED',metrics={'Siemens':score},new_model_samples=0))
    logs=rows(root/'SOURCE_DIAGNOSTICS.jsonl');assert [x['step'] for x in logs]==list(range(500,6001,500))
    vals=[x['EMA']['calibration_loss'] for x in logs[-3:]];changes=[abs(b-a)/max(1,abs(a)) for a,b in zip(vals,vals[1:])]
    stable=max(changes)<=.01;checks={'calibration_stable':stable};summaries={}
    for arm in ('Q','R'):
        v=compare(baseline,rows(root/'source'/arm/'private_Siemens_val.jsonl'));summaries[arm]=v
        checks[arm]=v['mean_pp']>=-.2 and v['CI95_pp'][0]>=-.3 and tail(v)
    result=dict(status='PASS' if all(checks.values()) else 'STOP',checks=checks,last_three_EMA_fixed_calibration=vals,relative_changes=changes,source_vs_original=summaries,original_source_score_percent=score['primary_dice']*100,thresholds_unchanged=True)
    write(root/'SOURCE_GATE.json',result);return result
