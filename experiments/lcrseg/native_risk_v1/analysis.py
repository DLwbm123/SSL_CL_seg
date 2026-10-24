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
    score=aggregate(baseline);base_dir=root/'source/B';base_dir.mkdir(parents=True,exist_ok=False);(base_dir/'private_Siemens_val.jsonl').write_text(Path(cfg['original_source_rows']).read_text());write(base_dir/'RESULTS.json',dict(status='REUSED_ORIGINAL_SOURCE',metrics={'Siemens':score},new_forward=0));assert abs(score['primary_dice']*100-80.36740663770262)<1e-9
    logs=rows(root/'SOURCE_DIAGNOSTICS_F.jsonl');assert [x['step'] for x in logs]==[1500,1750,2000]
    vals=[x['EMA']['audit_objective'] for x in logs];changes=[abs(b-a)/max(1,abs(a)) for a,b in zip(vals,vals[1:])]
    v=compare(baseline,rows(root/'source/F/private_Siemens_val.jsonl'));risk=read(root/'RISK_SOURCE_FINAL.json')
    checks=dict(training_sufficient=all(x<=.01 for x in changes),source_capability=v['mean_pp']>=-.2 and v['CI95_pp'][0]>=-.3 and tail(v),source_risk_feasible=risk['status']=='PASS')
    result=dict(status='PASS' if all(checks.values()) else 'STOP',checks=checks,source_vs_original=v,last_three_EMA_fixed_objective=vals,relative_changes=changes,training_components=logs,risk=risk,original_source_score_percent=score['primary_dice']*100,thresholds_unchanged=True)
    write(root/'SOURCE_GATE.json',result);return result
