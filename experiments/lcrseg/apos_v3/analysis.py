"""Saved-row patient comparisons; no model or image access."""
from pathlib import Path
from ..apos_v2.analysis import read,rows,compare,tail,ni,patients
from ..main_style_ssl_v1.runtime import write
from ..siemens_lonly_v1.run import aggregate

def source_gate(cfg):
    root=Path(cfg['root']);baseline=rows(cfg['original_source_rows']);score=aggregate(baseline);assert len(patients(baseline))==23 and abs(score['primary_dice']*100-80.36740663770262)<1e-9
    out=root/'source/N';out.mkdir(parents=True,exist_ok=False);(out/'private_Siemens_val.jsonl').write_text(Path(cfg['original_source_rows']).read_text());write(out/'RESULTS.json',dict(status='REUSED',metrics={'Siemens':score},new_model_samples=0))
    checks={};summary={}
    for arm in ('Q','R'):
        v=compare(baseline,rows(root/'source'/arm/'private_Siemens_val.jsonl'));summary[arm]=v;s=read(root/'source'/arm/'SOLVER_AGGREGATES.json')
        checks[arm]=abs(v['mean_pp'])<=1e-10 and max(abs(t) for t in v['CI95_pp'])<=1e-10 and abs(v['p10_pp'])<=1e-10 and v['over1']==v['over2']==0 and s['source_probability_max_error']['max']<=5e-7 and s['source_argmax_exact']['sum']==s['source_argmax_exact']['count']
    result=dict(status='PASS' if all(checks.values()) else 'STOP',checks=checks,source_vs_original=summary,source_function_and_source_ability_joint_gate=True,thresholds_unchanged=True)
    write(root/'SOURCE_GATE.json',result);return result
