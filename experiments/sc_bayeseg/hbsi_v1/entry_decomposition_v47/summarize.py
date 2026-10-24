"""Paired entry/learning/net effects from sealed patient records only."""
import argparse,json,math
from pathlib import Path
from statistics import mean,pstdev
from ..unet_transfer_v45.summarize import patients,METRICS


def split_effect(before,zero,end):
    j=zero-before;l=end-zero;g=end-before
    assert math.isclose(j+l,g,abs_tol=1e-12)
    return dict(J=j,L=l,G=g)


def trigger(values):
    return mean(values)<=-.01 and sum(v<0 for v in values)>=3


def extra_metrics(records):
    out={}
    for r in records:
        rows=[]
        for m in r['metrics']:
            c=m['classes'][0]
            rows.append(dict(precision=c['TP']/c['predicted'] if c['predicted'] else None,recall=c['TP']/c['GT'] if c['GT'] else None,volume_ratio=c['predicted']/c['GT'] if c['GT'] else None))
        out[r['patient']]={k:mean(v[k] for v in rows) if all(v[k] is not None for v in rows) else None for k in rows[0]}
    return out


def analyze(x):
    assert x['STATUS.json']['stage']=='COMPLETE' and x['EXIT.json']['exit_code']==0
    r=x['RESULTS.json'];assert r['formal_updates']==r['optimizer_updates']==r['reverse_calls']==0
    assert r['source_state_unchanged'] and r['new_evaluation_cells']==2 and r['reused_endpoint_cells']==4
    assert x['MATCHING.json']['status']=='PASS' and x['ledger']['updates_reserved']==0
    assert all(v['status']=='PASS' for v in x['ledger']['attempts'])
    assert x['parent_EXIT']['exit_code']==0 and x['parent_RESULTS']['new_formal_updates']==4800
    result={}
    for arm,phases in x['patients'].items():
        pp={phase:patients(rows,3) for phase,rows in phases.items()}
        ids=sorted(pp['source']);assert all(sorted(rows)==ids for rows in pp.values())
        cases=[]
        for n,i in enumerate(ids,1):
            effects={k:split_effect(pp['source'][i][k],pp['zero'][i][k],pp['target'][i][k]) for k in METRICS}
            cases.append(dict(case=n,effects=effects))
        agg={e:{k:dict(mean=mean(v['effects'][k][e] for v in cases),minimum=min(v['effects'][k][e] for v in cases),maximum=max(v['effects'][k][e] for v in cases),positive=sum(v['effects'][k][e]>1e-12 for v in cases),negative=sum(v['effects'][k][e]<-1e-12 for v in cases)) for k in METRICS} for e in ('J','L','G')}
        means={phase:{k:sum(v[k] for v in rows.values()) if k=='empty_predictions' else mean(v[k] for v in rows.values()) for k in METRICS} for phase,rows in pp.items()}
        extras={phase:extra_metrics(rows) for phase,rows in phases.items()}
        extras={phase:{k:dict(mean=mean(v[k] for v in rows.values() if v[k] is not None) if any(v[k] is not None for v in rows.values()) else None,valid_patients=sum(v[k] is not None for v in rows.values())) for k in ('precision','recall','volume_ratio')} for phase,rows in extras.items()}
        repeated={}
        for phase,rows in phases.items():
            scores=[mean(r['metrics'][j]['classes'][0]['Dice'] for r in rows) for j in range(3)]
            repeated[phase]=dict(Dice=scores,minimum=min(scores),maximum=max(scores),sd=pstdev(scores))
        mc_effects={e:[split_effect(repeated['source']['Dice'][j],repeated['zero']['Dice'][j],repeated['target']['Dice'][j])[e] for j in range(3)] for e in ('J','L','G')}
        result[arm]=dict(means=means,effects=agg,anonymous_patient_effects=cases,derived_foreground_metrics=extras,inference_repeats=repeated,paired_repeat_effects=mc_effects,warm_start_trigger=trigger([v['effects']['Dice']['J'] for v in cases]))
    triggered=[a for a in ('W','S') if result[a]['warm_start_trigger']]
    decision=dict(warm_start_eligible_arms=triggered,preferred_arm=triggered[0] if triggered else None,meaning='budget_trigger_not_causal_proof',next='one_warm_start_protocol' if triggered else 'check_v46_gate_then_conditional_fit_protocol')
    return dict(status='COMPLETE',arms=result,decision=decision,new_model_training_updates=0,old_models_rescored=0,training_seeds=1,patients=4,inference_repeats=3)


def main():
    p=argparse.ArgumentParser();p.add_argument('snapshot');args=p.parse_args();x=json.loads(Path(args.snapshot).read_text());a=analyze(x);out=Path(__file__).parent
    cost=dict(status='COMPLETE',original_start=x['LAUNCH.json']['started'],compute_started=x['DEPENDENCY_READY.json']['compute_started'],exit_time=x['EXIT.json']['time'],elapsed_including_queue=x['EXIT.json']['time']-x['LAUNCH.json']['started'],compute_elapsed=x['EXIT.json']['time']-x['DEPENDENCY_READY.json']['compute_started'],original_deadline=x['LAUNCH.json']['deadline'],root_bytes=x['root_bytes'],new_evaluation_cells=2,reused_endpoint_cells=4,formal_updates=0,optimizer_updates=0,reverse_calls=0,probe_semantic_forwards=16,stage_seconds={v['name']:v['seconds_actual'] for v in x['ledger']['attempts']})
    lines=['# v47 真实目标入口分解','','本报告仅新增S/W各一个BMC零步读出；source和target800指标直接复用v46，来源模型不更新、不重新评分。J为入口切换，L为随后学习，G为完整净变化；逐患者J+L=G。','','| 骨干 | source Dice % | zero Dice % | target Dice % | J pp | L pp | G pp | 入口负向患者 | warm-start触发 |','|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for arm,v in a['arms'].items():
        m=v['means'];e=v['effects'];lines.append(f"| {arm} | {100*m['source']['Dice']:.5f} | {100*m['zero']['Dice']:.5f} | {100*m['target']['Dice']:.5f} | {100*e['J']['Dice']['mean']:+.5f} | {100*e['L']['Dice']['mean']:+.5f} | {100*e['G']['Dice']['mean']:+.5f} | {e['J']['Dice']['negative']}/4 | {v['warm_start_trigger']} |")
    lines+=['','| 骨干 | 效应 | Dice范围 pp | 正/负 | NLL | Brier | HD95 mm |','|---|---|---:|---:|---:|---:|---:|']
    for arm,v in a['arms'].items():
        for name,e in v['effects'].items():
            d=e['Dice'];lines.append(f"| {arm} | {name} | [{100*d['minimum']:+.5f},{100*d['maximum']:+.5f}] | {d['positive']}/{d['negative']} | {e['NLL']['mean']:+.7f} | {e['Brier']['mean']:+.7f} | {e['HD95_penalized_mm']['mean']:+.5f} |")
    lines+=['','完整匿名逐患者效应、三个推断重复的波动及由封存TP/count推导的precision/recall/体积比见ANALYSIS.json。零分母记NA，三个推断重复不是三个训练seed。', '', f"预算触发：{a['decision']}。触发不是初始化主因或修复成功的证明；不满足时不直接运行warm-start。先结合v46完整六门槛决定是否仍需训练拟合诊断，任何后继训练另立协议。",'',f"排队及执行共{cost['elapsed_including_queue']:.2f}秒，实际计算阶段{cost['compute_elapsed']:.2f}秒，根{cost['root_bytes']}字节。0训练/副本Adam/反向，16语义前向用于旧memory路由一致性，2个完整8×4MC、3重复、4患者新读出；推断不是免费。",'', 'v46原判定和旧v21初始化非主因的负证据保留；单seed/4名开发患者不足以证明独立泛化、可靠不确定性或CL成功。', '', '[冻结协议](PREREG_ZH.md)；[完整匿名分析](ANALYSIS.json)；[成本](EXECUTION_SUMMARY.json)。']
    for n,v in [('ANALYSIS.json',a),('EXECUTION_SUMMARY.json',cost),('MATCHING.json',x['MATCHING.json'])]:(out/n).write_text(json.dumps(v,indent=2)+'\n')
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');print(json.dumps(a['decision']))


if __name__=='__main__':main()
