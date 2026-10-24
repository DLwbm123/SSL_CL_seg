"""Verify and summarize the sealed empty-only intervention."""
import argparse,json
from pathlib import Path
from ..pixel_ce_v49.summarize import enriched,aggregate,effect,gates,public_train_diagnostics,public_train_definition
from ..unet_transfer_v45.summarize import p10


def analyze(x):
    assert x['STATUS.json']['stage']=='COMPLETE' and x['EXIT.json']['exit_code']==0
    assert x['PREFLIGHT.json']['status']==x['MATCHING.json']['status']=='PASS'
    assert x['PREFLIGHT.json']['actual_empty_batch'] and x['PREFLIGHT.json']['control_loss_and_gradient_exact']
    r=x['RESULTS.json'];l=x['ledger']
    assert r['formal_updates']==l['updates_reserved']==800 and len(r['cells'])==2
    assert len(l['training_sessions'])==1 and l['training_sessions']['C3']['status']=='COMPLETE' and l['training_sessions']['C3']['completed_steps']==800
    assert len(l['attempts'])==5 and all(v['status']=='PASS' for v in l['attempts'])
    assert r['old_memory_and_Sb_unchanged'] and r['common_entrance'] and r['counter']['outer_backward_calls']==800
    assert r['train_diagnostics']['C3']['complete'] and len(r['train_diagnostics']['C3']['patients'])==4 and x['training']['rows']==800
    assert x['training']['empty_exposures']==sum(n for d in x['TRAIN_DEFINITION.json']['cases'] for n,g in zip(d['exposures'],d['strata']) if g=='empty')
    raw=dict(x['controls'],C3=x['patients'])
    pp={a:{d:enriched(rows,1 if a=='U' else 3) for d,rows in ds.items()} for a,ds in raw.items()}
    means={a:{d:aggregate(rows) for d,rows in ds.items()} for a,ds in pp.items()}
    pairs={'C3_minus_C0':('C3','C0'),'C3_minus_C2':('C3','C2'),'C3_minus_source':('C3','W_source'),'C3_minus_U':('C3','U')}
    effects={name:{d:effect(pp[a][d],pp[b][d]) for d in ('RUNMC','BMC')} for name,(a,b) in pairs.items()}
    tails={a:{d:dict(minimum=min(v['Dice'] for v in rows.values()),p10=p10([v['Dice'] for v in rows.values()])) for d,rows in ds.items()} for a,ds in pp.items()}
    train_raw=dict(x['control_train_diagnostics'],C3=r['train_diagnostics']['C3'])
    train={arm:enriched([dict(patient=str(v['case']),metrics=v['metrics']) for v in diag['patients']],3) for arm,diag in train_raw.items()}
    train_means={arm:aggregate(rows) for arm,rows in train.items()}
    anonymous={arm:{domain:[dict(case=i+1,metrics=rows[key]) for i,key in enumerate(sorted(rows))] for domain,rows in ds.items()} for arm,ds in pp.items()}
    return dict(status='COMPLETE',means=means,effects=effects,patient_tails=tails,mechanism_gate=gates(effects['C3_minus_C0'],means,pp['C3'],pp['C0']),anonymous_development_patients=anonymous,train_means=train_means,train_effect_C3_minus_C0=effect(train['C3'],train['C0']),train_effect_C3_minus_C2=effect(train['C3'],train['C2']),training=x['training'],training_seeds=1,development_patients_per_domain=4,old_controls_rescored=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('snapshot');args=parser.parse_args()
    x=json.loads(Path(args.snapshot).read_text());a=analyze(x);out=Path(__file__).parent;r=x['RESULTS.json']
    cost=dict(status='COMPLETE',original_start=x['LAUNCH.json']['started'],deadline=x['LAUNCH.json']['deadline'],exit_time=x['EXIT.json']['time'],elapsed_seconds=x['EXIT.json']['time']-x['LAUNCH.json']['started'],root_bytes=x['root_bytes'],formal_updates=800,preflight_model_reverse_calls=3,preflight_fixture_reverse_calls=1,train_diagnostic_semantic_sample_forwards=12864,stage_seconds={v['name']:v['seconds_actual'] for v in x['ledger']['attempts']},counter=r['counter'])
    fmt=lambda v,scale=1:'NA' if v is None else f'{scale*v:.5f}'
    lines=['# v51 仅空切片抑制终点','','C3从同一BMC目标入口训练800步。相比原C0，只在有标签空片添加最大前景罚；C0、v50 C2、U和自身源模型均只读封存。','','| 臂 | 数据 | Dice % | precision % | recall % | 体积比 | NLL | Brier | HD95 mm |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm,ds in a['means'].items():
        for domain,m in ds.items():lines.append('| '+arm+' | '+domain+' dev | '+' | '.join(fmt(m[k]['mean'],100 if k in ('Dice','precision','recall') else 1) for k in ('Dice','precision','recall','volume_ratio','NLL','Brier','HD95_penalized_mm'))+' |')
    for arm,m in a['train_means'].items():lines.append('| '+arm+' | BMC train | '+' | '.join(fmt(m[k]['mean'],100 if k in ('Dice','precision','recall') else 1) for k in ('Dice','precision','recall','volume_ratio','NLL','Brier','HD95_penalized_mm'))+' |')
    lines+=['',f"冻结七门继续信号：**{a['mechanism_gate']['status']}**。",'','| 门槛 | 判定 |','|---|---|']
    for k,v in a['mechanism_gate']['checks'].items():lines.append(f"| {k} | {'PASS' if v else 'FAILED'} |")
    lines+=['','| 对比 | 域 | Dice pp | 患者范围 pp | 正/负 | precision pp | recall pp |','|---|---|---:|---:|---|---:|---:|']
    for name,ds in a['effects'].items():
        for domain,e in ds.items():
            d=e['Dice'];lines.append(f"| {name} | {domain} | {fmt(d['mean'],100)} | [{fmt(d['minimum'],100)}, {fmt(d['maximum'],100)}] | {d['positive']}/{d['negative']} | {fmt(e['precision']['mean'],100)} | {fmt(e['recall']['mean'],100)} |")
    lines+=['','最终科学解释须在封存结果后补充，包含训练空片FP/非空FN、开发七门、旧域绝对遗忘和U-Net差距。',f"{cost['elapsed_seconds']:.2f}秒、根{cost['root_bytes']}字节；800正式更新、3预检模型反向+1fixture反向、12864训练读出样本前向。",'','单训练seed、每域4名反复开发患者。旧控制无重训/重评分；不得以训练切片或MC重复当独立患者。','', '[冻结协议](PREREG_ZH.md)；[匿名分析](ANALYSIS.json)；[成本](EXECUTION_SUMMARY.json)。']
    train_raw=dict(x['control_train_diagnostics'],C3=r['train_diagnostics']['C3'])
    outputs={'ANALYSIS.json':a,'EXECUTION_SUMMARY.json':cost,'TRAIN_DIAGNOSTICS.json':public_train_diagnostics(train_raw),'TRAIN_DEFINITION.json':public_train_definition(x['TRAIN_DEFINITION.json'])}
    for n,v in outputs.items():(out/n).write_text(json.dumps(v,indent=2)+'\n')
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(a['mechanism_gate']))


if __name__=='__main__':main()
