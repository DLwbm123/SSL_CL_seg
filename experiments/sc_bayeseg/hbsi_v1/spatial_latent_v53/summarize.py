"""Paired source-to-target endpoint for the single R-resolution change."""
import argparse,json
from pathlib import Path
from ..pixel_ce_v49.summarize import enriched,aggregate,effect,public_train_diagnostics,public_train_definition
from ..unet_transfer_v45.summarize import p10


def analyze(x):
    assert x['STATUS.json']['stage']=='COMPLETE' and x['EXIT.json']['exit_code']==0
    assert x['PREFLIGHT.json']['status']==x['MATCHING.json']['status']=='PASS'
    assert x['PREFLIGHT.json']['only_resolution_changed'] and all(x['PREFLIGHT.json']['initial_weights_q_prior_domain_rng_matching'].values())
    r=x['RESULTS.json'];l=x['ledger']
    assert r['formal_updates']==l['updates_reserved']==2400 and r['counter']['outer_backward_calls']==2400
    assert len(r['cells'])==4 and len(l['attempts'])==8 and all(v['status']=='PASS' for v in l['attempts'])
    assert {k:(v['status'],v['completed_steps']) for k,v in l['training_sessions'].items()}=={'R64_source':('COMPLETE',1600),'R64_target':('COMPLETE',800)}
    assert x['training']['source']['rows']==1600 and x['training']['target']['rows']==800
    assert r['source_target_pair'] and r['old_memory_and_Sb_unchanged'] and r['train_diagnostics']['R64']['complete'] and len(r['train_diagnostics']['R64']['patients'])==4
    raw=dict(R32_source=x['controls']['W_source'],R32_target=x['controls']['W_target'],R64_source=x['patients']['source'],R64_target=x['patients']['target'],U=x['controls']['U'])
    pp={a:{d:enriched(rows,1 if a=='U' else 3) for d,rows in ds.items()} for a,ds in raw.items()}
    means={a:{d:aggregate(rows) for d,rows in ds.items()} for a,ds in pp.items()}
    pairs={'R64_minus_R32_source':('R64_source','R32_source'),'R64_minus_R32_target':('R64_target','R32_target'),'R64_target_minus_source':('R64_target','R64_source'),'R64_target_minus_U':('R64_target','U')}
    effects={name:{d:effect(pp[a][d],pp[b][d]) for d in ('RUNMC','BMC')} for name,(a,b) in pairs.items()}
    tails={a:{d:dict(minimum=min(v['Dice'] for v in rows.values()),p10=p10([v['Dice'] for v in rows.values()])) for d,rows in ds.items()} for a,ds in pp.items()}
    source=effects['R64_minus_R32_source'];target=effects['R64_minus_R32_target'];b=target['BMC'];old=target['RUNMC']
    checks=dict(source_RUNMC_loss_le_0_5pp=source['RUNMC']['Dice']['mean']>=-.005,source_BMC_cold_loss_le_0_5pp=source['BMC']['Dice']['mean']>=-.005,
        BMC_Dice_ge_2pp=b['Dice']['mean']>=.02,BMC_positive_patients_ge_3=b['Dice']['positive']>=3,
        precision_ge_5pp=b['precision']['valid']==4 and b['precision']['mean']>=.05,
        recall_loss_le_2pp=b['recall']['valid']==4 and b['recall']['mean']>=-.02,
        RUNMC_extra_loss_le_0_5pp=old['Dice']['mean']>=-.005,
        no_new_empty=all(not(pp['R64_target'][d][i]['empty_predictions']>0 and pp['R32_target'][d][i]['empty_predictions']==0) for d in ('RUNMC','BMC') for i in pp['R64_target'][d]),
        HD95_worsening_le_10pct=all(means['R64_target'][d]['HD95_penalized_mm']['mean']<=1.1*means['R32_target'][d]['HD95_penalized_mm']['mean'] for d in ('RUNMC','BMC')))
    train_raw=dict(R32=x['control_train_diagnostics'],R64=r['train_diagnostics']['R64'])
    train={arm:enriched([dict(patient=str(v['case']),metrics=v['metrics']) for v in diag['patients']],3) for arm,diag in train_raw.items()}
    train_means={arm:aggregate(rows) for arm,rows in train.items()}
    anonymous={arm:{domain:[dict(case=i+1,metrics=rows[key]) for i,key in enumerate(sorted(rows))] for domain,rows in ds.items()} for arm,ds in pp.items()}
    return dict(status='COMPLETE',means=means,effects=effects,patient_tails=tails,mechanism_gate=dict(status='PASS' if all(checks.values()) else 'FAILED',checks=checks),anonymous_development_patients=anonymous,train_means=train_means,train_effect_R64_minus_R32=effect(train['R64'],train['R32']),training=x['training'],training_seeds=1,development_patients_per_domain=4,old_controls_rescored=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('snapshot');args=parser.parse_args()
    x=json.loads(Path(args.snapshot).read_text());a=analyze(x);out=Path(__file__).parent;r=x['RESULTS.json']
    cost=dict(status='COMPLETE',original_start=x['LAUNCH.json']['started'],deadline=x['LAUNCH.json']['deadline'],exit_time=x['EXIT.json']['time'],elapsed_seconds=x['EXIT.json']['time']-x['LAUNCH.json']['started'],root_bytes=x['root_bytes'],formal_updates=2400,preflight_reverse_calls=2,train_diagnostic_semantic_sample_forwards=12864,stage_seconds={v['name']:v['seconds_actual'] for v in x['ledger']['attempts']},counter=r['counter'])
    fmt=lambda v,scale=1:'NA' if v is None else f'{scale*v:.5f}'
    lines=['# v53 R64 结构潜变量源到目标配对','','新臂只将R空间网格从32×32变为64×64，同初始化独立完成源1600/目标800更新；封存R32与U-Net只读，均未重评分。','','| 臂 | 数据 | Dice % | precision % | recall % | 体积比 | NLL | Brier | HD95 mm |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm,ds in a['means'].items():
        for domain,m in ds.items():lines.append('| '+arm+' | '+domain+' dev | '+' | '.join(fmt(m[k]['mean'],100 if k in ('Dice','precision','recall') else 1) for k in ('Dice','precision','recall','volume_ratio','NLL','Brier','HD95_penalized_mm'))+' |')
    for arm,m in a['train_means'].items():lines.append('| '+arm+' | BMC train | '+' | '.join(fmt(m[k]['mean'],100 if k in ('Dice','precision','recall') else 1) for k in ('Dice','precision','recall','volume_ratio','NLL','Brier','HD95_penalized_mm'))+' |')
    lines+=['',f"冻结九门继续信号：**{a['mechanism_gate']['status']}**。",'','| 门槛 | 判定 |','|---|---|']
    for k,v in a['mechanism_gate']['checks'].items():lines.append(f"| {k} | {'PASS' if v else 'FAILED'} |")
    lines+=['','| 对比 | 域 | Dice pp | 患者范围 pp | 正/负 | precision pp | recall pp |','|---|---|---:|---:|---|---:|---:|']
    for name,ds in a['effects'].items():
        for domain,e in ds.items():
            d=e['Dice'];lines.append(f"| {name} | {domain} | {fmt(d['mean'],100)} | [{fmt(d['minimum'],100)}, {fmt(d['maximum'],100)}] | {d['positive']}/{d['negative']} | {fmt(e['precision']['mean'],100)} | {fmt(e['recall']['mean'],100)} |")
    lines+=['','完成后按封存结果补充源/目标九门、训练与开发FP/FN、旧域绝对遗忘和U-Net差距。',f"{cost['elapsed_seconds']:.2f}秒、根{cost['root_bytes']}字节；源1600/目标800正式更新、2预检反向、12864训练读出样本前向。",'','单训练seed、每域4名反复开发患者；MC重复与训练切片不是独立患者。','', '[冻结协议](PREREG_ZH.md)；[匿名分析](ANALYSIS.json)；[成本](EXECUTION_SUMMARY.json)。']
    train_raw=dict(R32=x['control_train_diagnostics'],R64=r['train_diagnostics']['R64'])
    outputs={'ANALYSIS.json':a,'EXECUTION_SUMMARY.json':cost,'TRAIN_DIAGNOSTICS.json':public_train_diagnostics(train_raw),'TRAIN_DEFINITION.json':public_train_definition(x['TRAIN_DEFINITION.json'])}
    for n,v in outputs.items():(out/n).write_text(json.dumps(v,indent=2)+'\n')
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(a['mechanism_gate']))


if __name__=='__main__':main()
