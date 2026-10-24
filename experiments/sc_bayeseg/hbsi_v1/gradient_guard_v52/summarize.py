"""Verify and summarize the sealed projected-gradient intervention."""
import argparse,json
from pathlib import Path
from ..pixel_ce_v49.summarize import enriched,aggregate,effect,public_train_diagnostics,public_train_definition
from ..unet_transfer_v45.summarize import p10


def gate_for_arm(e,means,new,old):
    b=e['BMC'];a=e['RUNMC']
    checks=dict(BMC_Dice_ge_2pp=b['Dice']['mean']>=.02,BMC_positive_patients_ge_3=b['Dice']['positive']>=3,
        precision_ge_5pp=b['precision']['valid']==4 and b['precision']['mean']>=.05,
        recall_loss_le_2pp=b['recall']['valid']==4 and b['recall']['mean']>=-.02,
        RUNMC_extra_loss_le_0_5pp=a['Dice']['mean']>=-.005,
        no_new_empty=all(not(new[d][i]['empty_predictions']>0 and old[d][i]['empty_predictions']==0) for d in new for i in new[d]),
        HD95_worsening_le_10pct=all(means['G'][d]['HD95_penalized_mm']['mean']<=1.1*means['C0'][d]['HD95_penalized_mm']['mean'] for d in ('RUNMC','BMC')))
    return dict(status='PASS' if all(checks.values()) else 'FAILED',checks=checks)


def analyze(x):
    assert x['STATUS.json']['stage']=='COMPLETE' and x['EXIT.json']['exit_code']==0
    assert x['PREFLIGHT.json']['status']==x['MATCHING.json']['status']=='PASS'
    assert x['PREFLIGHT.json']['same_full_loss_and_gradient'] and x['PREFLIGHT.json']['projected_gradient_formula']
    r=x['RESULTS.json'];l=x['ledger']
    assert r['formal_updates']==l['updates_reserved']==800 and len(r['cells'])==2
    assert len(l['training_sessions'])==1 and l['training_sessions']['G']['status']=='COMPLETE' and l['training_sessions']['G']['completed_steps']==800
    assert len(l['attempts'])==5 and all(v['status']=='PASS' for v in l['attempts'])
    assert r['old_memory_and_Sb_unchanged'] and r['common_entrance'] and r['counter']['outer_backward_calls']==800 and r['training_extra_reverse_calls']==1600
    assert len(r['gradient_audit']['adam_audits'])==4 and all(v['steps']==800 for v in r['gradient_audit']['groups'].values())
    assert r['train_diagnostics']['G']['complete'] and len(r['train_diagnostics']['G']['patients'])==4 and x['training']['rows']==800
    raw=dict(x['controls'],G=x['patients'])
    pp={a:{d:enriched(rows,1 if a=='U' else 3) for d,rows in ds.items()} for a,ds in raw.items()}
    means={a:{d:aggregate(rows) for d,rows in ds.items()} for a,ds in pp.items()}
    pairs={'G_minus_C0':('G','C0'),'G_minus_source':('G','W_source'),'G_minus_U':('G','U')}
    effects={name:{d:effect(pp[a][d],pp[b][d]) for d in ('RUNMC','BMC')} for name,(a,b) in pairs.items()}
    tails={a:{d:dict(minimum=min(v['Dice'] for v in rows.values()),p10=p10([v['Dice'] for v in rows.values()])) for d,rows in ds.items()} for a,ds in pp.items()}
    train_raw=dict(x['control_train_diagnostics'],G=r['train_diagnostics']['G'])
    train={arm:enriched([dict(patient=str(v['case']),metrics=v['metrics']) for v in diag['patients']],3) for arm,diag in train_raw.items()}
    train_means={arm:aggregate(rows) for arm,rows in train.items()}
    anonymous={arm:{domain:[dict(case=i+1,metrics=rows[key]) for i,key in enumerate(sorted(rows))] for domain,rows in ds.items()} for arm,ds in pp.items()}
    gate=gate_for_arm(effects['G_minus_C0'],means,pp['G'],pp['C0'])
    return dict(status='COMPLETE',means=means,effects=effects,patient_tails=tails,mechanism_gate=gate,anonymous_development_patients=anonymous,train_means=train_means,train_effect_G_minus_C0=effect(train['G'],train['C0']),training=x['training'],training_seeds=1,development_patients_per_domain=4,old_controls_rescored=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('snapshot');args=parser.parse_args()
    x=json.loads(Path(args.snapshot).read_text());a=analyze(x);out=Path(__file__).parent;r=x['RESULTS.json']
    cost=dict(status='COMPLETE',original_start=x['LAUNCH.json']['started'],deadline=x['LAUNCH.json']['deadline'],exit_time=x['EXIT.json']['time'],elapsed_seconds=x['EXIT.json']['time']-x['LAUNCH.json']['started'],root_bytes=x['root_bytes'],formal_updates=800,preflight_model_reverse_calls=5,training_extra_reverse_calls=1600,train_diagnostic_semantic_sample_forwards=12864,stage_seconds={v['name']:v['seconds_actual'] for v in x['ledger']['attempts']},counter=r['counter'])
    fmt=lambda v,scale=1:'NA' if v is None else f'{scale*v:.5f}'
    lines=['# v52 分组任务梯度保护终点','','G从同一BMC目标入口训练800步。相比原C0，仅逐组投影与标签任务相反的其余梯度；原目标数值不变。C0、U和自身源模型均只读封存。','','| 臂 | 数据 | Dice % | precision % | recall % | 体积比 | NLL | Brier | HD95 mm |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm,ds in a['means'].items():
        for domain,m in ds.items():lines.append('| '+arm+' | '+domain+' dev | '+' | '.join(fmt(m[k]['mean'],100 if k in ('Dice','precision','recall') else 1) for k in ('Dice','precision','recall','volume_ratio','NLL','Brier','HD95_penalized_mm'))+' |')
    for arm,m in a['train_means'].items():lines.append('| '+arm+' | BMC train | '+' | '.join(fmt(m[k]['mean'],100 if k in ('Dice','precision','recall') else 1) for k in ('Dice','precision','recall','volume_ratio','NLL','Brier','HD95_penalized_mm'))+' |')
    lines+=['',f"冻结七门继续信号：**{a['mechanism_gate']['status']}**。",'','| 门槛 | 判定 |','|---|---|']
    for k,v in a['mechanism_gate']['checks'].items():lines.append(f"| {k} | {'PASS' if v else 'FAILED'} |")
    lines+=['','| 对比 | 域 | Dice pp | 患者范围 pp | 正/负 | precision pp | recall pp |','|---|---|---:|---:|---|---:|---:|']
    for name,ds in a['effects'].items():
        for domain,e in ds.items():
            d=e['Dice'];lines.append(f"| {name} | {domain} | {fmt(d['mean'],100)} | [{fmt(d['minimum'],100)}, {fmt(d['maximum'],100)}] | {d['positive']}/{d['negative']} | {fmt(e['precision']['mean'],100)} | {fmt(e['recall']['mean'],100)} |")
    lines+=['','**实际结论：七门失败四项，关闭 G 投影规则。** G 对 C0 的 BMC Dice −4.1418pp，4名患者仅1名改善、3名下降（最差−10.5525pp）；precision −5.1400pp、recall +6.1789pp。RUNMC Dice −1.2189pp，超过额外下降0.5pp门。BMC 对标准 U-Net 仍低21.4593pp，4/4患者均低；相对自身 W_source，RUNMC Dice 下降2.2271pp。保护局部任务梯度没有产生可接受的目标收益或旧域保留。', '完整 BMC 训练患者 Dice 比 C0 低15.0728pp、precision 低17.2692pp、recall 高2.6759pp。空片平均 FP/片由956.57升至2149.78，小/中/大片 FP也增加；非空片FN有所下降。这与开发集的高召回低precision一致，显示主要权衡转向过度预测前景。', '800个正式 batch 中，q/eta/phi 分组出现反向梯度并触发投影分别265/139/166次；四个抽样位置的实际Adam任务内积多为负，但末位置q、eta为正。欧氏投影的局部一阶性质不能保证Adam实际步或患者终点。冲突存在并不等于它是跨域差距主因；在该固定规则下，纠正这些冲突反而加重假阳性。',f"{cost['elapsed_seconds']:.2f}秒、根{cost['root_bytes']}字节；800正式backward/Adam、1600额外梯度分解反向、3目标预检前向/5反向、12864训练读出样本前向。",'','单训练seed、每域4名反复开发患者。旧控制无重训/重评分；没有U增量证据。','', '[冻结协议](PREREG_ZH.md)；[匿名分析](ANALYSIS.json)；[成本](EXECUTION_SUMMARY.json)；[梯度审计](GRADIENT_AUDIT.json)。']
    train_raw=dict(x['control_train_diagnostics'],G=r['train_diagnostics']['G'])
    outputs={'ANALYSIS.json':a,'EXECUTION_SUMMARY.json':cost,'TRAIN_DIAGNOSTICS.json':public_train_diagnostics(train_raw),'TRAIN_DEFINITION.json':public_train_definition(x['TRAIN_DEFINITION.json']),'GRADIENT_AUDIT.json':r['gradient_audit']}
    for n,v in outputs.items():(out/n).write_text(json.dumps(v,indent=2)+'\n')
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(a['mechanism_gate']))


if __name__=='__main__':main()
