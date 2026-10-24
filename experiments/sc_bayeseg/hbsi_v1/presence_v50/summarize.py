"""Sealed patient-level comparison for the slice-presence intervention."""
import argparse,json
from pathlib import Path
from statistics import mean
from ..pixel_ce_v49.summarize import enriched,aggregate,effect,gates,public_train_diagnostics,public_train_definition
from ..unet_transfer_v45.summarize import p10


def analyze(x):
    assert x['STATUS.json']['stage']=='COMPLETE' and x['EXIT.json']['exit_code']==0
    assert x['original_failure']['EXIT.json']['exit_code']==1 and x['original_failure']['STATUS.json']['stage']=='FAILED'
    assert x['PREFLIGHT.json']['status']==x['MATCHING.json']['status']=='PASS'
    assert x['REPAIR_RECORD.json']['status']=='BOUNDED_CONTINUATION'
    assert x['REPAIR_RECORD.json']['source_payload_unchanged'] and x['REPAIR_RECORD.json']['train_definition_equal']
    r=x['RESULTS.json'];l=x['ledger']
    assert r['formal_updates']==l['updates_reserved']==800 and len(r['cells'])==2
    assert len(l['training_sessions'])==1 and l['training_sessions']['C2']['status']=='COMPLETE' and l['training_sessions']['C2']['completed_steps']==800
    assert len(l['attempts'])==6 and l['attempts'][0]['name']=='preflight' and l['attempts'][0]['status']=='FAILED'
    assert all(v['status']=='PASS' for v in l['attempts'][1:])
    assert r['old_memory_and_Sb_unchanged'] and r['common_entrance'] and r['counter']['outer_backward_calls']==800
    assert r['train_diagnostics']['C2']['complete'] and len(r['train_diagnostics']['C2']['patients'])==4 and x['training']['rows']==800
    assert x['training']['empty_exposures']==sum(n for d in x['TRAIN_DEFINITION.json']['cases'] for n,g in zip(d['exposures'],d['strata']) if g=='empty')
    raw=dict(x['controls'],C2=x['patients'])
    pp={a:{d:enriched(rows,1 if a=='U' else 3) for d,rows in ds.items()} for a,ds in raw.items()}
    means={a:{d:aggregate(rows) for d,rows in ds.items()} for a,ds in pp.items()}
    comparisons={'C2_minus_C0':('C2','C0'),'C2_minus_C1':('C2','C1'),'C2_minus_source':('C2','W_source'),'C2_minus_U':('C2','U')}
    effects={name:{d:effect(pp[a][d],pp[b][d]) for d in ('RUNMC','BMC')} for name,(a,b) in comparisons.items()}
    tails={a:{d:dict(minimum=min(v['Dice'] for v in rows.values()),p10=p10([v['Dice'] for v in rows.values()])) for d,rows in ds.items()} for a,ds in pp.items()}
    train_raw=dict(x['control_train_diagnostics'],C2=r['train_diagnostics']['C2'])
    train={arm:enriched([dict(patient=str(v['case']),metrics=v['metrics']) for v in diag['patients']],3) for arm,diag in train_raw.items()}
    train_means={arm:aggregate(rows) for arm,rows in train.items()}
    anonymous={arm:{domain:[dict(case=i+1,metrics=rows[key]) for i,key in enumerate(sorted(rows))] for domain,rows in ds.items()} for arm,ds in pp.items()}
    return dict(status='COMPLETE',means=means,effects=effects,patient_tails=tails,mechanism_gate=gates(effects['C2_minus_C0'],means,pp['C2'],pp['C0']),anonymous_development_patients=anonymous,train_means=train_means,train_effect_C2_minus_C0=effect(train['C2'],train['C0']),training=x['training'],training_seeds=1,development_patients_per_domain=4,old_controls_rescored=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('snapshot');args=parser.parse_args()
    x=json.loads(Path(args.snapshot).read_text());a=analyze(x);out=Path(__file__).parent;r=x['RESULTS.json']
    cost=dict(status='COMPLETE_AFTER_BOUNDED_REPAIR',original_start=x['LAUNCH.json']['started'],deadline=x['LAUNCH.json']['deadline'],exit_time=x['EXIT.json']['time'],elapsed_seconds=x['EXIT.json']['time']-x['LAUNCH.json']['started'],root_bytes=x['root_bytes'],formal_updates=800,first_failed_preflight=True,first_failure_type=x['original_failure']['STATUS.json']['error_type'],preflight_model_reverse_calls=3,preflight_fixture_reverse_calls=1,repair_model_reverse_calls=0,train_diagnostic_semantic_sample_forwards=12864,stage_seconds={v['name']:v['seconds_actual'] for v in x['ledger']['attempts']},counter=r['counter'])
    fmt=lambda v,scale=1:'NA' if v is None else f'{scale*v:.5f}'
    lines=['# v50 切片存在损失与空片假阳性','','C2从与C0/v49 C1相同的BMC目标入口训练800步；原类平衡CE+Dice保留，只加由L标签定义的切片最大前景存在损失。C0、C1、W_source和标准U-Net只读封存，均未重训或重评分。','','| 臂 | 数据 | Dice % | precision % | recall % | 体积比 | NLL | Brier | HD95 mm |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm,ds in a['means'].items():
        for domain,m in ds.items():lines.append('| '+arm+' | '+domain+' dev | '+' | '.join(fmt(m[k]['mean'],100 if k in ('Dice','precision','recall') else 1) for k in ('Dice','precision','recall','volume_ratio','NLL','Brier','HD95_penalized_mm'))+' |')
    for arm,m in a['train_means'].items():lines.append('| '+arm+' | BMC train | '+' | '.join(fmt(m[k]['mean'],100 if k in ('Dice','precision','recall') else 1) for k in ('Dice','precision','recall','volume_ratio','NLL','Brier','HD95_penalized_mm'))+' |')
    lines+=['',f"冻结七门继续信号：**{a['mechanism_gate']['status']}**。",'','| 门槛 | 判定 |','|---|---|']
    for k,v in a['mechanism_gate']['checks'].items():lines.append(f"| {k} | {'PASS' if v else 'FAILED'} |")
    lines+=['','| 对比 | 域 | Dice pp | 患者范围 pp | 正/负 | precision pp | recall pp |','|---|---|---:|---:|---|---:|---:|']
    for name,ds in a['effects'].items():
        for domain,e in ds.items():
            d=e['Dice'];lines.append(f"| {name} | {domain} | {fmt(d['mean'],100)} | [{fmt(d['minimum'],100)}, {fmt(d['maximum'],100)}] | {d['positive']}/{d['negative']} | {fmt(e['precision']['mean'],100)} | {fmt(e['recall']['mean'],100)} |")
    lines+=['','**实际结论：七门失败五项，关闭 C2 配置。** 相对 C0，BMC 开发 Dice −3.8787pp、precision −1.5354pp、recall −12.3599pp；RUNMC Dice −21.5106pp。BMC 的两名患者改善、两名下降，不满足至少三名改善。相对标准 U-Net，C2 的 BMC Dice 低 21.1962pp；相对自身源模型，RUNMC 旧域 Dice 下降 22.5189pp。没有可据此声称的目标域提升或旧域保护。','完整 BMC 训练患者读出也未显示诊断到的权衡已解决：C2 对 C0 的平均 Dice −6.5350pp、recall −20.9762pp，四名训练患者 recall 全降；空片平均 FP 每片从 956.57 降到 400.16，但小/中/大非空面积层 FN 均升高。存在项抑制空片错误的局部效果未转化为开发患者净收益，且非空前景受损；单臂不能判定正例存在项与空例最大值罚各自的因果份额。', '第一次启动的控制目标loss/梯度、来源状态及存在项代数检查均PASS，随后因操作层将内存中患者定义列表与序列化摘要字典直接比较而在正式训练前AssertionError退出。保留原EXIT1、日志、ledger失败项与原始时钟；限定修复只比较两个序列化摘要，续跑前0模型forward/backward/Adam且源payload不变。该错误不是科学方法负结果，不把第二次入口写成首次无异常。',f"原始首次启动至最终退出{cost['elapsed_seconds']:.2f}秒，全根{cost['root_bytes']}字节；800正式backward/Adam、3预检模型反向+1fixture反向、12864训练读出样本前向。控制训练/读出/开发评分只读复用。",'','单训练seed、每域4名反复开发患者、3推断不是3训练seed。相对 C0 的七门固定，不能因 C1 失败改阈值或选中间checkpoint；训练患者和开发患者分开报告，不能把相关切片当独立患者。', '', '[冻结协议](PREREG_ZH.md)；[匿名分析](ANALYSIS.json)；[成本与修复](EXECUTION_SUMMARY.json)。']
    train_raw=dict(x['control_train_diagnostics'],C2=r['train_diagnostics']['C2'])
    outputs={'ANALYSIS.json':a,'EXECUTION_SUMMARY.json':cost,'TRAIN_DIAGNOSTICS.json':public_train_diagnostics(train_raw),'TRAIN_DEFINITION.json':public_train_definition(x['TRAIN_DEFINITION.json'])}
    for n,v in outputs.items():(out/n).write_text(json.dumps(v,indent=2)+'\n')
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(a['mechanism_gate']))


if __name__=='__main__':main()
