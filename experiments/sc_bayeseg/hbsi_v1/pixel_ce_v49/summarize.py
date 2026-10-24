"""Frozen gates and paired patient analysis; train and development stay distinct."""
import argparse,json
from pathlib import Path
from statistics import mean
from ..unet_transfer_v45.summarize import patients,p10,METRICS
from ..entry_decomposition_v47.summarize import extra_metrics

EXTRAS=('precision','recall','volume_ratio')


def enriched(rows,repeats):
    out=patients(rows,repeats);extra=extra_metrics(rows)
    for key in out:out[key].update(extra[key])
    return out


def aggregate(rows):
    return {k:dict(mean=mean(v[k] for v in rows.values() if v[k] is not None) if any(v[k] is not None for v in rows.values()) else None,valid=sum(v[k] is not None for v in rows.values())) for k in (*METRICS,*EXTRAS)}


def effect(a,b):
    assert a.keys()==b.keys();result={}
    for k in (*METRICS,*EXTRAS):
        delta=[a[i][k]-b[i][k] for i in sorted(a) if a[i][k] is not None and b[i][k] is not None]
        result[k]=dict(mean=mean(delta) if delta else None,minimum=min(delta) if delta else None,maximum=max(delta) if delta else None,p10=p10(delta) if delta else None,positive=sum(v>1e-12 for v in delta),negative=sum(v< -1e-12 for v in delta),valid=len(delta))
    return result


def gates(e,means,new,old):
    b=e['BMC'];a=e['RUNMC']
    checks=dict(BMC_Dice_ge_2pp=b['Dice']['mean']>=.02,BMC_positive_patients_ge_3=b['Dice']['positive']>=3,
        precision_ge_5pp=b['precision']['valid']==4 and b['precision']['mean']>=.05,
        recall_loss_le_2pp=b['recall']['valid']==4 and b['recall']['mean']>=-.02,
        RUNMC_extra_loss_le_0_5pp=a['Dice']['mean']>=-.005,
        no_new_empty=all(not(new[d][i]['empty_predictions']>0 and old[d][i]['empty_predictions']==0) for d in new for i in new[d]),
        HD95_worsening_le_10pct=all(means['C1'][d]['HD95_penalized_mm']['mean']<=1.1*means['C0'][d]['HD95_penalized_mm']['mean'] for d in ('RUNMC','BMC')))
    return dict(status='PASS' if all(checks.values()) else 'FAILED',checks=checks)


def analyze(x):
    assert x['STATUS.json']['stage']=='COMPLETE' and x['EXIT.json']['exit_code']==0
    r=x['RESULTS.json'];l=x['ledger'];assert r['formal_updates']==l['updates_reserved']==800
    assert len(l['training_sessions'])==1 and l['training_sessions']['C1']['status']=='COMPLETE' and l['training_sessions']['C1']['completed_steps']==800
    assert all(v['status']=='PASS' for v in l['attempts']) and x['MATCHING.json']['status']==x['PREFLIGHT.json']['status']=='PASS'
    assert r['old_memory_and_Sb_unchanged'] and r['common_entrance'] and r['counter']['outer_backward_calls']==800
    assert len(x['UPDATES.json'])==len(x['PROBES.json'])==4 and x['training']['rows']==800
    raw=dict(x['controls'],C1=x['patients']);pp={a:{d:enriched(rows,1 if a=='U' else 3) for d,rows in ds.items()} for a,ds in raw.items()}
    means={a:{d:aggregate(rows) for d,rows in ds.items()} for a,ds in pp.items()}
    comparisons={'C1_minus_C0':('C1','C0'),'C1_minus_source':('C1','W_source'),'C0_minus_source':('C0','W_source'),'C1_minus_U':('C1','U')}
    effects={name:{d:effect(pp[a][d],pp[b][d]) for d in ('RUNMC','BMC')} for name,(a,b) in comparisons.items()}
    tails={a:{d:dict(minimum=min(v['Dice'] for v in rows.values()),p10=p10([v['Dice'] for v in rows.values()])) for d,rows in ds.items()} for a,ds in pp.items()}
    train={}
    for arm,diag in r['train_diagnostics'].items():
        assert diag['complete'] and len(diag['patients'])==4
        rows=[dict(patient=str(v['case']),metrics=v['metrics']) for v in diag['patients']];train[arm]=enriched(rows,3)
    train_summary={arm:aggregate(rows) for arm,rows in train.items()}
    assert all(sum(d['exposures'])>=0 for d in x['TRAIN_DEFINITION.json']['cases'])
    expected_empty=sum(n for d in x['TRAIN_DEFINITION.json']['cases'] for n,g in zip(d['exposures'],d['strata']) if g=='empty')
    assert abs(x['training']['empty_exposures']-expected_empty)<1e-8
    anonymous={arm:{domain:[dict(case=i+1,metrics=rows[key]) for i,key in enumerate(sorted(rows))] for domain,rows in ds.items()} for arm,ds in pp.items()}
    return dict(status='COMPLETE',means=means,patient_tails=tails,effects=effects,mechanism_gate=gates(effects['C1_minus_C0'],means,pp['C1'],pp['C0']),anonymous_development_patients=anonymous,train_means=train_summary,train_effect=effect(train['C1'],train['C0']),training=x['training'],training_seeds=1,development_patients_per_domain=4,old_development_models_rescored=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('snapshot');args=p.parse_args();x=json.loads(Path(args.snapshot).read_text());a=analyze(x);out=Path(__file__).parent;r=x['RESULTS.json']
    cost=dict(status='COMPLETE',original_start=x['LAUNCH.json']['started'],deadline=x['LAUNCH.json']['deadline'],exit_time=x['EXIT.json']['time'],elapsed_seconds=x['EXIT.json']['time']-x['LAUNCH.json']['started'],root_bytes=x['root_bytes'],formal_updates=800,preflight_model_reverse_calls=3,preflight_fixture_reverse_calls=1,preflight_model_objective_forwards=3,audit_extra_reverse_calls=0,probe_semantic_sample_forwards=3072,train_diagnostic_semantic_sample_forwards=25728,stage_seconds={v['name']:v['seconds_actual'] for v in x['ledger']['attempts']},counter=r['counter'])
    fmt=lambda v,scale=1:'NA' if v is None else f'{scale*v:.5f}'
    lines=['# v49 单因素像素 CE 目标期修复','','C0为封存v46 W目标800，C1只换CE类别权重、同入口和完整134张trainL训练800步。训练患者新增读出与正式开发集分别报告，U仅复用。','','| 臂 | 数据 | Dice % | precision % | recall % | 体积比 | NLL | Brier | HD95 mm |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm,ds in a['means'].items():
        for domain,m in ds.items():lines.append('| '+arm+' | '+domain+' dev | '+' | '.join(fmt(m[k]['mean'],100 if k in ('Dice','precision','recall') else 1) for k in ('Dice','precision','recall','volume_ratio','NLL','Brier','HD95_penalized_mm'))+' |')
    for arm,m in a['train_means'].items():lines.append('| '+arm+' | BMC train | '+' | '.join(fmt(m[k]['mean'],100 if k in ('Dice','precision','recall') else 1) for k in ('Dice','precision','recall','volume_ratio','NLL','Brier','HD95_penalized_mm'))+' |')
    lines+=['',f"冻结继续门：**{a['mechanism_gate']['status']}**。",'','| 门槛 | 判定 |','|---|---|']
    for k,v in a['mechanism_gate']['checks'].items():lines.append(f"| {k} | {'PASS' if v else 'FAILED'} |")
    lines+=['','| 对比 | 域 | Dice pp | 患者范围 pp | 正/负 | precision pp | recall pp |','|---|---|---:|---:|---|---:|---:|']
    for name,ds in a['effects'].items():
        for domain,e in ds.items():
            d=e['Dice'];lines.append(f"| {name} | {domain} | {fmt(d['mean'],100)} | [{fmt(d['minimum'],100)}, {fmt(d['maximum'],100)}] | {d['positive']}/{d['negative']} | {fmt(e['precision']['mean'],100)} | {fmt(e['recall']['mean'],100)} |")
    lines+=['','相对C0保护不等于旧域无遗忘。C1/source的RUNMC差为负表示实际遗忘；源模型未变，v46 source BMC保护失败不会被本实验改写。所有患者尾部/辅助指标/NA分母见ANALYSIS。训练患者及分层FP/FN、FG/BG条件概率误差见TRAIN_DIAGNOSTICS，训练曝光定义见TRAIN_DEFINITION。', '', f"首次启动至退出{cost['elapsed_seconds']:.2f}秒，根{cost['root_bytes']}字节。800正式backward/Adam、3真实目标+1fixture预检反向0Adam、实际位移审计0额外反向；训练诊断25728+固定探针3072样本语义前向，两个正式开发格另计。原控制训练不重跑，控制train是新增诊断，不能写成完全无旧模型推断。", '', '单训练seed，每域4名反复开发患者、3推断重复不是3训练seed。CE改变的是整个监督风险权衡和梯度量级，不能把结果归因于唯一参数、可靠校准或SSL收益。未通过不搜权重/阈值/seed；通过只给独立从头一致风险验证资格，不称整体解决或追平U-Net。', '', '[协议](PREREG_ZH.md)；[匿名完整分析](ANALYSIS.json)；[成本](EXECUTION_SUMMARY.json)。']
    outputs={'ANALYSIS.json':a,'EXECUTION_SUMMARY.json':cost,'TRAIN_DIAGNOSTICS.json':r['train_diagnostics'],'TRAIN_DEFINITION.json':x['TRAIN_DEFINITION.json'],'UPDATES.json':x['UPDATES.json'],'PROBES.json':x['PROBES.json']}
    for n,v in outputs.items():(out/n).write_text(json.dumps(v,indent=2)+'\n')
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');print(json.dumps(a['mechanism_gate']))

if __name__=='__main__':main()
