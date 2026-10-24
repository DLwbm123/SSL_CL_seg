"""Prespecified I1 frozen-generator resource gates from anonymous evidence."""
import json
import math
import sys
from pathlib import Path


def main(path):
    x=json.loads(Path(path).read_text())
    assert x['status']=='COMPLETE' and x['exit']['exit_code']==0
    assert x['preflight']['status']=='PASS'
    assert x['run']['formal_updates']==3200 and not x['run']['target_training']
    assert x['ledger']['updates_reserved']==3200
    assert all(a['status']=='PASS' for a in x['ledger']['attempts'])
    assert all(v['complete_frozen_generator_max_error']==0 for v in x['receipts'].values())
    c=x['cells'];p=x['patients']
    assert len(p['RUNMC'])==len(p['BMC'])==4
    diff=lambda phase,a,b:[100*(v[a]['classes'][0]['Dice']-v[b]['classes'][0]['Dice'])
                           for v in p[phase]]
    mdiff=lambda phase,a,b:100*(c[a+'_'+phase]['Dice']-c[b+'_'+phase]['Dice'])
    finite=all(all(math.isfinite(v[k]) for k in
                   ('fused_task','direct_task','variational','reconstruction_mse',
                    'shape_logsd','appearance_logsd'))
               for arm in ('I_Appearance','I_Shape') for v in x['training'][arm].values())
    comparisons=('I_Appearance','H1_Shape','U')
    source={}
    for arm in comparisons:
        source['RUNMC_mean_vs_'+arm+'_ge_minus_1pp']=mdiff('RUNMC','I_Shape',arm)>=-1
        source['RUNMC_each_vs_'+arm+'_ge_minus_5pp']=min(diff('RUNMC','I_Shape',arm))>=-5
        source['RUNMC_train_mean_vs_'+arm+'_ge_minus_5pp']=\
            mdiff('RUNMC_train','I_Shape',arm)>=-5
    source['RUNMC_HD95_vs_all_le_1_10']=all(c['I_Shape_RUNMC']['HD95_penalized_mm']<=
                                           1.1*c[arm+'_RUNMC']['HD95_penalized_mm']
                                           for arm in comparisons)
    source['no_source_collapse']=c['I_Shape_RUNMC']['empty_predictions']==0 and \
                                 c['I_Shape_RUNMC_train']['empty_predictions']==0
    source['full_generator_finite_and_exactly_frozen']=finite
    di=diff('BMC','I_Shape','I_Appearance');du=diff('BMC','I_Shape','U')
    target={
        'BMC_mean_vs_all_ge_plus_2pp':c['I_Shape_BMC']['Dice']>=
            max(c[arm+'_BMC']['Dice'] for arm in comparisons)+.02,
        'BMC_3_of_4_nonnegative_vs_I_Appearance_and_U':
            sum(a>=0 and u>=0 for a,u in zip(di,du))>=3,
        'BMC_each_vs_I_Appearance_and_U_ge_minus_5pp':min(di+du)>=-5,
        'BMC_HD95_vs_I_Appearance_and_U_le_1_10':all(
            c['I_Shape_BMC']['HD95_penalized_mm']<=1.1*c[arm+'_BMC']['HD95_penalized_mm']
            for arm in ('I_Appearance','U')),
        'no_BMC_collapse':c['I_Shape_BMC']['empty_predictions']==0,
    }
    verdict='POSITIVE_RESOURCE_SIGNAL' if all(source.values()) and all(target.values()) else \
            'SOURCE_SAFETY_FAILED' if not all(source.values()) else 'NO_BMC_GAIN'
    analysis=dict(status=verdict,source_gates=source,BMC_signal_gates=target,
                  cells=c,patients=p,training=x['training'],receipts=x['receipts'],
                  I_Shape_minus_I_Appearance_pp={phase:mdiff(phase,'I_Shape','I_Appearance')
                                                 for phase in ('RUNMC','BMC','RUNMC_train')},
                  I_Shape_minus_H1_Shape_pp={phase:mdiff(phase,'I_Shape','H1_Shape')
                                             for phase in ('RUNMC','BMC','RUNMC_train')},
                  I_Shape_minus_U_pp={phase:mdiff(phase,'I_Shape','U')
                                      for phase in ('RUNMC','BMC','RUNMC_train')},
                  H1_and_prior_hard_stops_unchanged=True)
    out=Path(__file__).resolve().parent
    (out/'ANALYSIS.json').write_text(json.dumps(analysis,indent=2)+'\n')
    execution=dict(first_actual_start=x['launch']['started'],deadline=x['launch']['deadline'],
                   elapsed_seconds=x['exit']['time']-x['launch']['started'],
                   new_root_bytes=x['bytes'],new_formal_updates=3200,
                   reused_H1_initial_C1_source_updates=1600,
                   attempts=[{k:a[k] for k in ('name','status','seconds_actual')}
                             for a in x['ledger']['attempts']])
    (out/'EXECUTION_SUMMARY.json').write_text(json.dumps(execution,indent=2)+'\n')
    arms=('U','H1_Shape','I_Appearance','I_Shape')
    table='\n'.join(f"| {arm} | {100*c[arm+'_RUNMC']['Dice']:.3f} | "
                    f"{100*c[arm+'_BMC']['Dice']:.3f} | "
                    f"{100*c[arm+'_RUNMC_train']['Dice']:.3f} | "
                    f"{c[arm+'_RUNMC']['HD95_penalized_mm']:.3f} | "
                    f"{c[arm+'_BMC']['HD95_penalized_mm']:.3f} |" for arm in arms)
    gates=lambda obj:'\n'.join(f"| {name} | {'PASS' if ok else 'FAILED'} |"
                               for name,ok in obj.items())
    rows='\n'.join(f"| {phase} | {i+1} | {100*v['U']['classes'][0]['Dice']:.3f} | "
                   f"{100*v['H1_Shape']['classes'][0]['Dice']:.3f} | "
                   f"{100*v['I_Appearance']['classes'][0]['Dice']:.3f} | "
                   f"{100*v['I_Shape']['classes'][0]['Dice']:.3f} | "
                   f"{100*(v['I_Shape']['classes'][0]['Dice']-v['I_Appearance']['classes'][0]['Dice']):+.3f} | "
                   f"{100*(v['I_Shape']['classes'][0]['Dice']-v['U']['classes'][0]['Dice']):+.3f} |"
                   for phase in ('RUNMC','BMC') for i,v in enumerate(p[phase]))
    t=x['training']
    report=f"""# I1：冻结生成层级后的直接监督潜变量头

**冻结判定：{verdict}。** 两臂从H1共同零更新状态出发，完整BayeSeg形状/外观生成层级与标准U-Net全程逐张量冻结，仅同容量头各做1600次RUNMC L标签更新。H1 Joint-Shape和旧U只读参照；Dice为每域4患者等权三维结果（%）。

| 臂 | RUNMC开发Dice | BMC冷迁移Dice | RUNMC完整L训练Dice | RUNMC惩罚HD95 mm | BMC惩罚HD95 mm |
|---|---:|---:|---:|---:|---:|
{table}

I-Shape−I-Appearance RUNMC/BMC为{mdiff('RUNMC','I_Shape','I_Appearance'):+.3f}/{mdiff('BMC','I_Shape','I_Appearance'):+.3f}pp；I-Shape−H1 Joint-Shape为{mdiff('RUNMC','I_Shape','H1_Shape'):+.3f}/{mdiff('BMC','I_Shape','H1_Shape'):+.3f}pp；I-Shape−旧U为{mdiff('RUNMC','I_Shape','U'):+.3f}/{mdiff('BMC','I_Shape','U'):+.3f}pp。I1和H1同入口与任务数据流，但更新参数集合不同，历史对比只能描述整条训练策略效应。

源安全门：

| 门 | 判定 |
|---|---|
{gates(source)}

冷BMC积极门（须源门全过才有后继意义）：

| 门 | 判定 |
|---|---|
{gates(target)}

匿名逐患者：

| 域 | 索引 | U Dice | H1 Joint-Shape Dice | I-Appearance Dice | I-Shape Dice | I-Shape−I-Appearance pp | I-Shape−U pp |
|---|---:|---:|---:|---:|---:|---:|---:|
{rows}

完整生成层级的终态逐张量与H1入口完全相同。最后100个源L训练批的生成项损失I-Appearance为{t['I_Appearance']['last100']['variational']:.3f}、I-Shape为{t['I_Shape']['last100']['variational']:.3f}；样本图像重建MSE分别{t['I_Appearance']['last100']['reconstruction_mse']:.5f}/{t['I_Shape']['last100']['reconstruction_mse']:.5f}。冻结生成读数仅描述同一图像流，不构成潜变量解耦证明。BMC无训练，U隐藏标签/test未访问。H1/C/F/G旧硬停不变；重复使用的4名开发患者只能支持资源门。[冻结协议](PREREG_ZH.md)、[匿名分析](ANALYSIS.json)、[执行摘要](EXECUTION_SUMMARY.json)。
"""
    (out/'REPORT_ZH.md').write_text(report)
    print(verdict,f"I_Shape_RUNMC={100*c['I_Shape_RUNMC']['Dice']:.3f}",
          f"I_Shape_BMC={100*c['I_Shape_BMC']['Dice']:.3f}")


if __name__=='__main__':
    main(sys.argv[1])
