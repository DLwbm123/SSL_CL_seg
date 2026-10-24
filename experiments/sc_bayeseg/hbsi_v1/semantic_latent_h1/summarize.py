"""Prespecified H1 shape-versus-appearance gates from anonymous final evidence."""
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
    c=x['cells'];p=x['patients']
    assert len(p['RUNMC'])==len(p['BMC'])==4
    diff=lambda phase,a,b:[100*(v[a]['classes'][0]['Dice']-v[b]['classes'][0]['Dice'])
                           for v in p[phase]]
    mdiff=lambda phase,a,b:100*(c[a+'_'+phase]['Dice']-c[b+'_'+phase]['Dice'])
    finite=all(all(math.isfinite(v[k]) for k in
                   ('fused_task','direct_task','variational','reconstruction_mse',
                    'shape_logsd','appearance_logsd'))
               for arm in ('Appearance','Shape') for v in x['training'][arm].values())
    source={
        'RUNMC_mean_vs_Appearance_ge_minus_1pp':mdiff('RUNMC','Shape','Appearance')>=-1,
        'RUNMC_mean_vs_U_ge_minus_1pp':mdiff('RUNMC','Shape','U')>=-1,
        'RUNMC_each_vs_Appearance_ge_minus_5pp':min(diff('RUNMC','Shape','Appearance'))>=-5,
        'RUNMC_each_vs_U_ge_minus_5pp':min(diff('RUNMC','Shape','U'))>=-5,
        'RUNMC_train_mean_vs_Appearance_ge_minus_5pp':mdiff('RUNMC_train','Shape','Appearance')>=-5,
        'RUNMC_train_mean_vs_U_ge_minus_5pp':mdiff('RUNMC_train','Shape','U')>=-5,
        'RUNMC_HD95_vs_both_le_1_10':all(c['Shape_RUNMC']['HD95_penalized_mm']<=
                                          1.1*c[arm+'_RUNMC']['HD95_penalized_mm']
                                          for arm in ('Appearance','U')),
        'no_source_collapse':c['Shape_RUNMC']['empty_predictions']==0 and
                              c['Shape_RUNMC_train']['empty_predictions']==0,
        'full_generator_finite':finite,
    }
    da=diff('BMC','Shape','Appearance');du=diff('BMC','Shape','U')
    target={
        'BMC_mean_vs_both_ge_plus_2pp':c['Shape_BMC']['Dice']>=
                                       max(c[arm+'_BMC']['Dice'] for arm in ('Appearance','U'))+.02,
        'BMC_3_of_4_nonnegative_vs_both':sum(a>=0 and u>=0 for a,u in zip(da,du))>=3,
        'BMC_each_vs_both_ge_minus_5pp':min(da+du)>=-5,
        'BMC_HD95_vs_both_le_1_10':all(c['Shape_BMC']['HD95_penalized_mm']<=
                                       1.1*c[arm+'_BMC']['HD95_penalized_mm']
                                       for arm in ('Appearance','U')),
        'no_BMC_collapse':c['Shape_BMC']['empty_predictions']==0,
    }
    verdict='POSITIVE_RESOURCE_SIGNAL' if all(source.values()) and all(target.values()) else \
            'SOURCE_SAFETY_FAILED' if not all(source.values()) else 'NO_SHAPE_SPECIFIC_BMC_SIGNAL'
    analysis=dict(status=verdict,source_gates=source,BMC_signal_gates=target,
                  cells=c,patients=p,training=x['training'],gradient_audits=x['gradient_audits'],
                  Shape_minus_Appearance_pp={phase:mdiff(phase,'Shape','Appearance')
                                             for phase in ('RUNMC','BMC','RUNMC_train')},
                  Shape_minus_U_pp={phase:mdiff(phase,'Shape','U')
                                    for phase in ('RUNMC','BMC','RUNMC_train')},
                  C1_C2_E1_F1_G1_hard_stops_unchanged=True)
    out=Path(__file__).resolve().parent
    (out/'ANALYSIS.json').write_text(json.dumps(analysis,indent=2)+'\n')
    execution=dict(first_actual_start=x['launch']['started'],deadline=x['launch']['deadline'],
                   elapsed_seconds=x['exit']['time']-x['launch']['started'],
                   new_root_bytes=x['bytes'],new_formal_updates=3200,
                   reused_C1_source_updates=1600,
                   attempts=[{k:a[k] for k in ('name','status','seconds_actual')}
                             for a in x['ledger']['attempts']])
    (out/'EXECUTION_SUMMARY.json').write_text(json.dumps(execution,indent=2)+'\n')
    table='\n'.join(f"| {arm} | {100*c[arm+'_RUNMC']['Dice']:.3f} | "
                    f"{100*c[arm+'_BMC']['Dice']:.3f} | "
                    f"{100*c[arm+'_RUNMC_train']['Dice']:.3f} | "
                    f"{c[arm+'_RUNMC']['HD95_penalized_mm']:.3f} | "
                    f"{c[arm+'_BMC']['HD95_penalized_mm']:.3f} |"
                    for arm in ('U','Appearance','Shape'))
    gates=lambda obj:'\n'.join(f"| {name} | {'PASS' if ok else 'FAILED'} |"
                               for name,ok in obj.items())
    rows='\n'.join(f"| {phase} | {i+1} | {100*v['U']['classes'][0]['Dice']:.3f} | "
                   f"{100*v['Appearance']['classes'][0]['Dice']:.3f} | "
                   f"{100*v['Shape']['classes'][0]['Dice']:.3f} | "
                   f"{100*(v['Shape']['classes'][0]['Dice']-v['Appearance']['classes'][0]['Dice']):+.3f} | "
                   f"{100*(v['Shape']['classes'][0]['Dice']-v['U']['classes'][0]['Dice']):+.3f} |"
                   for phase in ('RUNMC','BMC') for i,v in enumerate(p[phase]))
    audits='\n'.join(f"| {step} | " + ' | '.join(
        f"{next(v for v in x['gradient_audits'][arm] if v['step']==step)['task_grad_norm']:.5f} / "
        f"{next(v for v in x['gradient_audits'][arm] if v['step']==step)['weighted_variational_grad_norm']:.5f}"
        for arm in ('Appearance','Shape'))+' |'
        for step in (1,400,800,1200,1600))
    t=x['training']
    report=f"""# H1：直接监督形状与外观潜变量

**冻结判定：{verdict}。** 两臂从同一C1 Cut完整形状/外观生成层级出发，各新增1600次RUNMC标签加完整生成目标更新；U-Net始终冻结。新增头仅读一个后验均值，主预测固定叠加到U，另有同权直接分割监督。Dice是每域4患者等权三维结果（%）；旧U只读、累计成本不同。

| 臂 | RUNMC开发Dice | BMC冷迁移Dice | RUNMC完整L训练Dice | RUNMC惩罚HD95 mm | BMC惩罚HD95 mm |
|---|---:|---:|---:|---:|---:|
{table}

Shape−Appearance RUNMC/BMC为{mdiff('RUNMC','Shape','Appearance'):+.3f}/{mdiff('BMC','Shape','Appearance'):+.3f}pp；Shape−U为{mdiff('RUNMC','Shape','U'):+.3f}/{mdiff('BMC','Shape','U'):+.3f}pp。形状与外观后验原容量不同，这些是完整配置差值，不单独证明解耦。

源安全门：

| 门 | 判定 |
|---|---|
{gates(source)}

冷BMC形状特异积极门（须源门全过才有后继意义）：

| 门 | 判定 |
|---|---|
{gates(target)}

匿名逐患者：

| 域 | 索引 | U Dice | Appearance Dice | Shape Dice | Shape−Appearance pp | Shape−U pp |
|---|---:|---:|---:|---:|---:|---:|
{rows}

预注册当时状态所选后验梯度审计（标签范数 / 实际0.01系数后生成范数；余弦见匿名分析）：

| 更新前步号 | Appearance | Shape |
|---|---:|---:|
{audits}

最后100个源L训练批的生成项损失Appearance为{t['Appearance']['last100']['variational']:.3f}、Shape为{t['Shape']['last100']['variational']:.3f}；重建MSE分别{t['Appearance']['last100']['reconstruction_mse']:.5f}/{t['Shape']['last100']['reconstruction_mse']:.5f}。该固定配置未访问BMC训练标签或隐藏U标签/test。旧C/F/G负判定不变；同一4名开发患者多轮使用，即使积极门通过也不能替代独立患者、多种子和长期CL证据。[冻结协议](PREREG_ZH.md)、[匿名分析](ANALYSIS.json)、[执行摘要](EXECUTION_SUMMARY.json)。
"""
    (out/'REPORT_ZH.md').write_text(report)
    print(verdict,f"Shape_RUNMC={100*c['Shape_RUNMC']['Dice']:.3f}",
          f"Shape_BMC={100*c['Shape_BMC']['Dice']:.3f}")


if __name__=='__main__':
    main(sys.argv[1])
