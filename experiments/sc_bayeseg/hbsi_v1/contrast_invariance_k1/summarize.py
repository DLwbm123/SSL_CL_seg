"""Prespecified K1 two-view mechanism and patient-level resource gates."""
import json
import math
import sys
from pathlib import Path


def main(path):
    x=json.loads(Path(path).read_text())
    assert x['status']=='COMPLETE' and x['exit']['exit_code']==0
    assert x['preflight']['status']=='PASS'
    assert x['run']['formal_updates']==3200 and x['run']['target_U_used_for_design_only']
    assert x['ledger']['updates_reserved']==3200
    assert all(a['status']=='PASS' for a in x['ledger']['attempts'])
    c=x['cells'];p=x['patients'];t=x['training']
    assert len(p['RUNMC'])==len(p['BMC'])==4
    diff=lambda phase,a,b:[100*(v[a]['classes'][0]['Dice']-v[b]['classes'][0]['Dice'])
                           for v in p[phase]]
    mdiff=lambda phase,a,b:100*(c[a+'_'+phase]['Dice']-c[b+'_'+phase]['Dice'])
    finite=all(all(math.isfinite(v[k]) for k in
                   ('task','variational','invariant_mse','shape_spatial_sd',
                    'reconstruction_mse','shape_logsd','appearance_logsd'))
               for arm in ('AugOnly','ShapeInv') for v in t[arm].values())
    mechanism={
        'last100_invariant_mse_le_0_8_AugOnly':
            t['ShapeInv']['last100']['invariant_mse']<=
            .8*t['AugOnly']['last100']['invariant_mse'],
        'last100_shape_spatial_sd_ge_0_5_AugOnly':
            t['ShapeInv']['last100']['shape_spatial_sd']>=
            .5*t['AugOnly']['last100']['shape_spatial_sd'],
    }
    source={}
    for arm in ('AugOnly','U'):
        source['RUNMC_mean_vs_'+arm+'_ge_minus_1pp']=mdiff('RUNMC','ShapeInv',arm)>=-1
        source['RUNMC_each_vs_'+arm+'_ge_minus_5pp']=min(diff('RUNMC','ShapeInv',arm))>=-5
        source['RUNMC_train_mean_vs_'+arm+'_ge_minus_5pp']=\
            mdiff('RUNMC_train','ShapeInv',arm)>=-5
    source['RUNMC_HD95_vs_both_le_1_10']=all(c['ShapeInv_RUNMC']['HD95_penalized_mm']<=
                                            1.1*c[arm+'_RUNMC']['HD95_penalized_mm']
                                            for arm in ('AugOnly','U'))
    source['no_source_collapse']=c['ShapeInv_RUNMC']['empty_predictions']==0 and \
                                 c['ShapeInv_RUNMC_train']['empty_predictions']==0
    source['full_generator_finite']=finite
    da=diff('BMC','ShapeInv','AugOnly');du=diff('BMC','ShapeInv','U')
    target={
        'BMC_mean_vs_all_ge_plus_2pp':c['ShapeInv_BMC']['Dice']>=
            max(c[arm+'_BMC']['Dice'] for arm in ('AugOnly','U','H1_Shape'))+.02,
        'BMC_3_of_4_nonnegative_vs_AugOnly_and_U':
            sum(a>=0 and u>=0 for a,u in zip(da,du))>=3,
        'BMC_each_vs_AugOnly_and_U_ge_minus_5pp':min(da+du)>=-5,
        'BMC_HD95_vs_AugOnly_and_U_le_1_10':all(
            c['ShapeInv_BMC']['HD95_penalized_mm']<=1.1*c[arm+'_BMC']['HD95_penalized_mm']
            for arm in ('AugOnly','U')),
        'no_BMC_collapse':c['ShapeInv_BMC']['empty_predictions']==0,
    }
    verdict='POSITIVE_RESOURCE_SIGNAL' if all(mechanism.values()) and all(source.values()) and all(target.values()) else \
            'MECHANISM_FAILED' if not all(mechanism.values()) else \
            'SOURCE_SAFETY_FAILED' if not all(source.values()) else 'NO_BMC_GAIN'
    analysis=dict(status=verdict,mechanism_gates=mechanism,source_gates=source,
                  BMC_signal_gates=target,cells=c,patients=p,training=t,
                  gradient_audits=x['gradient_audits'],
                  ShapeInv_minus_AugOnly_pp={phase:mdiff(phase,'ShapeInv','AugOnly')
                                              for phase in ('RUNMC','BMC','RUNMC_train')},
                  ShapeInv_minus_U_pp={phase:mdiff(phase,'ShapeInv','U')
                                      for phase in ('RUNMC','BMC','RUNMC_train')},
                  BMC_target_U_informed_exploratory=True,prior_hard_stops_unchanged=True)
    out=Path(__file__).resolve().parent
    (out/'ANALYSIS.json').write_text(json.dumps(analysis,indent=2)+'\n')
    execution=dict(first_actual_start=x['launch']['started'],deadline=x['launch']['deadline'],
                   elapsed_seconds=x['exit']['time']-x['launch']['started'],
                   new_root_bytes=x['bytes'],new_formal_updates=3200,
                   reused_H1_initial_C1_source_updates=1600,
                   attempts=[{k:a[k] for k in ('name','status','seconds_actual')}
                             for a in x['ledger']['attempts']])
    (out/'EXECUTION_SUMMARY.json').write_text(json.dumps(execution,indent=2)+'\n')
    table='\n'.join(f"| {arm} | {100*c[arm+'_RUNMC']['Dice']:.3f} | "
                    f"{100*c[arm+'_BMC']['Dice']:.3f} | "
                    f"{100*c[arm+'_RUNMC_train']['Dice']:.3f} | "
                    f"{c[arm+'_RUNMC']['HD95_penalized_mm']:.3f} | "
                    f"{c[arm+'_BMC']['HD95_penalized_mm']:.3f} |"
                    for arm in ('U','H1_Shape','AugOnly','ShapeInv'))
    gates=lambda obj:'\n'.join(f"| {name} | {'PASS' if ok else 'FAILED'} |"
                               for name,ok in obj.items())
    rows='\n'.join(f"| {phase} | {i+1} | {100*v['U']['classes'][0]['Dice']:.3f} | "
                   f"{100*v['AugOnly']['classes'][0]['Dice']:.3f} | "
                   f"{100*v['ShapeInv']['classes'][0]['Dice']:.3f} | "
                   f"{100*(v['ShapeInv']['classes'][0]['Dice']-v['AugOnly']['classes'][0]['Dice']):+.3f} | "
                   f"{100*(v['ShapeInv']['classes'][0]['Dice']-v['U']['classes'][0]['Dice']):+.3f} |"
                   for phase in ('RUNMC','BMC') for i,v in enumerate(p[phase]))
    audits='\n'.join(f"| {step} | " + ' | '.join(
        f"{next(v for v in x['gradient_audits'][arm] if v['step']==step)['task_grad_norm']:.5f} / "
        f"{next(v for v in x['gradient_audits'][arm] if v['step']==step)['weighted_variational_grad_norm']:.5f} / "
        f"{next(v for v in x['gradient_audits'][arm] if v['step']==step)['invariant_grad_norm']:.5f}"
        for arm in ('AugOnly','ShapeInv'))+' |'
        for step in (1,400,800,1200,1600))
    report=f"""# K1：形状后验灰度不变性

**冻结判定：{verdict}。** 两臂同入口、同RUNMC L原图与单调灰度变换图、同完整BayeSeg形状/外观/图像生成训练，每臂1600次更新；唯一干预是ShapeInv多了系数1.0的形状后验均值两视图一致项。U-Net冻结，主预测仅用原图。Dice为每域4患者等权三维结果（%）。BMC U强度统计参与变换设计，BMC开发结果是**目标无标签分布知情的探索**，不能称严格冷迁移。

| 臂 | RUNMC开发Dice | BMC开发Dice | RUNMC完整L训练Dice | RUNMC惩罚HD95 mm | BMC惩罚HD95 mm |
|---|---:|---:|---:|---:|---:|
{table}

ShapeInv−AugOnly RUNMC/BMC为{mdiff('RUNMC','ShapeInv','AugOnly'):+.3f}/{mdiff('BMC','ShapeInv','AugOnly'):+.3f}pp；ShapeInv−旧U为{mdiff('RUNMC','ShapeInv','U'):+.3f}/{mdiff('BMC','ShapeInv','U'):+.3f}pp。H1单视图Shape只读，累计训练成本不同。

机制门（最后100个训练批）：

| 门 | 判定 |
|---|---|
{gates(mechanism)}

AugOnly/ShapeInv不变性MSE={t['AugOnly']['last100']['invariant_mse']:.6f}/{t['ShapeInv']['last100']['invariant_mse']:.6f}；形状特征空间SD={t['AugOnly']['last100']['shape_spatial_sd']:.5f}/{t['ShapeInv']['last100']['shape_spatial_sd']:.5f}。

源安全门：

| 门 | 判定 |
|---|---|
{gates(source)}

BMC积极门（须机制和源门全过才有后继意义）：

| 门 | 判定 |
|---|---|
{gates(target)}

匿名逐患者：

| 域 | 索引 | U Dice | AugOnly Dice | ShapeInv Dice | ShapeInv−AugOnly pp | ShapeInv−U pp |
|---|---:|---:|---:|---:|---:|---:|
{rows}

预注册当时状态的形状后验梯度范数审计（标签 / 实际0.01系数后生成 / 未乘系数的一致项；余弦见匿名分析）：

| 更新前步号 | AugOnly | ShapeInv |
|---|---:|---:|
{audits}

最后100批生成项损失AugOnly/ShapeInv={t['AugOnly']['last100']['variational']:.3f}/{t['ShapeInv']['last100']['variational']:.3f}，样本重建MSE={t['AugOnly']['last100']['reconstruction_mse']:.5f}/{t['ShapeInv']['last100']['reconstruction_mse']:.5f}。这仍不能单独证明潜变量解耦或临床有效；BMC开发患者反复用于路线判断，隐藏U标签/test未访问。[冻结协议](PREREG_ZH.md)、[匿名分析](ANALYSIS.json)、[执行摘要](EXECUTION_SUMMARY.json)。旧C/F/G/H/I/J判定不变。
"""
    (out/'REPORT_ZH.md').write_text(report)
    print(verdict,f"ShapeInv_RUNMC={100*c['ShapeInv_RUNMC']['Dice']:.3f}",
          f"ShapeInv_BMC={100*c['ShapeInv_BMC']['Dice']:.3f}")


if __name__=='__main__':
    main(sys.argv[1])
