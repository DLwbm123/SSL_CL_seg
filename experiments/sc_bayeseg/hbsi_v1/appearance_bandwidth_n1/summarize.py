"""Apply the frozen appearance-bandwidth and patient-level gates."""
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
    c=x['cells'];p=x['patients'];s=x['swap'];sp=x['swap_patients'];t=x['training']
    assert len(p['RUNMC'])==len(p['BMC'])==4
    assert all(len(sp[a])==4 and s[a]['slices']==80 for a in ('Full','Smooth'))
    diff=lambda phase,a,b:[100*(v[a]['classes'][0]['Dice']-v[b]['classes'][0]['Dice'])
                           for v in p[phase]]
    mdiff=lambda phase,a,b:100*(c[a+'_'+phase]['Dice']-c[b+'_'+phase]['Dice'])
    finite=all(all(math.isfinite(v[k]) for k in
                   ('task','variational','reconstruction_mse','shape_mean_sd',
                    'shape_logsd','appearance_logsd'))
               for arm in ('Full','Smooth') for v in t[arm].values())
    swap_diffs=[100*(v['metrics']['swap']['classes'][0]['Dice']-
                     v['metrics']['original']['classes'][0]['Dice']) for v in sp['Smooth']]
    mechanism={
        'Smooth_self_reconstruction_mse_le_0_02':math.isfinite(s['Smooth']['reconstruction_mse'])
            and s['Smooth']['reconstruction_mse']<=.02,
        'Smooth_swap_outside_fraction_le_0_01':math.isfinite(s['Smooth']['swap_outside_fraction'])
            and s['Smooth']['swap_outside_fraction']<=.01,
        'Smooth_swap_Dice_vs_Full_ge_plus_10pp':
            100*(s['Smooth']['swap_Dice']-s['Full']['swap_Dice'])>=10,
        'Smooth_swap_Dice_vs_own_original_ge_minus_5pp':
            100*(s['Smooth']['swap_Dice']-s['Smooth']['original_Dice'])>=-5,
        'Smooth_swap_each_vs_original_ge_minus_10pp':min(swap_diffs)>=-10,
        'Smooth_swap_no_patient_empty':all(not v['metrics']['swap']['classes'][0]
                                            ['empty_prediction_failure'] for v in sp['Smooth']),
    }
    source={}
    for arm in ('Full','U'):
        source['Smooth_RUNMC_mean_vs_'+arm+'_ge_minus_1pp']=mdiff('RUNMC','Smooth',arm)>=-1
        source['Smooth_RUNMC_each_vs_'+arm+'_ge_minus_5pp']=min(diff('RUNMC','Smooth',arm))>=-5
        source['Smooth_RUNMC_train_mean_vs_'+arm+'_ge_minus_5pp']=\
            mdiff('RUNMC_train','Smooth',arm)>=-5
    source['Smooth_RUNMC_HD95_vs_both_le_1_10']=all(
        c['Smooth_RUNMC']['HD95_penalized_mm']<=1.1*c[arm+'_RUNMC']['HD95_penalized_mm']
        for arm in ('Full','U'))
    source['Smooth_no_source_collapse']=c['Smooth_RUNMC']['empty_predictions']==0 and \
                                        c['Smooth_RUNMC_train']['empty_predictions']==0
    source['Full_vs_U_mean_ge_minus_1pp']=mdiff('RUNMC','Full','U')>=-1
    source['Full_vs_U_each_ge_minus_5pp']=min(diff('RUNMC','Full','U'))>=-5
    source['Full_no_source_collapse']=c['Full_RUNMC']['empty_predictions']==0
    source['finite_training_and_scales']=finite
    dfull=diff('BMC','Smooth','Full');du=diff('BMC','Smooth','U')
    target={
        'Smooth_BMC_mean_vs_max_Full_U_H1_L1_ge_plus_2pp':c['Smooth_BMC']['Dice']>=
            max(c[arm+'_BMC']['Dice'] for arm in ('Full','U','H1_Shape','L1_Cut'))+.02,
        'Smooth_BMC_3_of_4_nonnegative_vs_Full_and_U':
            sum(a>=0 and b>=0 for a,b in zip(dfull,du))>=3,
        'Smooth_BMC_each_vs_Full_and_U_ge_minus_5pp':min(dfull+du)>=-5,
        'Smooth_BMC_HD95_vs_Full_and_U_le_1_10':all(
            c['Smooth_BMC']['HD95_penalized_mm']<=1.1*c[arm+'_BMC']['HD95_penalized_mm']
            for arm in ('Full','U')),
        'Smooth_no_BMC_collapse':c['Smooth_BMC']['empty_predictions']==0,
    }
    verdict=('POSITIVE_RESOURCE_SIGNAL' if all(mechanism.values()) and all(source.values())
             and all(target.values()) else
             'SWAP_MECHANISM_FAILED' if not all(mechanism.values()) else
             'SOURCE_SAFETY_FAILED' if not all(source.values()) else 'NO_BMC_GAIN')
    analysis=dict(status=verdict,mechanism_gates=mechanism,source_gates=source,
                  BMC_signal_gates=target,cells=c,patients=p,swap=s,
                  swap_patients=sp,training=t,gradient_audits=x['gradient_audits'],
                  Smooth_minus_Full_pp={phase:mdiff(phase,'Smooth','Full')
                                        for phase in ('RUNMC','BMC','RUNMC_train')},
                  Smooth_minus_U_pp={phase:mdiff(phase,'Smooth','U')
                                     for phase in ('RUNMC','BMC','RUNMC_train')},
                  BMC_reused_development_patients=True,prior_hard_stops_unchanged=True)
    out=Path(__file__).resolve().parent
    (out/'ANALYSIS.json').write_text(json.dumps(analysis,indent=2)+'\n')
    execution=dict(first_actual_start=x['launch']['started'],deadline=x['launch']['deadline'],
                   elapsed_seconds=x['exit']['time']-x['launch']['started'],
                   new_root_bytes=x['bytes'],new_formal_updates=3200,
                   reused_L1_zero_update_initial_C1_source_updates=1600,
                   attempts=[{k:a[k] for k in ('name','status','seconds_actual')}
                             for a in x['ledger']['attempts']])
    (out/'EXECUTION_SUMMARY.json').write_text(json.dumps(execution,indent=2)+'\n')
    table='\n'.join(f"| {arm} | {100*c[arm+'_RUNMC']['Dice']:.3f} | "
                    f"{100*c[arm+'_BMC']['Dice']:.3f} | "
                    f"{100*c[arm+'_RUNMC_train']['Dice']:.3f} | "
                    f"{c[arm+'_RUNMC']['HD95_penalized_mm']:.3f} | "
                    f"{c[arm+'_BMC']['HD95_penalized_mm']:.3f} |"
                    for arm in ('U','H1_Shape','L1_Cut','Full','Smooth'))
    swap_table='\n'.join(f"| {arm} | {100*s[arm]['original_Dice']:.3f} | "
                         f"{100*s[arm]['self_Dice']:.3f} | {100*s[arm]['swap_Dice']:.3f} | "
                         f"{s[arm]['reconstruction_mse']:.5f} | "
                         f"{100*s[arm]['swap_outside_fraction']:.3f} |"
                         for arm in ('Full','Smooth'))
    gates=lambda obj:'\n'.join(f"| {name} | {'PASS' if ok else 'FAILED'} |"
                               for name,ok in obj.items())
    patients='\n'.join(f"| {phase} | {i+1} | " + ' | '.join(
        f"{100*v[arm]['classes'][0]['Dice']:.3f}"
        for arm in ('U','Full','Smooth'))+' |'
        for phase in ('RUNMC','BMC') for i,v in enumerate(p[phase]))
    swap_patients='\n'.join(f"| {arm} | {v['index']} | " + ' | '.join(
        f"{100*v['metrics'][key]['classes'][0]['Dice']:.3f}"
        for key in ('original','self','swap'))+' |'
        for arm in ('Full','Smooth') for v in sp[arm])
    audits='\n'.join(f"| {step} | " + ' | '.join(
        f"{next(v for v in x['gradient_audits'][arm] if v['step']==step)['task_norm']:.5f} / "
        f"{next(v for v in x['gradient_audits'][arm] if v['step']==step)['weighted_variational_norm']:.5f}"
        for arm in ('Full','Smooth'))+' |' for step in (1,400,800,1200,1600))
    report=f"""# N1：限制外观后验空间带宽

**冻结判定：{verdict}。** 从L1共同零更新初态出发，两臂同RUNMC L4患者80片、各1600次更新，完整保留形状/外观图像生成及空间Gamma先验，U冻结，生成梯度到qS。Smooth仅将qA均值和尺度固定池化到4×4后上采样，Full原样128×128。Dice为每域4患者等权三维结果（%）；BMC开发患者被多次用于路线选择，仅为探索性资源门。

| 臂 | RUNMC开发Dice | BMC开发Dice | RUNMC完整L训练Dice | RUNMC惩罚HD95 mm | BMC惩罚HD95 mm |
|---|---:|---:|---:|---:|---:|
{table}

Smooth−Full RUNMC/BMC为{mdiff('RUNMC','Smooth','Full'):+.3f}/{mdiff('BMC','Smooth','Full'):+.3f}pp；Smooth−U为{mdiff('RUNMC','Smooth','U'):+.3f}/{mdiff('BMC','Smooth','U'):+.3f}pp。U/H1/L1仅历史参考，累积训练成本不匹配。

源L患者固定跨患者外观交换审计，冻结U读数：

| 臂 | 原图Dice | 自身重建Dice | 外观交换Dice | 重建MSE | 交换图越界像素% |
|---|---:|---:|---:|---:|---:|
{swap_table}

交换机制门：

| 门 | 判定 |
|---|---|
{gates(mechanism)}

源安全门：

| 门 | 判定 |
|---|---|
{gates(source)}

BMC积极门（须交换机制与源安全全过才允许后继）：

| 门 | 判定 |
|---|---|
{gates(target)}

匿名开发患者Dice：

| 域 | 索引 | U | Full | Smooth |
|---|---:|---:|---:|---:|
{patients}

匿名源L交换患者Dice：

| 臂 | 索引 | 原图 | 自身重建 | 外观交换 |
|---|---:|---:|---:|---:|
{swap_patients}

预注册更新前qS梯度范数（标签 / 实际0.01系数后生成；余弦见匿名分析）：

| 步号 | Full | Smooth |
|---|---:|---:|
{audits}

最后100批Full/Smooth生成损失={t['Full']['last100']['variational']:.3f}/{t['Smooth']['last100']['variational']:.3f}，重建MSE={t['Full']['last100']['reconstruction_mse']:.5f}/{t['Smooth']['last100']['reconstruction_mse']:.5f}。冻结U在合成图上的Dice只是可识别性审计，不证明合成标签真实有效或潜变量已因果解耦。未访问隐藏U标签/test。[冻结协议](PREREG_ZH.md)、[匿名分析](ANALYSIS.json)、[执行摘要](EXECUTION_SUMMARY.json)。旧路线硬停不变。
"""
    (out/'REPORT_ZH.md').write_text(report)
    print(verdict,f"Smooth_RUNMC={100*c['Smooth_RUNMC']['Dice']:.3f}",
          f"Smooth_BMC={100*c['Smooth_BMC']['Dice']:.3f}",
          f"Smooth_swap={100*s['Smooth']['swap_Dice']:.3f}")


if __name__=='__main__':main(sys.argv[1])
