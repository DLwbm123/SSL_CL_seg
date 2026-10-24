"""Frozen label-conditional shape prior endpoint gates."""
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
    c=x['cells'];p=x['patients'];a=x['latent_audits'];ap=x['latent_patients'];t=x['training']
    assert len(p['RUNMC'])==len(p['BMC'])==4
    assert all(len(ap[arm])==4 and a[arm]['slices']==80 for arm in ('ProbZ','LabelY'))
    diff=lambda phase,left,right:[100*(v[left]['classes'][0]['Dice']-
                                        v[right]['classes'][0]['Dice']) for v in p[phase]]
    mdiff=lambda phase,left,right:100*(c[left+'_'+phase]['Dice']-
                                        c[right+'_'+phase]['Dice'])
    finite=all(all(math.isfinite(v[k]) for k in
                   ('fused_task','direct_task','variational','reconstruction_mse',
                    'shape_logsd','appearance_logsd'))
               for arm in ('ProbZ','LabelY') for v in t[arm].values())
    pearson=[v['LabelY']['abs_mask_pearson']-v['ProbZ']['abs_mask_pearson']
             for v in ap]
    mechanism={
        'LabelY_abs_Pearson_mean_vs_ProbZ_ge_plus_0_05':
            a['LabelY']['mean_abs_mask_pearson']-a['ProbZ']['mean_abs_mask_pearson']>=.05,
        'LabelY_abs_Pearson_at_least_3_of_4_nonnegative':sum(v>=0 for v in pearson)>=3,
        'LabelY_mean_reconstruction_MSE_le_2x_ProbZ':
            a['LabelY']['mean_reconstruction_mse']<=2*a['ProbZ']['mean_reconstruction_mse'],
        'training_and_scales_finite':finite,
    }
    source={}
    for arm in ('ProbZ','U'):
        source['LabelY_RUNMC_mean_vs_'+arm+'_ge_minus_1pp']=mdiff('RUNMC','LabelY',arm)>=-1
        source['LabelY_RUNMC_each_vs_'+arm+'_ge_minus_5pp']=min(diff('RUNMC','LabelY',arm))>=-5
        source['LabelY_RUNMC_train_mean_vs_'+arm+'_ge_minus_5pp']=\
            mdiff('RUNMC_train','LabelY',arm)>=-5
    source['LabelY_RUNMC_HD95_vs_both_le_1_10']=all(
        c['LabelY_RUNMC']['HD95_penalized_mm']<=1.1*c[arm+'_RUNMC']['HD95_penalized_mm']
        for arm in ('ProbZ','U'))
    source['LabelY_no_source_collapse']=c['LabelY_RUNMC']['empty_predictions']==0 and \
                                        c['LabelY_RUNMC_train']['empty_predictions']==0
    source['ProbZ_vs_U_mean_ge_minus_1pp']=mdiff('RUNMC','ProbZ','U')>=-1
    source['ProbZ_vs_U_each_ge_minus_5pp']=min(diff('RUNMC','ProbZ','U'))>=-5
    source['ProbZ_no_source_collapse']=c['ProbZ_RUNMC']['empty_predictions']==0
    dprob=diff('BMC','LabelY','ProbZ');du=diff('BMC','LabelY','U')
    target={
        'LabelY_BMC_mean_vs_max_ProbZ_U_H1_ge_plus_2pp':c['LabelY_BMC']['Dice']>=
            max(c[arm+'_BMC']['Dice'] for arm in ('ProbZ','U','H1_Shape'))+.02,
        'LabelY_BMC_3_of_4_nonnegative_vs_ProbZ_and_U':
            sum(v>=0 and w>=0 for v,w in zip(dprob,du))>=3,
        'LabelY_BMC_each_vs_ProbZ_and_U_ge_minus_5pp':min(dprob+du)>=-5,
        'LabelY_BMC_HD95_vs_ProbZ_and_U_le_1_10':all(
            c['LabelY_BMC']['HD95_penalized_mm']<=1.1*c[arm+'_BMC']['HD95_penalized_mm']
            for arm in ('ProbZ','U')),
        'LabelY_no_BMC_collapse':c['LabelY_BMC']['empty_predictions']==0,
    }
    verdict=('POSITIVE_RESOURCE_SIGNAL' if all(mechanism.values()) and all(source.values())
             and all(target.values()) else
             'MECHANISM_FAILED' if not all(mechanism.values()) else
             'SOURCE_SAFETY_FAILED' if not all(source.values()) else 'NO_BMC_GAIN')
    analysis=dict(status=verdict,mechanism_gates=mechanism,source_gates=source,
                  BMC_signal_gates=target,cells=c,patients=p,latent_audits=a,
                  latent_patients=ap,training=t,gradient_audits=x['gradient_audits'],
                  LabelY_minus_ProbZ_pp={phase:mdiff(phase,'LabelY','ProbZ')
                                          for phase in ('RUNMC','BMC','RUNMC_train')},
                  LabelY_minus_U_pp={phase:mdiff(phase,'LabelY','U')
                                      for phase in ('RUNMC','BMC','RUNMC_train')},
                  BMC_reused_development_patients=True,prior_hard_stops_unchanged=True)
    out=Path(__file__).resolve().parent
    (out/'ANALYSIS.json').write_text(json.dumps(analysis,indent=2)+'\n')
    execution=dict(first_actual_start=x['launch']['started'],deadline=x['launch']['deadline'],
                   elapsed_seconds=x['exit']['time']-x['launch']['started'],
                   new_root_bytes=x['bytes'],new_formal_updates=3200,
                   reused_H1_common_initial_C1_source_updates=1600,
                   attempts=[{k:v[k] for k in ('name','status','seconds_actual')}
                             for v in x['ledger']['attempts']])
    (out/'EXECUTION_SUMMARY.json').write_text(json.dumps(execution,indent=2)+'\n')
    table='\n'.join(f"| {arm} | {100*c[arm+'_RUNMC']['Dice']:.3f} | "
                    f"{100*c[arm+'_BMC']['Dice']:.3f} | "
                    f"{100*c[arm+'_RUNMC_train']['Dice']:.3f} | "
                    f"{c[arm+'_RUNMC']['HD95_penalized_mm']:.3f} | "
                    f"{c[arm+'_BMC']['HD95_penalized_mm']:.3f} |"
                    for arm in ('U','H1_Shape','ProbZ','LabelY'))
    gates=lambda obj:'\n'.join(f"| {name} | {'PASS' if ok else 'FAILED'} |"
                               for name,ok in obj.items())
    rows='\n'.join(f"| {phase} | {i+1} | " + ' | '.join(
        f"{100*v[arm]['classes'][0]['Dice']:.3f}"
        for arm in ('U','ProbZ','LabelY'))+' |'
        for phase in ('RUNMC','BMC') for i,v in enumerate(p[phase]))
    latent_rows='\n'.join(f"| {v['index']} | {v['ProbZ']['abs_mask_pearson']:.4f} | "
                          f"{v['LabelY']['abs_mask_pearson']:.4f} | "
                          f"{v['ProbZ']['mean_reconstruction_mse']:.5f} | "
                          f"{v['LabelY']['mean_reconstruction_mse']:.5f} |"
                          for v in ap)
    audits='\n'.join(f"| {step} | " + ' | '.join(
        f"{next(v for v in x['gradient_audits'][arm] if v['step']==step)['task_grad_norm']:.5f} / "
        f"{next(v for v in x['gradient_audits'][arm] if v['step']==step)['weighted_variational_grad_norm']:.5f}"
        for arm in ('ProbZ','LabelY'))+' |' for step in (1,400,800,1200,1600))
    report=f"""# P1：标签条件形状先验与完整图像生成

**冻结判定：{verdict}。** 两臂同H1共同零更新初态、同RUNMC L4患者80片、各1600次更新；只改变训练时形状先验输入`sampled Z`或`observed Y`。`q(S|I)`、独立`q(A|I)`、`p(I|S,A)`全图像生成、qZ空间/Gamma先验均保留，U冻结；预测只输入影像，绝不使用测试标签。Dice为4患者等权三维结果（%）；BMC开发患者多次用于路线选择，属于探索性资源门。

| 臂 | RUNMC开发Dice | BMC开发Dice | RUNMC完整L训练Dice | RUNMC惩罚HD95 mm | BMC惩罚HD95 mm |
|---|---:|---:|---:|---:|---:|
{table}

LabelY−ProbZ RUNMC/BMC为{mdiff('RUNMC','LabelY','ProbZ'):+.3f}/{mdiff('BMC','LabelY','ProbZ'):+.3f}pp；LabelY−旧U为{mdiff('RUNMC','LabelY','U'):+.3f}/{mdiff('BMC','LabelY','U'):+.3f}pp。U/H1 Shape仅不同累积成本的历史参考。

源L患者形状后验与标签对齐、无采样均值重建：

| 匿名索引 | ProbZ绝对Pearson | LabelY绝对Pearson | ProbZ重建MSE | LabelY重建MSE |
|---|---:|---:|---:|---:|
{latent_rows}

机制门：

| 门 | 判定 |
|---|---|
{gates(mechanism)}

源安全门：

| 门 | 判定 |
|---|---|
{gates(source)}

BMC积极门（须机制/源安全全过才允许后继）：

| 门 | 判定 |
|---|---|
{gates(target)}

匿名开发逐患者Dice：

| 域 | 索引 | U | ProbZ | LabelY |
|---|---:|---:|---:|---:|
{rows}

预注册更新前qS梯度范数（标签 / 实际0.01系数后生成；余弦见匿名分析）：

| 步号 | ProbZ | LabelY |
|---|---:|---:|
{audits}

最后100批ProbZ/LabelY变分损失={t['ProbZ']['last100']['variational']:.3f}/{t['LabelY']['last100']['variational']:.3f}，样本重建MSE={t['ProbZ']['last100']['reconstruction_mse']:.5f}/{t['LabelY']['last100']['reconstruction_mse']:.5f}。源L相关性不是解耦证明，BMC4名患者不是独立验证。隐藏U标签和test未访问。[冻结协议](PREREG_ZH.md)、[匿名分析](ANALYSIS.json)、[执行摘要](EXECUTION_SUMMARY.json)。旧路线硬停不变。
"""
    (out/'REPORT_ZH.md').write_text(report)
    print(verdict,f"LabelY_RUNMC={100*c['LabelY_RUNMC']['Dice']:.3f}",
          f"LabelY_BMC={100*c['LabelY_BMC']['Dice']:.3f}")


if __name__=='__main__':main(sys.argv[1])
