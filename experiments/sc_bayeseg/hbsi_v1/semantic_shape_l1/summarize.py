"""Apply frozen L1 gates to an identity-free remote snapshot."""
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
    c=x['cells'];p=x['patients'];t=x['training']
    assert len(p['RUNMC'])==len(p['BMC'])==4
    diff=lambda phase,a,b:[100*(v[a]['classes'][0]['Dice']-v[b]['classes'][0]['Dice'])
                           for v in p[phase]]
    mdiff=lambda phase,a,b:100*(c[a+'_'+phase]['Dice']-c[b+'_'+phase]['Dice'])
    finite=all(all(math.isfinite(v[k]) for k in
                   ('task','variational','reconstruction_mse','shape_mean_sd',
                    'shape_logsd','appearance_logsd'))
               for arm in ('Cut','Joint') for v in t[arm].values())
    source={}
    for arm in ('Cut','U'):
        source['Joint_RUNMC_mean_vs_'+arm+'_ge_minus_1pp']=mdiff('RUNMC','Joint',arm)>=-1
        source['Joint_RUNMC_each_vs_'+arm+'_ge_minus_5pp']=min(diff('RUNMC','Joint',arm))>=-5
        source['Joint_RUNMC_train_mean_vs_'+arm+'_ge_minus_5pp']=\
            mdiff('RUNMC_train','Joint',arm)>=-5
    source['Joint_RUNMC_HD95_vs_both_le_1_10']=all(
        c['Joint_RUNMC']['HD95_penalized_mm']<=1.1*c[arm+'_RUNMC']['HD95_penalized_mm']
        for arm in ('Cut','U'))
    source['Joint_no_source_collapse']=c['Joint_RUNMC']['empty_predictions']==0 and \
                                       c['Joint_RUNMC_train']['empty_predictions']==0
    source['Cut_vs_U_mean_ge_minus_1pp']=mdiff('RUNMC','Cut','U')>=-1
    source['Cut_vs_U_each_ge_minus_5pp']=min(diff('RUNMC','Cut','U'))>=-5
    source['Cut_no_source_collapse']=c['Cut_RUNMC']['empty_predictions']==0
    source['finite_training_and_scale']=finite
    dcut=diff('BMC','Joint','Cut');du=diff('BMC','Joint','U')
    target={
        'Joint_BMC_mean_vs_max_Cut_U_H1_ge_plus_2pp':c['Joint_BMC']['Dice']>=
            max(c[arm+'_BMC']['Dice'] for arm in ('Cut','U','H1_Shape'))+.02,
        'Joint_BMC_3_of_4_nonnegative_vs_Cut_and_U':
            sum(a>=0 and b>=0 for a,b in zip(dcut,du))>=3,
        'Joint_BMC_each_vs_Cut_and_U_ge_minus_5pp':min(dcut+du)>=-5,
        'Joint_BMC_HD95_vs_Cut_and_U_le_1_10':all(
            c['Joint_BMC']['HD95_penalized_mm']<=1.1*c[arm+'_BMC']['HD95_penalized_mm']
            for arm in ('Cut','U')),
        'Joint_no_BMC_collapse':c['Joint_BMC']['empty_predictions']==0,
    }
    verdict=('POSITIVE_RESOURCE_SIGNAL' if all(source.values()) and all(target.values())
             else 'SOURCE_SAFETY_FAILED' if not all(source.values()) else 'NO_BMC_GAIN')
    analysis=dict(status=verdict,source_gates=source,BMC_signal_gates=target,
                  cells=c,patients=p,training=t,gradient_audits=x['gradient_audits'],
                  Joint_minus_Cut_pp={phase:mdiff(phase,'Joint','Cut')
                                      for phase in ('RUNMC','BMC','RUNMC_train')},
                  Joint_minus_U_pp={phase:mdiff(phase,'Joint','U')
                                    for phase in ('RUNMC','BMC','RUNMC_train')},
                  reused_BMC_development_patients=True,prior_hard_stops_unchanged=True)
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
                    for arm in ('U','H1_Shape','Cut','Joint'))
    gates=lambda obj:'\n'.join(f"| {name} | {'PASS' if ok else 'FAILED'} |"
                               for name,ok in obj.items())
    rows='\n'.join(f"| {phase} | {i+1} | {100*v['U']['classes'][0]['Dice']:.3f} | "
                   f"{100*v['Cut']['classes'][0]['Dice']:.3f} | "
                   f"{100*v['Joint']['classes'][0]['Dice']:.3f} | "
                   f"{100*(v['Joint']['classes'][0]['Dice']-v['Cut']['classes'][0]['Dice']):+.3f} | "
                   f"{100*(v['Joint']['classes'][0]['Dice']-v['U']['classes'][0]['Dice']):+.3f} |"
                   for phase in ('RUNMC','BMC') for i,v in enumerate(p[phase]))
    audits='\n'.join(f"| {step} | " + ' | '.join(
        f"{next(v for v in x['gradient_audits'][arm] if v['step']==step)['task_norm']:.5f} / "
        f"{next(v for v in x['gradient_audits'][arm] if v['step']==step)['weighted_variational_norm']:.5f}"
        for arm in ('Cut','Joint'))+' |' for step in (1,400,800,1200,1600))
    report=f"""# L1：标签锚定的 BayeSeg 形状场

**冻结判定：{verdict}。** 两臂同一RUNMC L4患者80片，各1600更新；qS本身是U初值锚定的分割logit场，qA独立，完整图像生成与层级先验保留。Cut截断生成梯度到qS，Joint允许反馈。Dice为4患者等权三维结果（%）；BMC4开发患者已反复用于路线选择，以下只是探索性资源门。

| 臂 | RUNMC开发Dice | BMC开发Dice | RUNMC完整L训练Dice | RUNMC惩罚HD95 mm | BMC惩罚HD95 mm |
|---|---:|---:|---:|---:|---:|
{table}

Joint−Cut RUNMC/BMC为{mdiff('RUNMC','Joint','Cut'):+.3f}/{mdiff('BMC','Joint','Cut'):+.3f}pp；Joint−旧U为{mdiff('RUNMC','Joint','U'):+.3f}/{mdiff('BMC','Joint','U'):+.3f}pp。U与H1 Shape为不同累积训练成本的历史参考，不是同轮匹配对照。

源安全门：

| 门 | 判定 |
|---|---|
{gates(source)}

BMC积极门（源门全过才允许后继）：

| 门 | 判定 |
|---|---|
{gates(target)}

匿名逐患者：

| 域 | 索引 | U Dice | Cut Dice | Joint Dice | Joint−Cut pp | Joint−U pp |
|---|---:|---:|---:|---:|---:|---:|
{rows}

预注册第1/400/800/1200/1600步更新前qS梯度范数（标签 / 实际0.01系数后生成；余弦见匿名分析）：

| 步号 | Cut | Joint |
|---|---:|---:|
{audits}

最后100批Cut/Joint生成损失={t['Cut']['last100']['variational']:.3f}/{t['Joint']['last100']['variational']:.3f}，重建MSE={t['Cut']['last100']['reconstruction_mse']:.5f}/{t['Joint']['last100']['reconstruction_mse']:.5f}，qS平均log标准差={t['Cut']['last100']['shape_logsd']:.3f}/{t['Joint']['last100']['shape_logsd']:.3f}，形状均值空间SD={t['Cut']['last100']['shape_mean_sd']:.3f}/{t['Joint']['last100']['shape_mean_sd']:.3f}。Joint生成目标下降与形状尺度扩张同时发生，不能从这些观测确定哪一项造成BMC下降；本轮继承的`p(S|Z)`此前拟合的是灰度形状场，语义logit场尺度不匹配也限制解释。这不证明形状/外观解耦，也不构成独立患者证据；隐藏U标签和test未访问。[冻结协议](PREREG_ZH.md)、[匿名分析](ANALYSIS.json)、[执行摘要](EXECUTION_SUMMARY.json)。旧路线判定不变。
"""
    (out/'REPORT_ZH.md').write_text(report)
    print(verdict,f"Joint_RUNMC={100*c['Joint_RUNMC']['Dice']:.3f}",
          f"Joint_BMC={100*c['Joint_BMC']['Dice']:.3f}")


if __name__=='__main__':main(sys.argv[1])
