"""Apply the frozen R1 paired target and source-safety gates."""
import json
import math
import sys
from pathlib import Path


def main(path):
    x = json.loads(Path(path).read_text())
    assert x['status'] == 'COMPLETE' and x['exit']['exit_code'] == 0
    assert x['preflight']['status'] == 'PASS'
    assert x['target_match']['status'] == 'PASS'
    assert x['run']['formal_updates'] == x['ledger']['updates_reserved'] == 1600
    assert not x['run']['target_unlabeled_used']
    assert all(a['status'] == 'PASS' for a in x['ledger']['attempts'])
    c=x['run']['cells'];p=x['patients']
    cut='RemainCut';joint='SwitchJoint'
    for domain in ('RUNMC','BMC','BMC_train'):
        assert len(p[domain])==4
        for arm in (cut,joint):
            assert c[f'{arm}_{domain}']['patients']==4
    assert all('SourceU' in row for row in p['RUNMC'])

    def pdiff(domain,reference):
        return [100*(v[joint]['Dice']-v[reference]['Dice']) for v in p[domain]]
    run_cut=pdiff('RUNMC',cut)
    run_source=pdiff('RUNMC','SourceU')
    bmc_cut=pdiff('BMC',cut)
    bmc_train=pdiff('BMC_train',cut)

    def mdiff(domain):
        return 100*(c[f'{joint}_{domain}']['Dice']-c[f'{cut}_{domain}']['Dice'])

    gates_source=dict(
        RUNMC_mean_vs_Cut_ge_minus_1pp=mdiff('RUNMC')>=-1,
        RUNMC_each_vs_Cut_ge_minus_5pp=min(run_cut)>=-5,
        RUNMC_HD95_vs_Cut_le_1_10=c[f'{joint}_RUNMC']['HD95_penalized_mm']
            <=1.1*c[f'{cut}_RUNMC']['HD95_penalized_mm'],
        RUNMC_mean_vs_source_ge_minus_0_5pp=
            100*(c[f'{joint}_RUNMC']['Dice']-x['source_U_RUNMC']['Dice'])>=-.5,
        RUNMC_each_vs_source_ge_minus_5pp=min(run_source)>=-5,
        no_RUNMC_patient_empty=all(c[f'{arm}_RUNMC']['empty_predictions']==0
                                   for arm in (cut,joint)))
    gates_target=dict(
        BMC_mean_vs_Cut_ge_plus_2pp=mdiff('BMC')>=2,
        BMC_at_least_3_of_4_nonnegative_vs_Cut=sum(v>=0 for v in bmc_cut)>=3,
        BMC_each_vs_Cut_ge_minus_5pp=min(bmc_cut)>=-5,
        BMC_train_mean_vs_Cut_ge_minus_5pp=mdiff('BMC_train')>=-5,
        BMC_HD95_vs_Cut_le_1_10=c[f'{joint}_BMC']['HD95_penalized_mm']
            <=1.1*c[f'{cut}_BMC']['HD95_penalized_mm'],
        no_BMC_patient_empty=all(c[f'{arm}_{domain}']['empty_predictions']==0
                                 for arm in (cut,joint)
                                 for domain in ('BMC','BMC_train')))
    assert all(math.isfinite(c[f'{arm}_{dom}']['Dice'])
               for arm in (cut,joint) for dom in ('RUNMC','BMC','BMC_train'))
    source_pass=all(gates_source.values())
    target_pass=all(gates_target.values())
    verdict='STAGED_FEEDBACK_SIGNAL' if source_pass and target_pass else 'STAGED_FEEDBACK_FAILED'
    out=Path(__file__).resolve().parent
    analysis=dict(status=verdict,source_gates=gates_source,target_gates=gates_target,
                  source_pass=source_pass,target_pass=target_pass,
                  cells=c,patients=p,source_U_RUNMC=x['source_U_RUNMC'],
                  last100=x['last100'],formal_updates=1600,
                  C1_source_Joint_hard_stop_unchanged=True,
                  repeated_four_patient_dev_exploratory=True)
    (out/'ANALYSIS.json').write_text(json.dumps(analysis,indent=2)+'\n')
    execution=dict(first_actual_start=x['launch']['started'],
                   deadline=x['launch']['deadline'],
                   elapsed_seconds=x['exit']['time']-x['launch']['started'],
                   new_root_bytes=x['bytes'],new_formal_updates=1600,
                   inherited_source_Cut_updates=1600,
                   attempts=[{k:a[k] for k in ('name','status','seconds_actual')}
                             for a in x['ledger']['attempts']])
    (out/'EXECUTION_SUMMARY.json').write_text(json.dumps(execution,indent=2)+'\n')
    table='\n'.join(f"| {arm} | {100*c[f'{arm}_RUNMC']['Dice']:.3f} | "
                    f"{100*c[f'{arm}_BMC']['Dice']:.3f} | "
                    f"{100*c[f'{arm}_BMC_train']['Dice']:.3f} | "
                    f"{c[f'{arm}_RUNMC']['HD95_penalized_mm']:.3f} | "
                    f"{c[f'{arm}_BMC']['HD95_penalized_mm']:.3f} |"
                    for arm in (cut,joint))
    gates='\n'.join(f"| {kind} | {name} | {'PASS' if good else 'FAILED'} |"
                    for kind,group in (('旧域',gates_source),('目标',gates_target))
                    for name,good in group.items())
    patient_rows='\n'.join(f"| {domain} | {i+1} | " +
        (' | '.join(f"{100*v[arm]['Dice']:.3f}"
                  for arm in ('SourceU',cut,joint)) if domain=='RUNMC'
         else '— | ' + ' | '.join(f"{100*v[arm]['Dice']:.3f}"
                                   for arm in (cut,joint))) +
        f" | {diff[i]:+.3f} |"
        for domain,rows,diff in (('RUNMC',p['RUNMC'],run_cut),
                                 ('BMC',p['BMC'],bmc_cut))
        for i,v in enumerate(rows))
    logs='\n'.join(f"| {arm} | {x['last100'][arm]['task']:.5f} | "
                   f"{x['last100'][arm]['bound_per_pixel']:.5f} | "
                   f"{x['last100'][arm]['shape_logsd']:.5f} | "
                   f"{x['last100'][arm]['z_logsd']:.5f} |"
                   for arm in (cut,joint))
    report=f"""# R1：目标阶段才接通 BayeSeg 生成反馈

**冻结判定：{verdict}。** 两目标臂从同一 C1 Cut 源第1600步状态出发，源 U 与 v45 U1600 逐张量相同；BMC L4患者134片各800更新，同数据、随机流及完整形状/外观图像生成层级。RemainCut 的目标 U 与 v45 U_B800 逐张量相同；SwitchJoint 只改变生成目标能否反馈 U，主要预测仍是双 logit U-Net。C1 Joint 源硬停及其未运行目标链不变。

| 目标臂 | 旧域 RUNMC Dice% | 新域 BMC Dice% | BMC完整L训练Dice% | 旧域惩罚HD95 mm | 新域惩罚HD95 mm |
|---|---:|---:|---:|---:|---:|
{table}

SwitchJoint−RemainCut 旧/新域为 {mdiff('RUNMC'):+.3f}/{mdiff('BMC'):+.3f} pp；相对共同源 U 的旧域变化 {100*(c[f'{joint}_RUNMC']['Dice']-x['source_U_RUNMC']['Dice']):+.3f} pp。每个 Dice 是四患者等权三维结果。

| 类别 | 预注册门 | 判定 |
|---|---|---|
{gates}

匿名逐患者 Dice（RUNMC 列依次为源 U、RemainCut、SwitchJoint；BMC 列依次为 RemainCut、SwitchJoint；末列为 SwitchJoint−RemainCut）：

| 域 | 索引 | 源 U | RemainCut | SwitchJoint | 差值pp |
|---|---:|---:|---:|---:|---:|
{patient_rows}

最后100批训练读数（生成 bound 为每像素值；它与分割终点不是同一指标）：

| 臂 | 标签风险 | 生成bound/像素 | qS均值logsd | qZ均值logsd |
|---|---:|---:|---:|---:|
{logs}

旧域安全门 **{'PASS' if source_pass else 'FAILED'}**，BMC积极门 **{'PASS' if target_pass else 'FAILED'}**。本实验只识别当前封存源生成状态下“目标阶段接通反馈”的整条条件效应；两臂训练参数许可不同，不能归因单独形状/外观因子。相同的是目标800更新和标签曝光，生成反馈多了计算与参数更新；C1 Cut 历史源1600更新不计作本轮新成本但必须计入方法总成本。四名开发患者已多次用于路线选择，任何增益仅为探索性，不证明独立泛化、长期CL、统计非劣或绝对旧域安全。隐藏U标签与test未访问。[冻结协议](PREREG_ZH.md)、[匿名分析](ANALYSIS.json)、[执行成本](EXECUTION_SUMMARY.json)。
"""
    (out/'REPORT_ZH.md').write_text(report)
    print(verdict,f"RUNMC={mdiff('RUNMC'):+.3f}",f"BMC={mdiff('BMC'):+.3f}")


if __name__=='__main__':
    main(sys.argv[1])
