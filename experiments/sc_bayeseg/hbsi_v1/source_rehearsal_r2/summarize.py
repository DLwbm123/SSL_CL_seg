"""Apply preregistered R2 source-safety and target-benefit gates."""
import json
import math
import sys
from itertools import product
from pathlib import Path


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (position - low) * (ordered[high] - ordered[low])


def main(path):
    x = json.loads(Path(path).read_text())
    assert x['status'] == 'COMPLETE' and x['exit']['exit_code'] == 0
    assert x['preflight']['status'] == 'PASS'
    assert x['run']['formal_updates'] == x['ledger']['updates_reserved'] == 1600
    assert all(a['status'] == 'PASS' for a in x['ledger']['attempts'])
    assert not x['run']['target_unlabeled_used']
    c, p = x['run']['cells'], x['patients']
    cut, joint = 'ReplayCut', 'ReplayJoint'
    for domain in ('RUNMC', 'BMC', 'RUNMC_train', 'BMC_train'):
        assert len(p[domain]) == 4
        for arm in (cut, joint):
            assert c[f'{arm}_{domain}']['patients'] == 4
            assert math.isfinite(c[f'{arm}_{domain}']['Dice'])
    assert all('SourceU' in row for row in p['RUNMC'])

    def patient_diffs(domain, reference):
        return [100 * (row[joint]['Dice'] - row[reference]['Dice'])
                for row in p[domain]]

    def mean_diff(domain):
        return 100 * (c[f'{joint}_{domain}']['Dice'] -
                      c[f'{cut}_{domain}']['Dice'])

    old = patient_diffs('RUNMC', cut)
    old_source = patient_diffs('RUNMC', 'SourceU')
    new = patient_diffs('BMC', cut)
    source_gates = dict(
        RUNMC_mean_vs_Cut_ge_minus_1pp=mean_diff('RUNMC') >= -1,
        RUNMC_each_vs_Cut_ge_minus_5pp=min(old) >= -5,
        RUNMC_HD95_vs_Cut_le_1_10=c[f'{joint}_RUNMC']['HD95_penalized_mm']
            <= 1.1 * c[f'{cut}_RUNMC']['HD95_penalized_mm'],
        RUNMC_mean_vs_source_ge_minus_0_5pp=
            100 * (c[f'{joint}_RUNMC']['Dice'] - x['source_U_RUNMC']['Dice']) >= -.5,
        RUNMC_each_vs_source_ge_minus_5pp=min(old_source) >= -5,
        no_RUNMC_patient_empty=all(c[f'{arm}_RUNMC']['empty_predictions'] == 0
                                   for arm in (cut, joint)))
    target_gates = dict(
        BMC_mean_vs_Cut_ge_plus_2pp=mean_diff('BMC') >= 2,
        BMC_at_least_3_of_4_nonnegative_vs_Cut=sum(v >= 0 for v in new) >= 3,
        BMC_each_vs_Cut_ge_minus_5pp=min(new) >= -5,
        BMC_HD95_vs_Cut_le_1_10=c[f'{joint}_BMC']['HD95_penalized_mm']
            <= 1.1 * c[f'{cut}_BMC']['HD95_penalized_mm'],
        BMC_train_mean_vs_Cut_ge_minus_5pp=mean_diff('BMC_train') >= -5,
        no_BMC_patient_empty=all(c[f'{arm}_{domain}']['empty_predictions'] == 0
                                 for arm in (cut, joint)
                                 for domain in ('BMC', 'BMC_train')))
    source_pass, target_pass = all(source_gates.values()), all(target_gates.values())
    verdict = ('REHEARSAL_FEEDBACK_SIGNAL' if source_pass and target_pass
               else 'REHEARSAL_FEEDBACK_FAILED')
    tails = {domain: {arm: dict(
        worst_Dice_percent=100 * min(row[arm]['Dice'] for row in p[domain]),
        empirical_p10_percent=100 * percentile(
            [row[arm]['Dice'] for row in p[domain]], .1))
        for arm in (cut, joint)} for domain in ('RUNMC', 'BMC')}
    ci = {}
    for domain, values in (('RUNMC', old), ('BMC', new)):
        means = [sum(sample) / 4 for sample in product(values, repeat=4)]
        ci[domain] = [percentile(means, .025), percentile(means, .975)]

    out = Path(__file__).resolve().parent
    r1 = json.loads((out.parent / 'staged_target_feedback_r1' / 'ANALYSIS.json').read_text())
    assert r1['status'] == 'STAGED_FEEDBACK_FAILED'
    r1_cells = r1['cells']
    analysis = dict(status=verdict, source_gates=source_gates,
        target_gates=target_gates, source_pass=source_pass,
        target_pass=target_pass, cells=c, patients=p,
        source_U_RUNMC=x['source_U_RUNMC'], last100=x['last100'],
        formal_updates=1600, descriptive_tail=tails,
        paired_four_patient_bootstrap95pp=ci,
        C1_source_Joint_hard_stop_unchanged=True,
        R1_and_S1_verdicts_unchanged=True,
        repeated_four_patient_dev_exploratory=True)
    (out / 'ANALYSIS.json').write_text(json.dumps(analysis, indent=2) + '\n')
    execution = dict(first_actual_start=x['launch']['started'],
        deadline=x['launch']['deadline'],
        elapsed_seconds=x['exit']['time'] - x['launch']['started'],
        new_root_bytes=x['bytes'], new_formal_updates=1600,
        inherited_source_Cut_updates=1600,
        source_replay_supervised_images=1600,
        target_supervised_images=1600,
        target_generator_images=1600,
        attempts=[{k: a[k] for k in ('name', 'status', 'seconds_actual')}
                  for a in x['ledger']['attempts']])
    (out / 'EXECUTION_SUMMARY.json').write_text(json.dumps(execution, indent=2) + '\n')
    table = '\n'.join(
        f"| {arm} | {100*c[f'{arm}_RUNMC']['Dice']:.3f} | "
        f"{100*c[f'{arm}_BMC']['Dice']:.3f} | "
        f"{100*c[f'{arm}_RUNMC_train']['Dice']:.3f} | "
        f"{100*c[f'{arm}_BMC_train']['Dice']:.3f} |"
        for arm in (cut, joint))
    quality = '\n'.join(
        f"| {arm} | {c[f'{arm}_BMC']['precision']*100:.3f} | "
        f"{c[f'{arm}_BMC']['recall']*100:.3f} | "
        f"{c[f'{arm}_BMC']['HD95_penalized_mm']:.3f} | "
        f"{c[f'{arm}_BMC']['NLL']:.5f} | "
        f"{c[f'{arm}_BMC']['Brier']:.5f} |"
        for arm in (cut, joint))
    gates = '\n'.join(f"| {kind} | {name} | {'PASS' if good else 'FAILED'} |"
        for kind, group in (('旧域', source_gates), ('目标', target_gates))
        for name, good in group.items())
    rows = '\n'.join(
        f"| {domain} | {i+1} | " +
        (f"{100*row['SourceU']['Dice']:.3f}" if domain == 'RUNMC' else '—') +
        f" | {100*row[cut]['Dice']:.3f} | {100*row[joint]['Dice']:.3f} | "
        f"{difference[i]:+.3f} |"
        for domain, difference in (('RUNMC', old), ('BMC', new))
        for i, row in enumerate(p[domain]))
    log_rows = '\n'.join(
        f"| {arm} | {x['last100'][arm]['source_task']:.5f} | "
        f"{x['last100'][arm]['target_task']:.5f} | "
        f"{x['last100'][arm]['bound_per_pixel']:.5f} |"
        for arm in (cut, joint))
    report = f"""# R2：源标签重放下的目标生成反馈

**冻结判定：{verdict}。** 两臂继承同一 C1 Cut 源1600状态，目标阶段各800次更新；每步均取一张RUNMC源L图和一张BMC目标L图，各占监督风险一半，目标图额外保留完整BayeSeg形状/外观生成目标。唯一臂间干预是目标生成梯度是否进入U-Net；主预测均为标准双logit U。每臂均有800源＋800目标标签曝光，不能与R1无重放臂当成等暴露性能比较。

| 臂 | RUNMC开发Dice% | BMC开发Dice% | RUNMC完整L训练Dice% | BMC完整L训练Dice% |
|---|---:|---:|---:|---:|
{table}

ReplayJoint−ReplayCut 的旧/新域开发均值变化为 {mean_diff('RUNMC'):+.3f}/{mean_diff('BMC'):+.3f}pp；相对封存源U旧域均值 {100*(c[f'{joint}_RUNMC']['Dice']-x['source_U_RUNMC']['Dice']):+.3f}pp。每个Dice是四患者等权三维结果。

只读R1无重放终点的RUNMC/BMC开发Dice为RemainCut {100*r1_cells['RemainCut_RUNMC']['Dice']:.3f}/{100*r1_cells['RemainCut_BMC']['Dice']:.3f}%，SwitchJoint {100*r1_cells['SwitchJoint_RUNMC']['Dice']:.3f}/{100*r1_cells['SwitchJoint_BMC']['Dice']:.3f}%。R2的ReplayCut相对R1 RemainCut为 {100*(c[f'{cut}_RUNMC']['Dice']-r1_cells['RemainCut_RUNMC']['Dice']):+.3f}/{100*(c[f'{cut}_BMC']['Dice']-r1_cells['RemainCut_BMC']['Dice']):+.3f}pp，ReplayJoint相对R1 SwitchJoint为 {100*(c[f'{joint}_RUNMC']['Dice']-r1_cells['SwitchJoint_RUNMC']['Dice']):+.3f}/{100*(c[f'{joint}_BMC']['Dice']-r1_cells['SwitchJoint_BMC']['Dice']):+.3f}pp。重放同时改变源标签曝光与每步任务混合，这些跨实验差异不构成单独生成反馈的因果估计。

| 臂 | BMC precision% | BMC recall% | BMC惩罚HD95 mm | BMC NLL | BMC Brier |
|---|---:|---:|---:|---:|---:|
{quality}

BMC惩罚HD95均值从 {c[f'{cut}_BMC']['HD95_penalized_mm']:.3f} 增至 {c[f'{joint}_BMC']['HD95_penalized_mm']:.3f}mm，比例 {c[f'{joint}_BMC']['HD95_penalized_mm']/c[f'{cut}_BMC']['HD95_penalized_mm']:.3f}，超过冻结1.10上限；Dice显著提高不能覆盖该边界损失。
匿名BMC四人的惩罚HD95 Cut→Joint 为 {', '.join(f"{v[cut]['HD95_penalized_mm']:.3f}→{v[joint]['HD95_penalized_mm']:.3f}" for v in p['BMC'])}mm；第3、4人明显上升。

| 类别 | 预注册门 | 判定 |
|---|---|---|
{gates}

匿名逐患者Dice，最后一列为ReplayJoint−ReplayCut百分点：

| 域 | 索引 | 源U | ReplayCut | ReplayJoint | 差值pp |
|---|---:|---:|---:|---:|---:|
{rows}

旧域相对ReplayCut的最小配对差 {min(old):+.3f}pp、下降 {sum(v<0 for v in old)}/4 患者；新域最小配对差 {min(new):+.3f}pp、获益 {sum(v>=0 for v in new)}/4。旧域绝对最差Dice Cut/Joint={tails['RUNMC'][cut]['worst_Dice_percent']:.3f}/{tails['RUNMC'][joint]['worst_Dice_percent']:.3f}%，四人插值p10={tails['RUNMC'][cut]['empirical_p10_percent']:.3f}/{tails['RUNMC'][joint]['empirical_p10_percent']:.3f}%；新域绝对最差={tails['BMC'][cut]['worst_Dice_percent']:.3f}/{tails['BMC'][joint]['worst_Dice_percent']:.3f}%。四名配对患者的4⁴有放回重采样均值95%经验描述区间：旧域 [{ci['RUNMC'][0]:+.3f},{ci['RUNMC'][1]:+.3f}]pp，新域 [{ci['BMC'][0]:+.3f},{ci['BMC'][1]:+.3f}]pp；这些并非独立患者总体推断，也不进入门槛。

最后100批训练读数（生成bound每像素，不等于分割）：

| 臂 | 源标签风险 | 目标标签风险 | 目标生成bound/像素 |
|---|---:|---:|---:|
{log_rows}

旧域安全门 **{'PASS' if source_pass else 'FAILED'}**，BMC积极门 **{'PASS' if target_pass else 'FAILED'}**。任一门失败则固定策略关闭，不调重放比例、生成系数、步数、阈值、种子或检查点。C1/R1/S1原冻结判定不变。四名源/目标开发患者已反复用于路线选择，本轮至多提供探索性资源信号；未取得独立患者、多种子、无重放或长序列持续学习证据。隐藏U标签与test未访问。[冻结协议](PREREG_ZH.md)、[匿名分析](ANALYSIS.json)、[成本](EXECUTION_SUMMARY.json)。
"""
    (out / 'REPORT_ZH.md').write_text(report)
    print(verdict, f"RUNMC={mean_diff('RUNMC'):+.3f}",
          f"BMC={mean_diff('BMC'):+.3f}")


if __name__ == '__main__':
    main(sys.argv[1])
