"""Apply frozen external-site spatial-safety and paired-benefit gates."""
import json
import math
import sys
from pathlib import Path


SITES = ('BIDMC', 'HK', 'I2CVB', 'UCL')
COUNTS = {'BIDMC': 2, 'HK': 2, 'I2CVB': 3, 'UCL': 2}
ARMS = ('ReplayCut', 'ReplayJoint')


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
    assert x['predictions_sealed']['status'] == 'PASS'
    assert x['run']['status'] == 'COMPLETE' and x['run']['formal_updates'] == 0
    rows = x['run']['rows']
    assert len(rows) == 18
    assert [(r['site'], r['index'], r['arm']) for r in rows] == [
        (site, i, arm) for site in SITES for i in range(1, COUNTS[site] + 1)
        for arm in ARMS]
    assert not any('PRIVATE_' in key or 'patient' in key.lower()
                   for row in rows for key in row)

    def row(site, index, arm):
        return next(r for r in rows if (r['site'], r['index'], r['arm']) ==
                    (site, index, arm))

    def values(site, arm, variant, metric):
        return [row(site, i, arm)[variant][metric]
                for i in range(1, COUNTS[site] + 1)]

    def site_mean(site, arm, variant, metric):
        data = values(site, arm, variant, metric)
        assert all(v is not None and math.isfinite(v) for v in data)
        return sum(data) / len(data)

    def site_equal(arm, variant, metric):
        return sum(site_mean(s, arm, variant, metric) for s in SITES) / 4

    raw_effects = {}
    spatial_effects = {}
    for site in SITES:
        raw_effects[site] = 100 * (site_mean(site, 'ReplayJoint', 'original', 'Dice') -
                                   site_mean(site, 'ReplayCut', 'original', 'Dice'))
        spatial_effects[site] = 100 * (site_mean(site, 'ReplayJoint', 'component', 'Dice') -
                                       site_mean(site, 'ReplayCut', 'component', 'Dice'))
    patient_safety = []
    patient_paired = []
    for site in SITES:
        for i in range(1, COUNTS[site] + 1):
            cut, joint = (row(site, i, arm) for arm in ARMS)
            patient_safety.append(100 * (joint['component']['Dice'] -
                                         joint['original']['Dice']))
            patient_paired.append(100 * (joint['component']['Dice'] -
                                         cut['component']['Dice']))

    safety_gates = dict(
        site_equal_Dice_vs_original_ge_minus_1pp=
            100 * (site_equal('ReplayJoint', 'component', 'Dice') -
                   site_equal('ReplayJoint', 'original', 'Dice')) >= -1,
        each_patient_Dice_vs_original_ge_minus_5pp=min(patient_safety) >= -5,
        site_equal_HD95_vs_original_le_1_10=
            site_equal('ReplayJoint', 'component', 'HD95_penalized_mm') <=
            1.1 * site_equal('ReplayJoint', 'original', 'HD95_penalized_mm'),
        no_new_empty_patient=all(not r['component']['empty_prediction_failure']
            for r in rows if r['arm'] == 'ReplayJoint'))
    positive_gates = dict(
        site_equal_Dice_Joint_minus_Cut_ge_plus_2pp=
            100 * (site_equal('ReplayJoint', 'component', 'Dice') -
                   site_equal('ReplayCut', 'component', 'Dice')) >= 2,
        at_least_3_of_4_sites_nonnegative=sum(v >= 0 for v in spatial_effects.values()) >= 3,
        each_patient_Joint_minus_Cut_ge_minus_5pp=min(patient_paired) >= -5,
        site_equal_HD95_Joint_le_Cut_1_10=
            site_equal('ReplayJoint', 'component', 'HD95_penalized_mm') <=
            1.1 * site_equal('ReplayCut', 'component', 'HD95_penalized_mm'),
        no_component_patient_empty=all(not r['component']['empty_prediction_failure']
                                       for r in rows))
    safety_pass, positive_pass = all(safety_gates.values()), all(positive_gates.values())
    verdict = ('EXTERNAL_SPATIAL_SIGNAL' if safety_pass and positive_pass
               else 'EXTERNAL_SPATIAL_FAILED')
    summary = {site: {arm: {variant: {
        metric: site_mean(site, arm, variant, metric)
        for metric in ('Dice', 'HD95_penalized_mm')}
        for variant in ('original', 'component')}
        for arm in ARMS} for site in SITES}
    site_equal_summary = {arm: {variant: {
        metric: site_equal(arm, variant, metric)
        for metric in ('Dice', 'HD95_penalized_mm')}
        for variant in ('original', 'component')}
        for arm in ARMS}
    original_probability_quality = {arm: {
        metric: site_equal(arm, 'original', metric)
        for metric in ('NLL', 'Brier')} for arm in ARMS}
    out = Path(__file__).resolve().parent
    analysis = dict(status=verdict, safety_gates=safety_gates,
        positive_gates=positive_gates, safety_pass=safety_pass,
        positive_pass=positive_pass, rows=rows,
        per_site=summary, site_equal=site_equal_summary,
        raw_effects_pp=raw_effects, component_effects_pp=spatial_effects,
        patient_safety_effects_pp=patient_safety,
        patient_paired_effects_pp=patient_paired,
        original_probability_quality=original_probability_quality,
        R2_scientific_verdict_unchanged='REHEARSAL_FEEDBACK_FAILED',
        candidate_selected_on_BMC_development=True)
    (out / 'ANALYSIS.json').write_text(json.dumps(analysis, indent=2) + '\n')
    execution = dict(first_actual_start=x['launch']['started'],
        deadline=x['launch']['deadline'],
        elapsed_seconds=x['exit']['time'] - x['launch']['started'],
        compute_seconds=x['run']['elapsed_compute_seconds'],
        new_root_bytes=x['bytes'], formal_updates=0,
        synthetic_forward=x['run']['synthetic_forward'],
        inference_patient_volumes=x['run']['inference_patient_volumes'],
        slice_forward_calls=x['run']['slice_forward_calls'],
        metric_calls=x['run']['metric_calls'])
    (out / 'EXECUTION_SUMMARY.json').write_text(json.dumps(execution, indent=2) + '\n')
    table = '\n'.join(
        f"| {site} | {COUNTS[site]} | "
        f"{100*summary[site]['ReplayCut']['original']['Dice']:.3f} | "
        f"{100*summary[site]['ReplayJoint']['original']['Dice']:.3f} | "
        f"{100*summary[site]['ReplayCut']['component']['Dice']:.3f} | "
        f"{100*summary[site]['ReplayJoint']['component']['Dice']:.3f} | "
        f"{spatial_effects[site]:+.3f} | "
        f"{summary[site]['ReplayCut']['component']['HD95_penalized_mm']:.3f} | "
        f"{summary[site]['ReplayJoint']['component']['HD95_penalized_mm']:.3f} |"
        for site in SITES)
    gates = '\n'.join(f"| {kind} | {name} | {'PASS' if good else 'FAILED'} |"
        for kind, group in (('空间安全', safety_gates), ('配对积极', positive_gates))
        for name, good in group.items())
    patient_rows = '\n'.join(
        f"| {site} | {i} | "
        f"{100*row(site,i,'ReplayCut')['original']['Dice']:.3f} | "
        f"{100*row(site,i,'ReplayJoint')['original']['Dice']:.3f} | "
        f"{100*row(site,i,'ReplayCut')['component']['Dice']:.3f} | "
        f"{100*row(site,i,'ReplayJoint')['component']['Dice']:.3f} | "
        f"{100*(row(site,i,'ReplayJoint')['component']['Dice']-row(site,i,'ReplayCut')['component']['Dice']):+.3f} |"
        for site in SITES for i in range(1, COUNTS[site] + 1))
    cut = site_equal_summary['ReplayCut']
    joint = site_equal_summary['ReplayJoint']
    joint_component = [row(s,i,'ReplayJoint')['component']['Dice']
                       for s in SITES for i in range(1, COUNTS[s] + 1)]
    cut_component = [row(s,i,'ReplayCut')['component']['Dice']
                     for s in SITES for i in range(1, COUNTS[s] + 1)]
    report = f"""# V1：封存R2与固定三维空间规则的额外前列腺站点评价

**冻结判定：{verdict}。** BIDMC/HK/I2CVB/UCL全部既有val患者2/2/3/2名均纳入，未用于此前HBSI两站点模型选择；两只封存R2第800步模型0更新。每臂原U双logit概率与固定三维6邻接最大预测前景组件变换各评一次；后者不使用标签，但此规则曾根据BMC开发诊断提出，因此本次即使阳性也不是未受选择影响的确认性试验。R2原 `REHEARSAL_FEEDBACK_FAILED` 不变。

| 站点 | 患者 | Cut原Dice% | Joint原Dice% | Cut组件Dice% | Joint组件Dice% | 组件Joint−Cut pp | Cut组件HD95 mm | Joint组件HD95 mm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

四站点等权：原预测Cut/Joint Dice={100*cut['original']['Dice']:.3f}/{100*joint['original']['Dice']:.3f}%；固定组件Cut/Joint Dice={100*cut['component']['Dice']:.3f}/{100*joint['component']['Dice']:.3f}%（Joint−Cut {100*(joint['component']['Dice']-cut['component']['Dice']):+.3f}pp）。Joint原→组件HD95={joint['original']['HD95_penalized_mm']:.3f}→{joint['component']['HD95_penalized_mm']:.3f}mm；组件Cut/Joint HD95={cut['component']['HD95_penalized_mm']:.3f}/{joint['component']['HD95_penalized_mm']:.3f}mm。原概率四站点等权NLL Cut/Joint={original_probability_quality['ReplayCut']['NLL']:.5f}/{original_probability_quality['ReplayJoint']['NLL']:.5f}，Brier={original_probability_quality['ReplayCut']['Brier']:.5f}/{original_probability_quality['ReplayJoint']['Brier']:.5f}；硬组件输出不拿来比较软概率质量。

| 类别 | 冻结门 | 判定 |
|---|---|---|
{gates}

匿名逐患者Dice（组件末列为Joint−Cut）：

| 站点 | 索引 | Cut原% | Joint原% | Cut组件% | Joint组件% | 差值pp |
|---|---:|---:|---:|---:|---:|---:|
{patient_rows}

组件Joint相对Cut的9患者最差配对差 {min(patient_paired):+.3f}pp、获益 {sum(v>=0 for v in patient_paired)}/9；Joint组件绝对最差Dice {100*min(joint_component):.3f}%、9人插值p10 {100*percentile(joint_component,.1):.3f}%，Cut对应 {100*min(cut_component):.3f}%/{100*percentile(cut_component,.1):.3f}%。Joint组件相对自身原预测最差患者差 {min(patient_safety):+.3f}pp。站点效果范围 {min(spatial_effects.values()):+.3f}至{max(spatial_effects.values()):+.3f}pp；患者与站点并非可互换的独立重复。

空间安全门 **{'PASS' if safety_pass else 'FAILED'}**，外部配对积极门 **{'PASS' if positive_pass else 'FAILED'}**。{'该固定空间候选获得额外站点探索性信号，仍需独立训练种子/前瞻患者与长序列CL检验。' if safety_pass and positive_pass else '本固定候选未达到预设外部联合门，不按站点结果删病例、调阈值或改连通定义。'}单训练seed、先前BMC开发驱动候选、9名患者和四站点异质性限制结论；不读取任何test或U隐藏标签。[协议](PREREG_ZH.md)、[匿名分析](ANALYSIS.json)、[成本](EXECUTION_SUMMARY.json)。
"""
    (out / 'REPORT_ZH.md').write_text(report)
    print(verdict)


if __name__ == '__main__':
    main(sys.argv[1])
