"""Evaluate the frozen old-domain safety question for the fixed component rule."""
import json
import sys
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
    assert x['run']['status'] == 'COMPLETE'
    assert x['run']['formal_updates'] == 0 and x['run']['baseline_exact']
    rows = x['run']['rows']
    assert len(rows) == 16
    assert [(r['group'], r['arm'], r['index']) for r in rows] == [
        (group, arm, i)
        for group in ('RUNMC_train', 'RUNMC') for i in range(1, 5)
        for arm in ('ReplayCut', 'ReplayJoint')]
    assert not any('PRIVATE_' in key or 'patient' in key.lower()
                   for row in rows for key in row)
    old = x['source_U_patients']
    assert len(old) == 4 and all(0 <= v['Dice'] <= 1 for v in old)
    def row(group, arm, index):
        return next(r for r in rows if (r['group'], r['arm'], r['index']) ==
                    (group, arm, index))
    joint = [row('RUNMC', 'ReplayJoint', i) for i in range(1, 5)]
    base = [r['baseline_Dice'] for r in joint]
    final = [r['largest_Dice'] for r in joint]
    changes = [100 * (b - a) for a,b in zip(base, final)]
    vs_source = [100 * (r['largest_Dice'] - v['Dice'])
                 for r,v in zip(joint, old)]
    original_hd = sum(r['baseline_HD95_mm'] for r in joint) / 4
    final_hd = sum(r['largest_HD95_mm'] for r in joint) / 4
    gates = dict(
        mean_vs_original_ge_minus_1pp=sum(changes)/4 >= -1,
        each_vs_original_ge_minus_5pp=min(changes) >= -5,
        HD95_vs_original_le_1_10=final_hd <= 1.1 * original_hd,
        no_patient_empty=all(r['predicted_components'] > 0 for r in joint),
        mean_vs_source_U_ge_minus_0_5pp=sum(vs_source)/4 >= -.5,
        each_vs_source_U_ge_minus_5pp=min(vs_source) >= -5)
    passed = all(gates.values())
    verdict = 'OLD_DOMAIN_COMPONENT_SAFETY_PASS' if passed else 'OLD_DOMAIN_COMPONENT_SAFETY_FAILED'
    out = Path(__file__).resolve().parent
    analysis = dict(status=verdict, safety_gates=gates, safety_pass=passed,
        rows=rows, source_U_patients=old,
        R2_scientific_verdict_unchanged='REHEARSAL_FEEDBACK_FAILED',
        T0_result_not_confirmatory=True)
    (out / 'ANALYSIS.json').write_text(json.dumps(analysis, indent=2) + '\n')
    execution = dict(first_actual_start=x['launch']['started'],
        deadline=x['launch']['deadline'],
        elapsed_seconds=x['exit']['time'] - x['launch']['started'],
        compute_seconds=x['run']['elapsed_compute_seconds'],
        new_root_bytes=x['bytes'], formal_updates=0,
        original_patient_volumes=16, reconstructed_metrics_calls=32)
    (out / 'EXECUTION_SUMMARY.json').write_text(json.dumps(execution, indent=2) + '\n')
    table = '\n'.join(
        f"| {'训练' if r['group']=='RUNMC_train' else '开发'} | {r['arm']} | {r['index']} | "
        f"{r['gt_components']}/{r['predicted_components']} | "
        f"{r['removed_FP']}/{r['removed_TP']} | "
        f"{100*r['baseline_Dice']:.3f}→{100*r['largest_Dice']:.3f} | "
        f"{r['baseline_HD95_mm']:.3f}→{r['largest_HD95_mm']:.3f} |"
        for r in rows)
    gate_rows = '\n'.join(f"| {name} | {'PASS' if good else 'FAILED'} |"
                          for name,good in gates.items())
    worst_original = 100 * min(base)
    worst_final = 100 * min(final)
    p10_original = 100 * percentile(base, .1)
    p10_final = 100 * percentile(final, .1)
    report = f"""# U1：R2固定连通体候选的旧域安全审计

**冻结判定：{verdict}。** 只读封存R2两臂的RUNMC完整L训练和开发各四名患者预测，0网络前向、0反向、0更新。16例原Dice/惩罚HD95与R2封存患者指标重算误差≤1e−9。变换固定为三维6邻接最大预测前景连通体，不用标签决定保留区域；该反事实没有成为R2正式预测。

| 集合 | 臂 | 匿名索引 | GT/预测组件数 | 被删FP/TP体素 | Dice原→最大组件% | 惩罚HD95原→最大组件mm |
|---|---|---:|---:|---:|---:|---:|
{table}

ReplayJoint开发旧域的变换相对原预测：等权Dice {sum(changes)/4:+.3f}pp，患者最小差 {min(changes):+.3f}pp，惩罚HD95 {original_hd:.3f}→{final_hd:.3f}mm（比例 {final_hd/original_hd:.3f}）。相对封存源U1600的Dice均值 {sum(vs_source)/4:+.3f}pp、患者最小差 {min(vs_source):+.3f}pp。绝对最差Dice原/变换后为 {worst_original:.3f}/{worst_final:.3f}%，四人插值p10为 {p10_original:.3f}/{p10_final:.3f}%。

| 冻结旧域安全门 | 判定 |
|---|---|
{gate_rows}

R2的 `REHEARSAL_FEEDBACK_FAILED` 与T0的探索性目标域连通体观察均不变。{'本次旧域开发安全探针没有推翻这个固定空间候选，但仍缺独立患者和前瞻性模型检验。' if passed else '本固定最大组件规则触发旧域安全失败，不作为当前主方法的直接推断规则。'}训练第3名ReplayJoint仍误删 {row('RUNMC_train','ReplayJoint',3)['removed_TP']} 个TP，Dice下降 {100*(row('RUNMC_train','ReplayJoint',3)['largest_Dice']-row('RUNMC_train','ReplayJoint',3)['baseline_Dice']):+.3f}pp，故不宣称绝对无害。RUNMC开发患者同样多次用于决策；不能把这次安全检查与T0目标域结果拼成一项确认性成功，也不扫组件连接方式、阈值或大小。[协议](PREREG_ZH.md)、[匿名分析](ANALYSIS.json)、[成本](EXECUTION_SUMMARY.json)。
"""
    (out / 'REPORT_ZH.md').write_text(report)
    print(verdict)


if __name__ == '__main__':
    main(sys.argv[1])
