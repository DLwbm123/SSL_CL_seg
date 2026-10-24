"""Summarize the frozen, read-only R2 component question."""
import json
import sys
from pathlib import Path


def main(path):
    x = json.loads(Path(path).read_text())
    assert x['status'] == 'COMPLETE' and x['exit']['exit_code'] == 0
    assert x['run']['status'] == 'COMPLETE'
    assert x['run']['formal_updates'] == 0 and x['run']['baseline_exact']
    rows = x['run']['rows']
    assert len(rows) == 16
    assert not any('PRIVATE_' in key or 'patient' in key.lower()
                   for row in rows for key in row)
    assert [(r['group'], r['arm'], r['index']) for r in rows] == [
        (group, arm, i)
        for group in ('BMC_train', 'BMC') for i in range(1, 5)
        for arm in ('ReplayCut', 'ReplayJoint')]
    def row(group, arm, index):
        return next(r for r in rows if (r['group'], r['arm'], r['index']) ==
                    (group, arm, index))
    probes = []
    for i in (3, 4):
        cut = row('BMC', 'ReplayCut', i)
        joint = row('BMC', 'ReplayJoint', i)
        probes.append(dict(index=i,
            HD95_restored=joint['largest_HD95_mm'] <= 1.1 * cut['baseline_HD95_mm'],
            removed_FP_dominant=(joint['removed_FP'] >= 100 and
                joint['removed_FP'] >= 10 * max(1, joint['removed_TP']))))
    component_candidate = all(all(v for k,v in p.items() if k!='index') for p in probes)
    verdict = 'DISCONNECTED_FP_CANDIDATE' if component_candidate else 'SIMPLE_COMPONENT_EXPLANATION_INSUFFICIENT'
    out = Path(__file__).resolve().parent
    analysis = dict(status=verdict, component_candidate=component_candidate,
        predetermined_patient_probes=probes, rows=rows,
        R2_scientific_verdict_unchanged='REHEARSAL_FEEDBACK_FAILED',
        posthoc_transform_not_a_primary_model=True)
    (out / 'ANALYSIS.json').write_text(json.dumps(analysis, indent=2) + '\n')
    execution = dict(first_actual_start=x['launch']['started'],
        deadline=x['launch']['deadline'],
        elapsed_seconds=x['exit']['time'] - x['launch']['started'],
        compute_seconds=x['run']['elapsed_compute_seconds'],
        new_root_bytes=x['bytes'], formal_updates=0,
        original_patient_volumes=16, reconstructed_metrics_calls=32)
    (out / 'EXECUTION_SUMMARY.json').write_text(json.dumps(execution, indent=2) + '\n')
    table = '\n'.join(
        f"| {'训练' if r['group']=='BMC_train' else '开发'} | {r['arm']} | {r['index']} | "
        f"{r['predicted_components']} | {r['largest_fraction']:.4f} | "
        f"{r['removed_FP']}/{r['removed_TP']} | "
        f"{100*r['baseline_Dice']:.3f}→{100*r['largest_Dice']:.3f} | "
        f"{r['baseline_HD95_mm']:.3f}→{r['largest_HD95_mm']:.3f} |"
        for r in rows)
    report = f"""# T0：R2目标域边界误差的三维连通性诊断

**冻结描述判定：{verdict}。** 仅重新读取封存R2两臂在BMC训练/开发各四名患者的完整三维预测，0网络前向、0反向、0更新。16例原Dice与惩罚HD95均用正式评分函数重新计算且与R2原患者指标误差≤1e−9。固定的三维6邻接最大预测前景组件保留不使用标签；标签仅用于事后核算被删FP/TP与指标。这是诊断性反事实，不是R2主预测、阈值选择或正式后处理配置。

| 集合 | 臂 | 匿名索引 | 原预测组件数 | 最大组件占预测比例 | 被删FP/TP体素 | Dice原→最大组件% | 惩罚HD95原→最大组件mm |
|---|---|---:|---:|---:|---:|---:|---:|
{table}

预定关注的开发第3人：ReplayJoint最大组件HD95 {row('BMC','ReplayJoint',3)['largest_HD95_mm']:.3f}mm，ReplayCut原HD95×1.10为 {1.1*row('BMC','ReplayCut',3)['baseline_HD95_mm']:.3f}mm，恢复门 {'PASS' if probes[0]['HD95_restored'] else 'FAILED'}；被删FP/TP={row('BMC','ReplayJoint',3)['removed_FP']}/{row('BMC','ReplayJoint',3)['removed_TP']}，FP主导门 {'PASS' if probes[0]['removed_FP_dominant'] else 'FAILED'}。

开发第4人：ReplayJoint最大组件HD95 {row('BMC','ReplayJoint',4)['largest_HD95_mm']:.3f}mm，ReplayCut原HD95×1.10为 {1.1*row('BMC','ReplayCut',4)['baseline_HD95_mm']:.3f}mm，恢复门 {'PASS' if probes[1]['HD95_restored'] else 'FAILED'}；被删FP/TP={row('BMC','ReplayJoint',4)['removed_FP']}/{row('BMC','ReplayJoint',4)['removed_TP']}，FP主导门 {'PASS' if probes[1]['removed_FP_dominant'] else 'FAILED'}。

R2原冻结判定 `REHEARSAL_FEEDBACK_FAILED` 保持不变。{'此诊断仅将断开FP列为后继空间先验的候选机制；尚未验证训练或独立泛化。' if component_candidate else '简单最大组件保留不足以同时解释并修复两名开发患者的HD95损失；不能据此单独认定主体过扩张或强度漂移为原因。'} 四名开发患者已反复用于路线选择，本轮仍只作探索性机制诊断；不启动阈值或组件大小搜索，不读取隐藏U标签或test。[冻结协议](PREREG_ZH.md)、[匿名分析](ANALYSIS.json)、[成本](EXECUTION_SUMMARY.json)。
"""
    (out / 'REPORT_ZH.md').write_text(report)
    print(verdict)


if __name__ == '__main__':
    main(sys.argv[1])
