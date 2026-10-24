"""Publish only anonymous complete-endpoint A1 evidence."""
import json
import sys
from pathlib import Path


def main(snapshot):
    x = json.loads(Path(snapshot).read_text())
    assert x['complete'] and x['exit']['exit_code'] == 0
    assert x['run']['formal_updates'] == 2400 and x['preflight']['optimizer_updates'] == 0
    assert x['ledger']['updates_reserved'] == 2400
    out = Path(__file__).parent
    cells = x['run']['cells']
    names = ('source_RUNMC', 'source_BMC', 'target_RUNMC', 'target_BMC')
    controls = {k: x['control'][k]['repeats'][0] for k in names}
    paired = {}
    for name in names:
        rows = x['paired'][name]
        assert len(rows) == 4 and [r['index'] for r in rows] == [1, 2, 3, 4]
        pairs = []
        for r in rows:
            a, u = r['A']['classes'][0], r['U']['classes'][0]
            pairs.append(dict(index=r['index'], A_Dice=a['Dice'], U_Dice=u['Dice'],
                              Dice_difference_pp=100 * (a['Dice'] - u['Dice']),
                              A_precision=a['TP'] / a['predicted'] if a['predicted'] else 0,
                              A_recall=a['TP'] / a['GT'] if a['GT'] else 0,
                              A_TP=a['TP'], A_FP=a['FP'], A_FN=a['FN'],
                              A_predicted_to_GT=a['predicted'] / a['GT'] if a['GT'] else None,
                              A_HD95_penalized_mm=a['HD95_penalized_mm']))
        paired[name] = pairs
    old_change = []
    for before, after in zip(x['paired']['source_RUNMC'], x['paired']['target_RUNMC']):
        assert before['index'] == after['index']
        old_change.append(dict(index=before['index'], source_Dice=before['A']['classes'][0]['Dice'],
                               target_Dice=after['A']['classes'][0]['Dice'],
                               change_pp=100 * (after['A']['classes'][0]['Dice'] - before['A']['classes'][0]['Dice'])))
    train = x['train_patient']
    assert len(train) == 4
    train_tp = sum(m['classes'][0]['TP'] for m in train)
    train_pred = sum(m['classes'][0]['predicted'] for m in train)
    gates = {
        'source_RUNMC_within_1pp_of_U': cells['source_RUNMC']['Dice'] >= controls['source_RUNMC']['Dice'] - .01,
        'target_BMC_within_2pp_of_U': cells['target_BMC']['Dice'] >= controls['target_BMC']['Dice'] - .02,
        'train_foreground_not_collapsed': train_pred > 0 and train_tp > 0 and cells['target_BMC_train']['empty_predictions'] == 0,
        'no_dev_whole_patient_empty_prediction': all(cells[k]['empty_predictions'] == 0 for k in names),
    }
    analysis = dict(status='PASS' if all(gates.values()) else 'FAILED', gates=gates,
                    cells={k: cells[k] for k in names}, U_controls=controls,
                    target_BMC_train=cells['target_BMC_train'], paired=paired,
                    old_domain_change=old_change,
                    train_diagnostics=x['train_diagnostics'],
                    source_RUNMC_minus_U_pp=100 * (cells['source_RUNMC']['Dice'] - controls['source_RUNMC']['Dice']),
                    target_BMC_minus_U_pp=100 * (cells['target_BMC']['Dice'] - controls['target_BMC']['Dice']),
                    old_domain_mean_change_pp=100 * (cells['target_RUNMC']['Dice'] - cells['source_RUNMC']['Dice']),
                    old_domain_worst_patient_change_pp=min(v['change_pp'] for v in old_change))
    execution = dict(first_actual_start=x['launch']['started'], original_deadline=x['launch']['deadline'],
                     elapsed_seconds=x['exit']['time'] - x['launch']['started'], new_root_bytes=x['size_bytes'],
                     first_import_failure_preserved=True, first_failure_updates=0,
                     preflight_reverse_calls=x['preflight']['real_source_target_backward'],
                     formal_updates=2400, attempt_statuses=[dict(name=a['name'], status=a['status'],
                                                                  seconds_actual=a.get('seconds_actual'))
                                                      for a in x['ledger']['attempts']])
    (out / 'ANALYSIS.json').write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + '\n')
    (out / 'EXECUTION_SUMMARY.json').write_text(json.dumps(execution, indent=2, ensure_ascii=False) + '\n')
    def pct(value): return f'{100 * value:.3f}'
    table = ['| 臂 | RUNMC Dice % | BMC Dice % |', '|---|---:|---:|',
             f"| 标准 U-Net 源/目标 | {pct(controls['source_RUNMC']['Dice'])} / {pct(controls['target_RUNMC']['Dice'])} | {pct(controls['source_BMC']['Dice'])} / {pct(controls['target_BMC']['Dice'])} |",
             f"| A1 源/目标 | {pct(cells['source_RUNMC']['Dice'])} / {pct(cells['target_RUNMC']['Dice'])} | {pct(cells['source_BMC']['Dice'])} / {pct(cells['target_BMC']['Dice'])} |"]
    gates_text = '\n'.join(f"| {k} | {'PASS' if v else 'FAILED'} |" for k, v in gates.items())
    patient_text = '\n'.join(f"| {r['index']} | {r['Dice_difference_pp']:+.3f} |"
                             for r in paired['target_BMC'])
    report = f"""# A1 完整分辨率层次模型基础能力关口

**冻结四门联合判定：{analysis['status']}。** 源 RUNMC 相对标准 U-Net {analysis['source_RUNMC_minus_U_pp']:+.3f} pp；目标 BMC 相对标准 U-Net {analysis['target_BMC_minus_U_pp']:+.3f} pp。A1 是新独立模型，不能追认旧 W32/R64 的失败门或严格 HBSI 资格。

## 同数据、同更新数对照

每格为同域 4 名开发患者等权 Dice。源训练 RUNMC 1600 次更新，随后目标 BMC 800 次更新；U-Net 封存结果只读复用。结构、目标、计算量不等价。

{chr(10).join(table)}

| 冻结继续门 | 判定 |
|---|---|
{gates_text}

目标 BMC 相对 U-Net 的匿名患者差值（pp）：

| 患者索引 | A1−U |
|---|---:|
{patient_text}

A1 目标阶段后 RUNMC 相对自身源模型平均 {analysis['old_domain_mean_change_pp']:+.3f} pp；最差单患者变化 {analysis['old_domain_worst_patient_change_pp']:+.3f} pp。旧域保持在本 A1 仅描述，不能解释为完整 CL 安全证据。目标 BMC 全训练患者 Dice {pct(cells['target_BMC_train']['Dice'])}%；空真值切片每片 FP 均值 {x['train_diagnostics']['empty_FP_per_slice']:.2f}（{x['train_diagnostics']['empty_slices']} 片），非空真值切片每片 FN 均值 {x['train_diagnostics']['positive_FN_per_slice']:.2f}（{x['train_diagnostics']['positive_slices']} 片）。目标 BMC precision/recall 为 {pct(cells['target_BMC']['precision'])}% / {pct(cells['target_BMC']['recall'])}%；详细匿名患者计数、NLL、Brier、HD95 见 [分析](ANALYSIS.json)。

首次导入因部署快照漏现有工具模块而失败，未进入预检或优化；保留原失败记录并在同一根、同一截止时间内补齐，正式结果来自第二次启动。预检2次真实反向、0次 Adam，之后完成2400次正式更新；首次启动至结束 {execution['elapsed_seconds']:.1f} 秒，根占用 {execution['new_root_bytes']} 字节。失败和修复不重置时钟或科研预算。[冻结协议](PREREG_ZH.md)、[执行摘要](EXECUTION_SUMMARY.json)。

本轮单训练种子、每域4名反复使用的开发患者，4次局部 MC 不是独立训练种子；没有 U 收益、长序列 CL、独立测试、统计非劣、临床安全或不确定性质量结论。{'四门全过只允许另立 A2，需冻结外观重组模型及生成合格性后再运行。' if all(gates.values()) else '四门未全过，本 A1 固定配置关闭，不按开发患者结果回选权重、阈值、种子或中间检查点。'}
"""
    (out / 'REPORT_ZH.md').write_text(report)
    print(analysis['status'], {k: round(v, 3) for k, v in [('source_minus_U_pp', analysis['source_RUNMC_minus_U_pp']),
                                                         ('target_minus_U_pp', analysis['target_BMC_minus_U_pp'])]})


if __name__ == '__main__':
    main(sys.argv[1])
