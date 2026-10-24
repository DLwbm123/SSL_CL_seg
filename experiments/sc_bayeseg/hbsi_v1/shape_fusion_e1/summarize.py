"""Anonymous, frozen-gate analysis of the E1 shape-posterior fusion experiment."""
import json
import sys
from pathlib import Path


def main(snapshot):
    x = json.loads(Path(snapshot).read_text())
    assert x['status'] == 'COMPLETE' and x['exit']['exit_code'] == 0
    assert x['preflight']['status'] == 'PASS'
    assert x['run']['formal_updates'] == x['ledger']['updates_reserved'] == 3200
    assert all(a['status'] == 'PASS' for a in x['ledger']['attempts'])
    cells = x['cells']
    assert set(cells) == {f'{arm}_{domain}' for arm in ('U', 'Raw', 'Shape')
                          for domain in ('RUNMC', 'BMC', 'RUNMC_train')}
    patients = x['patients']
    assert set(patients) == {'RUNMC', 'BMC', 'RUNMC_train'}

    def difference(phase, arm):
        return 100 * (cells[f'Shape_{phase}']['Dice'] - cells[f'{arm}_{phase}']['Dice'])

    def deltas(phase, arm):
        return [100 * (p['Shape']['classes'][0]['Dice'] - p[arm]['classes'][0]['Dice'])
                for p in patients[phase]]

    source_gates = dict(
        RUNMC_mean_vs_Raw_ge_minus_1pp=difference('RUNMC', 'Raw') >= -1,
        RUNMC_mean_vs_U_ge_minus_1pp=difference('RUNMC', 'U') >= -1,
        RUNMC_each_vs_Raw_ge_minus_5pp=min(deltas('RUNMC', 'Raw')) >= -5,
        RUNMC_each_vs_U_ge_minus_5pp=min(deltas('RUNMC', 'U')) >= -5,
        RUNMC_train_mean_vs_Raw_ge_minus_5pp=difference('RUNMC_train', 'Raw') >= -5,
        RUNMC_train_mean_vs_U_ge_minus_5pp=difference('RUNMC_train', 'U') >= -5,
        RUNMC_HD95_vs_Raw_le_1_10=(cells['Shape_RUNMC']['HD95_penalized_mm'] <=
                                    1.1 * cells['Raw_RUNMC']['HD95_penalized_mm']),
        RUNMC_HD95_vs_U_le_1_10=(cells['Shape_RUNMC']['HD95_penalized_mm'] <=
                                  1.1 * cells['U_RUNMC']['HD95_penalized_mm']),
        no_collapse_source=(all(cells[f'{arm}_{phase}']['empty_predictions'] == 0
                                for arm in ('U', 'Raw', 'Shape') for phase in ('RUNMC', 'RUNMC_train'))
                            and all(p[arm]['classes'][0]['TP'] > 0
                                    for phase in ('RUNMC', 'RUNMC_train') for p in patients[phase]
                                    for arm in ('U', 'Raw', 'Shape'))),
    )
    bmc_gates = dict(
        BMC_mean_vs_both_ge_plus_2pp=all(difference('BMC', arm) >= 2 for arm in ('U', 'Raw')),
        BMC_3_of_4_positive_vs_both=all(sum(v >= 0 for v in deltas('BMC', arm)) >= 3
                                       for arm in ('U', 'Raw')),
        BMC_each_vs_both_ge_minus_5pp=all(min(deltas('BMC', arm)) >= -5
                                           for arm in ('U', 'Raw')),
        BMC_HD95_vs_both_le_1_10=all(cells['Shape_BMC']['HD95_penalized_mm'] <=
                                     1.1 * cells[f'{arm}_BMC']['HD95_penalized_mm']
                                     for arm in ('U', 'Raw')),
        no_collapse_BMC=(all(cells[f'{arm}_BMC']['empty_predictions'] == 0
                             for arm in ('U', 'Raw', 'Shape'))
                         and all(p[arm]['classes'][0]['TP'] > 0 for p in patients['BMC']
                                 for arm in ('U', 'Raw', 'Shape'))),
    )
    status = ('FEASIBILITY_SIGNAL' if all(source_gates.values()) and all(bmc_gates.values()) else
              'SOURCE_SAFETY_FAILED' if not all(source_gates.values()) else 'NO_COLD_BMC_SIGNAL')
    out = Path(__file__).parent
    analysis = dict(status=status, source_gates=source_gates, BMC_signal_gates=bmc_gates,
                    cells=cells, patients=patients,
                    Shape_minus_Raw_pp={p: difference(p, 'Raw') for p in patients},
                    Shape_minus_U_pp={p: difference(p, 'U') for p in patients},
                    C1_C2_hard_stops_unchanged=True)
    execution = dict(first_actual_start=x['launch']['started'], deadline=x['launch']['deadline'],
                     elapsed_seconds=x['exit']['time'] - x['launch']['started'],
                     new_root_bytes=x['bytes'], formal_head_updates=3200,
                     reused_C1_source_updates=1600, production_preflight_updates=0,
                     attempts=[dict(name=a['name'], status=a['status'], seconds_actual=a.get('seconds_actual'))
                               for a in x['ledger']['attempts']])
    (out / 'ANALYSIS.json').write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + '\n')
    (out / 'EXECUTION_SUMMARY.json').write_text(json.dumps(execution, indent=2, ensure_ascii=False) + '\n')

    def pct(v):
        return f'{100 * v:.3f}'

    table = '\n'.join(f"| {arm} | {pct(cells[arm + '_RUNMC']['Dice'])} | "
                      f"{pct(cells[arm + '_BMC']['Dice'])} | "
                      f"{pct(cells[arm + '_RUNMC_train']['Dice'])} | "
                      f"{cells[arm + '_RUNMC']['HD95_penalized_mm']:.3f} | "
                      f"{cells[arm + '_BMC']['HD95_penalized_mm']:.3f} |"
                      for arm in ('U', 'Raw', 'Shape'))
    patient_tables = {}
    for phase in ('RUNMC', 'BMC'):
        patient_tables[phase] = '\n'.join(
            f"| {p['index']} | {pct(p['U']['classes'][0]['Dice'])} | "
            f"{pct(p['Raw']['classes'][0]['Dice'])} | {pct(p['Shape']['classes'][0]['Dice'])} | "
            f"{100 * (p['Shape']['classes'][0]['Dice'] - p['Raw']['classes'][0]['Dice']):+.3f} | "
            f"{100 * (p['Shape']['classes'][0]['Dice'] - p['U']['classes'][0]['Dice']):+.3f} |"
            for p in patients[phase])
    gates = lambda d: '\n'.join(f"| {k} | {'PASS' if v else 'FAILED'} |" for k, v in d.items())
    report = f"""# E1：保留形状/外观生成层级的形状后验融合

**冻结判定：{status}。** 本轮只给封存C1 Cut的标准U-Net/完整生成层级增加小型残差头；原模型全部冻结。Raw与Shape各有1600次新增头更新，同数据、同初始化与容量。U为只读旧绝对参考，未获得同等新增更新；所有Dice为各域4名患者等权三维指标（%）。

| 臂 | RUNMC开发Dice | 冷BMC Dice | RUNMC完整L训练Dice | RUNMC惩罚HD95 mm | BMC惩罚HD95 mm |
|---|---:|---:|---:|---:|---:|
{table}

形状臂相对原图臂RUNMC开发Dice {difference('RUNMC', 'Raw'):+.3f}pp、冷BMC {difference('BMC', 'Raw'):+.3f}pp；相对旧U分别{difference('RUNMC', 'U'):+.3f}/{difference('BMC', 'U'):+.3f}pp。源安全门：

| 门 | 结果 |
|---|---|
{gates(source_gates)}

冷BMC积极信号门（需源安全门全过才有后继意义）：

| 门 | 结果 |
|---|---|
{gates(bmc_gates)}

RUNMC匿名患者：

| 索引 | U Dice | Raw Dice | Shape Dice | Shape−Raw pp | Shape−U pp |
|---|---:|---:|---:|---:|---:|
{patient_tables['RUNMC']}

BMC匿名患者：

| 索引 | U Dice | Raw Dice | Shape Dice | Shape−Raw pp | Shape−U pp |
|---|---:|---:|---:|---:|---:|
{patient_tables['BMC']}

[匿名分析](ANALYSIS.json)保留逐患者TP/FP/FN、惩罚HD95、NLL/Brier和完整L训练指标；[执行摘要](EXECUTION_SUMMARY.json)保留实际时间、字节和全部阶段。q(S|I)、q(A|I)、图像重建及形状—分割生成关系完整来自C1 Cut，但本轮主要预测仅使用形状后验**均值**，不证明采样或不确定性有益，也不证明潜变量已解耦。两臂使用C1历史预训练成本，不能把本轮新增头更新当作总成本。重复使用的4名开发患者只支持资源门，不是独立患者证据；隐藏U标签和test未访问。C1/C2原硬停及负结果不被本轮修改。[冻结协议](PREREG_ZH.md)
"""
    (out / 'REPORT_ZH.md').write_text(report)
    print(status, f"RUNMC Shape={pct(cells['Shape_RUNMC']['Dice'])}",
          f"BMC Shape={pct(cells['Shape_BMC']['Dice'])}")


if __name__ == '__main__':
    main(sys.argv[1])
