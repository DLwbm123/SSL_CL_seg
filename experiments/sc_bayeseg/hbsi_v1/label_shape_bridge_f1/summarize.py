"""Anonymous, prespecified F1 source and cold-domain gates."""
import json
import math
import sys
from pathlib import Path


def main(snapshot):
    x = json.loads(Path(snapshot).read_text())
    assert x['status'] == 'COMPLETE' and x['exit']['exit_code'] == 0
    assert x['preflight']['status'] == 'PASS'
    assert x['run']['formal_updates'] == x['ledger']['updates_reserved'] == 3200
    assert all(a['status'] == 'PASS' for a in x['ledger']['attempts'])
    for arm in ('Cut', 'Anchor'):
        assert [row['step'] for row in x['gradient_audits'][arm]] == [1, 400, 800, 1200, 1600]
        assert all(math.isfinite(v) for part in x['training'][arm].values()
                   for v in part.values() if isinstance(v, (int, float)))
    cells, patients = x['cells'], x['patients']
    assert set(patients) == {'RUNMC', 'BMC', 'RUNMC_train'}

    def diff(phase, arm):
        return 100 * (cells[f'Anchor_{phase}']['Dice'] - cells[f'{arm}_{phase}']['Dice'])

    def patient_diff(phase, arm):
        return [100 * (p['Anchor']['classes'][0]['Dice'] - p[arm]['classes'][0]['Dice'])
                for p in patients[phase]]

    source = dict(
        RUNMC_mean_vs_Cut_ge_minus_1pp=diff('RUNMC', 'Cut') >= -1,
        RUNMC_mean_vs_U_ge_minus_1pp=diff('RUNMC', 'U') >= -1,
        RUNMC_each_vs_Cut_ge_minus_5pp=min(patient_diff('RUNMC', 'Cut')) >= -5,
        RUNMC_each_vs_U_ge_minus_5pp=min(patient_diff('RUNMC', 'U')) >= -5,
        RUNMC_train_mean_vs_Cut_ge_minus_5pp=diff('RUNMC_train', 'Cut') >= -5,
        RUNMC_train_mean_vs_U_ge_minus_5pp=diff('RUNMC_train', 'U') >= -5,
        RUNMC_HD95_vs_Cut_le_1_10=(cells['Anchor_RUNMC']['HD95_penalized_mm'] <=
                                    1.1 * cells['Cut_RUNMC']['HD95_penalized_mm']),
        RUNMC_HD95_vs_U_le_1_10=(cells['Anchor_RUNMC']['HD95_penalized_mm'] <=
                                  1.1 * cells['U_RUNMC']['HD95_penalized_mm']),
        no_source_collapse=(all(cells[f'{arm}_{phase}']['empty_predictions'] == 0
                                for arm in ('Cut', 'Anchor') for phase in ('RUNMC', 'RUNMC_train'))
                            and all(p[arm]['classes'][0]['TP'] > 0
                                    for phase in ('RUNMC', 'RUNMC_train') for p in patients[phase]
                                    for arm in ('Cut', 'Anchor'))),
        full_generator_finite=all(math.isfinite(v) for arm in ('Cut', 'Anchor')
                                  for part in x['training'][arm].values()
                                  for v in part.values() if isinstance(v, (int, float))),
    )
    cold = dict(
        BMC_mean_vs_all_ge_plus_2pp=all(diff('BMC', arm) >= 2
                                        for arm in ('Cut', 'U', 'E1_Shape')),
        BMC_3_of_4_nonnegative_vs_Cut_U=all(sum(v >= 0 for v in patient_diff('BMC', arm)) >= 3
                                           for arm in ('Cut', 'U')),
        BMC_each_vs_Cut_U_ge_minus_5pp=all(min(patient_diff('BMC', arm)) >= -5
                                            for arm in ('Cut', 'U')),
        BMC_HD95_vs_Cut_U_le_1_10=all(cells['Anchor_BMC']['HD95_penalized_mm'] <=
                                      1.1 * cells[f'{arm}_BMC']['HD95_penalized_mm']
                                      for arm in ('Cut', 'U')),
        no_BMC_collapse=(all(cells[f'{arm}_BMC']['empty_predictions'] == 0
                             for arm in ('Cut', 'Anchor'))
                         and all(p[arm]['classes'][0]['TP'] > 0 for p in patients['BMC']
                                 for arm in ('Cut', 'Anchor'))),
    )
    status = ('FEASIBILITY_SIGNAL' if all(source.values()) and all(cold.values()) else
              'SOURCE_SAFETY_FAILED' if not all(source.values()) else 'NO_COLD_BMC_SIGNAL')
    out = Path(__file__).parent
    analysis = dict(status=status, source_gates=source, BMC_signal_gates=cold,
                    cells=cells, patients=patients, training=x['training'],
                    gradient_audits=x['gradient_audits'],
                    Anchor_minus_Cut_pp={phase: diff(phase, 'Cut') for phase in patients},
                    Anchor_minus_U_pp={phase: diff(phase, 'U') for phase in patients},
                    C1_C2_E1_hard_stops_unchanged=True)
    execution = dict(first_actual_start=x['launch']['started'], deadline=x['launch']['deadline'],
                     elapsed_seconds=x['exit']['time'] - x['launch']['started'],
                     new_root_bytes=x['bytes'], new_formal_updates=3200,
                     reused_C1_source_updates=1600,
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
                      for arm in ('U', 'E1_Shape', 'Cut', 'Anchor'))
    pair = {}
    for phase in ('RUNMC', 'BMC'):
        pair[phase] = '\n'.join(f"| {p['index']} | {pct(p['U']['classes'][0]['Dice'])} | "
                                f"{pct(p['Cut']['classes'][0]['Dice'])} | "
                                f"{pct(p['Anchor']['classes'][0]['Dice'])} | "
                                f"{100 * (p['Anchor']['classes'][0]['Dice'] - p['Cut']['classes'][0]['Dice']):+.3f} | "
                                f"{100 * (p['Anchor']['classes'][0]['Dice'] - p['U']['classes'][0]['Dice']):+.3f} |"
                                for p in patients[phase])
    gates = lambda d: '\n'.join(f"| {k} | {'PASS' if v else 'FAILED'} |" for k, v in d.items())
    audit = '\n'.join(f"| {step} | " + ' | '.join(
        f"{next(v for v in x['gradient_audits'][arm] if v['step'] == step)['task_grad_norm']:.5f} / "
        f"{next(v for v in x['gradient_audits'][arm] if v['step'] == step)['weighted_variational_grad_norm']:.5f}"
        for arm in ('Cut', 'Anchor')) + ' |'
        for step in (1, 400, 800, 1200, 1600))
    report = f"""# F1：标签锚定 BayeSeg 形状后验

**冻结判定：{status}。** 两新臂从相同C1 Cut完整形状/外观生成层级和E1零残差头初态出发，U-Net保持v45 U1600逐项不变。两臂均新增1600次生成+头更新；唯一干预是标签梯度是否进入形状后验。U、E1 Shape为只读历史参考，累计成本不同。Dice是每域4名患者等权三维结果（%）。

| 臂 | RUNMC开发Dice | 冷BMC Dice | RUNMC完整L训练Dice | RUNMC惩罚HD95 mm | BMC惩罚HD95 mm |
|---|---:|---:|---:|---:|---:|
{table}

Anchor−Cut RUNMC开发Dice {diff('RUNMC', 'Cut'):+.3f}pp、冷BMC {diff('BMC', 'Cut'):+.3f}pp；相对只读U分别{diff('RUNMC', 'U'):+.3f}/{diff('BMC', 'U'):+.3f}pp。

源安全门：

| 门 | 判定 |
|---|---|
{gates(source)}

冷BMC积极信号门（需源门全过才有后继意义）：

| 门 | 判定 |
|---|---|
{gates(cold)}

RUNMC匿名患者：

| 索引 | U Dice | Cut Dice | Anchor Dice | Anchor−Cut pp | Anchor−U pp |
|---|---:|---:|---:|---:|---:|
{pair['RUNMC']}

BMC匿名患者：

| 索引 | U Dice | Cut Dice | Anchor Dice | Anchor−Cut pp | Anchor−U pp |
|---|---:|---:|---:|---:|---:|
{pair['BMC']}

预注册当时状态的形状后验梯度审计（每格为标签范数 / 实际0.01系数后生成范数；余弦与完整日志见匿名分析）：

| 更新前步号 | Cut | Anchor |
|---|---:|---:|
{audit}

最后100个训练批的生成项损失Cut为{x['training']['Cut']['last100']['variational']:.3f}、Anchor为{x['training']['Anchor']['last100']['variational']:.3f}（越小越好）；样本图像重建MSE分别为{x['training']['Cut']['last100']['reconstruction_mse']:.5f}和{x['training']['Anchor']['last100']['reconstruction_mse']:.5f}。Anchor的标签分割小幅改善伴随这些训练生成读数变差。400、800、1200、1600步形状后验两目标梯度余弦符号并不恒定，单点冲突或吻合不能解释整段轨迹，也不能把此相关性单独认定为因果主因。

[匿名分析](ANALYSIS.json)保留逐患者TP/FP/FN、NLL/Brier、训练首尾100步生成及重建读数和审计；[执行摘要](EXECUTION_SUMMARY.json)保留运行成本。用户要求的形状/外观生成层级继续接受生成训练，但本轮只检验标签进入形状后验的**整条策略效应**，不能证明每个潜变量已解耦或后验采样有益。C1/C2/E1负结果保持原判；同一4名开发患者多轮使用，积极信号也不能替代独立患者与多种子确认。隐藏U标签/test未访问。[冻结协议](PREREG_ZH.md)
"""
    (out / 'REPORT_ZH.md').write_text(report)
    print(status, f"Anchor_RUNMC={pct(cells['Anchor_RUNMC']['Dice'])}",
          f"Anchor_BMC={pct(cells['Anchor_BMC']['Dice'])}")


if __name__ == '__main__':
    main(sys.argv[1])
