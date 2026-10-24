"""Summarize only complete, deidentified BMC C2 endpoints."""
import json
import sys
from pathlib import Path


def main(snapshot):
    x = json.loads(Path(snapshot).read_text())
    assert x['status'] == 'COMPLETE' and x['exit']['exit_code'] == 0
    assert x['preflight']['status'] == 'PASS'
    assert x['run']['formal_updates'] == x['ledger']['updates_reserved'] == 3200
    assert all(a['status'] == 'PASS' for a in x['ledger']['attempts'])
    assert set(x['patients']) == {'Cut_BMC', 'Joint_BMC', 'Cut_RUNMC', 'Joint_RUNMC',
                                  'Cut_BMC_train', 'Joint_BMC_train'}
    out = Path(__file__).parent
    cells = x['run']['cells']
    paired = []
    for i in range(4):
        cut = x['patients']['Cut_BMC'][i]['classes'][0]
        joint = x['patients']['Joint_BMC'][i]['classes'][0]
        paired.append(dict(index=i + 1, Cut_Dice=cut['Dice'], Joint_Dice=joint['Dice'],
                           Joint_minus_Cut_pp=100 * (joint['Dice'] - cut['Dice']),
                           Cut_TP=cut['TP'], Cut_FP=cut['FP'], Cut_FN=cut['FN'],
                           Joint_TP=joint['TP'], Joint_FP=joint['FP'], Joint_FN=joint['FN'],
                           Cut_HD95_penalized_mm=cut['HD95_penalized_mm'],
                           Joint_HD95_penalized_mm=joint['HD95_penalized_mm']))
    cold_paired = []
    for i in range(4):
        cut = x['patients']['Cut_RUNMC'][i]['classes'][0]
        joint = x['patients']['Joint_RUNMC'][i]['classes'][0]
        cold_paired.append(dict(index=i + 1, Cut_Dice=cut['Dice'], Joint_Dice=joint['Dice'],
                                Joint_minus_Cut_pp=100 * (joint['Dice'] - cut['Dice'])))
    diff = 100 * (cells['Joint_BMC']['Dice'] - cells['Cut_BMC']['Dice'])
    gates = dict(
        BMC_mean_gain_ge_1pp=diff >= 1,
        BMC_at_least_3_of_4_nonnegative=sum(p['Joint_minus_Cut_pp'] >= 0 for p in paired) >= 3,
        BMC_each_decline_le_5pp=min(p['Joint_minus_Cut_pp'] for p in paired) >= -5,
        BMC_HD95_within_10pct_of_Cut=(cells['Joint_BMC']['HD95_penalized_mm'] <=
                                       1.1 * cells['Cut_BMC']['HD95_penalized_mm']),
        no_full_patient_collapse=(all(cells[k]['empty_predictions'] == 0 for k in
                                      ('Cut_BMC', 'Joint_BMC', 'Cut_BMC_train', 'Joint_BMC_train'))
                                  and all(p['classes'][0]['TP'] > 0 for arm in ('Cut', 'Joint')
                                          for p in x['patients'][arm + '_BMC'] +
                                          x['patients'][arm + '_BMC_train'])),
    )
    status = 'EXPLORATORY_POSITIVE' if all(gates.values()) else 'NO_ROBUST_POSITIVE_SIGNAL'
    analysis = dict(status=status, paired_BMC=paired, paired_cold_RUNMC=cold_paired, gates=gates,
                    mean_effect_BMC_pp=diff,
                    cold_RUNMC_effect_pp=100 * (cells['Joint_RUNMC']['Dice'] - cells['Cut_RUNMC']['Dice']),
                    cells=cells,
                    patient_metrics=x['patients'],
                    C1_RUNMC_source_gate='FAILED_PREVIOUSLY_AND_UNCHANGED')
    execution = dict(first_actual_start=x['launch']['started'], deadline=x['launch']['deadline'],
                     elapsed_seconds=x['exit']['time'] - x['launch']['started'],
                     new_root_bytes=x['bytes'], formal_updates=3200,
                     preflight_updates=0,
                     attempts=[dict(name=a['name'], status=a['status'], seconds_actual=a.get('seconds_actual'))
                               for a in x['ledger']['attempts']])
    (out / 'ANALYSIS.json').write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + '\n')
    (out / 'EXECUTION_SUMMARY.json').write_text(json.dumps(execution, indent=2, ensure_ascii=False) + '\n')
    def pct(v):
        return f'{100 * v:.3f}'
    table = '\n'.join(f"| {arm} | {pct(cells[arm + '_BMC']['Dice'])} | "
                      f"{pct(cells[arm + '_RUNMC']['Dice'])} | "
                      f"{pct(cells[arm + '_BMC_train']['Dice'])} | "
                      f"{pct(cells[arm + '_BMC_MC4']['Dice'])} |"
                      for arm in ('Cut', 'Joint'))
    rows = '\n'.join(f"| {p['index']} | {pct(p['Cut_Dice'])} | {pct(p['Joint_Dice'])} | "
                     f"{p['Joint_minus_Cut_pp']:+.3f} |" for p in paired)
    cold_rows = '\n'.join(f"| {p['index']} | {pct(p['Cut_Dice'])} | {pct(p['Joint_Dice'])} | "
                          f"{p['Joint_minus_Cut_pp']:+.3f} |" for p in cold_paired)
    gate_rows = '\n'.join(f"| {k} | {'PASS' if v else 'FAILED'} |" for k, v in gates.items())
    report = f"""# C2：BMC 小样本跨中心复核

**冻结判定：{status}。** BMC 4 名标注患者、134 张切片从同一初值各训练1600步；以下 Dice 为各域4名开发患者等权三维结果，单位%。Cut 主干是本轮从零训练的标准 U-Net。RUNMC 列为未适应迁移，MC4 为次要读出。

| 臂 | BMC 域内 | RUNMC 冷迁移 | BMC 全训练患者 | BMC MC4 |
|---|---:|---:|---:|---:|
{table}

Joint−Cut BMC均值 **{diff:+.3f}pp**；RUNMC冷迁移均值 **{analysis['cold_RUNMC_effect_pp']:+.3f}pp**。域内每名患者：

| 索引 | Cut Dice % | Joint Dice % | Joint−Cut pp |
|---|---:|---:|---:|
{rows}

| 预定积极信号门 | 判定 |
|---|---|
{gate_rows}

RUNMC冷迁移的均值优势由患者差异共同构成，特别是第4名患者；它不是BMC域内主门的替代结果。RUNMC逐患者：

| 索引 | Cut Dice % | Joint Dice % | Joint−Cut pp |
|---|---:|---:|---:|
{cold_rows}

RUNMC惩罚HD95均值Cut {cells['Cut_RUNMC']['HD95_penalized_mm']:.3f}mm、Joint {cells['Joint_RUNMC']['HD95_penalized_mm']:.3f}mm；Brier分别为{cells['Cut_RUNMC']['Brier']:.4f}和{cells['Joint_RUNMC']['Brier']:.4f}。冷迁移Dice提高伴随这些指标变差，不能称为全面跨域改善。

完整匿名患者计数、NLL/Brier、惩罚HD95及次要MC4见[分析](ANALYSIS.json)，资源收据见[执行摘要](EXECUTION_SUMMARY.json)。C1原RUNMC源门失败和未做目标训练的结论保持不变；本轮只检验另一中心的单域从零训练，BMC开发患者已经被历史路线多次观察，不是外部独立验证。BMC和RUNMC的训练切片数也不同，不能把域内Dice差异单独归因于中心。单种子4名开发患者及层次反馈整项对比，不足以识别某个潜变量的因果贡献或声称普遍优势。隐藏U标签和test未访问；原图、标注、预测和模型均留NAS。[冻结协议](PREREG_ZH.md)
"""
    (out / 'REPORT_ZH.md').write_text(report)
    print(status, f'Cut_BMC={pct(cells["Cut_BMC"]["Dice"])}',
          f'Joint_BMC={pct(cells["Joint_BMC"]["Dice"])}', f'diff_pp={diff:+.3f}')


if __name__ == '__main__':
    main(sys.argv[1])
