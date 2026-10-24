"""Anonymous full-endpoint B0 factorial and preregistered gate."""
import json
import sys
from pathlib import Path


LOCATIONS = ('source_RUNMC', 'source_BMC', 'target_RUNMC', 'target_BMC')
ARMS = ('U_Std', 'U_Bal', 'H_Bal', 'H_Std')


def main(snapshot):
    x = json.loads(Path(snapshot).read_text())
    assert x['complete'] and x['exit']['exit_code'] == 0
    assert x['run']['formal_updates'] == x['ledger']['updates_reserved'] == 4800
    assert x['preflight']['status'] == 'PASS' and x['preflight']['optimizer_updates'] == 0
    out = Path(__file__).parent
    cells = {arm: {} for arm in ARMS}
    for loc in LOCATIONS:
        cells['U_Std'][loc] = x['U_Std'][loc]['repeats'][0]
        cells['H_Bal'][loc] = x['H_Bal'][loc]
        for arm in ('U_Bal', 'H_Std'):
            cells[arm][loc] = x['run']['cells'][arm + '_' + loc]
    effects = {}
    for loc in LOCATIONS:
        h = 100 * (cells['H_Std'][loc]['Dice'] - cells['H_Bal'][loc]['Dice'])
        u = 100 * (cells['U_Std'][loc]['Dice'] - cells['U_Bal'][loc]['Dice'])
        effects[loc] = dict(H_Std_minus_H_Bal_pp=h, U_Std_minus_U_Bal_pp=u,
                            family_by_risk_interaction_pp=h - u)
    pairs = {}
    for loc in LOCATIONS:
        assert len(x['patients'][loc]) == 4
        pairs[loc] = []
        for row in x['patients'][loc]:
            assert row['index'] in (1, 2, 3, 4)
            masks = {arm: row[arm]['classes'][0] for arm in ARMS}
            pairs[loc].append(dict(index=row['index'],
                                   Dice={arm: masks[arm]['Dice'] for arm in ARMS},
                                   H_Std_minus_U_Std_pp=100 * (masks['H_Std']['Dice'] - masks['U_Std']['Dice']),
                                   H_Std_minus_H_Bal_pp=100 * (masks['H_Std']['Dice'] - masks['H_Bal']['Dice']),
                                   U_Std_minus_U_Bal_pp=100 * (masks['U_Std']['Dice'] - masks['U_Bal']['Dice']),
                                   H_Std_TP=masks['H_Std']['TP'], H_Std_FP=masks['H_Std']['FP'],
                                   H_Std_FN=masks['H_Std']['FN'],
                                   H_Std_precision=masks['H_Std']['TP'] / masks['H_Std']['predicted'] if masks['H_Std']['predicted'] else 0,
                                   H_Std_recall=masks['H_Std']['TP'] / masks['H_Std']['GT'] if masks['H_Std']['GT'] else 0,
                                   H_Std_volume_ratio=masks['H_Std']['predicted'] / masks['H_Std']['GT'] if masks['H_Std']['GT'] else None,
                                   H_Std_HD95_penalized_mm=masks['H_Std']['HD95_penalized_mm']))
    old_change = []
    for old, new in zip(pairs['source_RUNMC'], pairs['target_RUNMC']):
        assert old['index'] == new['index']
        old_change.append(dict(index=old['index'], change_pp=100 * (new['Dice']['H_Std'] - old['Dice']['H_Std'])))
    train = x['train']['H_Std']
    train_masks = [p['classes'][0] for p in train]
    gates = dict(
        source_RUNMC_within_1pp_of_U=cells['H_Std']['source_RUNMC']['Dice'] >= cells['U_Std']['source_RUNMC']['Dice'] - .01,
        target_BMC_within_2pp_of_U=cells['H_Std']['target_BMC']['Dice'] >= cells['U_Std']['target_BMC']['Dice'] - .02,
        target_RUNMC_within_2pp_of_U=cells['H_Std']['target_RUNMC']['Dice'] >= cells['U_Std']['target_RUNMC']['Dice'] - .02,
        old_mean_change_ge_minus_0_5pp=cells['H_Std']['target_RUNMC']['Dice'] >= cells['H_Std']['source_RUNMC']['Dice'] - .005,
        old_each_change_ge_minus_5pp=min(v['change_pp'] for v in old_change) >= -5,
        BMC_at_least_3_of_4_within_5pp_of_U=sum(v['H_Std_minus_U_Std_pp'] >= -5 for v in pairs['target_BMC']) >= 3,
        BMC_each_within_10pp_of_U=min(v['H_Std_minus_U_Std_pp'] for v in pairs['target_BMC']) >= -10,
        no_empty_foreground_collapse=(sum(v['TP'] for v in train_masks) > 0
                                      and x['run']['cells']['H_Std_target_BMC_train']['empty_predictions'] == 0
                                      and all(cells['H_Std'][loc]['empty_predictions'] == 0 for loc in LOCATIONS)))
    analysis = dict(status='PASS' if all(gates.values()) else 'FAILED', gates=gates,
                    cells=cells, risk_effects=effects, patient_pairs=pairs,
                    H_Std_old_change=old_change, train_cells={arm: x['run']['cells'][arm + '_target_BMC_train']
                                                               for arm in ('U_Bal', 'H_Std')},
                    train_strata=x['train_strata'],
                    U_initial_matching='procedural_only_historical_initial_state_not_saved',
                    H_initial_matching='exact_initial_state_and_function')
    execution = dict(first_actual_start=x['launch']['started'], original_deadline=x['launch']['deadline'],
                     elapsed_seconds=x['exit']['time'] - x['launch']['started'], new_root_bytes=x['bytes'],
                     preflight_reverse_calls=x['preflight']['real_backward_calls'],
                     preflight_updates=0, formal_updates=4800,
                     attempt_statuses=[dict(name=a['name'], status=a['status'],
                                            seconds_actual=a.get('seconds_actual'))
                                       for a in x['ledger']['attempts']])
    (out / 'ANALYSIS.json').write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + '\n')
    (out / 'EXECUTION_SUMMARY.json').write_text(json.dumps(execution, indent=2, ensure_ascii=False) + '\n')
    def pct(v): return f'{100 * v:.3f}'
    table = ['| 臂 | 源RUNMC | 源BMC | 目标RUNMC | 目标BMC |',
             '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        table.append('| ' + arm + ' | ' + ' | '.join(pct(cells[arm][loc]['Dice']) for loc in LOCATIONS) + ' |')
    gate_table = '\n'.join(f"| {k} | {'PASS' if v else 'FAILED'} |" for k, v in gates.items())
    effect_table = '\n'.join(f"| {loc} | {effects[loc]['H_Std_minus_H_Bal_pp']:+.3f} | {effects[loc]['U_Std_minus_U_Bal_pp']:+.3f} | {effects[loc]['family_by_risk_interaction_pp']:+.3f} |" for loc in LOCATIONS)
    patient_table = '\n'.join(f"| {row['index']} | {row['H_Std_minus_U_Std_pp']:+.3f} | {row['H_Std_minus_H_Bal_pp']:+.3f} | {row['H_Std_precision']*100:.2f} | {row['H_Std_recall']*100:.2f} | {row['H_Std_volume_ratio']:.3f} |" for row in pairs['target_BMC'])
    report = f"""# B0 层次模型 × 监督风险缺角实验

**H-Std 新基座联合继续门：{analysis['status']}。** 本报告在两个新增臂各完成RUNMC1600＋BMC800后封存四格；旧 U-Std 与 H-Bal 结果只读复用。源/目标每格为4名开发患者等权三维Dice（%）。

{chr(10).join(table)}

| 冻结继续门 | 判定 |
|---|---|
{gate_table}

## 风险配置的家族内作用

正值表示将监督风险从Bal改为Std的Dice提高；最后一列是H和U两家族风险效应之差，不能解释为单个 Bayesian 元件的因果贡献。

| 位置 | H-Std−H-Bal pp | U-Std−U-Bal pp | 差中之差 pp |
|---|---:|---:|---:|
{effect_table}

H-Std目标BMC匿名患者结果：

| 索引 | H-Std−U-Std Dice pp | H-Std−H-Bal Dice pp | precision % | recall % | 体积比 |
|---|---:|---:|---:|---:|---:|
{patient_table}

## 判读

H-Std目标BMC均值60.407%，比H-Bal高3.045pp、比U-Std低1.700pp，均值距离门槛已经足够近；但源RUNMC比U-Std低4.696pp，目标RUNMC低5.085pp。目标BMC只有2/4名患者达到相对U-Std的−5pp界限，最差患者低11.253pp，因此联合继续门只通过4/8项。H-Std自身旧域源→目标4名患者均提高，说明本轮失败不是这一定义下的旧域遗忘；它也未证明对旧域的独立泛化。

两家族换成标准风险后的目标BMC均值都提高（H +3.045pp，U +10.512pp），但同一监督风险在两个家族的效应不同；U-Bal目标RUNMC仅22.944%，提示平衡风险本身可造成严重跨域损失。四格不能分离架构、层次变分项和推断策略的单独贡献，也不能把目标BMC均值接近U-Std误作患者尾部达标。下一步遵守冻结失败分支：关闭该B0配置，另立q(Z|S,I)结构与同结构无变分项对照的独立协议。

H-Std目标BMC全训练患者Dice {pct(analysis['train_cells']['H_Std']['Dice'])}%；空真值切片每片FP {x['train_strata']['H_Std']['empty_FP_per_slice']:.2f}（{x['train_strata']['H_Std']['empty_slices']}片），非空真值切片每片FN {x['train_strata']['H_Std']['positive_FN_per_slice']:.2f}（{x['train_strata']['H_Std']['positive_slices']}片）。U-Bal训练读出和两臂详细NLL/Brier/HD95、匿名逐患者计数见[分析](ANALYSIS.json)。

H-Std与旧H-Bal初始张量、初始后验输出及变分项在同随机流下逐项一致；U-Bal和旧U-Std使用相同源码及确定性构造程序，但历史U-Std没有保存初始状态文件，无法回溯逐项核对。该差别限制U家族的精确初值配对声明。BAL与STD同时改变CE权重及空片Dice约定，不能把效应全部归给CE。固定患者反复用于开发，单训练种子和MC抽样不构成独立泛化；本轮L-only不能证明半监督或持续学习。结果与实际资源记录见[执行摘要](EXECUTION_SUMMARY.json)，冻结定义见[协议](PREREG_ZH.md)。

{'H-Std过门仅允许另立同推断结构的无变分损失实验；仍需证明层次机制对后续U或CL的独立贡献。' if all(gates.values()) else 'H-Std未过门；本B0配置关闭，需先独立冻结下一结构假设与预算，不按开发结果回选权重、阈值、种子或中途检查点。'}
"""
    (out / 'REPORT_ZH.md').write_text(report)
    print(analysis['status'], {k: round(cells['H_Std'][k]['Dice'] * 100, 3) for k in LOCATIONS})


if __name__ == '__main__':
    main(sys.argv[1])
