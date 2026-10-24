"""Anonymous B1 full endpoint readout against frozen B0/U-Std controls."""
import json
import sys
from pathlib import Path


LOCATIONS = ('source_RUNMC', 'source_BMC', 'target_RUNMC', 'target_BMC')
ARMS = ('U_Std', 'H_Std', 'H_IS', 'H_IS_NoVar')


def main(snapshot):
    x = json.loads(Path(snapshot).read_text())
    assert x['complete'] and x['exit']['exit_code'] == 0
    assert x['run']['formal_updates'] == x['ledger']['updates_reserved'] == 4800
    assert x['preflight']['status'] == 'PASS' and x['preflight']['optimizer_updates'] == 0
    assert all(a['status'] == 'PASS' for a in x['ledger']['attempts'])
    out = Path(__file__).parent
    cells = {arm: {} for arm in ARMS}
    for loc in LOCATIONS:
        cells['U_Std'][loc] = x['U_Std'][loc]['repeats'][0]
        cells['H_Std'][loc] = x['H_Std'][loc]
        for arm in ('H_IS', 'H_IS_NoVar'):
            cells[arm][loc] = x['run']['cells'][arm + '_' + loc]
    pairs = {}
    for loc in LOCATIONS:
        assert len(x['patients'][loc]) == 4
        pairs[loc] = []
        for row in x['patients'][loc]:
            assert row['index'] in (1, 2, 3, 4)
            masks = {arm: row[arm]['classes'][0] for arm in ARMS}
            pairs[loc].append(dict(index=row['index'],
                                   Dice={arm: masks[arm]['Dice'] for arm in ARMS},
                                   H_IS_minus_U_Std_pp=100*(masks['H_IS']['Dice']-masks['U_Std']['Dice']),
                                   H_IS_minus_H_Std_pp=100*(masks['H_IS']['Dice']-masks['H_Std']['Dice']),
                                   H_IS_minus_NoVar_pp=100*(masks['H_IS']['Dice']-masks['H_IS_NoVar']['Dice']),
                                   H_IS_TP=masks['H_IS']['TP'], H_IS_FP=masks['H_IS']['FP'],
                                   H_IS_FN=masks['H_IS']['FN'],
                                   H_IS_HD95_penalized_mm=masks['H_IS']['HD95_penalized_mm']))
    old_change = []
    for old, new in zip(pairs['source_RUNMC'], pairs['target_RUNMC']):
        assert old['index'] == new['index']
        old_change.append(dict(index=old['index'], change_pp=100*(new['Dice']['H_IS']-old['Dice']['H_IS'])))
    train = x['train']['H_IS']
    gates = dict(
        source_RUNMC_within_1pp_of_U=cells['H_IS']['source_RUNMC']['Dice'] >= cells['U_Std']['source_RUNMC']['Dice']-.01,
        target_BMC_within_2pp_of_U=cells['H_IS']['target_BMC']['Dice'] >= cells['U_Std']['target_BMC']['Dice']-.02,
        target_RUNMC_within_2pp_of_U=cells['H_IS']['target_RUNMC']['Dice'] >= cells['U_Std']['target_RUNMC']['Dice']-.02,
        old_mean_change_ge_minus_0_5pp=cells['H_IS']['target_RUNMC']['Dice'] >= cells['H_IS']['source_RUNMC']['Dice']-.005,
        old_each_change_ge_minus_5pp=min(p['change_pp'] for p in old_change) >= -5,
        BMC_at_least_3_of_4_within_5pp_of_U=sum(p['H_IS_minus_U_Std_pp'] >= -5 for p in pairs['target_BMC']) >= 3,
        BMC_each_within_10pp_of_U=min(p['H_IS_minus_U_Std_pp'] for p in pairs['target_BMC']) >= -10,
        no_empty_foreground_collapse=(sum(p['classes'][0]['TP'] for p in train) > 0
                                      and x['run']['cells']['H_IS_target_BMC_train']['empty_predictions'] == 0
                                      and all(cells['H_IS'][loc]['empty_predictions'] == 0 for loc in LOCATIONS)))
    analysis = dict(status='PASS' if all(gates.values()) else 'FAILED', gates=gates, cells=cells,
                    patient_pairs=pairs, H_IS_old_change=old_change,
                    train_cells={arm:x['run']['cells'][arm+'_target_BMC_train'] for arm in ('H_IS','H_IS_NoVar')},
                    train_strata=x['train_strata'],
                    initial_function='H_IS_equals_H_Std_and_H_IS_NoVar_with_fp64_tolerance')
    execution = dict(first_actual_start=x['launch']['started'], original_deadline=x['launch']['deadline'],
                     elapsed_seconds=x['exit']['time']-x['launch']['started'], new_root_bytes=x['bytes'],
                     preflight_reverse_calls=x['preflight']['real_backward_calls'],
                     preflight_updates=0, formal_updates=4800,
                     attempt_statuses=[dict(name=a['name'],status=a['status'],seconds_actual=a.get('seconds_actual'))
                                       for a in x['ledger']['attempts']])
    (out/'ANALYSIS.json').write_text(json.dumps(analysis,indent=2,ensure_ascii=False)+'\n')
    (out/'EXECUTION_SUMMARY.json').write_text(json.dumps(execution,indent=2,ensure_ascii=False)+'\n')
    def pct(v): return f'{100*v:.3f}'
    table=['| 臂 | 源RUNMC | 源BMC | 目标RUNMC | 目标BMC |','|---|---:|---:|---:|---:|']
    for arm in ARMS:
        table.append('| '+arm+' | '+' | '.join(pct(cells[arm][loc]['Dice']) for loc in LOCATIONS)+' |')
    gate_table='\n'.join(f"| {k} | {'PASS' if v else 'FAILED'} |" for k,v in gates.items())
    delta_table='\n'.join(f"| {loc} | {100*(cells['H_IS'][loc]['Dice']-cells['H_Std'][loc]['Dice']):+.3f} | {100*(cells['H_IS'][loc]['Dice']-cells['H_IS_NoVar'][loc]['Dice']):+.3f} | {100*(cells['H_IS'][loc]['Dice']-cells['U_Std'][loc]['Dice']):+.3f} |" for loc in LOCATIONS)
    patient_table='\n'.join(f"| {p['index']} | {p['H_IS_minus_U_Std_pp']:+.3f} | {p['H_IS_minus_H_Std_pp']:+.3f} | {p['H_IS_minus_NoVar_pp']:+.3f} |" for p in pairs['target_BMC'])
    report=f"""# B1 图像条件后验与同结构目标对照

**H-IS 新基座联合继续门：{analysis['status']}（{sum(gates.values())}/8通过）。** 两新增臂各完成源RUNMC1600＋目标BMC800更新。旧U-Std和H-Std终点只读复用；四格均为4名开发患者等权三维Dice（%）。

{chr(10).join(table)}

| 冻结继续门 | 判定 |
|---|---|
{gate_table}

| 位置 | H-IS−H-Std pp | H-IS−NoVar pp | H-IS−U-Std pp |
|---|---:|---:|---:|
{delta_table}

目标BMC匿名患者差：

| 索引 | H-IS−U-Std pp | H-IS−H-Std pp | H-IS−NoVar pp |
|---|---:|---:|---:|
{patient_table}

H-IS目标BMC全训练患者Dice {pct(analysis['train_cells']['H_IS']['Dice'])}%；空真值切片每片FP {x['train_strata']['H_IS']['empty_FP_per_slice']:.2f}，非空真值切片每片FN {x['train_strata']['H_IS']['positive_FN_per_slice']:.2f}。[匿名完整分析](ANALYSIS.json)含患者TP/FP/FN、HD95及四格NLL/Brier；[执行摘要](EXECUTION_SUMMARY.json)含时间、字节与每阶段状态。

初始权重及函数与H-Std匹配，新图像通道初始为零；NoVar拥有相同网络、初始概率与采样，只去掉变分目标。任何性能差都仅为本种子/当前四名开发患者的诊断，不足以证明层次目标在U或CL中的作用。固定患者反复用于开发；MC4不是独立训练种子，隐藏U标签和test未访问。[冻结协议](PREREG_ZH.md)定义全部门槛。

{'H-IS过门，锁定本候选；后续种子、U/CL与独立患者确认须另立协议。' if all(gates.values()) else 'H-IS未过门，本批关闭；不追加风险权重、旁路、阈值、长度或中途模型搜索。'}
"""
    (out/'REPORT_ZH.md').write_text(report)
    print(analysis['status'], {k:round(cells['H_IS'][k]['Dice']*100,3) for k in LOCATIONS})


if __name__=='__main__':
    main(sys.argv[1])
