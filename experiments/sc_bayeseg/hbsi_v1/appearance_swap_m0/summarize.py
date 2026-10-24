"""Apply the frozen zero-update appearance-swap admission gates."""
import json
import math
import sys
from pathlib import Path


def main(path):
    x=json.loads(Path(path).read_text())
    assert x['status']=='COMPLETE' and x['exit']['exit_code']==0
    assert x['preflight']['status']=='PASS'
    assert x['run']['formal_updates']==0 and not x['run']['target_data_read']
    assert x['ledger']['updates_reserved']==0
    assert all(a['status']=='PASS' for a in x['ledger']['attempts'])
    arms=x['run']['arms'];patients=x['patients']
    assert all(len(patients[a])==4 and arms[a]['slices']==80 for a in ('Cut','Joint'))
    gates={}
    for arm in ('Cut','Joint'):
        a=arms[arm];rows=patients[arm]
        def diff(key):
            return [100*(v['metrics'][key]['classes'][0]['Dice']-
                         v['metrics']['original']['classes'][0]['Dice']) for v in rows]
        own,swapped=diff('self'),diff('swap')
        gates[arm]={
            'self_reconstruction_mse_le_0_02':math.isfinite(a['reconstruction_mse'])
                and a['reconstruction_mse']<=.02,
            'swap_outside_fraction_le_0_01':math.isfinite(a['swap_outside_fraction'])
                and a['swap_outside_fraction']<=.01,
            'self_mean_Dice_drop_le_5pp':100*(a['self_Dice']-a['original_Dice'])>=-5,
            'self_each_Dice_drop_le_10pp':min(own)>=-10,
            'swap_mean_Dice_drop_le_5pp':100*(a['swap_Dice']-a['original_Dice'])>=-5,
            'swap_each_Dice_drop_le_10pp':min(swapped)>=-10,
            'swap_no_patient_empty':all(not v['metrics']['swap']['classes'][0]
                                         ['empty_prediction_failure'] for v in rows),
        }
    verdict='CUT_ADMISSION_PASS' if all(gates['Cut'].values()) else 'CUT_ADMISSION_FAILED'
    analysis=dict(status=verdict,gates=gates,arms=arms,patients=patients,
                  source_training_patients=4,source_training_slices=80,formal_updates=0,
                  target_data_read=False,synthetic_labels_not_proven=True)
    out=Path(__file__).resolve().parent
    (out/'ANALYSIS.json').write_text(json.dumps(analysis,indent=2)+'\n')
    execution=dict(first_actual_start=x['launch']['started'],deadline=x['launch']['deadline'],
                   elapsed_seconds=x['exit']['time']-x['launch']['started'],
                   new_root_bytes=x['bytes'],formal_updates=0,
                   attempts=[{k:a[k] for k in ('name','status','seconds_actual')}
                             for a in x['ledger']['attempts']])
    (out/'EXECUTION_SUMMARY.json').write_text(json.dumps(execution,indent=2)+'\n')
    table='\n'.join(f"| {arm} | {100*a['original_Dice']:.3f} | {100*a['self_Dice']:.3f} | "
                    f"{100*a['swap_Dice']:.3f} | {a['reconstruction_mse']:.5f} | "
                    f"{100*a['swap_outside_fraction']:.3f} |"
                    for arm,a in arms.items())
    checks='\n'.join(f"| {arm} | {name} | {'PASS' if passed else 'FAILED'} |"
                     for arm in ('Cut','Joint') for name,passed in gates[arm].items())
    rows='\n'.join(f"| {arm} | {v['index']} | " + ' | '.join(
        f"{100*v['metrics'][key]['classes'][0]['Dice']:.3f}"
        for key in ('original','self','swap'))+' |'
        for arm in ('Cut','Joint') for v in patients[arm])
    report=f"""# M0：源域形状/外观交换零更新审计

**冻结判定：{verdict}。** 用L1 Cut和Joint封存终点，对RUNMC四名L训练患者80片做确定性自身重建与跨患者外观交换；伙伴是循环下一名患者，按轴向相对位置配片。没有优化器更新，也未读取BMC数据。用冻结标准U预测原图、重建图及交换图，Dice为四患者等权三维结果（%）。

| L1模型 | 原图U Dice | 自身重建U Dice | 外观交换U Dice | 重建MSE | 交换图越界像素% |
|---|---:|---:|---:|---:|---:|
{table}

预注册合成准入门（仅Cut全通过才考虑另立合成训练协议）：

| 模型 | 门 | 判定 |
|---|---|---|
{checks}

匿名逐患者Dice：

| L1模型 | 索引 | 原图 | 自身重建 | 外观交换 |
|---|---:|---:|---:|---:|
{rows}

Cut自身重建相对原图U为{100*(arms['Cut']['self_Dice']-arms['Cut']['original_Dice']):+.3f}pp，交换后为{100*(arms['Cut']['swap_Dice']-arms['Cut']['original_Dice']):+.3f}pp。重建误差和越界比例都很低，但交换后4名患者的Dice均大幅下降；这一组合与外观场携带空间结构或模型不能组合跨患者潜变量均相容，不能仅凭此审计确定原因。交换后的分割读数仅检验冻结U是否仍能识别源标签；它不是合成图真实标签正确性的直接证明，更不说明外观潜变量已与形状解耦。Cut失败，本轮关闭直接外观交换合成路线，不对伙伴、裁剪、权重或checkpoint作结果后调整。身份、图像、标签、合成图和原始预测留NAS。[冻结协议](PREREG_ZH.md)、[匿名分析](ANALYSIS.json)、[执行摘要](EXECUTION_SUMMARY.json)。
"""
    (out/'REPORT_ZH.md').write_text(report)
    print(verdict,'Cut swap Dice',100*arms['Cut']['swap_Dice'])


if __name__=='__main__':main(sys.argv[1])
