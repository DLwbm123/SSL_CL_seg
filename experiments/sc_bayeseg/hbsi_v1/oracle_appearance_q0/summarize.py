"""Apply Q0's frozen source-only oracle appearance gates."""
import json
import math
import sys
from pathlib import Path


def main(path):
    x = json.loads(Path(path).read_text())
    assert x['status'] == 'COMPLETE' and x['exit']['exit_code'] == 0
    assert x['preflight']['status'] == 'PASS'
    r = x['run']
    assert r['formal_updates'] == 0 and r['target_data_read'] is False
    assert r['oracle_source_labels_used'] is True
    assert x['ledger']['updates_reserved'] == 0
    assert all(a['status'] == 'PASS' for a in x['ledger']['attempts'])
    patients = x['patients']
    assert len(patients) == 4 and sum(v['slices'] for v in patients) == 80

    def diff(name):
        return [100 * (v['metrics'][name]['classes'][0]['Dice'] -
                       v['metrics']['original']['classes'][0]['Dice'])
                for v in patients]

    class_diff = diff('class_only')
    delta_diff = diff('delta')
    gates = dict(
        class_only_mse_le_0_02=math.isfinite(r['class_only_mse'])
            and r['class_only_mse'] <= .02,
        class_only_mean_Dice_drop_le_5pp=min(class_diff) > -math.inf
            and 100 * (r['class_only_Dice'] - r['original_Dice']) >= -5,
        class_only_each_Dice_drop_le_10pp=min(class_diff) >= -10,
        class_only_no_patient_empty=all(not v['metrics']['class_only']['classes'][0]
                                        ['empty_prediction_failure'] for v in patients),
        delta_outside_fraction_le_0_01=math.isfinite(r['delta_outside_fraction'])
            and r['delta_outside_fraction'] <= .01,
        delta_mean_Dice_drop_le_5pp=100 * (r['delta_Dice'] - r['original_Dice']) >= -5,
        delta_each_Dice_drop_le_10pp=min(delta_diff) >= -10,
        delta_no_patient_empty=all(not v['metrics']['delta']['classes'][0]
                                   ['empty_prediction_failure'] for v in patients),
    )
    class_pass = all(v for k,v in gates.items() if k.startswith('class_only_'))
    delta_pass = all(v for k,v in gates.items() if k.startswith('delta_'))
    verdict = 'ORACLE_ADMISSION_PASS' if class_pass and delta_pass else 'ORACLE_ADMISSION_FAILED'
    out = Path(__file__).resolve().parent
    analysis = dict(status=verdict, gates=gates, class_only_pass=class_pass,
                    delta_pass=delta_pass, aggregate=r, patients=patients,
                    formal_updates=0, target_data_read=False,
                    oracle_source_labels_used=True,
                    learned_shape_or_target_transfer_unproven=True)
    (out / 'ANALYSIS.json').write_text(json.dumps(analysis, indent=2) + '\n')
    execution = dict(first_actual_start=x['launch']['started'],
        deadline=x['launch']['deadline'],
        elapsed_seconds=x['exit']['time']-x['launch']['started'],
        new_root_bytes=x['bytes'], formal_updates=0,
        attempts=[{k:a[k] for k in ('name','status','seconds_actual')}
                  for a in x['ledger']['attempts']])
    (out / 'EXECUTION_SUMMARY.json').write_text(json.dumps(execution, indent=2) + '\n')
    gate_rows='\n'.join(f"| {name} | {'PASS' if passed else 'FAILED'} |"
                        for name,passed in gates.items())
    patient_rows='\n'.join(f"| {i+1} | " + ' | '.join(
        f"{100*v['metrics'][key]['classes'][0]['Dice']:.3f}"
        for key in ('original','class_only','delta')) + ' |'
        for i,v in enumerate(patients))
    report=f"""# Q0：真值形状下的类条件外观审计

**冻结判定：{verdict}。** RUNMC 四名 L 训练患者 80 片，0 次正式更新；只用源训练标签固定解剖形状，不读取 BMC 或 test。`C` 为真值掩膜加本患者两类全局灰度均值；`D` 在原图空间残差上替换成下一名患者的两类均值。原图、C、D 均由相同冻结 U-Net 评分，Dice 为四患者等权三维结果（%）。

| 原图 U Dice | C 的 U Dice | D 的 U Dice | C 重建 MSE | D 越界像素% |
|---:|---:|---:|---:|---:|
| {100*r['original_Dice']:.3f} | {100*r['class_only_Dice']:.3f} | {100*r['delta_Dice']:.3f} | {r['class_only_mse']:.6f} | {100*r['delta_outside_fraction']:.4f} |

| 预注册门 | 判定 |
|---|---|
{gate_rows}

| 匿名源训练患者 | 原图 Dice | C Dice | D Dice |
|---:|---:|---:|---:|
{patient_rows}

仅类均值发射门 **{'PASS' if class_pass else 'FAILED'}**，保留原图残差的类均值交换门 **{'PASS' if delta_pass else 'FAILED'}**。C 相对原图 Dice {100*(r['class_only_Dice']-r['original_Dice']):+.3f} pp；D 为 {100*(r['delta_Dice']-r['original_Dice']):+.3f} pp。C 的大幅重建与分割损失说明两类全局灰度均值不足以表示这组 MRI；D 虽在均值上接近 5 pp 限制，最差个体仍下降 {-min(delta_diff):.3f} pp，且越界像素超过预设线。两门须同时通过才考虑另立对应生成模型训练协议；本轮不训练，也不调整平滑、混合比例、伙伴或阈值。D 保留的是**原患者**空间残差，因此即使 D 通过，也不证明完整外观潜变量能跨患者交换；真值标签提供了学得 qS 所没有的理想解剖条件。源训练患者分数不是独立验证或目标域证据。[冻结协议](PREREG_ZH.md)、[匿名分析](ANALYSIS.json)、[执行摘要](EXECUTION_SUMMARY.json)。
"""
    (out / 'REPORT_ZH.md').write_text(report)
    print(verdict, f"C={100*r['class_only_Dice']:.3f}",
          f"D={100*r['delta_Dice']:.3f}")


if __name__ == '__main__':
    main(sys.argv[1])
