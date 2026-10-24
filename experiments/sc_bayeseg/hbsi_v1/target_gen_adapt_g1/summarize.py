"""Prespecified G1 gates from a remotely anonymized final snapshot."""
import json
import sys
from pathlib import Path


def main(path):
    x = json.loads(Path(path).read_text())
    assert x['status'] == 'COMPLETE' and x['exit']['exit_code'] == 0
    assert x['preflight']['status'] == 'PASS'
    assert x['run']['formal_updates'] == 1600 and x['run']['target_U_unlabeled_only']
    assert x['ledger']['updates_reserved'] == 1600
    assert all(a['status'] == 'PASS' for a in x['ledger']['attempts'])
    assert len(x['patients']['RUNMC']) == len(x['patients']['BMC']) == 4
    assert all(x['A_only_equals_F1_Anchor'].values())

    c = x['cells']
    p = x['patients']
    diff = lambda phase, a, b: [100 * (v[a]['classes'][0]['Dice'] - v[b]['classes'][0]['Dice'])
                                for v in p[phase]]
    mdiff = lambda phase, a, b: 100 * (c[a+'_'+phase]['Dice']-c[b+'_'+phase]['Dice'])
    src_vs_a = diff('RUNMC', 'S_plus_A', 'A_only')
    target_vs_a = diff('BMC', 'S_plus_A', 'A_only')
    target_vs_u = diff('BMC', 'S_plus_A', 'U')
    finite = all(all(__import__('math').isfinite(v[k]) for k in
                     ('variational', 'shape_logsd', 'appearance_logsd', 'reconstruction_mse'))
                 for arm in ('A_only', 'S_plus_A')
                 for v in x['training'][arm].values())
    source = {
        'RUNMC_mean_vs_A_only_ge_minus_1pp': mdiff('RUNMC','S_plus_A','A_only') >= -1,
        'RUNMC_each_vs_A_only_ge_minus_5pp': min(src_vs_a) >= -5,
        'RUNMC_train_mean_vs_A_only_ge_minus_5pp': mdiff('RUNMC_train','S_plus_A','A_only') >= -5,
        'RUNMC_HD95_vs_A_only_le_1_10': c['S_plus_A_RUNMC']['HD95_penalized_mm'] <=
                                     1.1*c['A_only_RUNMC']['HD95_penalized_mm'],
        'no_source_collapse': c['S_plus_A_RUNMC']['empty_predictions'] == 0 and
                              c['S_plus_A_RUNMC_train']['empty_predictions'] == 0,
        'full_generator_finite': finite,
    }
    target = {
        'BMC_mean_vs_all_ge_plus_2pp': c['S_plus_A_BMC']['Dice'] >=
                                       max(c[arm+'_BMC']['Dice'] for arm in ('A_only','Anchor','U'))+.02,
        'BMC_3_of_4_nonnegative_vs_A_and_U': sum(a >= 0 and u >= 0
                                                 for a,u in zip(target_vs_a,target_vs_u)) >= 3,
        'BMC_each_vs_A_and_U_ge_minus_5pp': min(target_vs_a+target_vs_u) >= -5,
        'BMC_HD95_vs_A_and_U_le_1_10': all(c['S_plus_A_BMC']['HD95_penalized_mm'] <=
                                          1.1*c[arm+'_BMC']['HD95_penalized_mm']
                                          for arm in ('A_only','U')),
        'no_BMC_collapse': c['S_plus_A_BMC']['empty_predictions'] == 0,
    }
    verdict = 'POSITIVE_RESOURCE_SIGNAL' if all(source.values()) and all(target.values()) else \
              'SOURCE_SAFETY_FAILED' if not all(source.values()) else 'NO_BMC_GAIN'
    analysis = dict(status=verdict, source_gates=source, BMC_signal_gates=target,
                    cells=c, patients=p, training=x['training'],
                    A_only_equals_F1_Anchor=x['A_only_equals_F1_Anchor'],
                    S_plus_A_minus_A_only_pp={phase: mdiff(phase,'S_plus_A','A_only')
                                              for phase in ('RUNMC','BMC','RUNMC_train')},
                    S_plus_A_minus_U_BMC_pp=mdiff('BMC','S_plus_A','U'),
                    F1_original_failure_unchanged=True)
    out = Path(__file__).resolve().parent
    (out/'ANALYSIS.json').write_text(json.dumps(analysis,indent=2)+'\n')
    ledger=x['ledger']
    execution=dict(first_actual_start=x['launch']['started'], deadline=x['launch']['deadline'],
                   elapsed_seconds=x['exit']['time']-x['launch']['started'],
                   new_root_bytes=x['bytes'], new_formal_updates=1600,
                   reused_F1_anchor_source_updates=1600,
                   attempts=[{k:a[k] for k in ('name','status','seconds_actual')} for a in ledger['attempts']])
    (out/'EXECUTION_SUMMARY.json').write_text(json.dumps(execution,indent=2)+'\n')
    arms=('U','Anchor','A_only','S_plus_A')
    table='\n'.join(f"| {arm} | {100*c[arm+'_RUNMC']['Dice']:.3f} | "
                    f"{100*c[arm+'_BMC']['Dice']:.3f} | "
                    f"{100*c[arm+'_RUNMC_train']['Dice']:.3f} | "
                    f"{c[arm+'_RUNMC']['HD95_penalized_mm']:.3f} | "
                    f"{c[arm+'_BMC']['HD95_penalized_mm']:.3f} |" for arm in arms)
    gates=lambda obj:'\n'.join(f"| {name} | {'PASS' if ok else 'FAILED'} |"
                               for name,ok in obj.items())
    rows='\n'.join(f"| {phase} | {i+1} | {100*v['U']['classes'][0]['Dice']:.3f} | "
                   f"{100*v['A_only']['classes'][0]['Dice']:.3f} | "
                   f"{100*v['S_plus_A']['classes'][0]['Dice']:.3f} | "
                   f"{100*(v['S_plus_A']['classes'][0]['Dice']-v['A_only']['classes'][0]['Dice']):+.3f} | "
                   f"{100*(v['S_plus_A']['classes'][0]['Dice']-v['U']['classes'][0]['Dice']):+.3f} |"
                   for phase in ('RUNMC','BMC') for i,v in enumerate(p[phase]))
    t=x['training']
    report=f"""# G1：BMC无标签生成适应

**冻结判定：{verdict}。** 两臂从相同F1 Anchor完整形状/外观层级出发，每臂仅用BMC U无标签图像做800次生成目标更新；U-Net与分割头冻结。A_only只更新外观和其他生成参数，S_plus_A额外更新形状后验。Dice为每域4名患者等权三维结果（%），旧U/F1 Anchor只读，累计成本不同。

| 臂 | RUNMC开发Dice | BMC冷迁移Dice | RUNMC完整L训练Dice | RUNMC惩罚HD95 mm | BMC惩罚HD95 mm |
|---|---:|---:|---:|---:|---:|
{table}

S_plus_A−A_only RUNMC/BMC分别{mdiff('RUNMC','S_plus_A','A_only'):+.3f}/{mdiff('BMC','S_plus_A','A_only'):+.3f}pp；BMC相对旧U为{mdiff('BMC','S_plus_A','U'):+.3f}pp。A_only的逐患者预测与封存F1 Anchor相同：{all(x['A_only_equals_F1_Anchor'].values())}。

源安全门：

| 门 | 判定 |
|---|---|
{gates(source)}

冷BMC积极门（须源门全过才有后继意义）：

| 门 | 判定 |
|---|---|
{gates(target)}

匿名逐患者：

| 域 | 索引 | U Dice | A_only Dice | S_plus_A Dice | S_plus_A−A_only pp | S_plus_A−U pp |
|---|---:|---:|---:|---:|---:|---:|
{rows}

BMC U训练的生成项最后100批：A_only {t['A_only']['last100']['variational']:.3f}、S_plus_A {t['S_plus_A']['last100']['variational']:.3f}；重建MSE分别{t['A_only']['last100']['reconstruction_mse']:.5f}/{t['S_plus_A']['last100']['reconstruction_mse']:.5f}。生成拟合读数不等于分割改善，也不能凭本轮证明形状/外观潜变量已解耦。F1原硬停不变；同一4名开发患者多轮使用，任何正信号都需要独立患者和多种子确认。BMC U隐藏标签/test未访问。[冻结协议](PREREG_ZH.md)、[匿名分析](ANALYSIS.json)、[执行摘要](EXECUTION_SUMMARY.json)。
"""
    (out/'REPORT_ZH.md').write_text(report)
    print(verdict, f"RUNMC={100*c['S_plus_A_RUNMC']['Dice']:.3f}",
          f"BMC={100*c['S_plus_A_BMC']['Dice']:.3f}")


if __name__=='__main__':
    main(sys.argv[1])
