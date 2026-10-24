"""Anonymous C source gate and conditional target result."""
import json
import sys
from pathlib import Path


def main(snapshot):
    x=json.loads(Path(snapshot).read_text());assert x['complete'] and x['exit']['exit_code']==0
    assert x['preflight']['status']=='PASS' and x['ledger']['updates_reserved']==x['run']['formal_updates']
    assert all(a['status']=='PASS' for a in x['ledger']['attempts'])
    out=Path(__file__).parent
    locations=['source_RUNMC','source_BMC']+([] if x['status']=='SOURCE_GATE_FAILED' else ['target_RUNMC','target_BMC'])
    cells={arm:{} for arm in ('U_Std','Cut','Joint')};mc={arm:{} for arm in ('Cut','Joint')}
    for loc in locations:
        cells['U_Std'][loc]=x['U_Std'][loc]['repeats'][0]
        for arm in ('Cut','Joint'):
            cells[arm][loc]=x['run']['cells'][arm+'_'+loc]
            mc[arm][loc]=x['run']['cells'][arm+'_'+loc+'_MC4']
    pairs={}
    for loc in locations:
        assert len(x['patients'][loc])==4
        pairs[loc]=[]
        for p in x['patients'][loc]:
            assert p['index'] in (1,2,3,4)
            masks={arm:p[arm]['classes'][0] for arm in ('U_Std','Cut','Joint')}
            pairs[loc].append(dict(index=p['index'],Dice={arm:masks[arm]['Dice'] for arm in masks},
                                   Joint_minus_Cut_pp=100*(masks['Joint']['Dice']-masks['Cut']['Dice']),
                                   Joint_minus_U_pp=100*(masks['Joint']['Dice']-masks['U_Std']['Dice']),
                                   Joint_TP=masks['Joint']['TP'],Joint_FP=masks['Joint']['FP'],Joint_FN=masks['Joint']['FN'],
                                   Joint_HD95_penalized_mm=masks['Joint']['HD95_penalized_mm']))
    source_gates=x['source_gate']['gates'];assert x['source_gate']['status']==('PASS' if all(source_gates.values()) else 'FAILED')
    target_gates=None;positive_signal=None;old_change=[]
    if x['status']=='COMPLETE':
        assert x['target_match']['status']=='PASS' and x['target_match']['max_tensor_error']<=1e-10
        for a,b in zip(pairs['source_RUNMC'],pairs['target_RUNMC']):
            assert a['index']==b['index']
            old_change.append(dict(index=a['index'],change_pp=100*(b['Dice']['Joint']-a['Dice']['Joint'])))
        target_gates=dict(
            Cut_exact_v45_target=True,
            BMC_Joint_ge_Cut_minus_2pp=cells['Joint']['target_BMC']['Dice']>=cells['Cut']['target_BMC']['Dice']-.02,
            RUNMC_Joint_ge_Cut_minus_2pp=cells['Joint']['target_RUNMC']['Dice']>=cells['Cut']['target_RUNMC']['Dice']-.02,
            BMC_Joint_absolute_ge_U_minus_2pp=cells['Joint']['target_BMC']['Dice']>=cells['U_Std']['target_BMC']['Dice']-.02,
            RUNMC_Joint_absolute_ge_U_minus_2pp=cells['Joint']['target_RUNMC']['Dice']>=cells['U_Std']['target_RUNMC']['Dice']-.02,
            old_mean_change_ge_minus_0_5pp=cells['Joint']['target_RUNMC']['Dice']>=cells['Joint']['source_RUNMC']['Dice']-.005,
            old_each_change_ge_minus_5pp=min(v['change_pp'] for v in old_change)>=-5,
            BMC_at_least_3_of_4_within_5pp_of_Cut=sum(v['Joint_minus_Cut_pp']>=-5 for v in pairs['target_BMC'])>=3,
            BMC_each_within_10pp_of_Cut=min(v['Joint_minus_Cut_pp'] for v in pairs['target_BMC'])>=-10,
            BMC_HD95_within_10pct_of_Cut=cells['Joint']['target_BMC']['HD95_penalized_mm']<=1.1*cells['Cut']['target_BMC']['HD95_penalized_mm'],
            RUNMC_HD95_within_10pct_of_Cut=cells['Joint']['target_RUNMC']['HD95_penalized_mm']<=1.1*cells['Cut']['target_RUNMC']['HD95_penalized_mm'],
            no_empty_collapse=(all(cells[a][loc]['empty_predictions']==0 for a in ('Cut','Joint') for loc in ('target_RUNMC','target_BMC'))
                               and all(x['run']['cells'][a+'_target_BMC_train']['empty_predictions']==0 for a in ('Cut','Joint'))
                               and all(sum(p[a]['classes'][0]['TP'] for p in x['train'][a+'_target'])>0 for a in ('Cut','Joint'))))
        positive_signal=(cells['Joint']['target_BMC']['Dice']-cells['Cut']['target_BMC']['Dice'])>=.01 and all(target_gates.values())
    status=('SOURCE_FAILED' if x['status']=='SOURCE_GATE_FAILED' else
            'QUALIFIED_POSITIVE' if all(target_gates.values()) and positive_signal else
            'QUALIFIED_NO_POSITIVE_SIGNAL' if all(target_gates.values()) else 'TARGET_FAILED')
    analysis=dict(status=status,source_gates=source_gates,target_gates=target_gates,
                  positive_BMC_signal=positive_signal,cells=cells,MC4_secondary=mc,
                  patient_pairs=pairs,Joint_old_change=old_change,
                  train_cells={k:x['run']['cells'][k.replace('_source','_source_RUNMC_train').replace('_target','_target_BMC_train')]
                               for k in x['train']},
                  training_patients={k:[dict(index=i+1,classes=v['classes']) for i,v in enumerate(rows)]
                                     for k,rows in x['train'].items()},
                  cut_v45_source_max_tensor_error=x['source_gate']['cut_v45_max_tensor_error'],
                  cut_v45_target_max_tensor_error=(x['target_match']['max_tensor_error'] if x['target_match'] else None))
    execution=dict(first_conservative_start=x['launch']['started'],root_created=x['launch']['root_created'],
                   original_deadline=x['launch']['deadline'],elapsed_seconds=x['exit']['time']-x['launch']['started'],
                   new_root_bytes=x['bytes'],formal_updates=x['run']['formal_updates'],
                   production_preflight_updates=0,
                   attempt_statuses=[dict(name=a['name'],status=a['status'],seconds_actual=a.get('seconds_actual'))
                                     for a in x['ledger']['attempts']])
    (out/'ANALYSIS.json').write_text(json.dumps(analysis,indent=2,ensure_ascii=False)+'\n')
    (out/'EXECUTION_SUMMARY.json').write_text(json.dumps(execution,indent=2,ensure_ascii=False)+'\n')
    def pct(v):return f'{100*v:.3f}'
    table=['| 臂 | 源RUNMC | 源BMC | 目标RUNMC | 目标BMC |','|---|---:|---:|---:|---:|']
    for arm in ('U_Std','Cut','Joint'):
        table.append('| '+arm+' | '+' | '.join(pct(cells[arm][loc]['Dice']) if loc in cells[arm] else '未执行'
                                              for loc in ('source_RUNMC','source_BMC','target_RUNMC','target_BMC'))+' |')
    source_table='\n'.join(f"| {k} | {'PASS' if v else 'FAILED'} |" for k,v in source_gates.items())
    target_table='\n'.join(f"| {k} | {'PASS' if v else 'FAILED'} |" for k,v in (target_gates or {}).items())
    patient_table='\n'.join(f"| {v['index']} | {v['Joint_minus_Cut_pp']:+.3f} | {v['Joint_minus_U_pp']:+.3f} |"
                            for v in pairs['target_BMC' if target_gates else 'source_RUNMC'])
    secondary='\n'.join(f"| {arm} | {loc} | {pct(mc[arm][loc]['Dice'])} |"
                        for arm in ('Cut','Joint') for loc in locations)
    source_effect=100*(cells['Joint']['source_RUNMC']['Dice']-cells['Cut']['source_RUNMC']['Dice'])
    cold_effect=100*(cells['Joint']['source_BMC']['Dice']-cells['Cut']['source_BMC']['Dice'])
    source_interpretation=(f"源RUNMC Joint−Cut {source_effect:+.3f}pp，低于预定−1pp界限"
                           f" {(-1-source_effect):.3f}pp；Cut与封存U-Std逐项相同，故表中相对Cut和相对旧U的两条均值失败是同一个事实，"
                           f"不能计作两个独立失败。四名患者差值均≥−5pp，最差{min(v['Joint_minus_Cut_pp'] for v in pairs['source_RUNMC']):+.3f}pp；"
                           f"未适应BMC Joint−Cut {cold_effect:+.3f}pp。源完整训练患者Dice Cut"
                           f"{pct(analysis['train_cells']['Cut_source']['Dice'])}%、Joint"
                           f"{pct(analysis['train_cells']['Joint_source']['Dice'])}%，差距已出现在训练患者，"
                           "不能只归咎于开发集泛化。固定MC4的主次读出接近，没有替代失败的均值门。"
                           if x['status']=='SOURCE_GATE_FAILED' else '')
    report=f"""# C1 判别均值路径 × 层次生成反馈

**冻结结论：{status}。** Cut/Joint的主要预测为标准双logit softmax，MC4是独立次要读出；所有患者指标为4名开发患者等权三维结果，百分数。

{chr(10).join(table)}

## 源域硬停门

| 门 | 判定 |
|---|---|
{source_table}

Cut与封存v45 U1600主分割参数最大绝对差 {analysis['cut_v45_source_max_tensor_error']:.3g}；生产零更新梯度准入PASS。[C0只读诊断](C0_DIAGNOSTIC_ZH.md)没有用于选择配置。

{source_interpretation}

{('## 目标域继续门'+chr(10)+chr(10)+'| 门 | 判定 |'+chr(10)+'|---|---|'+chr(10)+target_table+chr(10)+chr(10)+'Cut与封存v45 U_B800主分割参数最大绝对差 '+format(analysis['cut_v45_target_max_tensor_error'],'.3g')+'。目标BMC Joint−Cut达到预定+1pp积极信号：'+('是' if positive_signal else '否')+'。') if target_gates else '源门未过，协议规定的两条目标链均未启动。'}

{'目标BMC' if target_gates else '源RUNMC'}匿名患者差：

| 索引 | Joint−Cut Dice pp | Joint−旧U Dice pp |
|---|---:|---:|
{patient_table}

次要MC4后验预测Dice（不得替代上述主要门）：

| 臂 | 位置 | MC4 Dice % |
|---|---|---:|
{secondary}

[匿名分析](ANALYSIS.json)含逐患者TP/FP/FN/HD95、precision/recall、NLL/Brier及全训练患者读出；[执行摘要](EXECUTION_SUMMARY.json)列原时钟、正式更新、字节和全部阶段。Joint−Cut只识别整个生成反馈策略在当前训练种子与重复使用的四名开发患者上的效果，不能拆成单个潜变量的贡献；通过非劣门也不自动证明层次机制有益。隐藏U标签和test未访问，私有预测与模型未发布。[协议](PREREG_ZH.md)定义停止与后继边界。

补充的[源终点梯度审计](GRADIENT_AUDIT.json)在固定最后一批上0更新执行：Cut层次项到U-Net梯度仍为零，Joint非零；Joint监督梯度范数1.390、未乘系数的变分梯度范数0.848、余弦−0.035。单批接近正交不证明全程无冲突。训练日志逐步保留task/变分项/尺度，但运行时未保留预注册中间步数的梯度审计快照，故无法补做1/400/800/1200的原状态梯度分析；此解释性诊断缺口不改变预先计算的源门与停止动作，也不以重跑填补。
"""
    (out/'REPORT_ZH.md').write_text(report)
    print(status, {loc:round(cells['Joint'][loc]['Dice']*100,3) for loc in locations})


if __name__=='__main__':main(sys.argv[1])
