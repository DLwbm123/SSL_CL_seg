"""Prespecified anonymous J0 latent audit and bounded decisions."""
import json
import sys
from pathlib import Path


def main(path):
    x=json.loads(Path(path).read_text())
    assert x['status']=='COMPLETE' and x['exit']['exit_code']==0
    assert x['preflight']['status']=='PASS'
    assert x['run']['production_updates']==0 and x['run']['source_L_slices']==80
    assert x['ledger']['updates_reserved']==0
    assert all(a['status']=='PASS' for a in x['ledger']['attempts'])
    c=x['patients']['C1'];h=x['patients']['H1_Shape']
    assert len(c)==len(h)==4
    assert [v['index'] for v in c]==[v['index'] for v in h]==[1,2,3,4]
    assert [v['slices'] for v in c]==[v['slices'] for v in h]
    shape_specific=[v['abs_shape_label_pearson']-v['abs_appearance_label_pearson']>=.05
                    for v in c]
    conditional=[v['mean_log_p_S_given_Y_minus_Z_per_pixel']>=.01 for v in c]
    tension=[a['abs_shape_label_pearson']-b['abs_shape_label_pearson']>=.05 and
             a['mean_reconstruction_mse']-b['mean_reconstruction_mse']>=.01
             for a,b in zip(h,c)]
    decisions=dict(C1_shape_specific_training_signal=sum(shape_specific)>=3,
                   C1_label_condition_candidate=sum(conditional)>=3,
                   H1_semantic_reconstruction_tension_signal=sum(tension)>=3,
                   counts=dict(shape_specific=sum(shape_specific),conditional=sum(conditional),
                               tension=sum(tension)),patients=4)
    analysis=dict(status='DIAGNOSTIC_COMPLETE_NO_OPTIMIZATION',decisions=decisions,
                  patients=x['patients'],source_L_slices=80,source_L_patients=4,
                  BMC_and_U_labels_not_accessed=True,prior_hard_stops_unchanged=True)
    out=Path(__file__).resolve().parent
    (out/'ANALYSIS.json').write_text(json.dumps(analysis,indent=2)+'\n')
    execution=dict(first_actual_start=x['launch']['started'],deadline=x['launch']['deadline'],
                   elapsed_seconds=x['exit']['time']-x['launch']['started'],
                   new_root_bytes=x['bytes'],new_formal_updates=0,
                   attempts=[{k:a[k] for k in ('name','status','seconds_actual')}
                             for a in x['ledger']['attempts']])
    (out/'EXECUTION_SUMMARY.json').write_text(json.dumps(execution,indent=2)+'\n')
    rows='\n'.join(f"| {arm} | {v['index']} | {v['slices']} | {v['foreground_fraction']:.4f} | "
                   f"{v['shape_label_pearson']:+.3f} | {v['appearance_label_pearson']:+.3f} | "
                   f"{v['shape_appearance_pearson']:+.3f} | {v['shape_mean_std']:.3f} | "
                   f"{v['appearance_mean_std']:.3f} | {v['mean_reconstruction_mse']:.5f} | "
                   f"{v['mean_log_p_S_given_Y_minus_Z_per_pixel']:+.4f} |"
                   for arm,items in (('C1',c),('H1_Shape',h)) for v in items)
    report=f"""# J0：BayeSeg形状与外观潜变量零更新审计

**完成，正式参数更新0次。** 仅用RUNMC四名L训练患者80张切片，比较封存C1 Cut与H1 Shape终态；未读取BMC开发、U隐藏标签或test。下表相关系数由每名患者所有128×128训练像素汇总，不把像素当独立患者。形状条件差为同一抽样形状下`log p(S|真值一热标签)−log p(S|抽样Z)`每像素平均，真值条件是训练分布外替代输入，不是一个已训练条件模型。

| 状态 | 匿名患者 | 切片 | 前景比例 | 形状-标签Pearson | 外观-标签Pearson | 形状-外观Pearson | 形状均值SD | 外观均值SD | 均值重建MSE | 真值−Z条件对数密度/像素 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{rows}

预定描述规则：

| 规则 | 达标患者 | 判定 |
|---|---:|---|
| C1形状绝对标签相关比外观高至少0.05 | {sum(shape_specific)}/4 | {'达标' if decisions['C1_shape_specific_training_signal'] else '未达标'} |
| C1真值条件对数密度每像素比抽样Z高至少0.01 | {sum(conditional)}/4 | {'达标' if decisions['C1_label_condition_candidate'] else '未达标'} |
| H1形状绝对标签相关提高≥0.05且重建MSE恶化≥0.01 | {sum(tension)}/4 | {'达标' if decisions['H1_semantic_reconstruction_tension_signal'] else '未达标'} |

以上规则需要至少3/4名患者同时满足才作为下一机制候选线索；不宣称统计显著、临床有效或形状/外观解耦。qS和qA后验容量不同，像素相关不等于部署分割性能；条件替换还可能存在分布外效应。旧C/F/G/H/I实验原判保留。[冻结协议](PREREG_ZH.md)、[匿名分析](ANALYSIS.json)、[执行摘要](EXECUTION_SUMMARY.json)。
"""
    (out/'REPORT_ZH.md').write_text(report)
    print('DIAGNOSTIC_COMPLETE',decisions)


if __name__=='__main__':
    main(sys.argv[1])
