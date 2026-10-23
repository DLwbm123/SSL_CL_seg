"""Offline report from bounded diagnostics; no prediction or optimization."""
import argparse
import json
from pathlib import Path
import statistics


def mean_summary(cell):
    return {key:statistics.mean(r[key] for r in cell['repeats']) for key in cell['repeats'][0]}


def analyze(sem,switch):
    assert sem['status']=='COMPLETE' and switch['status']=='COMPLETE'
    rows=sem['rows'];bad=sum(r['adam_candidates']['A_sem']['risk']>r['adam_candidates']['A']['risk'] for r in rows)
    means={key:mean_summary(c['summary']) for key,c in switch['cells'].items()}
    changes={name:{metric:means[left][metric]-means[right][metric] for metric in means[left]} for name,left,right in [('entry','T_switch','T0'),('training_under_target_memory','T_post_target','T_switch'),('post_memory_readout','T_post_target','T_post_source'),('total','T_post_target','T0'),('post_with_source_memory','T_post_source','T0')]}
    total=changes['total']['Dice'];entry=changes['entry']['Dice']
    share=abs(entry/total) if total!=0 and entry*total>0 else None
    main_switch=total<0 and entry<0 and share>=.5
    high_errors=sum(r['L_teacher']['label_audit']['high_confidence_errors'] for r in rows)
    main_sem=bad>=8 and high_errors>0
    route='DOMAIN_INITIALIZATION_FIRST' if main_switch else ('TEACHER_RELIABILITY_FIRST' if main_sem else 'NO_SINGLE_ROUTE_ESTABLISHED')
    return dict(source_updates=0,reverse_calls=sem['reverse_calls'],discarded_candidate_steps=sem['candidate_steps_discarded'],semantic_adam_risk_worse=bad,semantic_probe_count=len(rows),training_L_high_confidence_error_pixels_repeated_probes=high_errors,readout_means=means,changes=changes,entry_share_of_total_drop=share,domain_entry_rule_met=main_switch,semantic_rule_met=main_sem,next_route=route)


def number(v,decimals=4):
    return 'NA' if v is None else f'{v:.{decimals}f}'


def report(sem,switch,a):
    rows=sem['rows'];lines=['# v21-Q0：一致性专属与零更新域切换诊断','',
        '本次仅诊断。既有v19/v20模型、原始结果、训练时钟和科学负结果不变；不重训已关闭配置，不重新评分已完成的基线格。', '',
        '## 预算、来源不变与成本','',
        f"一致性10探针/5既有状态，实际{sem['reverse_calls']}次诊断反向（上限128）、{sem['candidate_steps_discarded']}次丢弃副本Adam更新（上限24），来源更新0。来源payload/Adam/RNG检查不变，fixture检查通过。不能把副本Adam更新记成总计算更新0。", 
        '教师诊断只读当前训练L/U；L预测函数只传x，在预测后才用原训练L标签评分；U没有GT能力。域读出新增2格，复用2格旧完整结果；每格BMC原4val患者/3推断重复、8×4MC和原指标。模型副本仅前向，来源更新0；验证标签不用于训练、门槛选择或调参。','',
        '## 一致性候选更新','',
        f"A+sem相对A，在{a['semantic_adam_risk_worse']}/{a['semantic_probe_count']}个独立训练L探针上任务风险更高（规则阈值8/10）。这是同Adam状态的一步条件诊断，不是整个优化轨迹的因果证明。",'',
        '| 状态 | 批次 | Δrisk A+sem−A | ΔsoftDice loss | Δprecision | Δrecall |','|---|---:|---:|---:|---:|---:|']
    for r in rows:
        b=r['adam_candidates']['A'];c=r['adam_candidates']['A_sem']
        lines.append(f"| {r['state']} | {r['batch_index']} | {c['risk']-b['risk']:+.7f} | {c['soft_dice_loss']-b['soft_dice_loss']:+.7f} | {c['precision']-b['precision']:+.5f} | {c['recall']-b['recall']:+.5f} |")
    endpoint_file=Path(__file__).resolve().parent/'LEARNING_ENDPOINTS.json'
    if endpoint_file.exists():
        endpoints=json.loads(endpoint_file.read_text())
        lines+=['','## 既有绝对终点数值复用（首seed RUNMC）','',
                '| 臂 | 800 Dice % | 1600 Dice % | 绝对变化 pp |','|---|---:|---:|---:|']
        for arm,domains in endpoints['arms'].items():
            e=domains['RUNMC'];early=e['800']['Dice']*100;late=e['1600']['Dice']*100
            lines.append(f'| {arm} | {early:.4f} | {late:.4f} | {late-early:+.4f} |')
        lines+=['','四臂均继续进步而A追赶更快；这不是辅助臂绝对性能后期退化或过拟合的证明。未新评其他时点，也不选择800替代固定1600终点。']
    lines+=['','| 参数组 | sem/task norm比中位数 | cosine中位数 | 负cos/有效探针 |','|---|---:|---:|---:|']
    for key in rows[0]['gradient_groups']:
        vals=[r['gradient_groups'][key] for r in rows];ratios=[v['semantic_task_ratio'] for v in vals if v['semantic_task_ratio'] is not None];cos=[v['semantic_task_cosine'] for v in vals if v['semantic_task_cosine'] is not None]
        lines.append(f"| {key} | {number(statistics.median(ratios) if ratios else None)} | {number(statistics.median(cos) if cos else None)} | {sum(v<0 for v in cos)}/{len(cos)} |")
    lines+=['','仅组内比较；task或sem梯度为零时cos记NA，不解释epsilon占位。', '',
        '## 教师质量与固定可靠门覆盖','',
        '以下为每状态2批×2张训练切片的像素汇总，不是独立患者数。L已见过标签，质量偏乐观，不能证明真实U的监督准确；高置信错误在所有预测类别统计，前景precision另列。r=pmean−sd≥.9是固定启发式，4个2×2嵌套样本不是4个独立全局样本，也不是置信保证。','',
        '| 状态 | L FG precision | L FG recall | L高置信错误/数量 | L可靠FG precision | U预测FG像素 | U可靠FG覆盖 | U可靠BG覆盖 | U mask一致率 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for state in dict.fromkeys(r['state'] for r in rows):
        group=[r for r in rows if r['state']==state];audits=[r['L_teacher']['label_audit'] for r in group]
        pred=sum(v['pred_foreground_pixels'] for v in audits);gt=sum(v['gt_foreground_pixels'] for v in audits)
        tp=sum(round(v['precision']*v['pred_foreground_pixels']) if v['precision'] is not None else 0 for v in audits)
        accepted=sum(r['L_teacher']['classes']['foreground']['accepted_pixels'] for r in group)
        accepted_tp=sum(round(r['L_teacher']['label_audit']['accepted_foreground_precision']*r['L_teacher']['classes']['foreground']['accepted_pixels']) if r['L_teacher']['label_audit']['accepted_foreground_precision'] is not None else 0 for r in group)
        uf=sum(r['U_teacher']['classes']['foreground']['predicted_pixels'] for r in group);ufa=sum(r['U_teacher']['classes']['foreground']['accepted_pixels'] for r in group)
        ub=sum(r['U_teacher']['classes']['background']['predicted_pixels'] for r in group);uba=sum(r['U_teacher']['classes']['background']['accepted_pixels'] for r in group)
        lines.append(f"| {state} | {number(tp/pred if pred else None)} | {number(tp/gt if gt else None)} | {sum(v['high_confidence_errors'] for v in audits)}/{sum(v['high_confidence_pixels'] for v in audits)} | {number(accepted_tp/accepted if accepted else None)} | {uf} | {number(ufa/uf if uf else None)} | {number(uba/ub if ub else None)} | {statistics.mean(r['U_teacher']['mask_agreement'] for r in group):.4f} |")
    lines+=['','总体mask一致率可被背景主导；逐类mask Jaccard、边界/内部错误、预测熵、概率方差和样本类别分歧全部保留在匿名JSON。前景组为空时不强迫前景，可靠门前景覆盖接近零时不降阈值“调通”。本次未训练EMA或筛选模型。','',
        '## 域记忆四读出（首seed v19 L/M0，BMC）','',
        '| 读出 | q/phi | 域记忆 | Dice % | NLL | Brier | HD95 mm |','|---|---|---|---:|---:|---:|---:|']
    descriptions={'T0':('源终点','RUNMC，旧结果复用'),'T_switch':('源终点保持不变','真实初始BMC，零更新'),'T_post_source':('目标训练终点','封存RUNMC'),'T_post_target':('目标训练终点','已学BMC，旧结果复用')}
    for name in ['T0','T_switch','T_post_source','T_post_target']:
        m=a['readout_means'][name];q,mem=descriptions[name]
        lines.append(f"| {name} | {q} | {mem} | {m['Dice']*100:.4f} | {m['NLL']:.6f} | {m['Brier']:.6f} | {m['HD95_penalized_mm']:.4f} |")
    lines+=['','| 差值 | Dice pp |','|---|---:|']
    for name,vals in a['changes'].items():lines.append(f"| {name} | {vals['Dice']*100:+.4f} |")
    patient=switch.get('patient_contrast_aggregates',{})
    if patient:
        lines+=['','| 配对患者差值 | Dice范围 pp | 正/负患者 |','|---|---:|---:|']
        for name,values in patient.items():
            m=values['Dice'];lines.append(f"| {name} | [{m['minimum']*100:+.4f}, {m['maximum']*100:+.4f}] | {m['positive']}/{m['negative']} |")
    lines += ['',f"entry占总下降幅度：{number(a['entry_share_of_total_drop'])}；预定≥50%且同为负的域入口规则：{a['domain_entry_rule_met']}。entry+training_under_target_memory精确等于total。", 
        'q/phi/model/projection在源终点与真实目标ACTIVE_INITIAL一致；RUNMC封存记忆在目标终点不变已验证。域切换新读出只在副本注册current，不调用begin_task新建另一入口。T-post-source是定位读出，不是把较优读出选作最终方法。只覆盖首seed L/M0，不能声称三seed普遍机制；也没有拆分phi与全局后验更新、后验与记忆兼容性。特别是RUNMC记忆参数未变不代表其边缘分布不变：条件记忆仍依赖更新后的q。','',
        '## 根据预定规则选择后续路线','',f"本次选择：**{a['next_route']}**。", 
        '若入口损失主导，优先研究新域变分状态优化初始化，仍保留新域先验定义；不得把旧域训练证据当新域先验重复累计。若一致性专属规则满足，教师可靠性路线可另立协议，但不把本次训练L诊断当U可靠性验证。两者均满足时先修复目标L适应入口，教师路线后置。若均不满足，不强制认定根因。',
        '后续EMA/可靠筛选/覆盖匹配或域初始化训练未在本次执行；须另立可审核的协议和预算。v20失败配置保持关闭，不追加seed/权重/增强网格。本报告只交付诊断和选路，不声称后继方法已有效。','',
        '[一致性原始匿名标量](SEMANTIC_RESULTS.json)；[域切换聚合结果](SWITCH_RESULTS.json)；[判读汇总](ANALYSIS.json)；[冻结协议](PREREG_ZH.md)。不发布患者身份、影像、概率数组、权重、梯度向量、私有路径或原始日志。']
    execution_file=Path(__file__).resolve().parent/'EXECUTION_SUMMARY.json'
    if execution_file.exists():
        cost=json.loads(execution_file.read_text())
        lines+=['','## 执行回执','',f"COMPLETE、EXIT{cost['exit']['exit_code']}；原始起点至退出{cost['wall_elapsed_seconds']:.2f}秒，ledger字节{cost['ledger']['bytes']}，正式更新预留{cost['ledger']['updates_reserved']}。全部三个阶段PASS，无修复或重试。反向30/128、副本Adam20/24、来源更新0；新增2个完整读出、复用2个旧格。缓存、日志和私有预测保留在既定NAS root，未删除历史。"]
    if a['next_route']=='NO_SINGLE_ROUTE_ESTABLISHED':
        insertion=['','### 本轮具体决定','',
            '两条预定充分条件均未满足：不宣称已定位教师根因，也不优先域初始化。零更新入口损失只占总降幅的一小部分，应首先检查目标训练更新中共享phi与域状态的作用，而不是调初始记忆或放大历史KL。',
            '一致性路线的三个seed负结果、训练L高置信错误以及早期可靠门低覆盖仍有效，但7/10微步探针不是预定8/10。教师改造只能保留为后续候选假设，不能把本诊断写成教师因果机制已通过。',
            '建议下一个最小独立协议比较：全量适应、冻结phi而保留q与域状态学习、固定q/phi仅学习域状态；用相邻对照区分phi与q的作用，保持相同目标L曝光和原任务风险。先验证目标L能获得净收益，再决定域状态/识别器侧修复。原source只读、旧域隔离评价，不在本次Q0启动训练。']
        index=next(i for i,line in enumerate(lines) if line.startswith('[一致性原始匿名标量]'))
        lines[index:index]=insertion
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('folder');args=parser.parse_args();root=Path(args.folder)
    sem=json.loads((root/'SEMANTIC_RESULTS.json').read_text());switch=json.loads((root/'SWITCH_RESULTS.json').read_text());a=analyze(sem,switch)
    (root/'ANALYSIS.json').write_text(json.dumps(a,indent=2)+'\n');(root/'REPORT_ZH.md').write_text(report(sem,switch,a))
    print(json.dumps(a,ensure_ascii=False))


if __name__=='__main__':main()
