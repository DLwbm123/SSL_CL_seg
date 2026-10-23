"""Read sealed metrics only; reuse the unchanged paired endpoints and gates."""
import argparse,json
from pathlib import Path
from ..semantic_evidence_v22.summarize import analyze


def main():
    parser=argparse.ArgumentParser();parser.add_argument('snapshot');args=parser.parse_args()
    x=json.loads(Path(args.snapshot).read_text());a=analyze(x);out=Path(__file__).parent
    assert x['RESULTS.json']['preflight_reverse_calls']==5
    assert len(x['RESULTS.json']['cells'])==4 and all(v['complete'] and v['patients']==4 for v in x['RESULTS.json']['cells'].values())
    assert x['RESULTS.json']['counter']['outer_backward_calls']==2400
    a['trajectory_analysis']=x['trajectory_analysis']
    previous=json.loads((out.parent/'semantic_evidence_v22/ANALYSIS.json').read_text())
    a['historical_v22_comparison']={arm:{d:100*(a['means'][arm][d]['Dice']-previous['means'][arm][d]['Dice']) for d in ('RUNMC','BMC')} for arm in ('L','LU')}
    launch=x['LAUNCH.json'];ledger=x['ledger']
    cost=dict(status='COMPLETE',exit=x['EXIT.json'],launch=launch,
              training_wall_seconds=x['EXIT.json']['time']-launch['started'],
              combined_elapsed_seconds=x['EXIT.json']['time']-launch['combined_origin'],
              root_bytes=x['root_bytes'],diagnostic_root_bytes=x['diagnostic_root_bytes'],
              ledger={k:ledger[k] for k in ('limits','seconds_charged','updates_reserved','bytes')},
              formal_updates=2400,preflight_reverse_calls=5,preflight_optimizer_updates=0,
              diagnostic_reverse_calls=42,diagnostic_optimizer_updates=0,
              common_payload_and_prior_equal=True,attempts=[{k:v for k,v in row.items() if k in ('name','status','seconds_actual','updates_reserved')} for row in ledger['attempts']])
    lines=['# v23：预测匹配候选与语义路径 U 指导结果','',
       '两臂完成固定1600终点，共同L400入口，各新增1200步；只有LU增加新U指导。BMC仅未适应迁移。', '',
       '| 臂 | 域 | Dice % | NLL | Brier | HD95 mm | 空预测 |','|---|---|---:|---:|---:|---:|---:|']
    for arm,ds in a['means'].items():
        for d,m in ds.items():lines.append(f"| {arm} | {d} | {m['Dice']*100:.4f} | {m['NLL']:.6f} | {m['Brier']:.6f} | {m['HD95_penalized_mm']:.4f} | {m['empty_predictions']:.3f} |")
    lines+=['','| LU−L | Dice pp | NLL | Brier | HD95 mm | 患者Dice差值范围 pp | 正/负患者 |','|---|---:|---:|---:|---:|---:|---:|']
    for d,v in a['U_effect'].items():
        m=v['metrics'];q=v['patient_aggregates']['Dice'];lines.append(f"| {d} | {100*m['Dice']:+.4f} | {m['NLL']:+.6f} | {m['Brier']:+.6f} | {m['HD95_penalized_mm']:+.4f} | [{100*q['minimum']:+.4f}, {100*q['maximum']:+.4f}] | {q['positive']}/{q['negative']} |")
    lines+=['','原门槛：源增益≥1pp、迁移下降≤0.5pp、无新增空预测、HD95恶化≤10%。不追改旧结果。','', '| 门槛 | 判定 |','|---|---|']
    for k,v in a['gate']['checks'].items():lines.append(f"| {k} | {'PASS' if v else 'FAILED'} |")
    lines+=['',f"总判定：**{a['gate']['status']}**。",'失败不追加seed/权重/增强求过关；通过也仅是一seed探索信号，需独立确认，不是稳定泛化证明。', '',
       '候选配对、有效区域归一化、去除额外U局部KL和U梯度路由为本次组合改造；L/LU只分离新增U指导的条件增量，不分离组合内部因素，也不证明精确Bayesian后验。L仍保留原任务损失、局部ELBO与全局/域KL。U标量是条件图像评分，不能与L的局部ELBO混同。', '',
       '| 臂 | 最后100步任务损失 | anchor | 有效候选比例 | 支持质量归一化评分差 | L局部ELBO/HW | U条件评分/HW |','|---|---:|---:|---:|---:|---:|---:|']
    for arm,t in a['training_summaries'].items():
        m=t['last100_means'];lines.append(f"| {arm} | {m['task']:.6f} | {m['anchor']:.6f} | {m['valid_candidate_fraction']:.6f} | {m['structure_margin']:+.6f} | {m['image_L']:+.6f} | {m['image_U']:+.6f} |")
    lines+=['','## 修正是否有效：已修复覆盖和路径，但没有建立 U 分割收益','',
       '| 臂 | 全1200步有效候选比例均值/最小值 | 平均评分间隔为正的步数 | 最后100步间隔为正 | 首100步anchor | 末100步anchor |','|---|---:|---:|---:|---:|---:|']
    for arm,t in x['trajectory_analysis'].items():
        lines.append(f"| {arm} | {t['valid_candidate_fraction_mean']:.4f}/{t['valid_candidate_fraction_min']:.4f} | {t['mean_margin_positive_steps']}/1200 | {t['last100_mean_margin_positive_steps']}/100 | {t['first100_means']['anchor']:.6f} | {a['training_summaries'][arm]['last100_means']['anchor']:.6f} |")
    lines+=['',
       '有效候选比例是按训练L标签定义的正负质量顺序，不是评分器排序准确率。全程为1说明本次全部被采用配对有真实质量差异，原空前景平移恒等问题不再出现。评分间隔是每步跨候选与MC的平均；1194/1192个正均值步骤不能当作每个候选均正确或独立样本准确率。首末训练批次不同，这些仅为训练轨迹，不是固定探针上的能力增长检验。', '',
       '路径预检证实：L锚定不通过负例生成路径更新phi；U直接评分theta/a路径被阻断，仍能经Pi传到phi。结合有效候选覆盖，说明预定机制已实际实现。可是U的源域和迁移Dice增量仍接近零、两域均2名患者正和2名负，未达到方法有效性的原门槛。不能把训练评分信号建立等同于真实U上正确语义梯度已经建立。', '',
       '与v22比较仅作同首seed历史背景：','', '| 臂 | v23−v22 RUNMC Dice pp | v23−v22 BMC Dice pp |','|---|---:|---:|']
    for arm,ds in a['historical_v22_comparison'].items():lines.append(f"| {arm} | {ds['RUNMC']:+.4f} | {ds['BMC']:+.4f} |")
    lines+=['',
       '本次观察到的U差值从v22的−0.0882/−0.1984pp收缩为−0.0068/+0.0037pp；可描述为本次首seed的负增量减小，不能据单次跨版本比较宣布稳定消除伤害。v23的L本身也变化了，不能把LU跨版本提高全部算成U收益。候选、归一化与U目标/路由同时改变，没有独立消融，不能认定哪一项造成变化。v22约0.693与v23约0.11的anchor也不能直接比较，因为配对与尺度不同。', '',
       '概率指标方面：两域NLL的患者配对差值均4/4下降；Brier在RUNMC为3/4下降、BMC为4/4下降，均值均小幅改善。这是本次配置的概率误差变化，尚不足以声称可靠校准或分割性能提升，不能替代Dice门槛。', '',
       '当前证据支持“无效候选覆盖已修复，预定的新梯度路由已落实”，不支持“生成评分已给U提供足够有效的分割学习信号”。可能的剩余问题包括训练L的纠错排序向U分布迁移不足、评分局部梯度与分割质量不一致或指导量级不足；本轮没有独立区分，不能选定一个根因后自动调权重。本配置按FAILED关闭扩样，保留负结果，不追加seed/权重/增强来求过关。', '',
       '训练轨迹不能当独立探针或U标签质量证明；两臂总目标不同，不比较总损失的高低宣布成功。NLL为分割预测NLL，不是图像证据或可靠校准的单项证明。两臂输入计数都包含接口送入的U图像，但L没有U损失，不能把该计数当作U学习曝光。', '',
       '每域原4名开发val患者，3次推断不是3训练seed或12独立患者；共享开发集反复使用，未访问test/U隐藏标签。身份仅私有配对，公开全指标均值和患者差值范围/方向。', '',
       f"训练根实际{cost['root_bytes']}字节；D0根{cost['diagnostic_root_bytes']}字节；诊断原始起点至训练退出{cost['combined_elapsed_seconds']:.2f}秒。D0 42次梯度/0更新、预检5次反向/0Adam、正式2400更新；来源400步成本保留。",
       '', '[训练协议](PREREG_ZH.md)；[D0解释与选择](DIAGNOSTIC_REPORT_ZH.md)；[匿名结果](ANALYSIS.json)；[执行成本](EXECUTION_SUMMARY.json)。']
    for n,v in [('ANALYSIS.json',a),('RESULTS.json',x['RESULTS.json']),('EXECUTION_SUMMARY.json',cost)]:
        (out/n).write_text(json.dumps(v,indent=2)+'\n')
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(gate=a['gate'],U_Dice_pp={d:100*v['metrics']['Dice'] for d,v in a['U_effect'].items()})))

if __name__=='__main__':main()
