"""Offline anonymous report; unchanged paired patient metrics and original gates."""
import argparse,json
from pathlib import Path
from ..semantic_evidence_v22.summarize import analyze


def main():
    p=argparse.ArgumentParser();p.add_argument('snapshot');args=p.parse_args()
    x=json.loads(Path(args.snapshot).read_text());a=analyze(x);out=Path(__file__).parent
    assert len(x['RESULTS.json']['cells'])==4 and all(v['complete'] and v['patients']==4 for v in x['RESULTS.json']['cells'].values())
    assert x['RESULTS.json']['preflight_reverse_calls']==3 and x['RESULTS.json']['counter']['outer_backward_calls']==2400
    old=json.loads((out.parent/'semantic_alignment_v23/ANALYSIS.json').read_text())
    a['historical_v23_L_delta_pp']={arm:{d:100*(a['means'][arm][d]['Dice']-old['means']['L'][d]['Dice']) for d in ('RUNMC','BMC')} for arm in ('L','LU')}
    cost=dict(status='COMPLETE',wall_seconds=x['EXIT.json']['time']-x['LAUNCH.json']['started'],root_bytes=x['root_bytes'],formal_updates=2400,preflight_reverse_calls=3,preflight_optimizer_updates=0,source_updates=0,common_payload_and_prior=True,ledger={k:x['ledger'][k] for k in ('limits','seconds_charged','updates_reserved','bytes')})
    lines=['# v24：L供体对照与U供体外观迁移结果','', '共同L400入口，两个新臂固定1600终点。L=来自训练L池的外观迁移；LU=来自训练U池的外观迁移。两臂真实L标签、风险、视图数和更新数匹配。', '', '| 臂 | 域 | Dice % | NLL | Brier | HD95 mm | 空预测 |','|---|---|---:|---:|---:|---:|---:|']
    for arm,ds in a['means'].items():
        for d,m in ds.items():lines.append(f"| {arm} | {d} | {100*m['Dice']:.5f} | {m['NLL']:.7f} | {m['Brier']:.7f} | {m['HD95_penalized_mm']:.5f} | {m['empty_predictions']:.3f} |")
    lines+=['','| LU−L | Dice pp | 患者范围 pp | 正/负患者 | NLL变化 | Brier变化 |','|---|---:|---:|---:|---:|---:|']
    for d,v in a['U_effect'].items():
        m=v['metrics'];q=v['patient_aggregates']['Dice'];lines.append(f"| {d} | {100*m['Dice']:+.5f} | [{100*q['minimum']:+.5f}, {100*q['maximum']:+.5f}] | {q['positive']}/{q['negative']} | {m['NLL']:+.7f} | {m['Brier']:+.7f} |")
    lines+=['',f"原门槛总判定：**{a['gate']['status']}**。",'', '| 原门槛 | 判定 |','|---|---|']
    for k,v in a['gate']['checks'].items():lines.append(f"| {k} | {'PASS' if v else 'FAILED'} |")
    lines+=['','| 臂 | 相对已封存v23 L源Dice pp | 相对v23 L迁移Dice pp |','|---|---:|---:|']
    for arm,v in a['historical_v23_L_delta_pp'].items():lines.append(f"| {arm} | {v['RUNMC']:+.5f} | {v['BMC']:+.5f} |")
    lines+=['','LU−L只检验额外U供体外观相对L供体的条件增量。它不能验证旧生成评分U目标、历史先验或CL机制。相位保留不是严格MRI语义不变证明；没有读取U隐藏标签。历史比较不是单因素对照；若仅对照退化制造优势，不宣称净可行性。', '', '失败不扩seed/频带/权重/增强网格；通过也仅为单seed开发信号。4名患者/域，3推断不是3训练seed，共享开发val反复使用，未验证独立泛化。', '', '| 臂 | 最后100步原图风险 | 迁移图风险 | 迁移RMS |','|---|---:|---:|---:|']
    for arm,t in a['training_summaries'].items():
        m=t['last100_means'];lines.append(f"| {arm} | {m['task_original']:.6f} | {m['task_transported']:.6f} | {m['transport_rms']:.6f} |")
    lines+=['',f"2400正式更新，3预检反向/0Adam；全根{cost['root_bytes']}字节，总历时{cost['wall_seconds']:.2f}秒。来源400步成本保留。只读封存指标，无新推断。",'', '[冻结协议](PREREG_ZH.md)；[匿名全指标](ANALYSIS.json)；[执行摘要](EXECUTION_SUMMARY.json)。']
    for n,v in [('ANALYSIS.json',a),('RESULTS.json',x['RESULTS.json']),('EXECUTION_SUMMARY.json',cost)]:(out/n).write_text(json.dumps(v,indent=2)+'\n')
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');print(json.dumps(a['gate']))

if __name__=='__main__':main()
