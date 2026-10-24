"""Close out sealed records without model inference or private-label access."""
import argparse,json
from pathlib import Path
from statistics import mean
from ..trainability_v21.summarize import patients,METRICS

ARMS=('FIX_Q','FIX_PHI')
DOMAINS=('RUNMC','BMC')


def analyze(x):
    old=x['controls_v25']
    for s,arms,updates in ((old,('A','B','C'),1200),(x,ARMS,800)):
        assert s['STATUS.json']['stage']=='COMPLETE' and s['EXIT.json']['exit_code']==0
        assert s['RESULTS.json']['new_formal_updates']==updates and s['RESULTS.json']['counter']['outer_backward_calls']==updates
        assert len(s['RESULTS.json']['cells'])==2*len(arms) and all(v['complete'] and v['patients']==4 for v in s['RESULTS.json']['cells'].values())
        assert len(s['ledger']['training_sessions'])==len(arms) and s['ledger']['updates_reserved']==updates
        assert all(v['status']=='COMPLETE' and v['completed_steps']==2000 for v in s['ledger']['training_sessions'].values())
        assert all(v['status']=='PASS' for v in s['ledger']['attempts'])
    assert x['MATCHING.json']['status']==x['PREFLIGHT.json']['status']=='PASS'
    assert x['RESULTS.json']['preflight_model_reverse_calls']==2 and x['RESULTS.json']['training_audit_reverse_calls']==0
    assert x['RESULTS.json']['unused_heads_unchanged']
    for k,v in dict(global_updates=400,phi_updates=400,domain_updates=800).items():assert x['RESULTS.json']['counter'][k]==v
    for arm in ARMS:
        assert x[f'FROZEN_CHECK_{arm}.json']['status']=='PASS'
        assert all(x[f'BRANCH_{arm}.json'][k]==old['BRANCH_A.json'][k] for k in ('common_payload','prior_hash'))
    pp={arm:{d:patients(rows) for d,rows in s['patients'][arm].items()} for s,arm in ((old,'A'),(x,'FIX_Q'),(x,'FIX_PHI'))}
    means={arm:{d:{k:sum(v[k] for v in rows.values()) if k=='empty_predictions' else mean(v[k] for v in rows.values()) for k in METRICS} for d,rows in ds.items()} for arm,ds in pp.items()}
    effects={};gates={}
    for arm in ARMS:
        effects[arm]={}
        for d in DOMAINS:
            assert pp[arm][d].keys()==pp['A'][d].keys();agg={}
            for k in METRICS:
                v=[pp[arm][d][i][k]-pp['A'][d][i][k] for i in pp[arm][d]]
                agg[k]=dict(n=4,mean=mean(v),minimum=min(v),maximum=max(v),positive=sum(t>1e-12 for t in v),negative=sum(t< -1e-12 for t in v))
            effects[arm][d]=dict(metrics={k:means[arm][d][k]-means['A'][d][k] for k in METRICS},patient_aggregates=agg)
        checks=dict(transfer_recovery_ge_0_5pp=effects[arm]['BMC']['metrics']['Dice']*100>=.5,source_drop_le_0_5pp=effects[arm]['RUNMC']['metrics']['Dice']*100>=-.5,no_new_empty=all(means[arm][d]['empty_predictions']<=means['A'][d]['empty_predictions'] for d in DOMAINS),HD95_worsening_le_10pct=all(means[arm][d]['HD95_penalized_mm']<=1.1*means['A'][d]['HD95_penalized_mm'] for d in DOMAINS))
        gates[arm]=dict(status='PASS' if all(checks.values()) else 'FAILED',checks=checks)
    return dict(status='COMPLETE',seed=20261023,training_seeds=1,patients_per_domain=4,inference_repeats=3,means=means,effects_vs_A=effects,mechanism_gates=gates,training=x['training'],matching=x['MATCHING.json'],frozen_checks={arm:x[f'FROZEN_CHECK_{arm}.json'] for arm in ARMS},reused_control='v25_A2000',U_signal_tested=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('snapshot');args=p.parse_args();x=json.loads(Path(args.snapshot).read_text());a=analyze(x);out=Path(__file__).parent
    old=json.loads((out.parent/'semantic_alignment_v23/ANALYSIS.json').read_text())
    a['relative_to_v23_L1600_pp']={arm:{d:100*(a['means'][arm][d]['Dice']-old['means']['L'][d]['Dice']) for d in DOMAINS} for arm in ('A',*ARMS)}
    cost=dict(status='COMPLETE',original_start=x['LAUNCH.json']['started'],hard_deadline=x['LAUNCH.json']['deadline'],exit_time=x['EXIT.json']['time'],elapsed_seconds=x['EXIT.json']['time']-x['LAUNCH.json']['started'],root_bytes=x['root_bytes'],formal_updates=800,training_model_reverse_calls=800,preflight_model_reverse_calls=2,preflight_complete_objective_forward_evaluations=3,preflight_optimizer_updates=0,additional_training_audit_reverse_calls=0,source_updates=0,old_models_rescored=0,ledger={k:x['ledger'][k] for k in ('limits','seconds_charged','updates_reserved','bytes')})
    lines=['# v44 源域继续训练的更新许可干预结果','','同v23L1600完整入口与原目标；FIX_Q、FIX_PHI各400新更新到固定2000，A复用封存v25A2000。无U目标，不评价SSL收益。','','| 臂 | 域 | Dice % | NLL | Brier | HD95 mm | 空预测 |','|---|---|---:|---:|---:|---:|---:|']
    for arm,ds in a['means'].items():
        for d,m in ds.items():lines.append(f"| {arm} | {d} | {100*m['Dice']:.5f} | {m['NLL']:.7f} | {m['Brier']:.7f} | {m['HD95_penalized_mm']:.5f} | {m['empty_predictions']:.3f} |")
    lines+=['','| 相对A | 域 | Dice pp | 患者范围 pp | 正/负 | NLL变化 | Brier变化 | HD95变化 |','|---|---|---:|---:|---:|---:|---:|---:|']
    for arm,ds in a['effects_vs_A'].items():
        for d,v in ds.items():
            m=v['metrics'];q=v['patient_aggregates']['Dice'];lines.append(f"| {arm} | {d} | {100*m['Dice']:+.5f} | [{100*q['minimum']:+.5f},{100*q['maximum']:+.5f}] | {q['positive']}/{q['negative']} | {m['NLL']:+.7f} | {m['Brier']:+.7f} | {m['HD95_penalized_mm']:+.5f} |")
    lines+=['','| 臂 | 机制门 | BMC恢复 | 源保护 | 无新增空 | HD95保护 |','|---|---|---|---|---|---|']
    for arm,g in a['mechanism_gates'].items():lines.append('| '+arm+' | '+g['status']+' | '+' | '.join('PASS' if v else 'FAILED' for v in g['checks'].values())+' |')
    lines+=['','| 臂 | 相对入口源 pp | 相对入口BMC pp |','|---|---:|---:|']
    for arm,v in a['relative_to_v23_L1600_pp'].items():lines.append(f"| {arm} | {v['RUNMC']:+.5f} | {v['BMC']:+.5f} |")
    lines+=['','原入口BMC约44.07538%，未达到时仅为部分恢复。冻结q仍MC采样，冻结phi仍传输入梯度；eta都学习。参数与继承Adam槽冻结、另一模块及域状态改变、S/b不变均通过。许可干预包含其优化轨迹效应，不足以识别完整交互、唯一根因或eta独立作用。','','| 臂 | 标量 | 全400均值 | 末100均值 |','|---|---|---:|---:|']
    for arm,t in a['training'].items():
        for k in ('task','image_L','anchor','scaled'):lines.append(f"| {arm} | {k} | {t['full_means'][k]:.8f} | {t['last100_means'][k]:.8f} |")
    lines+=['',f"800正式反向/Adam、预检3完整目标前向与2模型反向/0Adam、0额外训练反向探针；历时{cost['elapsed_seconds']:.2f}秒，全根{cost['root_bytes']}字节。原A额外3次基础梯度探针为历史成本，新臂没有。LU适配器读取U但无U损失；没有新BMC训练或旧评分。",'','单训练seed、每域4开发患者、3推断重复不证明独立泛化、精确后验、可靠不确定性或严格CL遗忘。全部患者指标配对范围/方向见匿名JSON。门槛不改；两臂均失败即关闭冻结诊断，不补组合/seed/权重网格；后继U机制不以L-only通过作永久前提。','','[冻结协议](PREREG_ZH.md)；[完整匿名结果](ANALYSIS.json)；[成本](EXECUTION_SUMMARY.json)。']
    for n,v in [('ANALYSIS.json',a),('RESULTS.json',x['RESULTS.json']),('EXECUTION_SUMMARY.json',cost)]:(out/n).write_text(json.dumps(v,indent=2)+'\n')
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');print(json.dumps(a['mechanism_gates']))

if __name__=='__main__':main()
