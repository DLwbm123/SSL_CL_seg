"""Anonymous offline closeout; never loads a model or private labels."""
import argparse,json
from pathlib import Path
from statistics import mean
from ..trainability_v21.summarize import patients,METRICS


def gate(means,effects):
    checks=dict(transfer_recovery_ge_0_5pp=effects['BMC']['metrics']['Dice']*100>=.5,source_drop_le_0_5pp=effects['RUNMC']['metrics']['Dice']*100>=-.5,no_new_empty=all(means['T'][d]['empty_predictions']<=means['A'][d]['empty_predictions'] for d in effects),HD95_worsening_le_10pct=all(means['T'][d]['HD95_penalized_mm']<=1.1*means['A'][d]['HD95_penalized_mm'] for d in effects))
    return dict(status='PASS' if all(checks.values()) else 'FAILED',checks=checks)


def analyze(x):
    old=x['controls_v25']
    for s,arms,updates in ((old,'ABC',1200),(x,'T',400)):
        assert s['STATUS.json']['stage']=='COMPLETE' and s['EXIT.json']['exit_code']==0
        assert s['RESULTS.json']['new_formal_updates']==updates and s['RESULTS.json']['counter']['outer_backward_calls']==updates
        assert len(s['RESULTS.json']['cells'])==2*len(arms) and all(c['complete'] and c['patients']==4 for c in s['RESULTS.json']['cells'].values())
        assert len(s['ledger']['training_sessions'])==len(arms) and s['ledger']['updates_reserved']==updates
        assert all(v['completed_steps']==2000 and v['status']=='COMPLETE' for v in s['ledger']['training_sessions'].values())
        assert all(v['status']=='PASS' for v in s['ledger']['attempts'])
    assert x['RESULTS.json']['preflight_model_reverse_calls']==2 and x['RESULTS.json']['training_audit_reverse_calls']==0
    assert x['MATCHING.json']['status']=='PASS' and x['PREFLIGHT.json']['retained_gradient_match'] and x['RESULTS.json']['unused_heads_unchanged']
    assert all(old['BRANCH_A.json'][k]==x['BRANCH_T.json'][k] for k in ('common_payload','prior_hash'))
    pp={arm:{d:patients(rows) for d,rows in s['patients'][arm].items()} for s,arm in ((old,'A'),(x,'T'))}
    means={arm:{d:{k:sum(v[k] for v in rows.values()) if k=='empty_predictions' else mean(v[k] for v in rows.values()) for k in METRICS} for d,rows in ds.items()} for arm,ds in pp.items()}
    effects={}
    for d in ('RUNMC','BMC'):
        assert pp['A'][d].keys()==pp['T'][d].keys();agg={}
        for k in METRICS:
            v=[pp['T'][d][i][k]-pp['A'][d][i][k] for i in pp['T'][d]]
            agg[k]=dict(n=4,mean=mean(v),minimum=min(v),maximum=max(v),positive=sum(t>1e-12 for t in v),negative=sum(t< -1e-12 for t in v))
        effects[d]=dict(metrics={k:means['T'][d][k]-means['A'][d][k] for k in METRICS},patient_aggregates=agg)
    return dict(status='COMPLETE',seed=20261023,training_seeds=1,patients_per_domain=4,inference_repeats=3,means=means,T_minus_A=effects,mechanism_gate=gate(means,effects),training=x['training'],matching=x['MATCHING.json'],reused_control='v25_A2000',U_signal_tested=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('snapshot');args=p.parse_args();x=json.loads(Path(args.snapshot).read_text());a=analyze(x);out=Path(__file__).parent
    old=json.loads((out.parent/'semantic_alignment_v23/ANALYSIS.json').read_text())
    a['relative_to_v23_L1600_pp']={arm:{d:100*(a['means'][arm][d]['Dice']-old['means']['L'][d]['Dice']) for d in ('RUNMC','BMC')} for arm in ('A','T')}
    cost=dict(status='COMPLETE',original_start=x['LAUNCH.json']['started'],hard_deadline=x['LAUNCH.json']['deadline'],exit_time=x['EXIT.json']['time'],elapsed_seconds=x['EXIT.json']['time']-x['LAUNCH.json']['started'],root_bytes=x['root_bytes'],formal_updates=400,training_model_reverse_calls=400,preflight_model_reverse_calls=2,preflight_complete_objective_forward_evaluations=3,preflight_optimizer_updates=0,additional_training_audit_reverse_calls=0,source_updates=0,old_models_rescored=0,removed_block_computed_for_logs=True,ledger={k:x['ledger'][k] for k in ('limits','seconds_charged','updates_reserved','bytes')})
    lines=['# v43 观测评分与排序目标块移除结果','', '共同v23L1600完整入口，T新增400更新至固定2000；A复用v25A2000封存结果。此为目标块条件干预，无U目标，不评价SSL收益。', '', '| 臂 | 域 | Dice % | NLL | Brier | HD95 mm | 空预测 |','|---|---|---:|---:|---:|---:|---:|']
    for arm,ds in a['means'].items():
        for d,m in ds.items():lines.append(f"| {arm} | {d} | {100*m['Dice']:.5f} | {m['NLL']:.7f} | {m['Brier']:.7f} | {m['HD95_penalized_mm']:.5f} | {m['empty_predictions']:.3f} |")
    lines+=['', '| T−A | Dice pp | 患者范围 pp | 正/负 | NLL变化 | Brier变化 | HD95变化 |','|---|---:|---:|---:|---:|---:|---:|']
    for d,v in a['T_minus_A'].items():
        m=v['metrics'];q=v['patient_aggregates']['Dice'];lines.append(f"| {d} | {100*m['Dice']:+.5f} | [{100*q['minimum']:+.5f},{100*q['maximum']:+.5f}] | {q['positive']}/{q['negative']} | {m['NLL']:+.7f} | {m['Brier']:+.7f} | {m['HD95_penalized_mm']:+.5f} |")
    lines+=['',f"新机制干预信号：**{a['mechanism_gate']['status']}**。旧U门槛与全部历史FAILED不变。",'', '| 冻结机制标准 | 判定 |','|---|---|']
    for k,v in a['mechanism_gate']['checks'].items():lines.append(f"| {k} | {'PASS' if v else 'FAILED'} |")
    lines+=['', '| 臂 | 相对入口源 pp | 相对入口BMC pp |','|---|---:|---:|']
    for arm,v in a['relative_to_v23_L1600_pp'].items():lines.append(f"| {arm} | {v['RUNMC']:+.5f} | {v['BMC']:+.5f} |")
    lines+=['','原入口BMC约44.07538%；相对A改善但未回到入口仅是部分恢复。联合移除score_x与anchor，不能分别归因。局部c/R先验减后验项仍保留，S/b原冻结、其余原Adam继承与采样不变，不是确定性baseline或原ELBO。', '', 'T仍计算移除项用于detach日志，原A额外有3次梯度审计而T没有，评分块反向成本亦不同。同样本完整原目标是T模型路径的标量，不是A的反事实训练轨迹；变化批次非固定泛化探针。LU适配器读取的U图像没有进入损失。', '', '| 训练标量 | 全400均值 | 末100均值 |','|---|---:|---:|']
    t=a['training']['T']
    for k in ('task','local_prior_minus_q','score_x','anchor','scaled','full_objective_same_samples'):lines.append(f"| {k} | {t['full_means'][k]:.8f} | {t['last100_means'][k]:.8f} |")
    lines+=['',f"400正式模型反向/Adam；实现预检3完整目标前向评估、2模型反向、0Adam；0额外训练反向探针。历时{cost['elapsed_seconds']:.2f}秒，全根{cost['root_bytes']}字节。仅读封存指标，无旧评分。", '', '单训练seed、每域4开发患者、3推断重复，不证明独立泛化、精确后验、可靠不确定性或严格CL遗忘；不追加seed/权重/增强网格，不以此作为新U研究永久准入。', '', '[冻结协议](PREREG_ZH.md)；[全部匿名指标](ANALYSIS.json)；[成本](EXECUTION_SUMMARY.json)。']
    for n,v in [('ANALYSIS.json',a),('RESULTS.json',x['RESULTS.json']),('EXECUTION_SUMMARY.json',cost)]:(out/n).write_text(json.dumps(v,indent=2)+'\n')
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');print(json.dumps(a['mechanism_gate']))

if __name__=='__main__':main()
