"""Offline anonymous report; unchanged paired patient metrics and original gates."""
import argparse,json
from pathlib import Path
from statistics import mean
from ..trainability_v21.summarize import patients,METRICS

def analyze(x):
    assert x['STATUS.json']['stage']=='COMPLETE' and x['EXIT.json']['exit_code']==0
    assert x['RESULTS.json']['new_formal_updates']==800
    assert x['RESULTS.json']['prior_failed_fixture_reverse_calls']==1
    assert x['REPAIR_RECORD.json']['formal_updates_before_failure']==0
    assert x['REPAIR_LAUNCH_002.json']['original_started']==x['LAUNCH.json']['started']
    assert x['REPAIR_LAUNCH_002.json']['original_deadline']==x['LAUNCH.json']['deadline']
    assert x['BRANCH_L.json']['common_payload']==x['BRANCH_LU.json']['common_payload']
    assert x['BRANCH_L.json']['prior_hash']==x['BRANCH_LU.json']['prior_hash']
    l=x['ledger'];assert len(l['training_sessions'])==2 and l['updates_reserved']==800
    failed=[v for v in l['attempts'] if v['status']!='PASS']
    assert len(failed)==1 and failed[0]['name']=='preflight' and failed[0]['status']=='FAILED' and failed[0]['updates_reserved']==0
    assert all(v['completed_steps']==2000 and v['status']=='COMPLETE' for v in l['training_sessions'].values())
    p={arm:{d:patients(v) for d,v in ds.items()} for arm,ds in x['patients'].items()}
    means={arm:{d:{k:mean(v[k] for v in rows.values()) if k!='empty_predictions' else sum(v[k] for v in rows.values()) for k in METRICS} for d,rows in ds.items()} for arm,ds in p.items()}
    effects={}
    for d in ('RUNMC','BMC'):
        assert p['L'][d].keys()==p['LU'][d].keys()
        changes={k:means['LU'][d][k]-means['L'][d][k] for k in METRICS}
        patient_effects={}
        for k in METRICS:
            v=[p['LU'][d][identity][k]-p['L'][d][identity][k] for identity in p['L'][d]]
            patient_effects[k]=dict(n=4,mean=mean(v),minimum=min(v),maximum=max(v),positive=sum(t>1e-12 for t in v),negative=sum(t< -1e-12 for t in v))
        effects[d]=dict(metrics=changes,patient_aggregates=patient_effects)
    checks=dict(source_U_gain_ge_1pp=effects['RUNMC']['metrics']['Dice']*100>=1,transfer_drop_le_0_5pp=effects['BMC']['metrics']['Dice']*100>=-.5,no_new_empty=all(means['LU'][d]['empty_predictions']<=means['L'][d]['empty_predictions'] for d in effects),HD95_worsening_le_10pct=all(means['LU'][d]['HD95_penalized_mm']<=1.1*means['L'][d]['HD95_penalized_mm'] for d in effects))
    return dict(status='COMPLETE',seed=20261023,training_seeds=1,patients_per_domain=4,inference_repeats=3,means=means,U_effect=effects,gate=dict(status='PASS' if all(checks.values()) else 'FAILED',checks=checks),training_summaries=x['training'])




def main():
    p=argparse.ArgumentParser();p.add_argument('snapshot');args=p.parse_args()
    x=json.loads(Path(args.snapshot).read_text());a=analyze(x);out=Path(__file__).parent
    assert len(x['RESULTS.json']['cells'])==4 and all(v['complete'] and v['patients']==4 for v in x['RESULTS.json']['cells'].values())
    assert x['RESULTS.json']['preflight_real_reverse_calls']==2 and x['RESULTS.json']['preflight_fixture_reverse_calls']==1 and x['RESULTS.json']['counter']['outer_backward_calls']==800
    old=json.loads((out.parent/'semantic_alignment_v23/ANALYSIS.json').read_text())
    a['historical_v23_L_delta_pp']={arm:{d:100*(a['means'][arm][d]['Dice']-old['means']['L'][d]['Dice']) for d in ('RUNMC','BMC')} for arm in ('L','LU')}
    previous=json.loads((out.parent/'posterior_teacher_v25/ANALYSIS.json').read_text())
    a['historical_v25_A_delta_pp']={arm:{d:100*(a['means'][arm][d]['Dice']-previous['means']['A'][d]['Dice']) for d in ('RUNMC','BMC')} for arm in ('L','LU')}
    cost=dict(status='COMPLETE',repair=x['REPAIR_RECORD.json'],failed_forward_calls=None,original_started=x['LAUNCH.json']['started'],original_deadline=x['LAUNCH.json']['deadline'],wall_seconds=x['EXIT.json']['time']-x['LAUNCH.json']['started'],root_bytes=x['root_bytes'],formal_updates=800,preflight_real_reverse_calls=2,preflight_fixture_reverse_calls=2,failed_attempt_model_reverse_calls=0,failed_attempt_optimizer_updates=0,preflight_optimizer_updates=0,source_updates=0,common_payload_and_prior=True,ledger={k:x['ledger'][k] for k in ('limits','seconds_charged','updates_reserved','bytes')})
    lines=['# v27：固定监督区域的L/U供体上下文结果','', '共同v23L1600入口，两个新臂各400步到固定2000。L/LU仅供体池不同，半幅L原像素保留，只监督其中距接缝4像素以外的区域。原图与有效区域任务风险各半。', '', '| 臂 | 域 | Dice % | NLL | Brier | HD95 mm | 空预测 |','|---|---|---:|---:|---:|---:|---:|']
    for arm,ds in a['means'].items():
        for d,m in ds.items():lines.append(f"| {arm} | {d} | {100*m['Dice']:.5f} | {m['NLL']:.7f} | {m['Brier']:.7f} | {m['HD95_penalized_mm']:.5f} | {m['empty_predictions']:.3f} |")
    lines+=['','| LU−L | Dice pp | 患者范围 pp | 正/负患者 | NLL变化 | Brier变化 |','|---|---:|---:|---:|---:|---:|']
    for d,v in a['U_effect'].items():
        m=v['metrics'];q=v['patient_aggregates']['Dice'];lines.append(f"| {d} | {100*m['Dice']:+.5f} | [{100*q['minimum']:+.5f}, {100*q['maximum']:+.5f}] | {q['positive']}/{q['negative']} | {m['NLL']:+.7f} | {m['Brier']:+.7f} |")
    lines+=['',f"原门槛总判定：**{a['gate']['status']}**。",'', '| 原门槛 | 判定 |','|---|---|']
    for k,v in a['gate']['checks'].items():lines.append(f"| {k} | {'PASS' if v else 'FAILED'} |")
    lines+=['','| 臂 | 相对已封存v23 L源Dice pp | 相对v23 L迁移Dice pp |','|---|---:|---:|']
    for arm,v in a['historical_v23_L_delta_pp'].items():lines.append(f"| {arm} | {v['RUNMC']:+.5f} | {v['BMC']:+.5f} |")
    lines+=['','LU−L只检验额外U供体上下文相对L供体的条件增量。它不能验证旧生成评分U目标、历史先验或CL机制。监督区域像素和标签保留，但拼接上下文可能不自然；没有读取供体标签或U隐藏标签。L供体也做同样变换，LU−L不是与无增强模型比较。', '', '失败不扩seed/掩码比例/接缝/权重/增强网格；通过也仅为单seed开发信号。4名患者/域，3推断不是3训练seed，共享开发val反复使用，未验证独立泛化。', '', '| 臂 | 最后100步原图风险 | 保留区域风险 | 上下文替换RMS |','|---|---:|---:|---:|']
    for arm,t in a['training_summaries'].items():
        m=t['last100_means'];lines.append(f"| {arm} | {m['task_original']:.6f} | {m['task_context']:.6f} | {m['context_rms']:.6f} |")
    lines+=['',f"800正式更新，2模型预检反向+2小fixture反向（含失败尝试1次）/0Adam；全根{cost['root_bytes']}字节，总历时{cost['wall_seconds']:.2f}秒。来源1600步成本保留。只读封存指标，无新推断。",'', '[冻结协议](PREREG_ZH.md)；[匿名全指标](ANALYSIS.json)；[执行摘要](EXECUTION_SUMMARY.json)。']
    lines+=['','| 臂 | 相对封存v25A2000源 pp | 相对v25A2000迁移 pp |','|---|---:|---:|']
    for arm,v in a['historical_v25_A_delta_pp'].items():lines.append(f"| {arm} | {v['RUNMC']:+.5f} | {v['BMC']:+.5f} |")
    lines+=['','v25A只作同入口同更新数的无混合背景，其视图计算少一份；不能隐藏计算差异或把新对照退化当净改进。']
    lines += ['', '首次启动在真实目标首次反向前因监督mask与ranking标量同名覆盖而断言退出；已完成1次小fixture autograd、0模型反向/0Adam。失败entry保留，修复变量名后沿原时钟/原ledger运行；再次预检另含1次小fixture+2模型反向。失败阶段已有模型前向但次数未封存；失败不记作零计算，不重置预算。']
    for n,v in [('ANALYSIS.json',a),('RESULTS.json',x['RESULTS.json']),('EXECUTION_SUMMARY.json',cost)]:(out/n).write_text(json.dumps(v,indent=2)+'\n')
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');print(json.dumps(a['gate']))

if __name__=='__main__':main()
