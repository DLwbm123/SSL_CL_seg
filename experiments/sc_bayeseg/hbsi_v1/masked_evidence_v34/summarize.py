"""Offline anonymous report; unchanged paired patient metrics and original gates."""
import argparse,json
from pathlib import Path
from statistics import mean
from ..trainability_v21.summarize import patients,METRICS

def analyze(x):
    assert x['STATUS.json']['stage']=='COMPLETE' and x['EXIT.json']['exit_code']==0
    assert x['RESULTS.json']['new_formal_updates']==800
    assert x['BRANCH_L.json']['common_payload']==x['BRANCH_LU.json']['common_payload']
    assert x['BRANCH_L.json']['prior_hash']==x['BRANCH_LU.json']['prior_hash']
    l=x['ledger'];assert len(l['training_sessions'])==2 and l['updates_reserved']==800
    failed=[v for v in l['attempts'] if v['status']!='PASS']
    assert not failed
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
    parser=argparse.ArgumentParser();parser.add_argument('snapshot');args=parser.parse_args()
    x=json.loads(Path(args.snapshot).read_text());a=analyze(x);out=Path(__file__).parent
    r=x['RESULTS.json']
    assert len(r['cells'])==4 and all(v['complete'] and v['patients']==4 for v in r['cells'].values())
    assert r['counter']['outer_backward_calls']==800 and r['preflight_real_reverse_calls']==2 and r['auxiliary_path_reverse_calls']==1 and r['preflight_fixture_reverse_calls']==1
    for tag,folder,arm in [('v23L1600','semantic_alignment_v23','L'),('v25A2000','posterior_teacher_v25','A')]:
        old=json.loads((out.parent/folder/'ANALYSIS.json').read_text())
        a[tag+'_historical_delta_pp']={new:{d:100*(a['means'][new][d]['Dice']-old['means'][arm][d]['Dice']) for d in ('RUNMC','BMC')} for new in ('L','LU')}
    cost=dict(status='COMPLETE',original_started=x['LAUNCH.json']['started'],original_deadline=x['LAUNCH.json']['deadline'],wall_seconds=x['EXIT.json']['time']-x['LAUNCH.json']['started'],root_bytes=x['root_bytes'],formal_updates=800,preflight_model_reverse_calls=3,preflight_fixture_reverse_calls=1,preflight_optimizer_updates=0,source_updates=0,common_payload_and_prior=True,ledger={k:x['ledger'][k] for k in ('limits','seconds_charged','updates_reserved','bytes')})
    lines=['# v34：遮挡区域图像边缘风险 L/U 池对照结果','', '共同v23L1600入口，各400步到固定2000；两臂均有相同辅助目标，只辅助图像池不同。', '', '| 臂 | 域 | Dice % | NLL | Brier | HD95 mm | 空预测 |','|---|---|---:|---:|---:|---:|---:|']
    for arm,ds in a['means'].items():
        for d,m in ds.items():lines.append(f"| {arm} | {d} | {100*m['Dice']:.5f} | {m['NLL']:.7f} | {m['Brier']:.7f} | {m['HD95_penalized_mm']:.5f} | {m['empty_predictions']:.3f} |")
    lines+=['','| LU−L | Dice pp | 患者范围 pp | 正/负患者 | NLL变化 | Brier变化 |','|---|---:|---:|---:|---:|---:|']
    for d,v in a['U_effect'].items():
        m=v['metrics'];q=v['patient_aggregates']['Dice'];lines.append(f"| {d} | {100*m['Dice']:+.5f} | [{100*q['minimum']:+.5f}, {100*q['maximum']:+.5f}] | {q['positive']}/{q['negative']} | {m['NLL']:+.7f} | {m['Brier']:+.7f} |")
    lines+=['',f"原门槛总判定：**{a['gate']['status']}**。",'', '| 原门槛 | 判定 |','|---|---|']
    for k,v in a['gate']['checks'].items():lines.append(f"| {k} | {'PASS' if v else 'FAILED'} |")
    for tag in ('v23L1600','v25A2000'):
        lines+=['',f'| 臂 | 相对{tag}源 pp | 迁移 pp |','|---|---:|---:|']
        for arm,v in a[tag+'_historical_delta_pp'].items():lines.append(f"| {arm} | {v['RUNMC']:+.5f} | {v['BMC']:+.5f} |")
    lines+=['','历史v25A每步少4次辅助语义预测及2次区域协方差分解，是无辅助目标背景，不是新两臂等计算的主对照。', '', '| 臂 | 末100辅助风险 | 区域logscore/像素 | 区域残差MSE | logdet/像素 |','|---|---:|---:|---:|---:|']
    for arm,t in a['training_summaries'].items():
        m=t['last100_means'];lines.append(f"| {arm} | {m['masked_risk']:.6f} | {m['heldout_logscore_per_pixel']:.6f} | {m['heldout_residual_mse']:.6f} | {m['heldout_logdet_per_pixel']:.6f} |")
    lines+=['','区域图像风险来自不同训练批次和变化模型，不能当固定探针、分割质量或可靠后验。此处是SAR高斯的主子矩阵边缘分布，不是给定可见残差的条件高斯。识别器只看遮挡输入，但仍可能利用坐标、纹理和强度捷径；直接评分theta/a被停止，Pi条件路径仍可向q/phi/域状态传梯度。', '',f"800正式更新，3模型预检反向+1小fixture反向、0预检Adam。总历时{cost['wall_seconds']:.2f}秒，全根{cost['root_bytes']}字节。只读封存指标，无新评分。",'', '单训练seed、每域4患者、3推断不是3训练seed。共享开发val反复使用，不能宣称稳定泛化/精确后验/CL成功；原门槛与负结果保留，失败不追加seed/patch位置尺寸/遮挡值/协方差尺度/权重/MC/增强搜索。', '', '[冻结协议](PREREG_ZH.md)；[全部匿名指标](ANALYSIS.json)；[执行摘要](EXECUTION_SUMMARY.json)。']
    for name,value in [('ANALYSIS.json',a),('RESULTS.json',r),('EXECUTION_SUMMARY.json',cost)]:
        (out/name).write_text(json.dumps(value,indent=2)+'\n')
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');print(json.dumps(a['gate']))

if __name__=='__main__':main()
