"""Offline sealed-result analysis. No model calls or public patient identifiers."""
import argparse,json
from pathlib import Path
from statistics import mean
from ..trainability_v21.summarize import patients,METRICS


def analyze(x):
    old=x['controls_v25'];neighbor=x['controls_v30']
    for s,arms,updates in ((old,'ABC',1200),(neighbor,'N',400),(x,'IE',800)):
        assert s['STATUS.json']['stage']=='COMPLETE' and s['EXIT.json']['exit_code']==0
        assert s['RESULTS.json']['new_formal_updates']==updates and s['RESULTS.json']['counter']['outer_backward_calls']==updates
        assert len(s['RESULTS.json']['cells'])==2*len(arms) and all(c['complete'] and c['patients']==4 for c in s['RESULTS.json']['cells'].values())
        assert len(s['ledger']['training_sessions'])==len(arms)
        assert all(v['completed_steps']==2000 and v['status']=='COMPLETE' for v in s['ledger']['training_sessions'].values())
        assert all(v['status']=='PASS' for v in s['ledger']['attempts'])
    assert x['PREFLIGHT.json']['status']=='PASS' and x['MATCHING.json']['status']=='PASS'
    assert x['RESULTS.json']['preflight_reverse_calls']==2 and x['RESULTS.json']['training_audit_reverse_calls']==12
    assert x['TARGET_SUMMARY.json']['model_forwards']==0 and not x['TARGET_SUMMARY.json']['U_labels_read']
    assert x['TARGET_SUMMARY.json']['graph_solves']==542
    for field in ('common_payload','prior_hash','teacher_payload'):
        assert len({s[f'BRANCH_{arm}.json'][field] for s,arms in ((old,'A'),(neighbor,'N'),(x,'IE')) for arm in arms})==1
    assert all([r['step'] for r in x['audit'][arm]]==[1600,1799,1999] for arm in 'IE')
    assert sum(r['extra_reverse_calls'] for arm in 'IE' for r in x['audit'][arm])==12
    pp={arm:{d:patients(rows) for d,rows in s['patients'][arm].items()} for s,arms in ((old,'A'),(neighbor,'N'),(x,'IE')) for arm in arms}
    means={arm:{d:{k:(sum(v[k] for v in rows.values()) if k=='empty_predictions' else mean(v[k] for v in rows.values())) for k in METRICS} for d,rows in ds.items()} for arm,ds in pp.items()}
    effects={}
    for arm,control in (('E','A'),('E','N'),('E','I'),('I','A'),('I','N')):
        ds={}
        for d in ('RUNMC','BMC'):
            assert pp[arm][d].keys()==pp[control][d].keys();agg={}
            for k in METRICS:
                v=[pp[arm][d][i][k]-pp[control][d][i][k] for i in pp[arm][d]]
                agg[k]=dict(n=4,mean=mean(v),minimum=min(v),maximum=max(v),positive=sum(t>1e-12 for t in v),negative=sum(t< -1e-12 for t in v))
            ds[d]=dict(metrics={k:means[arm][d][k]-means[control][d][k] for k in METRICS},patient_aggregates=agg)
        checks=dict(source_gain_ge_1pp=ds['RUNMC']['metrics']['Dice']*100>=1,transfer_drop_le_0_5pp=ds['BMC']['metrics']['Dice']*100>=-.5,no_new_empty=all(means[arm][d]['empty_predictions']<=means[control][d]['empty_predictions'] for d in ds),HD95_worsening_le_10pct=all(means[arm][d]['HD95_penalized_mm']<=1.1*means[control][d]['HD95_penalized_mm'] for d in ds))
        effects[arm+'_minus_'+control]=dict(domains=ds,gate=dict(status='PASS' if all(checks.values()) else 'FAILED',checks=checks))
    return dict(status='COMPLETE',seed=20261023,training_seeds=1,patients_per_domain=4,inference_repeats=3,means=means,effects=effects,training=x['training'],audit=x['audit'],matching=x['MATCHING.json'],target_summary=x['TARGET_SUMMARY.json'],reused_controls=['v25_A2000','v30_N2000'])


def main():
    p=argparse.ArgumentParser();p.add_argument('snapshot');args=p.parse_args();x=json.loads(Path(args.snapshot).read_text());a=analyze(x);out=Path(__file__).parent
    old=json.loads((out.parent/'semantic_alignment_v23/ANALYSIS.json').read_text())
    a['relative_to_v23_L1600_pp']={arm:{d:100*(a['means'][arm][d]['Dice']-old['means']['L'][d]['Dice']) for d in ('RUNMC','BMC')} for arm in 'ANIE'}
    cost=dict(status='COMPLETE',original_start=x['LAUNCH.json']['started'],exit_time=x['EXIT.json']['time'],hard_deadline=x['LAUNCH.json']['deadline'],elapsed_seconds=x['EXIT.json']['time']-x['LAUNCH.json']['started'],root_bytes=x['root_bytes'],formal_updates=800,preflight_reverse_calls=2,training_audit_reverse_calls=12,graph_solves=542,new_target_model_forwards=0,extra_optimizer_updates=0,old_models_rescored=0,ledger={k:x['ledger'][k] for k in ('limits','seconds_charged','updates_reserved','bytes')})
    lines=['# v32 图像条件空间耦合','', 'I/E均从共同v23L1600续400步；A/N只读复用v25/v30固定2000。图目标解不等于U真实语义或原模型精确后验。','', '| 臂 | 域 | Dice % | NLL | Brier | HD95 mm | 空预测 |','|---|---|---:|---:|---:|---:|---:|']
    for arm,ds in a['means'].items():
        for d,m in ds.items():lines.append(f"| {arm} | {d} | {100*m['Dice']:.5f} | {m['NLL']:.7f} | {m['Brier']:.7f} | {m['HD95_penalized_mm']:.5f} | {m['empty_predictions']:.3f} |")
    lines+=['','| 对比 | 源Dice pp | 迁移Dice pp | 原门槛 |','|---|---:|---:|---|']
    for name,v in a['effects'].items():lines.append(f"| {name} | {100*v['domains']['RUNMC']['metrics']['Dice']:+.5f} | {100*v['domains']['BMC']['metrics']['Dice']:+.5f} | {v['gate']['status']} |")
    lines+=['','E−A整套净收益、E−N结构条件增量、E−I图像条件性分开判断，不相互替代。','', '| 对比 | 域 | 患者Dice范围 pp | 正/负 | NLL变化 | Brier变化 | HD95变化 |','|---|---|---:|---:|---:|---:|---:|']
    for name,v in a['effects'].items():
        for d,z in v['domains'].items():
            m=z['metrics'];r=z['patient_aggregates']['Dice'];lines.append(f"| {name} | {d} | [{100*r['minimum']:+.5f},{100*r['maximum']:+.5f}] | {r['positive']}/{r['negative']} | {m['NLL']:+.7f} | {m['Brier']:+.7f} | {m['HD95_penalized_mm']:+.5f} |")
    lines+=['','逐项判定：','']
    for name,v in a['effects'].items():lines.append('- '+name+': '+'; '.join(k+'='+('PASS' if b else 'FAILED') for k,b in v['gate']['checks'].items()))
    lines+=['','| 臂 | 相对v23L1600源 pp | 迁移 pp |','|---|---:|---:|']
    for arm,v in a['relative_to_v23_L1600_pp'].items():lines.append(f"| {arm} | {v['RUNMC']:+.5f} | {v['BMC']:+.5f} |")
    lines+=['',f"800正式Adam更新，2预检反向0Adam，12训练内额外autograd、0额外Adam；542图求解，0新目标模型forward，旧351骨干建库成本仍属v30。历时{cost['elapsed_seconds']:.2f}秒，全根{cost['root_bytes']}字节。",'', 'E−I同时改变权重空间分布和总耦合强度，非纯边缘位置消融；均值保持或求解残差不证明语义正确。单训练seed、每域4开发患者、3推断重复；不宣称稳定泛化、可靠不确定性或CL成功。失败不搜seed/邻接/尺度/权重/求解阈值/增强。', '', '[冻结协议](PREREG_ZH.md)；[全部匿名指标/轨迹](ANALYSIS.json)；[执行摘要](EXECUTION_SUMMARY.json)。']
    lines+=['', '## 实测解释与配置收口', '',
        'E−A源/迁移为+0.42511/+0.15488pp，E−N为−0.06865/+0.00057pp，E−I为+0.15866/+0.03691pp；I−A和I−N也都未通过。五项总判定均FAILED，均仅源≥1pp未达标，迁移/空预测/HD95保护PASS、空预测0。均匀空间耦合相对N两域下降，图像条件耦合只补回部分均匀耦合损失，不能替代E−A或E−N成功。',
        'E−N与E−I在两域均为2正2负患者；E−A两域各3正1负，迁移差值范围−0.64928至+1.07813pp，平均保护门通过不表示每名患者都受保护。E较N的NLL、Brier、HD95两域均值均改善，但源Dice下降、迁移近零；图像条件目标改变了概率和几何表现，未建立原定分割收益。相对v23L1600迁移仍−0.83742pp，没有双域绝对净收益。',
        'I/E初始q均值U/基础梯度范数约1.981/1.258、余弦+0.985/+0.943；中点约1.079/1.041且余弦−0.859/−0.866；末点约0.705/0.616且余弦−0.841/−0.871。不是始终梯度太小或始终反向；三个训练时点不能定位总体根因，基础项含全部L目标，实际Adam位移不是U反事实。末100步U风险I0.06218/E0.06473，是不同目标与训练状态的摘要，不是固定探针或U正确率。',
        '542次图求解通过数值检查且每图平均前景票保持，只证明计算符合冻结二次模型，不证明MRI边缘对应类别边界。E−I还改变总耦合强度，无法独立归因边缘位置。该配置关闭，不扩seed、邻接、尺度、耦合权重或求解阈值。后继转向后验风险积分方式的独立假设，保留原近邻证据而不继续改图。']
    for n,v in [('ANALYSIS.json',a),('RESULTS.json',x['RESULTS.json']),('EXECUTION_SUMMARY.json',cost)]:(out/n).write_text(json.dumps(v,indent=2)+'\n')
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:v['gate'] for k,v in a['effects'].items()}))

if __name__=='__main__':main()
