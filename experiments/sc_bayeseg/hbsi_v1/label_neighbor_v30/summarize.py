"""Sealed N plus matched historical A/B; anonymous offline summaries only."""
import argparse,json
from pathlib import Path
from statistics import mean
from ..trainability_v21.summarize import patients,METRICS


def analyze(x):
    assert x['RESULTS.json']['preflight_reverse_calls']==1 and x['RESULTS.json']['training_audit_reverse_calls']==6
    old=x['controls_v25']
    for snapshot,arms,steps in ((old,'ABC',1200),(x,'N',400)):
        assert snapshot['STATUS.json']['stage']=='COMPLETE' and snapshot['EXIT.json']['exit_code']==0
        assert snapshot['RESULTS.json']['new_formal_updates']==steps
        assert snapshot['RESULTS.json']['counter']['outer_backward_calls']==steps
        assert len(snapshot['RESULTS.json']['cells'])==2*len(arms)
        assert all(c['complete'] and c['patients']==4 for c in snapshot['RESULTS.json']['cells'].values())
        assert len(snapshot['ledger']['training_sessions'])==len(arms)
        assert all(s['completed_steps']==2000 and s['status']=='COMPLETE' for s in snapshot['ledger']['training_sessions'].values())
        assert all(a['status']=='PASS' for a in snapshot['ledger']['attempts'])
    assert x['MATCHING.json']['status']=='PASS' and x['PREFLIGHT.json']['status']=='PASS'
    for field in ('common_payload','prior_hash','teacher_payload'):
        assert len({snapshot[f'BRANCH_{arm}.json'][field] for snapshot,arms in ((old,'AB'),(x,'N')) for arm in arms})==1
    assert sum(r['extra_reverse_calls'] for arm in 'N' for r in x['audit'][arm])==6
    assert all([r['step'] for r in x['audit'][arm]]==[1600,1799,1999] for arm in 'N')
    pp={arm:{d:patients(rows) for d,rows in snapshot['patients'][arm].items()} for snapshot,arms in ((old,'AB'),(x,'N')) for arm in arms}
    means={arm:{d:{k:(sum(v[k] for v in rows.values()) if k=='empty_predictions' else mean(v[k] for v in rows.values())) for k in METRICS} for d,rows in ds.items()} for arm,ds in pp.items()}
    effects={}
    for arm,control in (('N','A'),('N','B')):
        ds={}
        for d in ('RUNMC','BMC'):
            assert pp[arm][d].keys()==pp[control][d].keys()
            metrics={k:means[arm][d][k]-means[control][d][k] for k in METRICS};agg={}
            for k in METRICS:
                v=[pp[arm][d][i][k]-pp[control][d][i][k] for i in pp[arm][d]]
                agg[k]=dict(n=4,mean=mean(v),minimum=min(v),maximum=max(v),positive=sum(t>1e-12 for t in v),negative=sum(t< -1e-12 for t in v))
            ds[d]=dict(metrics=metrics,patient_aggregates=agg)
        checks=dict(source_gain_ge_1pp=ds['RUNMC']['metrics']['Dice']*100>=1,transfer_drop_le_0_5pp=ds['BMC']['metrics']['Dice']*100>=-.5,no_new_empty=all(means[arm][d]['empty_predictions']<=means[control][d]['empty_predictions'] for d in ds),HD95_worsening_le_10pct=all(means[arm][d]['HD95_penalized_mm']<=1.1*means[control][d]['HD95_penalized_mm'] for d in ds))
        effects[arm+'_minus_'+control]=dict(domains=ds,gate=dict(status='PASS' if all(checks.values()) else 'FAILED',checks=checks))
    return dict(status='COMPLETE',seed=20261023,training_seeds=1,patients_per_domain=4,inference_repeats=3,means=means,effects=effects,training=x['training'],audit=x['audit'],matching=x['MATCHING.json'],reused_controls=['v25_A2000','v25_B2000'])


def main():
    p=argparse.ArgumentParser();p.add_argument('snapshot');args=p.parse_args();x=json.loads(Path(args.snapshot).read_text());a=analyze(x);out=Path(__file__).parent
    assert x['TARGET_SUMMARY.json']['source_unchanged'] and not x['TARGET_SUMMARY.json']['U_labels_read']
    a['target_summary']=x['TARGET_SUMMARY.json']
    v23=json.loads((out.parent/'semantic_alignment_v23/ANALYSIS.json').read_text())
    a['relative_to_v23_L1600_pp']={arm:{d:100*(a['means'][arm][d]['Dice']-v23['means']['L'][d]['Dice']) for d in ('RUNMC','BMC')} for arm in ('A','B','N')}
    cost=dict(status='COMPLETE',original_start=x['LAUNCH.json']['started'],exit_time=x['EXIT.json']['time'],hard_deadline=x['LAUNCH.json']['deadline'],elapsed_seconds=x['EXIT.json']['time']-x['LAUNCH.json']['started'],root_bytes=x['root_bytes'],formal_updates=400,target_backbone_forwards=351,teacher_new_semantic_forwards=0,preflight_reverse_calls=1,training_telemetry_reverse_calls=6,extra_optimizer_updates=0,old_models_rescored=0,ledger={k:x['ledger'][k] for k in ('limits','seconds_charged','updates_reserved','bytes')})
    lines=['# v30 真实标签锚定局部近邻传递','', 'N=冻结L标签特征库向U传递的top8软类别，学生完全复用v25B。A/B复用匹配的v25固定2000终点，未重训/重评分；仅N从共同v23L1600续400步。','', '| 臂 | 域 | Dice % | NLL | Brier | HD95 mm | 空预测 |','|---|---|---:|---:|---:|---:|---:|']
    for arm,ds in a['means'].items():
        for d,m in ds.items():lines.append(f"| {arm} | {d} | {100*m['Dice']:.5f} | {m['NLL']:.7f} | {m['Brier']:.7f} | {m['HD95_penalized_mm']:.5f} | {m['empty_predictions']:.3f} |")
    lines+=['','| 对比 | 源Dice pp | 迁移Dice pp | 原门槛 |','|---|---:|---:|---|']
    for name,v in a['effects'].items():lines.append(f"| {name} | {100*v['domains']['RUNMC']['metrics']['Dice']:+.5f} | {100*v['domains']['BMC']['metrics']['Dice']:+.5f} | {v['gate']['status']} |")
    lines+=['','N−A判断这套U标签传递训练的净收益，N−B判断相对固定分割均值教师的条件改进。两者分开应用原门槛，N−B不能代替N−A。','', '| 对比 | 域 | 患者Dice范围 pp | 正/负 | NLL变化 | Brier变化 | HD95变化 |','|---|---|---:|---:|---:|---:|---:|']
    for name,v in a['effects'].items():
        for d,z in v['domains'].items():
            m=z['metrics'];r=z['patient_aggregates']['Dice'];lines.append(f"| {name} | {d} | [{100*r['minimum']:+.5f},{100*r['maximum']:+.5f}] | {r['positive']}/{r['negative']} | {m['NLL']:+.7f} | {m['Brier']:+.7f} | {m['HD95_penalized_mm']:+.5f} |")
    lines+=['','逐项判定：','']
    for name,v in a['effects'].items():lines.append('- '+name+': '+'; '.join(k+'='+('PASS' if b else 'FAILED') for k,b in v['gate']['checks'].items()))
    lines+=['','| 臂 | 相对v23L1600源 pp | 相对v23L1600迁移 pp |','|---|---:|---:|']
    for arm,v in a['relative_to_v23_L1600_pp'].items():lines.append(f"| {arm} | {v['RUNMC']:+.5f} | {v['BMC']:+.5f} |")
    lines+=['','额外400步不能全归U；绝对背景与严格同预算对照分开。近邻软票不等于U GT或校准概率。N−B改变信息来源、目标分布/熵和建库成本，不声称隔离单一纯梯度原因。L标签额外用于一次固定库构造，没有新增标签患者，但标签使用量不与A/B逐项相同。单seed、每域4开发患者、3推断不是3训练seed；BMC未适应，不是CL结果。失败不扩seed/k/网格/特征层/坐标/权重/增强网格。', '', f"新增400正式更新、1真实预检反向、6训练内遥测反向、0额外Adam；一次351骨干前向建库、0新增语义后验预测；全根{cost['root_bytes']}字节，历时{cost['elapsed_seconds']:.2f}秒。旧teacher与A/B来源成本保留；新目标缓存成本计入本run，未重评分旧格。",'', '[冻结协议](PREREG_ZH.md)；[全指标/遥测](ANALYSIS.json)；[执行摘要](EXECUTION_SUMMARY.json)。']
    # Post hoc descriptive precision/recall from already sealed confusion counts.
    pools={'A':x['controls_v25']['patients']['A'],'B':x['controls_v25']['patients']['B'],'N':x['patients']['N']}
    confusion={}
    for arm,ds in pools.items():
        confusion[arm]={}
        for d,rows in ds.items():
            summaries=[]
            for row in rows:
                metrics=[]
                for rep in row['metrics']:
                    c=rep['classes'][0]
                    metrics.append(dict(precision=c['TP']/max(1,c['TP']+c['FP']),recall=c['TP']/max(1,c['TP']+c['FN']),predicted_over_GT=c['predicted']/max(1,c['GT'])))
                summaries.append({k:mean(v[k] for v in metrics) for k in metrics[0]})
            confusion[arm][d]={k:mean(v[k] for v in summaries) for k in summaries[0]}
    a['posthoc_sealed_confusion_macro_means']=confusion
    lines+=['','补充描述（只读封存TP/FP/FN，患者内先平均3推断再平均4患者，不作新增门槛）：','', '| 臂 | 域 | Precision % | Recall % | 预测/GT体素比 |','|---|---|---:|---:|---:|']
    for arm,ds in confusion.items():
        for d,m in ds.items():lines.append(f"| {arm} | {d} | {100*m['precision']:.4f} | {100*m['recall']:.4f} | {m['predicted_over_GT']:.5f} |")
    lines += ['', '## 实测分析与后继取舍\n\nN−A源+0.49376pp、迁移+0.15431pp，N−B源+0.49799pp、迁移+0.27188pp。两个原门槛都因源未达1pp而FAILED，其他保护PASS、空预测0。N相对A源3/4、迁移2/4患者正，相对B源4/4、迁移3/4正；这支持在本条件下标签来源影响结果，不构成稳定可行性或独立泛化。N相对入口L1600迁移仍−0.83799pp，不能隐藏这部分绝对损失。\n\nN相对A的NLL均值源−0.017851、迁移−0.0313035，Brier源−0.0050006、迁移−0.0083777；概率误差改善比Dice差异更明显，但没有校准评估或独立测试，不能称可靠不确定性。封存混淆计数显示N源precision/recall均值较A略升，迁移precision略升而recall略降，预测/GT体素比两域下降。不能仅用近邻缓存前景少断言其质量更高；也不能把概率指标改善全部归因于减少假阳性。\n\nN起点q均值U/基础梯度范数0.612、余弦+0.012；中点1.040/−0.751，末点0.534/−0.834。骨干后两点余弦也为负，c/R三点均负，但N仍较A/B有平均Dice改善。这不能证明冲突是唯一原因，也不支持一概增大或减小U权重。基础梯度含原L任务、image、anchor和KL，不等于纯监督任务；三个点不能代表全程，实际Adam位移不是U反事实。\n\n本固定近邻配置按原FAILED关闭，不扩seed/k/网格/坐标/特征层/权重/增强。下一独立实验保持同一缓存、数据、损失、初始Adam和终点，仅改变基础目标与U梯度的合并规则：当全参数U梯度与基础梯度内积为负时，把U投影到基础梯度的正交平面，基础梯度不动。比较P−N检验该更新规则、P−A检验整套U净收益；均保留原门槛，不能用局部补偿代替整体成功。该非对称投影借鉴[PCGrad](https://arxiv.org/abs/2001.06782)，并非原论文标准对称随机任务实现；向量范数也会改变，不是纯方向消融。欧氏原梯度的一阶内积性质不保证带历史动量/自适应缩放的Adam实际步长、有限步损失或泛化改善，也不是原ELBO梯度或精确后验更新。无需另做P0，直接冻结独立协议并训练检验。\n']
    for n,v in [('ANALYSIS.json',a),('RESULTS.json',x['RESULTS.json']),('EXECUTION_SUMMARY.json',cost)]:(out/n).write_text(json.dumps(v,indent=2)+'\n')
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:v['gate'] for k,v in a['effects'].items()}))

if __name__=='__main__':main()
