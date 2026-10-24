"""Sealed multiscale-recognition comparison and frozen continuation rule."""
import argparse,json
from pathlib import Path
from statistics import mean
from ..unet_transfer_v45.summarize import patients,p10,DOMAINS,METRICS

CONTRASTS={**{f'{a}_minus_{b}_{stage}':(a+'_'+stage,b+'_'+stage) for a,b in [('W','S'),('W','U'),('S','U')] for stage in ('source','target')},**{a+'_adaptation':(a+'_target',a+'_source') for a in ('W','S','U')}}


def analyze(x):
    assert x['STATUS.json']['stage']=='COMPLETE' and x['EXIT.json']['exit_code']==0
    assert x['RESULTS.json']['new_formal_updates']==4800 and x['RESULTS.json']['counter']['outer_backward_calls']==4800
    assert len(x['RESULTS.json']['cells'])==8 and all(v['complete'] and v['patients']==4 for v in x['RESULTS.json']['cells'].values())
    assert x['ledger']['updates_reserved']==4800 and len(x['ledger']['training_sessions'])==4
    assert all(v['status']=='PASS' for v in x['ledger']['attempts'])
    assert all(v['status']=='COMPLETE' and v['completed_steps']==(1600 if k.endswith('source') else 800) for k,v in x['ledger']['training_sessions'].items())
    assert x['PREFLIGHT.json']['status']=='PASS' and x['RESULTS.json']['old_memory_and_Sb_unchanged']
    assert all(x['BRANCH_S.json'][k]==x['BRANCH_W.json'][k] for k in ('q','prior'))
    u=x['control_v45'];assert u['STATUS.json']['stage']=='COMPLETE' and u['EXIT.json']['exit_code']==0
    raw=dict(x['patients'],U_source=u['patients']['U1600'],U_target=u['patients']['U_B800'])
    pp={a:{d:patients(rows,1 if a.startswith('U') else 3) for d,rows in ds.items()} for a,ds in raw.items()}
    means={a:{d:{k:sum(v[k] for v in rows.values()) if k=='empty_predictions' else mean(v[k] for v in rows.values()) for k in METRICS} for d,rows in ds.items()} for a,ds in pp.items()}
    tails={a:{d:dict(min_Dice=min(v['Dice'] for v in rows.values()),p10_Dice=p10([v['Dice'] for v in rows.values()])) for d,rows in ds.items()} for a,ds in pp.items()};effects={}
    for name,(a,b) in CONTRASTS.items():
        effects[name]={}
        for d in DOMAINS:
            assert pp[a][d].keys()==pp[b][d].keys();agg={}
            for k in METRICS:
                v=[pp[a][d][i][k]-pp[b][d][i][k] for i in pp[a][d]]
                agg[k]=dict(mean=mean(v),minimum=min(v),maximum=max(v),p10=p10(v),positive=sum(t>1e-12 for t in v),negative=sum(t< -1e-12 for t in v),n=4)
            effects[name][d]=dict(metrics={k:means[a][d][k]-means[b][d][k] for k in METRICS},patients=agg)
    checks=dict(target_W_minus_S_ge_1pp=100*effects['W_minus_S_target']['BMC']['metrics']['Dice']>=1,W_target_gain_ge_0_5pp=100*effects['W_adaptation']['BMC']['metrics']['Dice']>=.5,W_old_drop_le_0_5pp=100*effects['W_adaptation']['RUNMC']['metrics']['Dice']>=-.5,source_transfer_W_minus_S_ge_minus_0_5pp=100*effects['W_minus_S_source']['BMC']['metrics']['Dice']>=-.5,no_new_empty=all(means['W_'+stage][d]['empty_predictions']<=means['S_'+stage][d]['empty_predictions'] for stage in ('source','target') for d in DOMAINS),HD95_worsening_le_10pct=all(means['W_'+stage][d]['HD95_penalized_mm']<=1.1*means['S_'+stage][d]['HD95_penalized_mm'] for stage in ('source','target') for d in DOMAINS))
    return dict(status='COMPLETE',means=means,patient_tails=tails,effects=effects,mechanism_gate=dict(status='PASS' if all(checks.values()) else 'FAILED',checks=checks),seed=20261023,training_seeds=1,patients_per_domain=4,training=x['training'],old_models_rescored=0)


def main():
    p=argparse.ArgumentParser();p.add_argument('snapshot');args=p.parse_args();x=json.loads(Path(args.snapshot).read_text());a=analyze(x);out=Path(__file__).parent
    repair=x.get('REPAIR_RECORD.json');cost=dict(status='COMPLETE_WITH_RECEIPT_REPAIR' if repair else 'COMPLETE',original_start=x['LAUNCH.json']['started'],exit_time=x['EXIT.json']['time'],elapsed_seconds=x['EXIT.json']['time']-x['LAUNCH.json']['started'],original_deadline=x['LAUNCH.json']['deadline'],root_bytes=x['root_bytes'],formal_updates=4800,preflight_model_reverse_calls=4,preflight_optimizer_updates=0,serialization_semantic_forwards=16,counter=x['RESULTS.json']['counter'],stage_seconds={v['name']:v['seconds_actual'] for v in x['ledger']['attempts']},repair=repair)
    lines=['# v46 多尺度识别骨干结果','','S为小骨干、W为U-Net多尺度骨干，二者从头完整v23L目标源1600后目标800；U是v45封存标准U-Net。旧v23前400步目标不同，仅历史背景，不作唯一骨干对照。','','| 臂 | 域 | Dice % | NLL | Brier | HD95 mm | 空 | 最差Dice % | p10 % |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ('S_source','W_source','U_source','S_target','W_target','U_target'):
        for d in DOMAINS:
            m=a['means'][arm][d];t=a['patient_tails'][arm][d];lines.append(f"| {arm} | {d} | {100*m['Dice']:.5f} | {m['NLL']:.7f} | {m['Brier']:.7f} | {m['HD95_penalized_mm']:.5f} | {m['empty_predictions']:.3f} | {100*t['min_Dice']:.5f} | {100*t['p10_Dice']:.5f} |")
    lines+=['','| 对比 | 域 | Dice pp | 患者差值范围 pp | 正/负 | NLL变化 | Brier变化 | HD95变化 |','|---|---|---:|---:|---:|---:|---:|---:|']
    for name,ds in a['effects'].items():
        for d,e in ds.items():
            m=e['metrics'];q=e['patients']['Dice'];lines.append(f"| {name} | {d} | {100*m['Dice']:+.5f} | [{100*q['minimum']:+.5f},{100*q['maximum']:+.5f}] | {q['positive']}/{q['negative']} | {m['NLL']:+.7f} | {m['Brier']:+.7f} | {m['HD95_penalized_mm']:+.5f} |")
    lines+=['',f"冻结机制继续标准：**{a['mechanism_gate']['status']}**。",'', '| 标准 | 判定 |','|---|---|']
    for k,v in a['mechanism_gate']['checks'].items():lines.append(f"| {k} | {'PASS' if v else 'FAILED'} |")
    lines+=['','| 阶段 | 实际秒（含I/O） |','|---|---:|']
    for k,v in cost['stage_seconds'].items():lines.append(f'| {k} | {v:.2f} |')
    lines+=['',f"原始启动至最终退出{cost['elapsed_seconds']:.2f}秒，根{cost['root_bytes']}字节；4800正式反向/Adam、4预检模型反向0Adam，额外16语义前向核验序列化，元数据修正检查0前向/反向/Adam。目标预检使用未训练源副本，正式目标结果由各实际源1600初始化。", '', '若有封存计步检查修正，原EXIT1与错误保留；只规范化inactive steps元数据，W源训练/评价/预检不重复，counter/ledger/原硬截止保持。具体已发生记录见执行JSON，不将修正过程写成首次无失败。', '', '单训练seed、每域4反复开发患者、3推断非3训练seed，p10仅小样本描述。不证明独立泛化、精确后验、可靠校准、普遍遗忘保护；无U目标不评价SSL增益。骨干同时改变深度/归一化/多尺度/容量，不能独立归因单项。所有原门槛与失败保留，不扩宽度/学习率/权重/seed网格；L-only或胜过U-Net不作后继U机制永久前提。', '', '[冻结协议](PREREG_ZH.md)；[全部匿名数据](ANALYSIS.json)；[成本与修正记录](EXECUTION_SUMMARY.json)。']
    for n,v in [('ANALYSIS.json',a),('RESULTS.json',x['RESULTS.json']),('EXECUTION_SUMMARY.json',cost)]:(out/n).write_text(json.dumps(v,indent=2)+'\n')
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');print(json.dumps(a['mechanism_gate']))

if __name__=='__main__':main()
