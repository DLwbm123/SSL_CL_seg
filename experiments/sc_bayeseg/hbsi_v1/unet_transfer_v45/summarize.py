"""Anonymous paired comparison from sealed, fixed-budget endpoints."""
import argparse,json
from pathlib import Path
from statistics import mean
from ..trainability_v21.summarize import METRICS

DOMAINS=('RUNMC','BMC')
CONTRASTS={'M1600_minus_U1600':('M1600','U1600'),'M2000_minus_U2000':('M2000','U2000'),'M_B800_minus_U_B800':('M_B800','U_B800'),'M_adaptation_minus_arrival':('M_B800','M1600'),'U_adaptation_minus_arrival':('U_B800','U1600'),'M_source_2000_minus_1600':('M2000','M1600'),'U_source_2000_minus_1600':('U2000','U1600')}


def patients(records,repeats):
    assert len(records)==4;out={}
    for r in records:
        assert len(r['metrics'])==repeats;vs=[]
        for m in r['metrics']:
            c=m['classes'][0];vs.append(dict(Dice=c['Dice'],NLL=m['NLL'],Brier=m['Brier'],HD95_penalized_mm=c['HD95_penalized_mm'],empty_predictions=int(c['empty_prediction_failure'])))
        out[r['patient']]={k:mean(v[k] for v in vs) for k in METRICS}
    assert len(out)==4;return out


def p10(values):
    v=sorted(values);position=.1*(len(v)-1);lo=int(position)
    return v[lo]+(position-lo)*(v[min(lo+1,len(v)-1)]-v[lo])


def analyze(x):
    for s in (x,x['v23'],x['v25']):assert s['STATUS.json']['stage']=='COMPLETE' and s['EXIT.json']['exit_code']==0
    r=x['RESULTS.json'];assert r['new_formal_updates']==3600 and r['unet_updates']==2800 and r['method_updates']==800
    assert len(r['cells'])==8 and all(v['complete'] and v['patients']==4 for v in r['cells'].values())
    assert x['ledger']['updates_reserved']==3600 and all(v['status']=='PASS' for v in x['ledger']['attempts'])
    assert r['method_counter']['outer_backward_calls']==800 and r['old_memory_and_Sb_unchanged'] and x['PREFLIGHT.json']['status']=='PASS'
    assert all(x['training'][a]['rows']==n for a,n in dict(U1600=1600,U2000=400,U_B800=800,M_B800=800).items())
    raw=dict(x['patients'],M1600=x['v23']['patients']['L'],M2000=x['v25']['patients']['A'])
    pp={a:{d:patients(v,1 if a.startswith('U') else 3) for d,v in ds.items()} for a,ds in raw.items()}
    means={a:{d:{k:sum(v[k] for v in rows.values()) if k=='empty_predictions' else mean(v[k] for v in rows.values()) for k in METRICS} for d,rows in ds.items()} for a,ds in pp.items()}
    tails={a:{d:dict(patient_Dice_min=min(v['Dice'] for v in rows.values()),patient_Dice_p10=p10([v['Dice'] for v in rows.values()]),n=4) for d,rows in ds.items()} for a,ds in pp.items()}
    effects={}
    for name,(a,b) in CONTRASTS.items():
        effects[name]={}
        for d in DOMAINS:
            assert pp[a][d].keys()==pp[b][d].keys();agg={}
            for k in METRICS:
                v=[pp[a][d][i][k]-pp[b][d][i][k] for i in pp[a][d]]
                agg[k]=dict(mean=mean(v),minimum=min(v),maximum=max(v),p10=p10(v),positive=sum(t>1e-12 for t in v),negative=sum(t< -1e-12 for t in v),n=4)
            effects[name][d]=dict(metrics={k:means[a][d][k]-means[b][d][k] for k in METRICS},patients=agg)
    return dict(status='COMPLETE',training_seeds=1,seed=20261023,patients_per_domain=4,means=means,patient_tails=tails,effects=effects,training=x['training'],inference_repeats={'U':1,'M':3},no_old_models_rescored=True,standard_method_comparison_not_single_factor=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('snapshot');args=p.parse_args();x=json.loads(Path(args.snapshot).read_text());a=analyze(x);out=Path(__file__).parent
    cost=dict(status='COMPLETE',original_start=x['LAUNCH.json']['started'],hard_deadline=x['LAUNCH.json']['deadline'],exit_time=x['EXIT.json']['time'],elapsed_seconds=x['EXIT.json']['time']-x['LAUNCH.json']['started'],root_bytes=x['root_bytes'],new_formal_updates=3600,unet_updates=2800,method_updates=800,preflight_model_reverse_calls=3,preflight_updates=0,stage_seconds={v['name']:v['seconds_actual'] for v in x['ledger']['attempts']},method_counter=x['RESULTS.json']['method_counter'],preflight=x['PREFLIGHT.json'])
    lines=['# v45 同数据、同更新预算：当前方法与标准 U-Net','','M为完整v23L目标的结构概率方法，U为固定标准U-Net。源1600、源2000、源1600后BMC800分别比较，禁止跨预算混比。M1600/M2000指标复用封存记录，其源训练成本均保留；新训练为U2800+M目标800。','','| 臂 | 域 | Dice % | NLL | Brier | HD95 mm | 空预测 | 最差患者Dice % | 患者p10 % |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ('U1600','M1600','U2000','M2000','U_B800','M_B800'):
        for d in DOMAINS:
            m=a['means'][arm][d];t=a['patient_tails'][arm][d]
            lines.append(f"| {arm} | {d} | {100*m['Dice']:.5f} | {m['NLL']:.7f} | {m['Brier']:.7f} | {m['HD95_penalized_mm']:.5f} | {m['empty_predictions']:.3f} | {100*t['patient_Dice_min']:.5f} | {100*t['patient_Dice_p10']:.5f} |")
    lines+=['','| 对比 | 域 | Dice pp | 患者差值范围 pp | 正/负 | NLL变化 | Brier变化 | HD95变化 |','|---|---|---:|---:|---:|---:|---:|---:|']
    for name,ds in a['effects'].items():
        for d,e in ds.items():
            m=e['metrics'];q=e['patients']['Dice'];lines.append(f"| {name} | {d} | {100*m['Dice']:+.5f} | [{100*q['minimum']:+.5f},{100*q['maximum']:+.5f}] | {q['positive']}/{q['negative']} | {m['NLL']:+.7f} | {m['Brier']:+.7f} | {m['HD95_penalized_mm']:+.5f} |")
    lines+=['','适应相对到达的RUNMC差为负表示旧域遗忘，BMC差为正表示新域净学习收益；必须分别报告，不能以两域间相减代表遗忘。p10为4患者样本的线性插值描述，不是总体尾部分位估计。','','| 新运行阶段（含I/O） | 实际秒 |','|---|---:|']
    for stage,seconds in cost['stage_seconds'].items():lines.append(f'| {stage} | {seconds:.2f} |')
    lines+=['',f"本次共{cost['elapsed_seconds']:.2f}秒、根{cost['root_bytes']}字节；3600正式反向/Adam、3真实目标预检反向/0Adam，另1个CPU等价CE值检查，无额外科学梯度探针。方法目标800的采样/反向计数见执行JSON，预检采样计数包含在累计counter中，不把它当免费计算。",'', '源相同4L、相同切片序列、相同更新步数/Adam学习率/float64/预处理/无增强；目标相同BMC4L、同800更新、都freshAdam。源2000分支保留源Adam。U普通CE+Dice与M类别平衡CE+Dice及概率项不同，网络容量也不同，属于标准整方法比较，不能归因单个概率机制。M最早400步为v19旧目标的实际历史，其后1200步为v23目标，不能宣称从头同目标训练。', '', '匹配的是更新与标注曝光预算，不是FLOPs或墙钟。U确定性一次，M每患者3次8×4MC；一次训练seed和反复使用的4患者开发集不支持普遍优越、显著性、等效、非劣或可靠不确定性。未新增U目标，不能将任何M−U差值当作SSL增益。旧域图像只评价，无test/U隐藏标签访问。', '', '[冻结协议](PREREG_ZH.md)；[全部匿名指标与患者差值](ANALYSIS.json)；[实际成本](EXECUTION_SUMMARY.json)。']
    for name,v in [('ANALYSIS.json',a),('RESULTS.json',x['RESULTS.json']),('EXECUTION_SUMMARY.json',cost)]:(out/name).write_text(json.dumps(v,indent=2)+'\n')
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:{d:e['metrics']['Dice']*100 for d,e in v.items()} for k,v in a['effects'].items()}))

if __name__=='__main__':main()
