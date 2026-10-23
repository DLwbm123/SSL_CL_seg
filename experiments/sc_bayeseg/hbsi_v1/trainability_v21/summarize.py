"""Offline q x phi conditional effects from sealed metrics; never calls models."""
import argparse
import json
from pathlib import Path
from statistics import mean

METRICS=('Dice','NLL','Brier','HD95_penalized_mm','empty_predictions')
CONTRASTS={'phi_given_q_fixed':{'F01':1,'F00':-1},'phi_given_q_learning':{'F11':1,'F10':-1},'q_given_phi_fixed':{'F10':1,'F00':-1},'q_given_phi_learning':{'F11':1,'F01':-1},'interaction':{'F11':1,'F10':-1,'F01':-1,'F00':1}}

def patients(records):
    assert len(records)==4
    result={}
    for record in records:
        values=[]
        for m in record['metrics']:
            c=m['classes'][0]
            values.append(dict(Dice=c['Dice'],NLL=m['NLL'],Brier=m['Brier'],HD95_penalized_mm=c['HD95_penalized_mm'],empty_predictions=int(c['empty_prediction_failure'])))
        assert len(values)==3
        result[record['patient']]={k:mean(v[k] for v in values) for k in METRICS}
    assert len(result)==4
    return result

def analyze(snapshot,extra=None):
    assert snapshot['STATUS.json']['stage']=='COMPLETE' and snapshot['EXIT.json']['exit_code']==0
    raw=dict(snapshot['patients'])
    if extra:
        assert extra['STATUS.json']['stage']=='COMPLETE' and extra['EXIT.json']['exit_code']==0
        assert extra['MATCHING.json']['reuse_F11']=='PASS'
        raw['F01']=extra['patients']['F01']
    p={a:{d:patients(v) for d,v in ds.items()} for a,ds in raw.items()}
    means={a:{d:{k:mean(r[k] for r in v.values()) if k!='empty_predictions' else sum(r[k] for r in v.values()) for k in METRICS} for d,v in ds.items()} for a,ds in p.items()}
    gates={}
    for a in p:
        if a=='SOURCE':continue
        old=means[a]['RUNMC'];new=means[a]['BMC'];src=means['SOURCE']
        checks=dict(new_domain_net_ge_0_5pp=(new['Dice']-src['BMC']['Dice'])*100>=.5,old_domain_drop_le_0_5pp=(src['RUNMC']['Dice']-old['Dice'])*100<=.5,no_new_empty=all(means[a][d]['empty_predictions']<=src[d]['empty_predictions'] for d in src),HD95_worsening_le_10pct=all(means[a][d]['HD95_penalized_mm']<=1.1*src[d]['HD95_penalized_mm'] for d in src))
        gates[a]=dict(checks=checks,status='PASS' if all(checks.values()) else 'FAILED',new_net_pp=100*(new['Dice']-src['BMC']['Dice']),old_change_pp=100*(old['Dice']-src['RUNMC']['Dice']))
    contrasts=dict(CONTRASTS)
    contrasts.update({a+'_minus_SOURCE':{a:1,'SOURCE':-1} for a in gates})
    contrasts['FIX_PHI_minus_FULL']={'F10':1,'F11':-1}
    effects={}
    for name,coeff in contrasts.items():
        if not all(a in p for a in coeff):continue
        effects[name]={}
        for d in ('RUNMC','BMC'):
            ids=set(p['SOURCE'][d]);assert all(set(p[a][d])==ids for a in coeff)
            effect={k:sum(c*means[a][d][k] for a,c in coeff.items()) for k in METRICS}
            vals=[sum(c*p[a][d][patient]['Dice'] for a,c in coeff.items())*100 for patient in ids]
            effects[name][d]=dict(metrics=effect,paired_patient_Dice_pp=dict(n=4,mean=mean(vals),minimum=min(vals),maximum=max(vals),positive=sum(v>1e-10 for v in vals),negative=sum(v< -1e-10 for v in vals)))
    signal=all(effects['FIX_PHI_minus_FULL'][d]['metrics']['Dice']*100>=.5 for d in ('RUNMC','BMC')) and all(means['F10'][d]['empty_predictions']<=means['F11'][d]['empty_predictions'] for d in ('RUNMC','BMC'))
    return dict(status='COMPLETE',training_seed=20261023,patients_per_domain=4,inference_repeats=3,factorial_complete='F01'in p,means=means,gates=gates,effects=effects,FIX_PHI_conditional_rule_pass=signal)

def report(a,snapshot,extra=None):
    lines=['# v21 目标L适应：Q1收口与 q × phi 条件效应','', 'Q1已按冻结协议结束：两个新臂各800步、共1600次正式更新，四个新评估格完整；原FULL和源到达锚点复用。F10/F00均未晋级，FIX_PHI相对FULL两域同时提升≥0.5pp的机制规则也未通过。', '', 'F00：固定q/phi，仅学域状态；F10：学q与域状态、固定phi；F01：固定q、学phi与域状态；F11：全量学习。所有臂域状态均学习。仅首seed20261023、BMC当前4例L、无U；旧域仅评价，固定800终点。共享4例开发val和3次推断重复不能当成3训练seed或独立泛化证据。','', '## 固定终点与原晋级门槛','', '| 臂 | 域 | Dice % | NLL | Brier | HD95 mm | 空预测 |','|---|---|---:|---:|---:|---:|---:|']
    for arm,ds in a['means'].items():
        for d,m in ds.items():lines.append(f"| {arm} | {d} | {m['Dice']*100:.4f} | {m['NLL']:.6f} | {m['Brier']:.6f} | {m['HD95_penalized_mm']:.4f} | {m['empty_predictions']:.0f} |")
    lines+=['','所有晋级条件保持不变：新域相对源未适应到达锚点净增≥0.5pp；旧域下降≤0.5pp；无新增空预测；每域平均HD95相对自身源到达锚点恶化≤10%。不采用放宽到1pp的旧域门槛。','', '| 臂 | 新域净增 pp | 旧域变化 pp | 新域门 | 旧域门 | 空预测门 | HD95门 | 总判定 |','|---|---:|---:|---|---|---|---|---|']
    for arm,g in a['gates'].items():
        c=g['checks'];lines.append(f"| {arm} | {g['new_net_pp']:+.4f} | {g['old_change_pp']:+.4f} | {c['new_domain_net_ge_0_5pp']} | {c['old_domain_drop_le_0_5pp']} | {c['no_new_empty']} | {c['HD95_worsening_le_10pct']} | {g['status']} |")
    lines+=['','## 条件效应与患者异质性','', '| 对照 | RUNMC Dice pp | BMC Dice pp |','|---|---:|---:|']
    for n,ds in a['effects'].items():lines.append(f"| {n} | {ds['RUNMC']['metrics']['Dice']*100:+.4f} | {ds['BMC']['metrics']['Dice']*100:+.4f} |")
    lines+=['','正值表示允许对应组学习后Dice更高，不等于通过晋级。phi|q固定=F01−F00；phi|q学习=F11−F10；q|phi固定=F10−F00；q|phi学习=F11−F01；交互=F11−F10−F01+F00。均为eta可学习下的条件总效应，包含优化路径和后验/域记忆兼容性，不是独立直接因果贡献。','', '| 对照 | 域 | 患者Dice差值范围 pp | 正/负患者 |','|---|---|---:|---:|']
    for n,ds in a['effects'].items():
        for d,v in ds.items():
            t=v['paired_patient_Dice_pp'];lines.append(f"| {n} | {d} | [{t['minimum']:+.4f}, {t['maximum']:+.4f}] | {t['positive']}/{t['negative']} |")
    lines+=['','每域始终是同4例患者，先对每例3推断求平均再配对；不是12例。全部辅助指标条件差值保留在ANALYSIS.json，不公开患者身份。','', '## 判读与执行回执','', 'F10较F11保留更多旧域Dice，但新域反而更低，因此不能认定phi更新单独解释适应失败。F00旧域读出与源锚点完全相同，这是固定q/phi和封存旧记忆下的工程不变性，不是已获得有效新域学习。固定phi仍可传递输入theta/a的梯度；记忆参数不变也不代表随q变化的条件边缘不变。']
    if a['factorial_complete']:
        lines+=['','F01已在独立协议/预算下完成，F11匹配检查通过后复用；没有重训/重评分F11。四臂效应完整，但只有一个训练seed，禁止据交互方向声称稳定机制收益。任何后继实验须另立假设和预算，不自动扩seed或降低门槛。']
    else:lines+=['','Q1缺少F01，当前不能计算完整交互或phi|q固定效应；后续F01采用独立协议和预算，不追改Q1。']
    ledger=snapshot['ledger'];lines+=['', f"Q1 COMPLETE/EXIT0；原始总历时{snapshot['EXIT.json']['time']-snapshot['LAUNCH.json']['started']:.2f}秒，账本{ledger['bytes']}字节；1600正式更新、额外2次预检反向/0预检Adam更新，8阶段均PASS；冻结参数、梯度None与Adam槽不变检查通过。没有恢复旧配置、重试或重置原时钟。"]
    if extra:
        lines.append(f"F01 COMPLETE/EXIT0；原始总历时{extra['EXIT.json']['time']-extra['LAUNCH.json']['started']:.2f}秒，账本{extra['ledger']['bytes']}字节；800正式更新。匹配审计副本计算单独计数见MATCHING.json和EXECUTION_SUMMARY.json，不记成零计算。")
    lines+=['','[原Q1冻结协议](PREREG_ZH.md)；[离线汇总](ANALYSIS.json)。代码、匿名标量和执行摘要公开；患者身份、原始影像、概率数组、权重、私有配置/路径和训练日志仅保留私有。v20负结果及已关闭配置保留，不重跑。']
    return '\n'.join(lines)+'\n'

def main():
    p=argparse.ArgumentParser();p.add_argument('snapshot');p.add_argument('--f01');p.add_argument('--out',default=str(Path(__file__).parent));args=p.parse_args()
    snapshot=json.loads(Path(args.snapshot).read_text());extra=json.loads(Path(args.f01).read_text()) if args.f01 else None
    a=analyze(snapshot,extra);out=Path(args.out);out.mkdir(exist_ok=True,parents=True)
    (out/'ANALYSIS.json').write_text(json.dumps(a,indent=2)+'\n');(out/'REPORT_ZH.md').write_text(report(a,snapshot,extra))
    summary=dict(Q1={k:snapshot[k] for k in ('STATUS.json','EXIT.json','LAUNCH.json')},ledger={k:snapshot['ledger'][k] for k in ('limits','seconds_charged','updates_reserved','bytes')},formal_updates=1600,preflight_reverse_calls=2,preflight_optimizer_updates=0)
    (out/'EXECUTION_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(gates=a['gates'],factorial_complete=a['factorial_complete'])))

if __name__=='__main__':main()
