import csv,json
from pathlib import Path


def main(folder,reference):
    folder=Path(folder);s=json.loads((folder/'SUMMARY.json').read_text());old=json.loads(Path(reference).read_text());r=s['arms'];d=s['decision']
    s['external_low_strength_references']={k:old['arms'][k] for k in ['G_U','M_U','dense_fixmatch']}
    s['strength_contrasts_pp']={}
    for name in ['G_U','M_U']:
        s['strength_contrasts_pp'][name]={domain:100*(r[name][domain]['primary_dice']-old['arms'][name][domain]['primary_dice']) for domain in ['Siemens','Philips']}
    (folder/'SUMMARY.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
    text='''# GYRO谱权重0.1与纯CE匹配对照：自主改进第1轮

本轮由用户12小时自主授权启动，协议dd1ee8f在优化器调用前公开并匿名HTTP200。原五组gamma0.01结论和成本保持不变。本轮同S†最后学生、同数据/图像库/全模型训练/原生头，gamma提高到0.1；CE_only为同入口当前L2纯CE，不使用历史标签或KD。固定2000步最后学生，EMA仅诊断。

'''
    text+=f"**冻结判断：{d['status']}**；对CE保持门槛{d['gates']['retention_vs_CE']}，对M的frame门槛{d['gates']['frame_vs_M']}，可塑性门槛{d['gates']['plasticity']}。单种子重复开发验证属探索，不是独立test或新颖性证明。\n\n"
    text+='## 两域与训练口径\n\n| 配方/组 | Siemens% | Philips% | 均值% | 源10L训练% | 目标13L训练% |\n|---|---:|---:|---:|---:|---:|\n'
    records=[]
    for label,rs in [('本轮gamma0.1',r),('前轮gamma0.01/参照',s['external_low_strength_references'])]:
        for name,x in rs.items():
            vals=[100*x['Siemens']['primary_dice'],100*x['Philips']['primary_dice'],100*x['two_domain_mean'],100*x['source_train']['primary_dice'],100*x['target_train']['primary_dice']]
            text+=f'| {label}/{name} | '+' | '.join(f'{v:.4f}' for v in vals)+' |\n';records.append([label,name,*vals])
    with (folder/'metrics.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['recipe','arm','Siemens_percent','Philips_percent','mean_percent','source10L_train_percent','target13L_train_percent']);w.writerows(records)
    text+='\n共同入口Siemens79.4337143%/Philips73.2043174%，源与目标验证患者分别23/31，EDES体46/62。原80.3674%到S†的-.9337pp单列，不算本轮遗忘。\n\n## 预设比较和强度变化\n\n'
    for name,v in d['contrasts'].items():text+=f'- {name}: {v:+.6f} pp\n'
    for name,v in s['strength_contrasts_pp'].items():text+=f"- {name}本轮减原gamma0.01：Siemens{v['Siemens']:+.6f} pp，Philips{v['Philips']:+.6f} pp。\n"
    text+='\n对CE和对M均预设旧域至少+.5pp、新域不下降超过.5pp；另对原Dense新域不低超过.5pp、目标13L训练不低超过1pp。强度变化可能影响保持，但必须同时看纯CE和无锚对照，不能把一般几何正则作用归为frame特有贡献。没有noNewU组，不能从本轮声称U类别语义收益；不能把原0.01的U消融拼接到0.1。\n\n'
    for domain in ['Siemens','Philips']:
        text+=f'## {domain}类别、ED/ES与绝对变化\n\n| 组 | LV% | MYO% | RV% | ED% | ES% | 较共同入口pp |\n|---|---:|---:|---:|---:|---:|---:|\n'
        for n,x in r.items():
            a=x[domain];vals=[*[100*a['classes'][k] for k in ['LV','MYO','RV']],*[100*a['phases'][k]['primary_dice'] for k in ['ED','ES']],x['old_delta_from_entry_pp' if domain=='Siemens' else 'target_delta_from_entry_pp']]
            text+='| '+n+' | '+' | '.join(f'{v:.4f}' for v in vals)+' |\n'
    text+='\n## 患者配对（各组相对共同入口）\n\n| 组/域 | 改善 | 下降 | 不变 | 配对中位数pp |\n|---|---:|---:|---:|---:|\n'
    for n in r:
        for dom in ['Siemens','Philips']:
            a=s['paired'][n+'_minus_entry'][dom]['overall'];text+=f"| {n}/{dom} | {a['increased']} | {a['decreased']} | {a['unchanged']} | {a['median_patient_delta_pp']:+.4f} |\n"
    text+='\n其余CE/M/G组间和类/EDES配对统计见SUMMARY.json。两域均值不得掩盖旧域下降，新域提升不等于抗遗忘成立。\n\n## 梯度与更新诊断\n\n| 组 | 几何/CE比中位数 | 比值最大值 | native累计更新路径 | projector累计更新路径 |\n|---|---:|---:|---:|---:|\n'
    for n,a in s['diagnostics'].items():text+=f"| {n} | {a['weighted_spectral_to_CE_ratio']['median']:.6f} | {a['weighted_spectral_to_CE_ratio']['max']:.6f} | {a['native_update_path_length']:.6f} | {a['projector_update_path_length']:.6f} |\n"
    text+='\n梯度比只作诊断，不自动调lambda；CE_only几何梯度为零、projector参数无目标梯度，BN缓冲因诊断前向变化合法，不影响原生分割推理。FLOPs/反传不完全匹配，明确披露。\n\n## 一次完成审计及成本\n\n三组各2000正式EXIT0，正式6000+工程96=6096实际调用；工程包含非空动量恢复重放及原入口完整目标，全部丢弃，预留/执行及各阶段账本一致，无在途。共用入口逐张量对应、双动量空、frame及旧常量不变、final/latest模型/EMA/optimizer/RNG/reads一致且有限；患者及EDES身份集合匹配，全部worker退出。CE_only的projector参数未变，BN缓冲合法更新。审计只读取既有记录，没有新增患者前向/反传/optimizer调用。\n\n空间保护按本窗口新增分配计数，20GB硬上限/18GB提前保护/2GB峰值预留，保存前及每50步检查；不保留每步历史checkpoint，不复制数据或图像库。旧五组10160和源历史成本另列。所有正负结果保留；后续改进另起协议和账本，不改本轮终点。与BayeSeg/CARE-Seg独立，不进入后续域。公开仅源码/完整协议/匿名聚合结果，私有患者/权重/库/frame/日志/配置不公开。\n'
    (folder/'REPORT_ZH.md').write_text(text)
