"""Format the single completion audit; no patient inference or checkpoint reads."""
import csv,json
from pathlib import Path


def main(folder):
    folder=Path(folder);s=json.loads((folder/'SUMMARY.json').read_text());r=s['arms'];d=s['decision']
    text='''# 原GYRO谱权重0.3：同256图库的匹配研究

本轮在用户12小时自主授权内启动，协议e5a2cc0f及实现eb4346f5在优化器调用前公开并匿名访问验证。与BayeSeg/CARE-Seg独立；Siemens→Philips，不进入后续域。固定2000步最后学生是主要终点，EMA仅诊断。原五组和强度实验终点不变。

唯一配置变量为gamma0.1提高到0.3，复用同一256图reservoir、每步仍回放8图、原GYRO beta1/3及无锚M beta0；两组共同S†、空动量、全原生允许参数训练、同原生头/当前采样/增强/学习率。没有历史标签CE或KD。图库和采样身份保持一致，gamma0.1参照来自已完成的非同期容量实验；相同验证集反复开发属于探索，不是独立泛化或新TMI贡献证明。

'''
    text+=f"**预设判断：{d['status']}**。谱强度门槛{d['gates']['strength']}，frame门槛{d['gates']['frame']}，对Dense两域实用门槛{d['gates']['practical']}。\n\n"
    text+='## 两域与训练口径\n\n| 配方/组 | Siemens% | Philips% | 均值% | 源10L训练% | 目标13L训练% |\n|---|---:|---:|---:|---:|---:|\n'
    records=[]
    for recipe,arms in [('本轮gamma0.3',r),('同库gamma0.1既有参照',s['strength_references']),('原五组实用参照',{'dense_fixmatch':s['dense_reference']})]:
        for n,x in arms.items():
            vals=[100*x['Siemens']['primary_dice'],100*x['Philips']['primary_dice'],100*x['two_domain_mean'],100*x['source_train']['primary_dice'],100*x['target_train']['primary_dice']]
            text+=f'| {recipe}/{n} | '+' | '.join(f'{v:.4f}' for v in vals)+' |\n';records.append([recipe,n,*vals])
    with (folder/'metrics.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['recipe','arm','Siemens_percent','Philips_percent','mean_percent','source10L_train_percent','target13L_train_percent']);w.writerows(records)
    text+='\n共同入口Siemens79.4337143%、Philips73.2043174%。原80.3674%至S†的-.9337pp是历史源接入变化，不计入本轮遗忘。\n\n## 冻结比较\n\n'
    for n,v in d['contrasts'].items():text+=f"- {n}：Siemens{v['Siemens']:+.6f} pp，Philips{v['Philips']:+.6f} pp。\n"
    text+='\n谱强度门槛为G0.3相对G0.1旧域至少+.5pp且新域不下降超过.5pp；frame为G0.3相对M0.3同一门槛；实用标准为相对原Dense两域均不低超过.5pp。没有noNewU，不能把本轮谱收益解释为U类别语义贡献。\n'
    for dom in ['Siemens','Philips']:
        text+=f'\n## {dom}类别、ED/ES及绝对变化\n\n| 组 | LV% | MYO% | RV% | ED% | ES% | 较入口pp |\n|---|---:|---:|---:|---:|---:|---:|\n'
        for n,x in r.items():
            a=x[dom];vals=[*[100*a['classes'][k] for k in ['LV','MYO','RV']],*[100*a['phases'][k]['primary_dice'] for k in ['ED','ES']],x['old_delta_from_entry_pp' if dom=='Siemens' else 'target_delta_from_entry_pp']]
            text+=f'| {n} | '+' | '.join(f'{v:.4f}' for v in vals)+' |\n'
    text+='\n## 患者配对\n\n| 比较/域 | 改善 | 下降 | 不变 | 中位数pp |\n|---|---:|---:|---:|---:|\n'
    for source,pairs in [('组间/入口',s['paired']),('gamma0.3减0.1',s['strength_paired'])]:
        for n,v in pairs.items():
            for dom,a in v.items():
                b=a['overall'];text+=f"| {source}/{n}/{dom} | {b['increased']} | {b['decreased']} | {b['unchanged']} | {b['median_patient_delta_pp']:+.4f} |\n"
    text+='\n类/EDES详细配对见SUMMARY.json。新域与旧域分开判断，不用均值掩盖代价。\n\n## 库与梯度诊断\n\n'
    b=s['bank'];text+=f"复用库{b['bank_size']}图覆盖{b['patients_covered']}/{b['source_training_patients']}名合法源训练患者，源L图{b['source_L_images']}、源U图{b['source_U_images']}，无标签读取。患者覆盖不能等同器官或语义覆盖。本轮未重建reservoir，未重复读取源图像准备库，未重复入口评估。\n\n"
    text+='| 组 | 加权谱/CE比中位数 | 比值最大值 | native更新路径 | projector更新路径 |\n|---|---:|---:|---:|---:|\n'
    for n,a in s['diagnostics'].items():text+=f"| {n} | {a['weighted_spectral_to_CE_ratio']['median']:.6f} | {a['weighted_spectral_to_CE_ratio']['max']:.6f} | {a['native_update_path_length']:.6f} | {a['projector_update_path_length']:.6f} |\n"
    text+='\n比值只诊断，不自适应调权或触发旧比值停止门槛。\n\n## 一次完成审计、成本和边界\n\n两组各2000正式更新且EXIT0；工程各32含实际恢复重放和干净入口完整目标，全部丢弃。总4064=正式4000+工程64，预留/执行/阶段账本一致，无在途。共同入口逐张量对应、双动量空，frame和原常量不变，final/latest模型/EMA/optimizer/RNG/reads有限且一致，全部worker退休。Siemens23患者/46EDES、Philips31/62及源10L/20、目标13L/26训练身份精确对应。一次审计只读既有记录与本轮checkpoint，没有额外患者前向或优化器调用，没有重复审计旧轮权重。\n\n所有新产物在12小时父目录内，SpaceGuard限制新增20GB、18GB提前保护、2GB原子保存余量，每50次更新及写前检查。所有正负结果保留，不改旧终点；不读test或U隐藏GT，不改原数据和源参考。公开源码/协议/匿名聚合报告，不公开患者、图像、权重、库/frame/特征、原始日志或服务器配置。\n'
    (folder/'REPORT_ZH.md').write_text(text)
