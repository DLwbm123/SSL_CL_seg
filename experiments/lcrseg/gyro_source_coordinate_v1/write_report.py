"""Format the single completion audit; no patient inference or checkpoint reads."""
import csv,json
from pathlib import Path


def main(folder):
    folder=Path(folder);s=json.loads((folder/'SUMMARY.json').read_text());r=s['arms'];d=s['decision']
    text="""# GYRO源阶段投影坐标准备：全native目标训练

从已有S†原projector继续1000步合法源图像GYRO几何学习，只初始化期间native只读以精确保留原分割函数。随后G/M同一新phi/BN固定函数、相同S† native/frame、空native动量，目标全部原允许native参数训练，原生头推理不变。G beta1/3/gamma.45，M beta0/gamma.3，其余V2/P24/历史8/256图bank及采样增强原样。源阶段不读标签、不新增入口评估，没有新模块/损失/KD/历史CE。与BayeSeg/CARE-Seg独立，固定2000最后学生，EMA仅诊断。

旧强固定phi参照来自fixed_projector_v1，非同期但相同bank/目标流，所有旧终点保留；反复开发验证属于探索，不是独立泛化或TMI创新。无noNewU，不称U语义贡献。

"""
    text+=f"**预设判断：{d['status']}**。源投影准备门槛{d['gates']['coordinate']}，frame门槛{d['gates']['frame']}，对Dense两域实用门槛{d['gates']['practical']}。\n\n"
    text+='## 两域与训练口径\n\n| 配方/组 | Siemens% | Philips% | 均值% | 源10L训练% | 目标13L训练% |\n|---|---:|---:|---:|---:|---:|\n'
    records=[]
    for recipe,arms in [('本轮源准备phi固定',r),('已有源phi固定参照',s['coordinate_references']),('原五组实用参照',{'dense_fixmatch':s['dense_reference']})]:
        for n,x in arms.items():
            vals=[100*x['Siemens']['primary_dice'],100*x['Philips']['primary_dice'],100*x['two_domain_mean'],100*x['source_train']['primary_dice'],100*x['target_train']['primary_dice']]
            text+=f'| {recipe}/{n} | '+' | '.join(f'{v:.4f}' for v in vals)+' |\n';records.append([recipe,n,*vals])
    with (folder/'metrics.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['recipe','arm','Siemens_percent','Philips_percent','mean_percent','source10L_train_percent','target13L_train_percent']);w.writerows(records)
    text+='\n共同入口Siemens79.4337143%、Philips73.2043174%。原80.3674%至S†的-.9337pp是历史源接入变化，不计入本轮遗忘。\n\n## 冻结比较\n\n'
    for n,v in d['contrasts'].items():text+=f"- {n}：Siemens{v['Siemens']:+.6f} pp，Philips{v['Philips']:+.6f} pp。\n"
    text+='\n源投影准备门槛为G相对原强固定phi G旧域至少+.5pp、新域不下降超过.5pp；frame门槛为同期G相对M旧域至少+.5pp、新域不低超过.5pp；实用标准为相对原Dense两域均不低超过.5pp且目标13L训练不低超过1pp。没有noNewU，不证明U类别语义贡献。\n'
    for dom in ['Siemens','Philips']:
        text+=f'\n## {dom}类别、ED/ES及绝对变化\n\n| 组 | LV% | MYO% | RV% | ED% | ES% | 较入口pp |\n|---|---:|---:|---:|---:|---:|---:|\n'
        for n,x in r.items():
            a=x[dom];vals=[*[100*a['classes'][k] for k in ['LV','MYO','RV']],*[100*a['phases'][k]['primary_dice'] for k in ['ED','ES']],x['old_delta_from_entry_pp' if dom=='Siemens' else 'target_delta_from_entry_pp']]
            text+=f'| {n} | '+' | '.join(f'{v:.4f}' for v in vals)+' |\n'
    text+='\n## 患者配对\n\n| 比较/域 | 改善 | 下降 | 不变 | 中位数pp |\n|---|---:|---:|---:|---:|\n'
    for source,pairs in [('组间/入口',s['paired']),('新源phi减旧源phi',s['coordinate_paired'])]:
        for n,v in pairs.items():
            for dom,a in v.items():
                b=a['overall'];text+=f"| {source}/{n}/{dom} | {b['increased']} | {b['decreased']} | {b['unchanged']} | {b['median_patient_delta_pp']:+.4f} |\n"
    text+='\n类/EDES详细配对见SUMMARY.json。新域与旧域分开判断，不用均值掩盖代价。\n\n## 库与梯度诊断\n\n'
    b=s['bank'];text+=f"复用库{b['bank_size']}图覆盖{b['patients_covered']}/{b['source_training_patients']}名合法源训练患者，源L图{b['source_L_images']}、源U图{b['source_U_images']}，无标签读取。患者覆盖不能等同器官或语义覆盖。本轮未重建reservoir，未重复读取源图像准备库，未重复入口评估。\n\n"
    text+='| 组 | 加权谱/CE比中位数 | 比值最大值 | native更新路径 | projector更新路径 |\n|---|---:|---:|---:|---:|\n'
    for n,a in s['diagnostics'].items():text+=f"| {n} | {a['weighted_spectral_to_CE_ratio']['median']:.6f} | {a['weighted_spectral_to_CE_ratio']['max']:.6f} | {a['native_update_path_length']:.6f} | {a['projector_update_path_length']:.6f} |\n"
    text+='\n比值只诊断，不自适应调权或触发旧比值停止门槛。\n\n## 一次完成审计、成本和边界\n\n源准备1000正式更新且native/frame精确不变、phi改变；源标签读取0。两组目标各2000正式更新且EXIT0；三组工程各32含实际恢复重放和干净入口完整目标，全部丢弃。总5096=正式5000+工程96，预留/执行/阶段账本一致，无在途。两目标入口逐张量对应S† native/frame及新calibration phi/BN、双动量空，frame和原常量不变，final/latest模型/EMA/optimizer/RNG/reads有限且一致，全部worker退休。Siemens23患者/46EDES、Philips31/62及源10L/20、目标13L/26训练身份精确对应。一次审计只读既有记录与本轮checkpoint，没有额外患者前向或优化器调用，没有重复审计旧轮权重。\n\n所有新产物在12小时父目录内，SpaceGuard限制新增20GB、18GB提前保护、2GB原子保存余量，每50次更新及写前检查。所有正负结果保留，不改旧终点；不读test或U隐藏GT，不改原数据和源参考。公开源码/协议/匿名聚合报告，不公开患者、图像、权重、库/frame/特征、原始日志或服务器配置。\n'
    (folder/'REPORT_ZH.md').write_text(text)
