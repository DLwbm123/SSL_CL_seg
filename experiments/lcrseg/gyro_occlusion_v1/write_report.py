"""Format the single completion audit; no patient inference or checkpoint reads."""
import csv,json
from pathlib import Path


def main(folder):
    folder=Path(folder);s=json.loads((folder/'SUMMARY.json').read_text());r=s['arms'];d=s['decision']
    text="""# GYRO目标几何遮挡消融：全native目标训练

仅取消目标几何强视图的Cutout写入。保留光度增强所有类型/强度/符号和包括遮挡中心在内的全部RNG抽取，确保逐图光度操作与旧轮匹配。复用已完成source_coordinate_v1的phi1000最终状态和S† native/frame组成相同入口；不再源准备，不续跑旧目标模型。G beta1/3/gamma.45，M beta0/gamma.3，nativeLR.001、V2/P24/历史8/256库与全部流/增强其余配置不变。phi整个函数固定，目标native全部原允许参数训练，原头推理不变。

旧参照为source_coordinate_v1带Cutout的G/M，非同期但相同入口/数据流；同期无Cutout G/M公平匹配。无新增模块、损失、KD/历史CE/伪类参照/读出绑定，与BayeSeg/CARE独立。U语义仍由当前L和共享编码器间接建立，没有noNewU不能证明U语义贡献。固定最后2000学生，EMA仅诊断；重复开发验证属于探索非独立泛化，不是TMI创新证明。\n\n"""
    text+=f"**预设判断：{d['status']}**。目标遮挡消融门槛{d['gates']['augmentation']}，frame门槛{d['gates']['frame']}，对Dense两域实用门槛{d['gates']['practical']}。\n\n"
    text+='## 两域与训练口径\n\n| 配方/组 | Siemens% | Philips% | 均值% | 源10L训练% | 目标13L训练% |\n|---|---:|---:|---:|---:|---:|\n'
    records=[]
    for recipe,arms in [('本轮目标无Cutout',r),('同入口有Cutout参照',s['augmentation_references']),('原五组实用参照',{'dense_fixmatch':s['dense_reference']})]:
        for n,x in arms.items():
            vals=[100*x['Siemens']['primary_dice'],100*x['Philips']['primary_dice'],100*x['two_domain_mean'],100*x['source_train']['primary_dice'],100*x['target_train']['primary_dice']]
            text+=f'| {recipe}/{n} | '+' | '.join(f'{v:.4f}' for v in vals)+' |\n';records.append([recipe,n,*vals])
    with (folder/'metrics.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['recipe','arm','Siemens_percent','Philips_percent','mean_percent','source10L_train_percent','target13L_train_percent']);w.writerows(records)
    text+='\n共同入口Siemens79.4337143%、Philips73.2043174%。原80.3674%至S†的-.9337pp是历史源接入变化，不计入本轮遗忘。\n\n## 冻结比较\n\n'
    for n,v in d['contrasts'].items():text+=f"- {n}：Siemens{v['Siemens']:+.6f} pp，Philips{v['Philips']:+.6f} pp。\n"
    text+='\naugmentation门槛为G无Cutout相对G有Cutout目标13L训练至少+.5pp、Philips至少+.3pp且Siemens不下降超过.5pp；frame门槛为同期G-M旧域至少+.5pp且新域不低超过.5pp；实用门槛为相对原Dense两域均不低超过.5pp、目标训练不低超过1pp。\n'
    for dom in ['Siemens','Philips']:
        text+=f'\n## {dom}类别、ED/ES及绝对变化\n\n| 组 | LV% | MYO% | RV% | ED% | ES% | 较入口pp |\n|---|---:|---:|---:|---:|---:|---:|\n'
        for n,x in r.items():
            a=x[dom];vals=[*[100*a['classes'][k] for k in ['LV','MYO','RV']],*[100*a['phases'][k]['primary_dice'] for k in ['ED','ES']],x['old_delta_from_entry_pp' if dom=='Siemens' else 'target_delta_from_entry_pp']]
            text+=f'| {n} | '+' | '.join(f'{v:.4f}' for v in vals)+' |\n'
    text+='\n## 患者配对\n\n| 比较/域 | 改善 | 下降 | 不变 | 中位数pp |\n|---|---:|---:|---:|---:|\n'
    for source,pairs in [('组间/入口',s['paired']),('无Cutout减有Cutout',s['augmentation_paired'])]:
        for n,v in pairs.items():
            for dom,a in v.items():
                b=a['overall'];text+=f"| {source}/{n}/{dom} | {b['increased']} | {b['decreased']} | {b['unchanged']} | {b['median_patient_delta_pp']:+.4f} |\n"
    text+='\n类/EDES详细配对见SUMMARY.json。新域与旧域分开判断，不用均值掩盖代价。\n\n## 库与梯度诊断\n\n'
    b=s['bank'];text+=f"复用库{b['bank_size']}图覆盖{b['patients_covered']}/{b['source_training_patients']}名合法源训练患者，源L图{b['source_L_images']}、源U图{b['source_U_images']}，无标签读取。患者覆盖不能等同器官或语义覆盖。本轮未重建reservoir，未重复读取源图像准备库，未重复入口评估。\n\n"
    text+='| 组 | 加权谱/CE比中位数 | 比值最大值 | native更新路径 | projector更新路径 |\n|---|---:|---:|---:|---:|\n'
    for n,a in s['diagnostics'].items():text+=f"| {n} | {a['weighted_spectral_to_CE_ratio']['median']:.6f} | {a['weighted_spectral_to_CE_ratio']['max']:.6f} | {a['native_update_path_length']:.6f} | {a['projector_update_path_length']:.6f} |\n"
    text+='\n比值只诊断，不自适应调权或触发旧比值停止门槛。每步Cutout=False与光度RNG合同核对；零更新合成验证legacy开启时逐值等于原增强，取消后mask外像素精确相同。\n\n## 一次完成审计、成本和边界\n\n两组目标各2000正式更新且EXIT0；工程各32含实际恢复重放和干净入口完整目标，全部丢弃。总4064=正式4000+工程64，预留/执行/阶段账本一致，无在途。两目标入口逐张量对应S† native/frame及只读既有source_coordinate phi/BN、双动量空，frame和原常量不变，final/latest模型/EMA/optimizer/RNG/reads有限且一致，全部worker退休。Siemens23患者/46EDES、Philips31/62及源10L/20、目标13L/26训练身份精确对应。一次审计只读既有记录与本轮checkpoint，没有额外患者前向或优化器调用，没有重复审计旧轮权重。\n\n所有新产物在12小时父目录内，SpaceGuard限制新增20GB、18GB提前保护、2GB原子保存余量，每50次更新及写前检查。所有正负结果保留，不改旧终点；不读test或U隐藏GT，不改原数据和源参考。公开源码/协议/匿名聚合报告，不公开患者、图像、权重、库/frame/特征、原始日志或服务器配置。\n'
    (folder/'REPORT_ZH.md').write_text(text)
