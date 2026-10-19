"""Render anonymous tables from the one audited aggregate summary."""
import csv,json
from pathlib import Path


def main(folder):
    folder=Path(folder);s=json.loads((folder/'SUMMARY.json').read_text());r=s['arms'];d=s['decision']
    text='''# GYRO Siemens→Philips 五组固定终点报告

2026-10-19。冻结协议e096386先经代理公开并匿名HTTP200，再执行五组。共同入口为源v1.2固定69步最后学生S†；原模型80.3674%与S†79.4337%的源接入差异单列，不拼接历史CL结果。与BayeSeg、CARE-Seg独立。

'''
    text+=f"**预设判断：{d['status']}。** U门槛{'通过' if d['gates']['U_benefit'] else '未通过'}，frame保持门槛{'通过' if d['gates']['frame_retention'] else '未通过'}，可塑性门槛{'通过' if d['gates']['plasticity'] else '未通过'}。这是一种子、短程、固定开发验证集的结果，不是独立test或TMI创新证明。\n\n"
    text+='## 整体终点：学生为主，EMA仅诊断\n\n| 组 | Siemens% | Philips% | 双域均值% | 旧域较入口pp | 新域较入口pp | 源10L训练% | 目标13L训练% |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    rows=[]
    for n,x in r.items():
        values=[100*x['Siemens']['primary_dice'],100*x['Philips']['primary_dice'],100*x['two_domain_mean'],x['old_delta_from_entry_pp'],x['target_delta_from_entry_pp'],100*x['source_train']['primary_dice'],100*x['target_train']['primary_dice']]
        text+='| '+n+' | '+' | '.join(f'{v:.4f}' for v in values)+' |\n';rows.append([n,*values])
    with (folder/'metrics.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['arm','Siemens_percent','Philips_percent','two_domain_mean_percent','old_delta_entry_pp','new_delta_entry_pp','source10L_train_percent','target13L_train_percent']);w.writerows(rows)
    text+='\n共同入口Siemens79.4337143%、Philips73.2043174%。主遗忘方向为Siemens入口减终点；较原源80.3674%的源准备损失0.9337pp不归因本轮Philips训练。\n\n'
    text+='## 按冻结标准判断U与frame\n\n| 比较 | Siemens差值pp | Philips差值pp | 冻结要求 |\n|---|---:|---:|---|\n'
    text+=f"| G_U−G_noNewU | {d['U_old_pp']:+.6f} | {d['U_new_pp']:+.6f} | 旧≥−0.5、新≥+0.5 |\n"
    text+=f"| G_U−M_U | {d['frame_old_pp']:+.6f} | {d['frame_new_pp']:+.6f} | 旧≥+0.5、新≥−0.5 |\n"
    old_dense=100*(r['G_U']['Siemens']['primary_dice']-r['dense_fixmatch']['Siemens']['primary_dice'])
    text+=f"| G_U−Dense | {old_dense:+.6f} | {d['practical_new_pp']:+.6f} | 新≥−0.5且目标训练≥−1 |\n"
    text+=f"\nG_U目标13L训练减Dense {d['target_training_vs_Dense_pp']:+.6f}pp；相对共同Philips入口增益{d['G_target_gain_from_entry_pp']:+.6f}pp。\n"
    text+=f"\nU×frame交互 `(G_U−G_noNewU)−(M_U−M_noNewU)`：Siemens {d['interactions_pp']['Siemens']:+.6f}pp、Philips {d['interactions_pp']['Philips']:+.6f}pp。交互为正也不独立证明因果；无正交互不能声称U特别增强抗遗忘。\n"
    text+='\n本实验估计额外Philips U图像替换额外L图像几何输入的贡献，不是完全有/无一切自监督；noNewU仍有源库和源准备。M为同回放量beta0的无锚谱对照，不是历史CE回放，不估计replay本身的独立效应。Dense无回放/projector计算，不是FLOP匹配。\n\n'
    if not d['gates']['U_benefit']:text+='本配方没有达到预先规定的U类别任务收益；不能凭谱下降或几何梯度非零称U学会了有用器官语义。\n\n'
    if not d['gates']['frame_retention']:text+='本配方没有达到frame带来旧域保持收益的门槛；不能把普通有标签适应的增益归给阶段frame。\n\n'
    text+='源域相对入口下降必须独立披露：若优于某对照但仍下降，只能称相对减少遗忘，不称绝对保持。双域均值不能抵销单域代价。\n\n'
    for domain in ['Siemens','Philips']:
        text+=f'## {domain}类别与ED/ES\n\n| 组 | LV% | MYO% | RV% | ED% | ES% |\n|---|---:|---:|---:|---:|---:|\n'
        for n,x in r.items():
            a=x[domain];vals=[*[100*a['classes'][k] for k in ['LV','MYO','RV']],*[100*a['phases'][k]['primary_dice'] for k in ['ED','ES']]]
            text+='| '+n+' | '+' | '.join(f'{v:.4f}' for v in vals)+' |\n'
        text+='\n'
    text+='## 相对共同入口的患者配对\n\n| 组/域 | 改善 | 下降 | 不变 | 配对中位数pp |\n|---|---:|---:|---:|---:|\n'
    for n in r:
        for dom in ['Siemens','Philips']:
            a=s['paired'][n+'_minus_entry'][dom]['overall'];text+=f"| {n}/{dom} | {a['increased']} | {a['decreased']} | {a['unchanged']} | {a['median_patient_delta_pp']:+.4f} |\n"
    text+='\n其余组间/类别及ED/ES配对人数和中位数完整保留在SUMMARY.json，方向均为键名B_minus_A所示。训练仅各域原10L或13L，不能合并成同一训练成绩。\n\n'
    text+='## 梯度与更新诊断\n\n| 组 | 加权U/CE梯度比中位数 | 比值最大值 | native累计更新路径L2 | projector累计更新路径L2 |\n|---|---:|---:|---:|---:|\n'
    for n,a in s['diagnostics'].items():
        text+=f"| {n} | {a['weighted_spectral_to_CE_ratio']['median']:.6f} | {a['weighted_spectral_to_CE_ratio']['max']:.6f} | {a['native_update_path_length']:.6f} | {a['projector_update_path_length']:.6f} |\n"
    text+='\nDense该列是FixMatch像素U梯度，不能当作谱项同尺度比较。比值只诊断，无0.25硬停或自适应调权。几何信号较弱/projector吸收是可检验的后续假设，现有结果不能单独归因为该原因；也不能靠扩大损失而宣称机制正确。\n\n'
    text+='## 成本、资格与一次完成审计\n\n五组各2000正式且EXIT0，总正式10000；各32工程含一次实际非空动量恢复重放及原入口完整目标，工程160全部丢弃；实际10160/10160，预留与执行token及各阶段账本一致、无在途。初次依赖路径缺失发生0调用，错误现场保留，不归为科学失败。\n\n所有组同一S†逐张量入口、空动量、同RNG；native完整原允许参数训练，旧sigma.weight/grad_update常量及frame未变。final/latest的model/EMA/optimizer/RNG/reads一致且有限。各组验证Siemens23/46、Philips31/62及源10L/20体、目标13L/26体身份EDES精确匹配，全部worker退出。审计没有新增患者前向/反传/optimizer调用。\n\n原协议、五组原终点和所有正负结果保持，不因后续自主探索覆盖或改判。没有进入后续域。用户新12小时授权允许后续另起改进协议/独立账本，遵守新增20GB保护；这不改变本试点预设结论。公开只有源码、完整协议和匿名聚合结果；图像、身份、权重、库/frame、原始诊断及配置均私有。\n'
    (folder/'REPORT_ZH.md').write_text(text)
