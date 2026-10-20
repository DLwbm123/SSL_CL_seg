"""CPU-only reporting from the completed, audited aggregate summary; no model/data access."""
import json
from pathlib import Path


def main(directory):
    p = Path(directory)
    x = json.loads((p / 'SPECTRUM_ALIGNMENT_PUBLIC_SUMMARY.json').read_text())
    assert x['audit']['status'] == 'PASS' and x['audit']['workers_retired']
    original_source = .8036740663770262
    analysis = {'source_reference_old_dice': original_source, 'absolute': {}, 'phase_contrasts': {}, 'gradient_summary': {}, 'gates': x['gates'], 'no_new_model_forwards': True}
    r = ['# 两目标流空间谱几何与一致性对照：完整实验报告', '',
         '六组固定2000步最后学生均完成，一次最终审计PASS。G00相对S的复现门槛、相对A-align的比较门槛均在两个新目标流上通过：旧域平均分别增加0.694747和1.453709个百分点，新域平均分别增加0.207224和0.953228个百分点。结果支持在当前冻结配置下优先研究谱几何；不等于证明所有一致性方法无效或已确定唯一因果机制。', '',
         '这是2026-10-20协议及运行的最终交付；报告整理于2026-10-21（北京时间）。旧A/B/C/M/N及此前实验终点均保留，条件后续未启动。', '',
         '## 冻结合同与工程证据', '',
         '两个目标流为20261021/20261022，共享源初始化163、S†入口和源训练的固定P，属于两个目标随机流而非两个完整独立源种子。S只用当前Philips L2监督CE；G00额外使用L池14幅图像（只读图像、不读补充标签），三视图48个encoder图像形成64行、D32/R4的谱几何；A-align使用完全相同图像、增强、位置和P，替换为centroid一致性。无U accessor、generator、code或历史原图训练；frame仅为不参与损失的兼容buffer。本轮真实输入不可学习，不使用输入STE。', '',
         '统一native训练范围、空SGD、LR .001、Nesterov .9、矩阵wd .0005/向量0、global clip1，sigma/grad_update固定。S计算较少；G00/A-align的图像暴露相同，谱计算额外开销仍存在。', '',
         '仅在共同入口用独立固定训练校准流20261024的16批匹配encoder梯度RMS，0更新、768次encoder图像暴露和32次求梯度；λ_A=0.11545419599809577，两流共用且不再校准。G/align梯度平方累计83.7699866668/62.8447501211，能量及权重范围门槛通过，模型和buffers不变。尺度匹配只约束入口，不保证后续梯度相等。', '',
         'S与额外前向但断梯度的S_detach各32更新，逐步native/opt/EMA/RNG精确一致；六组资格各32更新全部通过，工程终态全部丢弃。正式不选best、不补步、不用中间验证改配方。', '',
         '## 固定最后学生与统一EMA', '',
         'Dice为百分数，旧域Siemens、新域Philips；EMA .999仅统一辅助，不替换主终点。', '',
         '|目标流|组|学生旧/新|EMA旧/新|源训练/目标训练（学生）|', '|---|---|---:|---:|---:|']
    for seed, arms in x['metrics'].items():
        analysis['absolute'][seed] = {}
        for arm, m in arms.items():
            e = m['EMA_diagnostic']
            vals = [m['Siemens']['primary_dice']*100, m['Philips']['primary_dice']*100, e['Siemens']['primary_dice']*100, e['Philips']['primary_dice']*100]
            r.append(f"|{seed}|{arm}|{vals[0]:.6f}/{vals[1]:.6f}|{vals[2]:.6f}/{vals[3]:.6f}|{m['source_train']['primary_dice']*100:.6f}/{m['target_train']['primary_dice']*100:.6f}|")
            analysis['absolute'][seed][arm] = {'old_from_entry_pp': m['old_delta_from_entry_pp'], 'new_from_entry_pp': m['target_delta_from_entry_pp'], 'old_from_original_source_pp': (m['Siemens']['primary_dice']-original_source)*100}
    r += ['', '## 配对比较与冻结门槛', '', '所有区间为逐流患者配对bootstrap 10000次、seed163；单位pp。患者重采样不估计训练随机性，也不校正反复开发和多重比较。', '', '|流/口径|比较|旧域变化 [95%区间]|新域变化 [95%区间]|', '|---|---|---:|---:|']
    for seed, modes in x['contrasts'].items():
        for mode, domains in modes.items():
            for contrast in domains['Siemens']:
                cells=[]
                for domain in ['Siemens','Philips']:
                    c=domains[domain][contrast]; lo,hi=c['patient_bootstrap95_pp']
                    cells.append(f"{c['delta_B_minus_A_pp']:+.6f} [{lo:+.6f}, {hi:+.6f}]")
                r.append(f"|{seed}/{mode}|{contrast}|{'|'.join(cells)}|")
    r += ['', '主门槛：G00−S旧域两流平均≥+.5pp、每流旧>0、新各≥−.3pp；G00−A-align旧域平均≥+.3pp、每流旧>0、新各≥−.3pp。两项均PASS；EMA相应比较也均PASS。新域G00−S区间各跨0，不能称新域收益已稳定确认。', '',
          '旧域平均提升并非多数患者均改善。G00−S在20261021只有10/23患者改善、13下降，中位差−.039516pp；20261022为11改善、12下降，中位差−.017843pp。RV提升较大，不能用均值掩盖个体代价。', '',
          '### 类别、患者损益与ED/ES', '', '|流/域|学生比较|LV/MYO/RV变化pp|改善/下降/不变|患者中位差pp|ED/ES变化pp|', '|---|---|---:|---:|---:|---:|']
    for seed, modes in x['contrasts'].items():
        analysis['phase_contrasts'][seed] = {}
        for domain, contrasts in modes['student'].items():
            for contrast,c in contrasts.items():
                b,a=contrast.split('_minus_'); phases={ph:(x['metrics'][seed][b][domain]['phases'][ph]['primary_dice']-x['metrics'][seed][a][domain]['phases'][ph]['primary_dice'])*100 for ph in ['ED','ES']}
                analysis['phase_contrasts'][seed][domain+'_'+contrast]=phases
                classes='/'.join(f'{c["classes_pp"][k]:+.6f}' for k in ['LV','MYO','RV'])
                r.append(f"|{seed}/{domain}|{contrast}|{classes}|{c['increased']}/{c['decreased']}/{c['unchanged']}|{c['median_patient_delta_pp']:+.6f}|{phases['ED']:+.6f}/{phases['ES']:+.6f}|")
    r += ['', '## 绝对保持与训练口径', '',
          '共同S†旧域79.433714%，原源模型80.367407%；接入损失−.933692pp仍计入。下表学生相对入口及原源旧域的变化，不用相对S的收益替代绝对保持。', '', '|流|组|旧/新相对入口pp|旧域相对原源pp|', '|---|---|---:|---:|']
    for seed, arms in analysis['absolute'].items():
        for arm,a in arms.items():r.append(f"|{seed}|{arm}|{a['old_from_entry_pp']:+.6f}/{a['new_from_entry_pp']:+.6f}|{a['old_from_original_source_pp']:+.6f}|")
    r += ['', 'G00旧域相对入口分别−.530875/−1.008953pp，均未达到历史−.5pp保持参考线；第一流也不能四舍五入为通过。新域相对入口提高约10pp，但不能称遗忘已解决。G00源训练分数在两流均低于S，目标训练分数与S接近；验证收益不等于所有训练/验证口径同步改善。完整逐类/EDES、入口配对及EMA−student患者统计见PUBLIC_SUMMARY。', '',
          '## 梯度、裁剪与真实更新诊断', '',
          '只在正式step0/499/999/1499/1999取样，共30点，复用训练forward、额外60次求梯度。30点raw梯度均低于1，观测裁剪范数等于raw范数；仅能说这些点没有触发裁剪，不能外推全轨迹。', '', '|流/组|辅助encoder梯度范数（五点）|CE与辅助encoder余弦（五点）|实际encoder SGD位移（五点）|', '|---|---|---|---|']
    for key, rows in x['gradient_diagnostics'].items():
        g={'steps':[v['step'] for v in rows], 'aux_encoder_norm':[v['aux_gradient']['encoder'] for v in rows], 'aux_decoder_norm':[v['aux_gradient']['decoder'] for v in rows], 'encoder_CE_aux_cosine':[v['CE_aux_cosine']['encoder'] for v in rows], 'actual_encoder_displacement':[v['actual_SGD_displacement']['encoder'] for v in rows], 'raw_norm':[v['raw_global_gradient_norm'] for v in rows], 'clip_coefficients':[v['theoretical_clip_coefficient'] for v in rows]}
        analysis['gradient_summary'][key]=g
        fmt=lambda vs:', '.join('NA' if v is None else f'{v:.6g}' for v in vs)
        r.append(f"|{key}|{fmt(g['aux_encoder_norm'])}|{fmt(g['encoder_CE_aux_cosine'])}|{fmt(g['actual_encoder_displacement'])}|")
    r += ['', 'A-align辅助encoder梯度在499步以后明显衰减；G00保留更强辅助信号。终点G00/A-align范数比两流约15.28/21.53。入口匹配并未实现全程尺度匹配，这是比较解释的实际边界。CE与辅助方向余弦小且正负混合，不能声称一致的梯度冲突；采样点没有“过强裁剪”的证据，也没有新增特征塌缩检查，不能据此断言表征塌缩。辅助decoder梯度为0，但decoder仍沿各自CE轨迹更新。真实SGD位移包含动量和wd，不能由当步梯度范数直接等同推断。', '',
          '## 实际预算与最终审计', '',
          '总12256 optimizer=12000正式+192资格+64断梯度工程；functional/toy/pending均0，未增加更新或重复旧模型评价。受管运行4086.88秒（68.11分钟），审计前实际新增约193.44MB，8h/<4GB硬预算内；最终交付收据另记录交付后实际占用。所有受管timeout/controller/阶段worker已退休。', '', '|调用类别|实际执行数|', '|---|---:|']
    for key,value in x['audit']['calls']['executed'].items():r.append(f'|{key}|{value}|')
    r += ['', '调用reserved=executed、pending0；一次CPU最终审计0模型前向，核验共同入口空opt、冻结P/buffers/native常量、无G/code、finite final=latest（含opt/RNG/reads）、患者EDES集合及全部固定诊断点。各模型终点评价2892 native切片，总17352；校准与评价成本没有混入优化器更新数。', '',
          '## 结论及适用范围', '',
          '当前两条新目标流复现G00相对S的旧域收益，并优于入口梯度RMS校准的一致性对照。可以将谱几何列为当前配置下更有证据的候选，但尚不能区分整个谱目标中的所有成分，不能推广为所有一致性方法劣于谱方法。', '',
          '源初始化/P/入口共享、仅两个目标流、同一开发验证集反复使用；患者区间不是独立训练重复。S算力更低，G/A虽图像暴露匹配但优化动态不同。相对收益伴随多数旧域患者未改善及绝对旧域损失，后续必须保留这些代价。按授权，条件后续不展开；所有旧结果和固定终点保持。', '',
          '公开范围仅完整源码、冻结协议、匿名聚合与必要审计；不公开患者级记录、原图/合成医学图像或mask、P/frame/code/G/native权重、原始日志或服务器配置。']
    (p/'SPECTRUM_ALIGNMENT_ANALYSIS.json').write_text(json.dumps(analysis,ensure_ascii=False,indent=2)+'\n')
    (p/'SPECTRUM_ALIGNMENT_REPORT_ZH.md').write_text('\n'.join(r)+'\n')

if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('directory');main(parser.parse_args().directory)
