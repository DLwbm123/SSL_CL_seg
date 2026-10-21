"""CPU-only final report from audited aggregates; no model/data forward."""
import json
from pathlib import Path
import statistics


def main(directory):
    p=Path(directory);x=json.loads((p/'TEMPORAL_SPECTRUM_PUBLIC_SUMMARY.json').read_text());assert x['audit']['status']=='PASS'
    original=.8036740663770262;analysis={'absolute':{},'phase_contrasts':{},'gradient_summary':{},'gates':x['gates'],'new_model_forwards':0,'new_updates':0}
    r=['# 两目标流S/G/T/GT：完整实验报告（2026-10-21）','','本报告使用固定最后学生2000主终点，统一EMA辅助；两目标流全部完成、一次最终CPU审计PASS，旧终点不变。GT表示空间谱G与EMA教师特征一致性T组合，不是Ground Truth。','','## 冻结问题与执行合同','','S为当前L2 CE；G增加原.1空间谱目标；T增加弱EMA对学生两强视图的空间特征一致性；GT增加两者。目标流20261025/20261026，共享源初始化163、S†入口和固定源训练P，因此不是两个完整独立源种子。四组均L2+L池仅图14，同48学生+16教师encoder图像暴露、同增强/位置/D32R4，未启用项不连接损失；不称反向或FLOPs完全匹配。无U accessor、frame损失、generator/code、历史原图或在线历史teacher监督。P固定，真实输入无STE。','','教师使用本步opt前EMA弱特征，无梯度eval且不注册student子模块，opt后沿原.999 eligible/copy规则更新。学生native原允许参数全训练，sigma/grad_update固定；空SGD .001/Nesterov .9/matrixwd .0005/vector0/globalclip1。最后2000步学生固定，不选择best或中间验证改配方。', '',f"一次共同校准流20261027的16训练批得到lambda_T={x['calibration']['lambda_T']:.17g}；两流T/GT共用。分子已含G的.1，按.25倍入口encoder梯度RMS校准；不是全程尺度匹配，未用验证集调权。S/Sdetach64更新逐步model/opt/EMA/RNG exact，八资格256更新均通过并丢弃。",'','## 八组固定终点','','Dice为百分数；旧域Siemens、新域Philips。源/目标train指标与val严格分开，EMA不替代主学生。','','|目标流|组|学生旧/新%|EMA旧/新%|源train/目标train学生%|','|---|---|---:|---:|---:|']
    for seed,arms in x['metrics'].items():
        analysis['absolute'][seed]={}
        for m,v in arms.items():
            e=v['EMA_diagnostic'];r.append(f"|{seed}|{m}|{v['Siemens']['primary_dice']*100:.6f}/{v['Philips']['primary_dice']*100:.6f}|{e['Siemens']['primary_dice']*100:.6f}/{e['Philips']['primary_dice']*100:.6f}|{v['source_train']['primary_dice']*100:.6f}/{v['target_train']['primary_dice']*100:.6f}|")
            analysis['absolute'][seed][m]={'student_old_from_entry_pp':v['old_delta_from_entry_pp'],'student_new_from_entry_pp':v['target_delta_from_entry_pp'],'student_old_from_original_source_pp':100*(v['Siemens']['primary_dice']-original),'EMA_old_from_original_source_pp':100*(e['Siemens']['primary_dice']-original)}
    r+=['','## GT相对G和T的增量、原谱复现与T独立收益','','pp为百分点；逐流患者配对bootstrap10000次seed163，不覆盖训练随机性、反复开发或多重比较。均值点估计门槛不是区间非劣确认。','','|流/读出|比较|旧域pp [患者95%区间]|新域pp [患者95%区间]|','|---|---|---:|---:|']
    for seed,modes in x['contrasts'].items():
        for mode,domains in modes.items():
            for name in ['G_minus_S','T_minus_S','GT_minus_G','GT_minus_T','GT_minus_S']:
                cells=[]
                for d in ['Siemens','Philips']:
                    c=domains[d][name];lo,hi=c['patient_bootstrap95_pp'];cells.append(f"{c['delta_B_minus_A_pp']:+.6f} [{lo:+.6f},{hi:+.6f}]")
                r.append(f"|{seed}/{mode}|{name}|{'|'.join(cells)}|")
    r+=['','### 两流平均与冻结门槛','','G−S要求旧均>=+.5pp且每流>0、新各>=−.3；GT−G和GT−T要求旧均>=+.3pp且每流>0、新各>=−.3。不能在看到结果后降低阈值。','','|读出|比较|两流旧平均pp|两流新平均pp|冻结门槛|','|---|---|---:|---:|---|']
    for mode,gates in x['gates'].items():
        for name in ['G_minus_S','GT_minus_G','GT_minus_T']:
            g=gates[name];r.append(f"|{mode}|{name}|{g['old_mean_pp']:+.6f}|{statistics.mean(g['new_pp']):+.6f}|{'PASS' if g['pass_gate'] else '未过'}|")
    r+=['','## 交互：组合最好不等于协同','','交互I=(GT−T)−(G−S)；同患者四组差分后bootstrap，不能把两个独立区间简单相减。','','|流/读出|旧域交互pp [95%区间]|新域交互pp [95%区间]|','|---|---:|---:|']
    for seed,modes in x['contrasts'].items():
        for mode,domains in modes.items():
            cells=[]
            for d in ['Siemens','Philips']:
                c=domains[d]['interaction'];lo,hi=c['patient_bootstrap95_pp'];cells.append(f"{c['mean_pp']:+.6f} [{lo:+.6f},{hi:+.6f}]")
            r.append(f"|{seed}/{mode}|{'|'.join(cells)}|")
    r+=['','## 患者代价与绝对保持','','相对共同S†入口：旧79.433714%、新73.204317%；原源旧80.367407%，接入已损失−.933692pp，必须计入端到端变化。历史目标保持参考线旧78.933714%，原源保持参考线79.867407%，两者不可混淆。阈值用于开发分析，不是临床重要性标准。','','|流/组|学生旧/新相对入口pp|学生旧相对原源pp|EMA旧相对原源pp|','|---|---|---:|---:|']
    for seed,arms in analysis['absolute'].items():
        for m,a in arms.items():r.append(f"|{seed}/{m}|{a['student_old_from_entry_pp']:+.6f}/{a['student_new_from_entry_pp']:+.6f}|{a['student_old_from_original_source_pp']:+.6f}|{a['EMA_old_from_original_source_pp']:+.6f}|")
    r+=['','### 相对入口的全部患者损益','','|流/读出/域|组|均值/中位/p10 pp|改善/下降|下降>.5/1/2pp人数|','|---|---|---:|---:|---:|']
    for seed,modes in x['patient_damage'].items():
        for mode,domains in modes.items():
            for d,arms in domains.items():
                for m,c in arms.items():r.append(f"|{seed}/{mode}/{d}|{m}|{c['mean_pp']:+.6f}/{c['median_pp']:+.6f}/{c['p10_pp']:+.6f}|{c['increased']}/{c['decreased']}|{c['decline_over_pp']['0.5']}/{c['decline_over_pp']['1']}/{c['decline_over_pp']['2']}|")
    r+=['','冻结GT绝对目标保持逐流结果（student）：'+json.dumps(x['gates']['student']['absolute_target_retention'],ensure_ascii=False)+'；GT相对入口下降>1pp旧域患者数不高于G逐流结果：'+json.dumps(x['gates']['student']['no_more_over_1pp_patient_damage_than_G'],ensure_ascii=False)+'。通过损害人数门槛不等于没有损害，点估计保持也不等于置信区间非劣。','','### GT增量的类与ED/ES分解','','|流/域|学生比较|LV/MYO/RV pp|ED/ES pp|患者改善/下降、中位差pp|','|---|---|---:|---:|---:|']
    for seed,modes in x['contrasts'].items():
        analysis['phase_contrasts'][seed]={}
        for d in ['Siemens','Philips']:
            for name in ['GT_minus_G','GT_minus_T','G_minus_S']:
                b,a=name.split('_minus_');c=modes['student'][d][name];phase={ph:100*(x['metrics'][seed][b][d]['phases'][ph]['primary_dice']-x['metrics'][seed][a][d]['phases'][ph]['primary_dice']) for ph in ['ED','ES']};analysis['phase_contrasts'][seed][d+'_'+name]=phase
                classes='/'.join(f"{c['classes_pp'][k]:+.6f}" for k in ['LV','MYO','RV']);r.append(f"|{seed}/{d}|{name}|{classes}|{phase['ED']:+.6f}/{phase['ES']:+.6f}|{c['increased']}/{c['decreased']}, {c['median_patient_delta_pp']:+.6f}|")
    r+=['','## 梯度与实际SGD位移：固定采样点描述','','每组step0/499/999/1499/1999，共40点。辅助是各组实际总辅助，GT并未单独分解G与T梯度，不能由此声称单项冲突机制。这里只复用训练forward，额外80次求梯度，不新增评价。','','|流/组|raw全局范数最大值（5点）|触发裁剪点数|辅助encoder范数（5点）|实际encoder位移（5点）|','|---|---:|---:|---|---|']
    for key,rr in x['gradient_diagnostics'].items():
        d={'steps':[v['step'] for v in rr],'raw_norm':[v['raw_global_gradient_norm'] for v in rr],'clip_coefficients':[v['theoretical_clip_coefficient'] for v in rr],'aux_encoder_norm':[v['aux_gradient']['encoder'] for v in rr],'actual_encoder_displacement':[v['actual_SGD_displacement']['encoder'] for v in rr],'encoder_CE_aux_cosine':[v['CE_aux_cosine']['encoder'] for v in rr]};analysis['gradient_summary'][key]=d
        fmt=lambda vv:', '.join(f'{v:.6g}' for v in vv);r.append(f"|{key}|{max(d['raw_norm']):.6f}|{sum(v<1 for v in d['clip_coefficients'])}|{fmt(d['aux_encoder_norm'])}|{fmt(d['actual_encoder_displacement'])}|")
    r+=['','有限采样点不能推出全程无裁剪；实际SGD位移含动量与weight decay，不等于当步梯度乘LR。完整CE/辅助范数、余弦及decoder指标在PUBLIC_SUMMARY，不用诊断回改本轮配方。','','## 成本、工程异常与交付','','实际16320 optimizer=16000正式+256资格+64断梯度工程，functional/toy/pending0。前向/反向reserved=executed；下表单位为样本暴露或顶层反向调用，不能把求梯度当免费或只报告optimizer数。','','|调用类别|实际|','|---|---:|']
    for k,v in x['audit']['calls']['executed'].items():r.append(f'|{k}|{v}|')
    r+=['','运行时长与最终实际存储见DELIVERY收据，原10h/<4GB硬预算内；原始时钟未重置。全部timeout/controller/阶段worker退休后一次CPU审计PASS：共同入口空opt、固定P/buffers/native常量、无G/code或注册teacher、finite final=latest含opt/RNG/reads、患者EDES及所有固定诊断点。审计0模型前向，不重复旧模型评分。','','首组S评价的全部2892前向完成后，因缺只读入口preparation引用在摘要阶段退出。已保留错误/非零退出/日志/修复前源码，补齐symlink，仅用已有六份完整记录复用原聚合函数重建结果；额外模型前向和更新均0、原账本不变。恢复controller跳过已完成阶段，同原时钟/cap继续。详见[TEMPORAL_SPECTRUM_REPAIR_NOTE_ZH.md](TEMPORAL_SPECTRUM_REPAIR_NOTE_ZH.md)。','','P0诊断先前已交付，不重复执行；其兼容性/特征观察不能替代本轮功能对照结果。当前源/P/入口共享，两目标流及重复开发验证不构成完整独立源或临床泛化。EMA只作统一辅助，不追溯改写最后学生主终点。','','## 预声明后续边界','','若GT−G未过，则停止本T配方分支，不扫描EMA衰减或权重；若GT并未优于T，不声称谱在T上必要；交互无支持不称协同。无论结果正负，保留旧终点与全部成本。强VC/U/frame/记忆/独立源/多域等条件后续未启动。公开范围仅完整源码、协议、匿名聚合与必要审计，不公开患者级记录、图像mask、任何权重/P/frame/code、逐图特征、原始日志或服务器配置。']
    r[4:4]=['## 本轮结论', '',
        'G相对S的旧/新域两流平均收益为+.698710/+.364464pp，原谱复现门槛通过。GT相对G为+.161595/+.072707pp，两流方向一致，但旧域平均低于预设+.3pp，患者区间均跨0，教师增量门槛未过；这不等于教师效果严格为0。GT相对T为+.576586/+.484846pp，冻结门槛通过，说明在本T对照上谱项仍有增量。', '',
        'GT在两流学生旧/新域的点估计均为四组最高，但交互旧域−.081284/−.162964pp，新域+.120250/+.120515pp，全部患者区间跨0，不能称正协同。EMA下GT−G旧均+.061656pp、新均−.041996pp，同样未支持预设教师增量。', '',
        'GT学生相对S†旧域−.575283/−.501932pp，均未达到−.5pp目标保持参考线；第二流只差.001932pp也不能四舍五入通过。相对入口下降>1pp患者G与GT均为12/23（两流），人数门槛通过但不是患者损害已消失。GT旧域差值p10从G的−3.570667/−2.845495改善至−2.692020/−2.473204pp，存在描述性尾部改善。', '',
        'EMA是有竞争力的固定辅助读出：GT旧域80.928442/80.808769%，比原源点估计+.561035/+.441362pp；G EMA也已达到80.872339/80.741560%。EMA不替代学生主终点，不能只用GT EMA对比S学生来论证组合。', '',
        '按预声明规则，停止继续扩展本T配方，不扫描EMA衰减或教师权重；保留G作为更简且有复现证据的研究候选。GT的小幅一致点估计和部分患者尾部改善如实保留，不把门槛阴性写成组合普遍无效。其他条件后续未启动。', '']
    r+=['', '## 增量患者损害幅度与补充说明', '', '|流/域|学生比较|改善/下降|中位/p10 pp|下降>.5/1/2pp人数|', '|---|---|---:|---:|---:|']
    supplement=json.loads((p/'TEMPORAL_SPECTRUM_PATIENT_SUPPLEMENT.json').read_text())
    for seed,modes in supplement['comparisons'].items():
        for d in ['Siemens','Philips']:
            for name in ['GT_minus_G','GT_minus_T']:
                c=modes['student'][d][name];r.append(f"|{seed}/{d}|{name}|{c['increased']}/{c['decreased']}|{c['median_pp']:+.6f}/{c['p10_pp']:+.6f}|{c['decline_over_pp']['0.5']}/{c['decline_over_pp']['1']}/{c['decline_over_pp']['2']}|")
    r+=['', 'GT−G旧域改善16/23和15/23患者，新域改善19/31和20/31；虽然多数受益，但增量较小、仍有损害病例。新域LV的GT−G分别−.152923/−.126322pp，RV分别+.340096/+.409451pp；器官代价不能被新域总均值掩盖。完整EMA−student、相对原源患者区间和跨流差值一致性见PATIENT_SUPPLEMENT。', '',
        '40个梯度诊断点中，第二流四组step0均触发clip1（raw约3.58至3.61，系数约.277至.280），其余36点未触发。该现象不为GT独有，不能解释成仅教师造成过强裁剪，也不能外推全部训练步骤。T单独辅助encoder范数后期约.010至.018，低于初始约.058；GT记录的是总辅助，不能分离G/T各自的因果作用。', '',
        '实际受管总历时7523.36秒（125.39分钟），结束时含2MB元数据预留265385088bytes，原10h/<4GB硬预算内。最终交付后实际du另载DELIVERY。失败/恢复共享原始时钟，摘要修复不增加模型调用或更新。']
    (p/'TEMPORAL_SPECTRUM_ANALYSIS.json').write_text(json.dumps(analysis,ensure_ascii=False,indent=2)+'\n');(p/'TEMPORAL_SPECTRUM_REPORT_ZH.md').write_text('\n'.join(r)+'\n')

if __name__=='__main__':
    import argparse
    q=argparse.ArgumentParser();q.add_argument('directory');main(q.parse_args().directory)
