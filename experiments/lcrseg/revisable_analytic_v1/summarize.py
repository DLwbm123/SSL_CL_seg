"""Audit private receipts and emit aggregate-only delivery artifacts; no inference."""
import json
from collections import Counter
from pathlib import Path
import torch


def main(root):
    root=Path(root); job=root/'job'
    read=lambda name:json.loads((job/name).read_text())
    if read('EXIT.json')!={'exit_code':0}: raise ValueError('run did not finish successfully')
    result=read('RESULTS.json'); engineering=read('ENGINEERING.json')
    calls=[json.loads(line) for line in (job/'solve_calls.jsonl').read_text().splitlines()]
    counts=Counter(row['category'] for row in calls)
    if counts['formal']!=82 or counts['engineering']>16 or dict(counts)!=result['costs']['solve_calls']:
        raise ValueError('solve budget mismatch')
    for category,count in counts.items():
        if [r['call'] for r in calls if r['category']==category]!=list(range(1,count+1)):
            raise ValueError('nonsequential physical solve ledger')
    for stage in (0,1):
        for arm in ('l_only','fixed_q','revisable_q'):
            saved=torch.load(job/f'stage{stage}_{arm}.pt',map_location='cpu',weights_only=False)
            if saved['stage']!=stage or saved['arm']!=arm or saved['w'].shape!=(129,4) or not torch.isfinite(saved['w']).all():
                raise ValueError('bad endpoint')
            for domain in (('Siemens',) if not stage else ('Siemens','Philips')):
                key=domain+'_val'; metric=result['stages'][f'{stage}_{arm}'][key]
                expected=23 if domain=='Siemens' else 31
                if (metric['patients'],metric['volumes'])!=(expected,2*expected): raise ValueError('patient coverage')
    exposures=Counter(); pixels=Counter()
    for line in (job/'exposure.jsonl').read_text().splitlines():
        row=json.loads(line); key='/'.join([row['kind'],row['domain'],row['role']])
        exposures[key]+=row['forward_slices']; pixels[key]+=row.get('sampled_pixels',0)
    diagnostics=[json.loads(line) for line in (job/'state_diagnostics.jsonl').read_text().splitlines()]
    for row in diagnostics:
        row['head_implied_responsibility_change']=row.pop('total_responsibility_change')
        row['old_head_implied_responsibility_change']=row.pop('old_responsibility_change')
        row['optimization_responsibilities_revised']=row['arm']=='revisable_q'
    geometry=read('GEOMETRY_DIAGNOSTIC.json')
    iterations=[json.loads(line) for line in (job/'iterations.jsonl').read_text().splitlines()]
    if len(iterations)!=80 or any(r['objective_after']>r['objective_before']+1e-9*(1+abs(r['objective_before'])) for r in iterations):
        raise ValueError('alternating iteration receipt mismatch')
    s=result['stages']; p=result['paired']
    source_gain=p['0_revisable_q_minus_l_only']['Siemens_val']['delta_B_minus_A_pp']
    target_gain=p['1_revisable_q_minus_l_only']['Philips_val']['delta_B_minus_A_pp']
    mean_gain=(s['1_revisable_q']['two_domain_mean']-s['1_fixed_q']['two_domain_mean'])*100
    less_forgetting=result['retention']['revisable_q']['delta_B_minus_A_pp']>=result['retention']['fixed_q']['delta_B_minus_A_pp']
    gate=dict(source_SSL_gain_pp=source_gain,target_SSL_gain_pp=target_gain,
              two_domain_gain_vs_fixed_pp=mean_gain,old_domain_drop_no_worse_than_fixed=less_forgetting,
              joint_descriptive_gate=source_gain>0 and target_gain>0 and mean_gain>0 and less_forgetting)
    public={**result,'engineering':engineering,'audit':dict(status='PASS',formal_solves=counts['formal'],
        engineering_solves=counts['engineering'],finite_endpoints=6,conditional_updates=80,matched_validation_populations=True),
        'exposure_forward_slices':dict(exposures),'sampled_pixel_exposure':dict(pixels),'state_diagnostics':diagnostics,
        'geometry_diagnostic':geometry,'gate':gate}
    (root/'PUBLIC_SUMMARY.json').write_text(json.dumps(public,indent=2,ensure_ascii=False)+'\n')
    lines=['# 可修订U状态解析持续分割：首轮结果','',
           '**本实现首轮失败，首先是源端基础分割能力未保住。** 新源端点仅32.42%–33.62%，原DenseFixMatch固定最终EMA为80.3674%。这不是原模型续训退化，而是另换RBF解析头后的结果；不据此继续调参或扩展持续学习。',
           '组间小幅增益不能抵消这一基础问题，也不足以支持统一SSL/CL机制有效；本轮不能单独否定可修订历史U状态这个一般研究问题。','',
           '本轮为固定U-Net表示下的解析头筛查，独立于CARE-Seg。seed163；固定最后20轮；只报告既有验证集，不是独立test成绩。',
           '特征先验已通过源DenseFixMatch接触源U，因此L-only仅指解析阶段。新源端点单独评估，不与80.3674%拼接。','',
           '|组别|Siemens源端点|最终Siemens|最终Philips|最终双域均值|旧域变化pp|Siemens10L训练|Philips13L训练|',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ('l_only','fixed_q','revisable_q'):
        a=s['0_'+arm]; b=s['1_'+arm]
        vals=[a['Siemens_val']['primary_dice'],b['Siemens_val']['primary_dice'],b['Philips_val']['primary_dice'],b['two_domain_mean'],
              result['retention'][arm]['delta_B_minus_A_pp']/100,b['Siemens_train_labeled']['primary_dice'],b['Philips_train_labeled']['primary_dice']]
        lines.append('|'+arm+'|'+'|'.join(f'{100*v:.4f}' for v in vals)+'|')
    lines+=['','## 分结构与时相','','|阶段/组别/域|LV|MYO|RV|ED均值|ES均值|','|---|---:|---:|---:|---:|---:|']
    for key,stage in s.items():
        for name,value in stage.items():
            if not name.endswith('_val'): continue
            vals=[value['classes'][c] for c in ('LV','MYO','RV')]+[value['phases'][c]['primary_dice'] for c in ('ED','ES')]
            lines.append('|'+key+'/'+name+'|'+'|'.join(f'{100*v:.4f}' for v in vals)+'|')
    lines+=['','## 患者配对（后者减前者）','','|比较/域|差值pp|改善|下降|不变|患者中位数pp|','|---|---:|---:|---:|---:|---:|']
    for comparison,values in p.items():
        for name,value in values.items():
            if not name.endswith('_val'): continue
            lines.append(f'|{comparison}/{name}|{value["delta_B_minus_A_pp"]:.4f}|{value["increased"]}|{value["decreased"]}|{value["unchanged"]}|{value["median_patient_delta_pp"]:.4f}|')
    lines+=['','## 冻结判断与限制','',f'事先约定的联合描述性收益条件：**{"满足" if gate["joint_descriptive_gate"] else "未满足"}**。',
        f'源阶段相对L-only {source_gain:+.4f}pp；目标阶段相对L-only {target_gain:+.4f}pp；相对固定Q双域均值 {mean_gain:+.4f}pp；旧域下降不大于固定Q：{less_forgetting}。',
        '这只检验单一冻结配置，不能据此断言统计显著性、普遍因果或TMI新颖性。摘要责任变化不是责任正确率；压缩界只限制同一W的下一条件更新，不是Dice界。',
        '状态诊断中的责任变化是头隐含softmax概率的变化；fixed_q虽然其头隐含概率会变化，参与拟合的旧Q始终未更新，不能将其诊断量误读为训练责任已被修订。',
        '未满足联合条件则本轮没有同时验证SSL和持续学习收益，不自动调温度、扩大记忆、换seed或恢复负结果模块。固定表示/响应基与未标注目标均可能造成限制，此轮不能单独确定原因。','',
        '## 成本与工程核验','',f'正式{counts["formal"]}次、工程{counts["engineering"]}次线性解算；0 SGD；80轮Q/W循环。源DenseFixMatch43,600更新、旧SPACE目标30,000+工程96更新均为历史成本，单列不混入。',
        '4项单测、真实切片求解、两样本拟合及golden/checkpoint恢复通过；6个阶段头均有限，患者ED/ES集合配对通过。',
        f'另做一次不更新参数的源L切片几何核查：原生分类头与unfold patch矩阵读出最大logit差{geometry["maximum_logit_difference"]:.8g}；增加1次工程切片前向、0解算，耗时{geometry["seconds"]:.2f}秒，已纳入曝光账本。该检查验证提取/原生读出几何，不证明RBF头具有足够容量。',
        '采样曝光的源L背景质量约96.89%，三个前景各约1%；RBF解析头在源LV尤其失败。该现象与固定响应/平方风险下稀有前景拟合不足一致，但本轮没有消融，不能断言具体哪项造成失败。',
        '前向曝光与采样像素按分组列于SUMMARY；共享统计提取不重复记为每个对照各自实际执行。源L响应映射拟合另占一次完整切片前向。验证与L训练评估不进入拟合。',
        'G_U对每片固定1024个曝光位置精确，不是全图像素；历史摘要为私有特征近似记忆，不宣称严格无特征回放或隐私保证。','',
        '发布范围：源码、冻结协议和匿名聚合指标；不公开患者行、图像、模型、摘要/基、原始日志、运行配置或第三方全文/代码包。','']
    (root/'REPORT_ZH.md').write_text('\n'.join(lines))
    print(json.dumps(dict(audit='PASS',gate=gate,formal_solves=counts['formal'],engineering_solves=counts['engineering']),ensure_ascii=False))
