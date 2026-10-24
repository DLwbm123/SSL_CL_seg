"""Frozen S1 source-patient safety and BMC admission decision."""
import json
import math
import sys
from itertools import product
from pathlib import Path


def percentile(values,q):
    values=sorted(values);position=(len(values)-1)*q
    lo=int(position);hi=min(lo+1,len(values)-1)
    return values[lo]+(position-lo)*(values[hi]-values[lo])


def bootstrap_interval(values):
    means=[sum(s)/len(s) for s in product(values,repeat=len(values))]
    return [percentile(means,.025),percentile(means,.975)]


def main(path):
    x=json.loads(Path(path).read_text())
    assert x['status']=='COMPLETE' and x['exit']['exit_code']==0
    assert x['preflight']['status']=='PASS' and x['full_match']['status']=='PASS'
    assert x['ledger']['updates_reserved']==x['run']['formal_updates']==1600
    assert all(a['status']=='PASS' for a in x['ledger']['attempts'])
    assert not x['run']['target_unlabeled_used']
    assert not x['run']['source_images_read_during_target_updates']
    c=x['run']['cells'];p=x['patients']
    for domain in ('RUNMC','BMC','BMC_train'):
        assert len(p[domain])==4
    receipt=x['guard_receipt']
    mechanism=dict(
        Full_exact_R1=x['full_match']['max_tensor_error']==0,
        at_least_one_projection=receipt['projection_changed_steps']>=1,
        all_post_projection_constraints=receipt['max_post_projection_violation']<=1e-8,
        source_images_not_replayed=not receipt['source_images_read_during_target_updates'])

    def diff(domain,reference):
        return [100*(v['PatientGuard']['Dice']-v[reference]['Dice'])
                for v in p[domain]]
    run_u=diff('RUNMC','Utarget');run_source=diff('RUNMC','Usource')
    bmc_u=diff('BMC','Utarget');bmc_train_u=diff('BMC_train','Utarget')
    run_full=diff('RUNMC','Full');bmc_full=diff('BMC','Full')
    def mean_diff(domain,reference):
        return 100*(c['PatientGuard_'+domain]['Dice']-x['references'][reference+'_'+domain]['Dice'])
    source=dict(
        RUNMC_mean_vs_Utarget_ge_minus_1pp=mean_diff('RUNMC','Utarget')>=-1,
        RUNMC_each_vs_Utarget_ge_minus_5pp=min(run_u)>=-5,
        RUNMC_HD95_vs_Utarget_le_1_10=c['PatientGuard_RUNMC']['HD95_penalized_mm']
            <=1.1*x['references']['Utarget_RUNMC']['HD95_penalized_mm'],
        RUNMC_mean_vs_Usource_ge_minus_0_5pp=mean_diff('RUNMC','Usource')>=-.5,
        RUNMC_each_vs_Usource_ge_minus_5pp=min(run_source)>=-5,
        no_RUNMC_patient_empty=all(c[arm+'_RUNMC']['empty_predictions']==0
                                   for arm in ('Full','PatientGuard')))
    target=dict(
        BMC_mean_vs_Utarget_ge_plus_2pp=mean_diff('BMC','Utarget')>=2,
        BMC_at_least_3_of_4_nonnegative_vs_Utarget=sum(v>=0 for v in bmc_u)>=3,
        BMC_each_vs_Utarget_ge_minus_5pp=min(bmc_u)>=-5,
        BMC_HD95_vs_Utarget_le_1_10=c['PatientGuard_BMC']['HD95_penalized_mm']
            <=1.1*x['references']['Utarget_BMC']['HD95_penalized_mm'],
        BMC_train_mean_vs_Utarget_ge_minus_5pp=mean_diff('BMC_train','Utarget')>=-5,
        no_BMC_patient_empty=all(c[arm+'_'+domain]['empty_predictions']==0
                                 for arm in ('Full','PatientGuard')
                                 for domain in ('BMC','BMC_train')))
    assert all(math.isfinite(c[f'{arm}_{dom}']['Dice'])
               for arm in ('Full','PatientGuard')
               for dom in ('RUNMC','BMC','BMC_train'))
    verdict='PATIENT_GUARD_SIGNAL' if all(mechanism.values()) and all(source.values()) and all(target.values()) else 'PATIENT_GUARD_FAILED'
    tail={domain:{arm:dict(worst_Dice_percent=100*min(v[arm]['Dice'] for v in p[domain]),
                            empirical_p10_percent=100*percentile([v[arm]['Dice']
                                               for v in p[domain]],.1))
                  for arm in ('Utarget','Full','PatientGuard')}
          for domain in ('RUNMC','BMC')}
    ci={domain:{reference:bootstrap_interval(values)
                for reference,values in references.items()}
        for domain,references in (('RUNMC',{'Utarget':run_u,'Full':run_full}),
                                  ('BMC',{'Utarget':bmc_u,'Full':bmc_full}))}
    out=Path(__file__).resolve().parent
    analysis=dict(status=verdict,mechanism_gates=mechanism,source_gates=source,
                  target_gates=target,cells=c,patients=p,references=x['references'],
                  source_anchor_audit=x['source_anchor_audit'],
                  projection_receipt=receipt,last100=x['last100'],
                  descriptive_tail=tail,paired_bootstrap95pp=ci,
                  formal_updates=1600,old_dev_repeated=True)
    (out/'ANALYSIS.json').write_text(json.dumps(analysis,indent=2)+'\n')
    execution=dict(first_actual_start=x['launch']['started'],
        deadline=x['launch']['deadline'],
        elapsed_seconds=x['exit']['time']-x['launch']['started'],
        new_root_bytes=x['bytes'],new_formal_updates=1600,
        inherited_source_Cut_updates=1600,
        extra_source_patient_gradient_backwards=80,
        attempts=[{k:a[k] for k in ('name','status','seconds_actual')}
                  for a in x['ledger']['attempts']])
    (out/'EXECUTION_SUMMARY.json').write_text(json.dumps(execution,indent=2)+'\n')
    table='\n'.join(f"| {arm} | {100*x['references'][arm+'_RUNMC']['Dice']:.3f} | "
                    f"{100*x['references'][arm+'_BMC']['Dice']:.3f} | "
                    f"{100*x['references'][arm+'_BMC_train']['Dice']:.3f} |"
                    for arm in ('Utarget','Full','PatientGuard'))
    gates='\n'.join(f"| {name} | {key} | {'PASS' if passed else 'FAILED'} |"
                    for name,group in (('机制',mechanism),('旧域',source),('目标',target))
                    for key,passed in group.items())
    rows='\n'.join(f"| {domain} | {i+1} | " + ' | '.join(
        f"{100*v[arm]['Dice']:.3f}" for arm in ('Utarget','Full','PatientGuard')) +
        f" | {100*(v['PatientGuard']['Dice']-v['Utarget']['Dice']):+.3f} |"
        for domain in ('RUNMC','BMC') for i,v in enumerate(p[domain]))
    last='\n'.join(f"| {arm} | {x['last100'][arm]['task']:.5f} | "
                   f"{x['last100'][arm]['bound_per_pixel']:.5f} | "
                   f"{x['last100'][arm]['shape_logsd']:.5f} | "
                   f"{x['last100'][arm]['z_logsd']:.5f} |"
                   for arm in ('Full','PatientGuard'))
    report=f"""# S1：源患者梯度锥保护目标生成反馈

**冻结判定：{verdict}。** Full 与 PatientGuard 从同一 C1 Cut 源1600状态开始，在 BMC L4患者134片各做800次更新；两臂保留完整 BayeSeg 形状/外观图像生成，主要预测为同一 U 双logit。Full第800步逐张量等于封存R1 SwitchJoint；PatientGuard只对额外生成梯度施加四名RUNMC源训练患者风险梯度的欧氏半空间约束，目标期间不重放旧图/标签。Utarget为同更新/标注曝光的v45 U_B800；其 BMC训练读出取R1逐张量一致的RemainCut。

| 目标臂 | RUNMC开发Dice% | BMC开发Dice% | BMC完整L训练Dice% |
|---|---:|---:|---:|
{table}

| 类型 | 预注册门 | 判定 |
|---|---|---|
{gates}

匿名逐患者Dice（%）：

| 域 | 索引 | Utarget | Full | PatientGuard | Guard−U pp |
|---|---:|---:|---:|---:|---:|
{rows}

与Utarget相比，PatientGuard旧/新域均值变化 {mean_diff('RUNMC','Utarget'):+.3f}/{mean_diff('BMC','Utarget'):+.3f} pp；相对Full为 {100*(c['PatientGuard_RUNMC']['Dice']-c['Full_RUNMC']['Dice']):+.3f}/{100*(c['PatientGuard_BMC']['Dice']-c['Full_BMC']['Dice']):+.3f} pp。旧域相对Utarget的配对最差 {min(run_u):+.3f} pp、下降{sum(v<0 for v in run_u)}/4名；新域相对Utarget的配对最差 {min(bmc_u):+.3f} pp、获益{sum(v>=0 for v in bmc_u)}/4名。RUNMC绝对最差 Dice U/Full/Guard为 {tail['RUNMC']['Utarget']['worst_Dice_percent']:.3f}/{tail['RUNMC']['Full']['worst_Dice_percent']:.3f}/{tail['RUNMC']['PatientGuard']['worst_Dice_percent']:.3f}%，四人插值p10为 {tail['RUNMC']['Utarget']['empirical_p10_percent']:.3f}/{tail['RUNMC']['Full']['empirical_p10_percent']:.3f}/{tail['RUNMC']['PatientGuard']['empirical_p10_percent']:.3f}%。对4名患者配对均值的全部4⁴有放回重采样95%经验描述区间，Guard−U旧/新域为 [{ci['RUNMC']['Utarget'][0]:+.3f},{ci['RUNMC']['Utarget'][1]:+.3f}]/[{ci['BMC']['Utarget'][0]:+.3f},{ci['BMC']['Utarget'][1]:+.3f}] pp；并非独立患者推断区间，不进入门槛。

机制审计：正式800步中有 {receipt['projection_changed_steps']} 步投影改变了生成梯度；四源患者各自出现负内积的步数 {receipt['source_anchor_negative_dot_steps']}，投影后最大残留负内积 {receipt['max_post_projection_violation']:.3g}。这仅是源状态梯度的一阶欧氏保证，Adam实际步与固定终点没有患者安全定理。

最后100批训练读数（生成bound每像素，和分割Dice不是同一指标）：

| 臂 | 标签风险 | 生成bound/像素 | qS均值logsd | qZ均值logsd |
|---|---:|---:|---:|---:|
{last}

任一冻结门失败即关闭本固定患者梯度锥策略，不调锚、权重、种子、阈值或回选checkpoint。方法新成本另含80次源切片梯度反向与目标双反向；C1 Cut源1600更新为继承历史成本。四名开发患者反复用于路线决策，不能宣称独立泛化、统计非劣、长序列CL或绝对患者安全；隐藏U标签与test未访问。v52另一投影负结果与R1安全失败仍保留。[冻结协议](PREREG_ZH.md)、[匿名分析](ANALYSIS.json)、[成本](EXECUTION_SUMMARY.json)。
"""
    (out/'REPORT_ZH.md').write_text(report)
    print(verdict, f"RUNMC={mean_diff('RUNMC','Utarget'):+.3f}",
          f"BMC={mean_diff('BMC','Utarget'):+.3f}")


if __name__=='__main__':main(sys.argv[1])
