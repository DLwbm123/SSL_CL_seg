"""Read-only arithmetic over public aggregates; no models, data access or fitting."""
import csv
import json
import statistics as st
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
def read(name):
    return json.loads((ROOT / name).read_text())
def rows(name):
    with (ROOT / name).open() as f:
        return list(csv.DictReader(f))
def md_table(headers, values):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] + ['| ' + ' | '.join(map(str, r)) + ' |' for r in values])
def fmt(x):
    return f'{x:.6f}'

def main():
    metrics = rows('FINAL_METRICS.csv')
    decision = read('DECISION.json')
    accounting = read('TRAINING_AND_MEMORY_ACCOUNTING.json')
    verification = read('CLOSEOUT_VERIFICATION.json')
    arms = ['SUP', 'J_MSE', 'T_MSE', 'J_SCE', 'T_SCE']
    seeds = [31, 32, 33]
    domains = ['RIM_ONE_r3', 'Drishti_GS']
    assert len(metrics) == 30 and all(r['model'] == 'student' and r['epoch'] == '100' for r in metrics)
    assert verification['tasks'] == 30 and verification['snapshots'] == 150 and len(verification['deployments']) == 30
    assert verification['formal_updates'] == sum(int(r['updates']) for r in rows('RUN_LEDGER.csv')) == 79500
    assert len(accounting['child_exits']) == 60 and all(r['exit_code'] == 0 for r in accounting['child_exits'])
    assert not verification['source_worktree_status']
    q = {(s, a): st.mean(float(r['macro_fg_dice']) for r in metrics if r['seed'] == str(s) and r['arm'] == a) for s in seeds for a in arms}
    delta = [q[s, 'T_SCE'] - q[s, 'SUP'] for s in seeds]
    assert all(abs(x-y) < 1e-14 for x, y in zip(delta, decision['delta_SUP']['values']))
    assert decision['status'] == 'TARGET_PATH_SSL_NOT_ESTABLISHED'
    sections = ['# SSL_TARGET_PATH_V0_1 最终报告',
        '**最终学生没有建立稳定的 SSL 增益。** T_SCE 相对匹配 SUP 的三种子两域平均 Dice 差为 '
        f"{st.mean(delta):+.6f}，sample SD={st.stdev(delta):.6f}；仅 seed32 改善，seed31/33 下降。"
        '最大类别代价为 seed33/Drishti cup −0.042400。仅教师筛选确实保留更多纠错机会，'
        '也接受更多教师错误；机制改善不能替代最终模型收益。',
        '**科学终态：TARGET_PATH_SSL_NOT_ESTABLISHED。工程矩阵：ENGINEERING_COMPLETE。** '
        '30 个完整任务、150 个固定诊断、30 个单学生部署及 60 个成功子进程退出已交叉核验。'
        '无追加训练、阈值调整、候选替换或 CL。本报告保留运行器原终态；合成资格的细粒度计数限制见后文。',
        '## 最终学生结果',
        '指标为各患者 rim/cup 宏平均 Dice，先域内患者平均，再两域等权；三种子预先指定。'
        '全部取 epoch100 学生，不使用 EMA 或最佳 epoch。表中 SD 为三次优化种子的 sample SD。',
        md_table(['配方', 'seed31 Q', 'seed32 Q', 'seed33 Q', '三种子平均 Q', 'Q sample SD', '平均 ΔSUP'],
                 [[a, *[fmt(q[s,a]) for s in seeds], fmt(st.mean(q[s,a] for s in seeds)), fmt(st.stdev(q[s,a] for s in seeds)), f'{st.mean(q[s,a]-q[s,"SUP"] for s in seeds):+.6f}'] for a in arms]),
        '完整 30 行、各域 rim/cup 见 [FINAL_METRICS.csv](FINAL_METRICS.csv)；所有阶段学生/EMA 见 [EPOCH_METRICS.csv](EPOCH_METRICS.csv)。',
        '## 预定门槛逐项判定',
        md_table(['门槛', '实际', '判定'], [
            ['平均 ΔSUP ≥ +0.010', f'{st.mean(delta):+.6f}', '未通过'],
            ['每个 seed ΔSUP > 0', ', '.join(f'{s}: {v:+.6f}' for s,v in zip(seeds,delta)), '未通过'],
            ['平均 ΔJ_MSE ≥ +0.005', f"{decision['delta_J_MSE']['mean']:+.6f}", '未通过'],
            ['每 seed/domain Δmacro ≥ −0.005', f"最小 {min(r['macro_delta'] for r in decision['seed_domain']):+.6f}", '未通过'],
            ['每 seed/domain/class ΔDice ≥ −0.020', f"最小 {min(r[k] for r in decision['seed_domain'] for k in ['rim_delta','cup_delta']):+.6f}", '未通过'],
            ['完整矩阵与隔离、部署检查', '30/30；150/150；30/30', '通过']]),
        'T_SCE−SUP 的范围为 [−0.020327, +0.019430]；T_SCE−J_MSE 的平均差 +0.000731，'
        'sample SD=0.005865，范围 [−0.005749, +0.005676]。平均差不足，且跨种子不稳定。',
        '## 域和类别代价',
        md_table(['seed','域','T_SCE−SUP macro','rim','cup'], [[r['seed'],r['domain'],*[f'{r[k]:+.6f}' for k in ['macro_delta','rim_delta','cup_delta']]] for r in decision['seed_domain']]),
        'seed32 两域及两类均提高，不能表述为各处均无效；但 seed31 Drishti rim、seed33 Drishti rim/cup 超过预定类别损害上限。',
        '## 两因素效应',
        '下表为三种子平均的完整训练配方差，不是单像素因果效应。相同名义系数下，SCE 替换同时改变梯度形状与尺度，不能完全分离二者。']
    contrasts = rows('FACTORIAL_CONTRASTS.csv')
    effects = ['mask_effect_MSE', 'mask_effect_SCE', 'loss_effect_joint', 'loss_effect_teacher', 'interaction']
    definitions = ['T_MSE−J_MSE','T_SCE−J_SCE','J_SCE−J_MSE','T_SCE−T_MSE','(T_SCE−J_SCE)−(T_MSE−J_MSE)']
    sections += [md_table(['效应','定义','RIM 平均 Δmacro','Drishti 平均 Δmacro','两域平均'],
        [[effect, definition, *[f'{v:+.6f}' for v in values], f'{st.mean(values):+.6f}']
         for effect, definition in zip(effects, definitions)
         for values in [[float(next(r['mean'] for r in contrasts if r['seed']=='all' and r['domain']==d and r['metric']=='macro_fg_dice' and r['effect']==effect)) for d in domains]]]),
        '在 RIM 上，MSE→SCE 在 J 和 T 掩码下三个种子均改善；Drishti 的 J 掩码下三个种子均下降。'
        '仅教师掩码的收益也依赖域和种子。所有消融的三种子两域平均 Q 均低于 SUP；不将消融替换为主候选。'
        '完整逐 seed/domain/rim/cup 的效应、sample SD 和范围见 [FACTORIAL_CONTRASTS.csv](FACTORIAL_CONTRASTS.csv)。',
        '## 纠错机会与错误传播',
        '下表固定查看最终 T_SCE 轨迹的同一组输出，同时计算 raw/T/J；没有为比较掩码重新前向。'
        '单位是平均像素/患者：先将四个 training-like draw 在患者内平均，再患者平均、最后三个种子等权平均。'
        '仅汇总互斥的 teacher_predicted_class 三类，不重复加上 true_class 分组。表内比率是平均计数之比，不是患者比率的平均；原始逐类患者比率与 undefined 支持另见 CSV。']
    correction = rows('CORRECTION_PATH_DIAGNOSTICS.csv')
    path_values = []
    for mode in ['posterior_mean_clean','training_like']:
        for d in domains:
            for name in ['teacher_correct_student_wrong','teacher_wrong_student_correct','both_wrong_same_class']:
                selected=[r for r in correction if r['epoch']=='100' and r['arm']=='T_SCE' and r['mode']==mode and r['domain']==d and r['grouping']=='teacher_predicted_class' and r['stratum']==name]
                vals=[sum(float(r[k]) for r in selected)/3 for k in ['patient_mean_raw_support','patient_mean_T_accepted','patient_mean_J_accepted']]
                assert len(selected)==9 and vals[0]+1e-8>=vals[1]>=vals[2]-1e-8
                path_values.append([mode,d,name,*[f'{v:.3f}' for v in vals],f'{100*vals[1]/vals[0]:.2f}%',f'{100*vals[2]/vals[0]:.2f}%'])
    sections += [md_table(['视图','域','分层','raw','T接受','J接受','T/raw','J/raw'],path_values),
        'training-like 纠错机会保留率，RIM 从 J 的约 24.19% 增至 T 的 48.72%，Drishti 从 12.26% 增至 33.44%。'
        '教师置信度仍先拒绝约 51.28% / 66.56% 的原始纠错机会；学生置信度又拒绝 T 准入机会的约 50.35% / 63.35%。'
        '同时，“教师错、学生对”被接受数从 50.40→111.25（RIM）、126.34→271.85（Drishti）。'
        '扩大准入包含潜在有益和有害方向，不是正确性保证。',
        'clean 纠错机会极少被 T 接受：RIM 0.192/49.625，Drishti 6.840/158.760。'
        'training-like 下大量共同错误仍被接受；共同错误不等于零梯度：'
        'RIM 同类共同错误中约 5580.58/5595.59、Drishti 8049.73/8098.51 的 p/q 分布不同。'
        '只有 p≈q（冻结 L1≤1e−6）才接近一致性固定点，不能把相同 argmax 当成相同分布。',
        '## 局部梯度、实际训练贡献与概率质量',
        '四种解析候选均在同一输出上计算。下表固定 T_SCE 轨迹、epoch100、training-like、true_class=rim；'
        '为三种子各自患者平均的等权平均。未加权范数只在相应候选接受像素上统计；'
        'weighted dot 是该分层逐像素与隔离 GT-CE 方向的内积和，再作患者平均。'
        '正内积表示局部一阶方向一致，负内积表示冲突，不代表全网络参数梯度或实际 Dice 因果。']
    gradient = rows('CLASS_LOSS_AND_GRADIENT.csv')
    gr=[]
    for d in domains:
        for name in ['teacher_correct_student_wrong','teacher_wrong_student_correct']:
            for candidate in arms[1:]:
                rr=[r for r in gradient if r['kind']=='evaluator_local_logit' and r['epoch']=='100' and r['domain']==d and r['arm']=='T_SCE' and r['mode']=='training_like' and r['grouping']=='true_class' and r['class_id']=='1' and r['stratum']==name and r['candidate']==candidate]
                assert len(rr)==3
                gr.append([d,name,candidate,*[f'{st.mean(float(r[k]) for r in rr if r[k]):.6g}' for k in ['patient_mean_raw_pixel_logit_norm_mean','patient_mean_weighted_local_logit_norm','patient_mean_weighted_dot_GT_sum']]])
    sections += [md_table(['域','rim 分层','候选','raw norm','weighted norm','weighted dot GT'],gr),
        '实际训练按教师预测类统计 accepted、raw/weighted loss、SCE/entropy/KL 和 local logit norm，'
        '逐 epoch 原值均在 CLASS_LOSS_AND_GRADIENT.csv（kind=actual_training）。下表汇总 T_SCE 的 epoch21–100，'
        '三种子合计；loss 为逐 batch 类贡献之和，norm 为各 batch 类局部范数之和，不是合并后的参数梯度。']
    tr=[]
    for d in domains:
        for c in ['0','1','2']:
            rr=[r for r in gradient if r['kind']=='actual_training' and r['arm']=='T_SCE' and r['domain']==d and r['class_id']==c]
            values=[sum(float(r[k]) for r in rr) for k in ['accepted','raw_loss_contribution','weighted_loss_contribution','SCE','target_entropy','KL','local_logit_gradient_norm']]
            tr.append([d,c,*[f'{v:.6g}' for v in values]])
    sections += [md_table(['域','教师类 0/bg 1/rim 2/cup','accepted','raw loss','weighted loss','SCE','entropy','KL','local norm sum'],tr),
        '背景 accepted 数量不能用于断言背景参数梯度支配。原始 SCE 含目标熵；比较损失优化时需同时查看去熵 KL。']
    quality=rows('PROBABILITY_AND_PSEUDO_QUALITY.csv')
    qr=[]
    for d in domains:
        for c in ['0','1','2']:
            for f in ['teacher','joint']:
                rr=[r for r in quality if r['kind']=='pseudo_quality' and r['epoch']=='100' and r['arm']=='T_SCE' and r['mode']=='training_like' and r['domain']==d and r['class_id']==c and r['filter']==f and r['model']=='teacher']
                qr.append([d,c,f,*[fmt(st.mean(float(r[k]) for r in rr)) for k in ['patient_mean_precision','patient_mean_accepted_correct_recall']]])
    sections += [md_table(['域','类','mask','teacher precision','accepted-correct recall'],qr),
        '以上为隔离 val 的患者平均质量，不是隐藏 train-U GT 上的真实标签质量。T-mask 扩大覆盖，'
        '但前景精度约 0.719–0.780，显著低于背景约 0.986–0.988，不能用总体高置信替代前景正确性。']
    pr=[]
    for d in domains:
        for a in ['SUP','T_SCE']:
            rr=[r for r in quality if r['kind']=='probability' and r['epoch']=='100' and r['arm']==a and r['domain']==d and r['mode']=='posterior_mean_clean' and r['model']=='student']
            pr.append([d,a,*[fmt(st.mean(float(r[k]) for r in rr)) for k in ['patient_mean_NLL','patient_mean_multiclass_Brier']]])
    sections += [md_table(['域','最终学生','NLL','multiclass Brier'],pr),
        'T_SCE 在两域平均 NLL/Brier 均改善，但未建立 Dice 实用增益；概率指标与任务指标必须分开。'
        '完整 confidence 分位数、pmax>0.99、错误前景 confidence、所有阶段、学生/教师及逐类质量均保存在 '
        '[PROBABILITY_AND_PSEUDO_QUALITY.csv](PROBABILITY_AND_PSEUDO_QUALITY.csv)。']
    counters=Counter()
    for receipt in accounting['receipts']:
        counters.update(receipt['counters'])
    deploy_cases=sum(r['cases'] for r in verification['deployments'])
    q_attempts=accounting['qualification']['attempts']
    formal_forward=sum(counters.get(k,0) for k in ['student_l_forward','student_u_forward','ema_u_forward'])
    formal_images=sum(counters.get(k,0) for k in ['student_l_images','student_u_images','ema_u_images'])
    synthetic={k:sum(r['operation_counts'].get(k,0) for r in q_attempts) for k in ['model_forward','image_forward','backward','autograd_grad_calls']}
    summary=dict(formal_optimizer_updates=79500,formal_backward=79500,formal_EMA_updates=79500,
        training_model_forwards=formal_forward,training_image_forwards=formal_images,
        diagnostic_model_forwards=48750,diagnostic_image_forwards=48750,diagnostic_snapshots=150,
        deployment_model_forwards=deploy_cases,deployment_image_forwards=deploy_cases,
        synthetic_updates=616,synthetic_operations=synthetic,smoke_updates=10,
        current_all_updates=80126,historical_formal_total=248108,
        current_counted_model_forwards=formal_forward+48750+deploy_cases+synthetic['model_forward']+26,
        current_counted_image_forwards=formal_images+48750+deploy_cases+synthetic['image_forward']+52,
        current_backward=80126,additional_real_diagnostic_backward=0,
        training_labeled_asset_opens=sum(r['labeled_opens'] for r in accounting['receipts']),
        training_unlabeled_image_opens=sum(r['unlabeled_opens'] for r in accounting['receipts']),
        diagnostic_image_only_opens=4875,diagnostic_val_image_and_GT_sample_opens=4875,
        deployment_image_only_opens=deploy_cases,
        diagnostic_prediction_seals=24375,
        max_torch_allocated_bytes=max(r['peak_allocated'] for r in accounting['receipts']),
        max_torch_reserved_bytes=max(r['peak_reserved'] for r in accounting['receipts']),
        formal_wall_seconds=verification['parent']['finished_unix']-verification['launch']['started_unix'],
        qualification_accounting_limit='Synthetic EMA and detailed synthetic fixture I/O/diagnostic subtotals were not persisted by the qualification instrumentation. No fabricated counts or extra test runs. Synthetic optimizer/backward/model/image forwards are instrumented. Historical non-formal operation totals are not reconstructed.')
    (ROOT/'ACCOUNTING_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    sections += ['## 工程执行与计数边界',
        f"冻结执行源码 `{verification['source']}`；服务器执行 checkout 干净。任务耗时 {summary['formal_wall_seconds']/60:.2f} 分钟，GPU4/5/6 三条独立队列。"
        '公共输入、五臂初始化、20epoch warm-up、标签顺序与模型状态匹配；SUP 和 warm-up 无 U 图像读取；'
        '150 次诊断前后学生/EMA/优化器/RNG 状态保持，真实诊断额外 backward=0。',
        md_table(['阶段','optimizer / backward','模型 forward','image forward','EMA'],[
            ['正式训练','79500 / 79500',formal_forward,formal_images,79500],
            ['固定真实诊断','0 / 0',48750,48750,0],
            ['独立单学生部署','0 / 0',deploy_cases,deploy_cases,0],
            ['真实 labeled-only smoke','10 / 10',26,52,10],
            ['四次合成资格','616 / 616',synthetic['model_forward'],synthetic['image_forward'],'未单独持久化'],
            ['本轮合计','80126 / 80126',summary['current_counted_model_forwards'],summary['current_counted_image_forwards'],'已持久化真实合计 79510；另有合成 EMA']]),
        '正式训练 L 样本打开 159000 次、U 图像打开 99840 次；固定诊断 image-only 打开 4875 次，'
        '隔离 val 样本（含图像/GT）打开 4875 次，预测密封 24375 份；部署 image-only 打开 975 次，额外 GT=0。'
        '这些是 loader 级计数，不是操作系统 read 系统调用数。smoke 源码一次加载每域前两例 labeled，'
        '共 4 个样本打开后供五臂复用（源码推导，未单独 I/O 仪表计数）；不访问真实 U 图像。',
        '四次合成资格均通过，各 154 次成功更新，合计 616≤1000；冻结执行源码本地/服务器资格均通过，'
        '真实 smoke 10≤10。合成 autograd.grad 共 12 次，仅合成算术验证。'
        '**计数限制：资格仪表未单独持久化合成 EMA、fixture I/O 与合成诊断的细分小计；不能把这些标为 0 或声称计数无缺项。** '
        '合成总 forward/backward/optimizer 已实测保留。本次不为补细项重跑资格、不更改运行器原 ENGINEERING_COMPLETE；'
        '它表示固定真实矩阵及隔离部署核验完成，细粒度资格审计仍有上述限制。',
        '历史正式更新 168608 + 本轮 79500 = 248108；该数不含合成和 smoke。'
        '本轮包含合成/smoke 的成功更新总数 80126。历史全部非正式 forward/I/O/EMA 未在本轮重新构建，不能与正式历史总数混称。',
        f"实测训练最多 student+EMA 两份完整模型，部署一份；每份参数 1936064 bytes，Adam 3865412 bytes。"
        f"峰值 Torch allocated={summary['max_torch_allocated_bytes']/2**20:.3f} MiB、reserved={summary['max_torch_reserved_bytes']/2**20:.3f} MiB。"
        'sigma/grad_update 各 1728 bytes 为惰性状态，未入 Adam；无原型、GAS 或第三份完整模型。',
        '## 结论和停止边界',
        '纠错机会被学生置信度排除的机制得到描述性支持，但教师置信度同样排除了大量机会；'
        '扩大通路会同时引入更多错误监督。SCE 的局部梯度性质与概率质量改善未转化为稳定的 Dice 增益。'
        '因此结束本轮微型骨干、同视图轻扰动 EMA 下的监督路径检验，不追加参数、seed、PAS、分类头或投影，也不进入 CL。'
        '这不构成“半监督学习普遍无效”的结论。',
        '固定 split0 已经历多轮研究者暴露；三个优化种子衡量训练随机性，不是独立患者确认。'
        '患者 bootstrap 按预先声明未执行；不作事后显著性检验，不将像素或四个 draw 当独立患者。',
        '## 交付与复现',
        '原冻结协议和 SOURCE_FREEZE 不变。`EXECUTOR_FINAL_REPORT.md` 保留运行器原报告；'
        '`DECISION.json` 保留原终态。`CLOSEOUT_VERIFICATION.json` 保存只读重验及部署/退出凭据；'
        '`ACCOUNTING_SUMMARY.json` 提供明确计数和不可用细项。运行 `python3 build_final_report.py` '
        '只读取这些公开汇总表，重建本报告并核对 30 行结果、79500 步及原判定；不加载模型、不读 GT、不训练。',
        '全部逐类/逐阶段汇总和资格证据公开。图像、GT、患者 ID、逐病例像素、checkpoint 和模型权重保留 NAS，不进入 GitHub。'
        '具体发布提交、匿名访问结果及 NAS 归档以 PUBLICATION_VERIFICATION.json 为准。']
    (ROOT/'FINAL_REPORT.md').write_text('\n\n'.join(sections)+'\n')
    print(json.dumps(summary,indent=2))

if __name__ == '__main__':
    main()
