"""Rebuild the closeout using public aggregate evidence only; no model or GT access."""
import csv,json,statistics as st
from collections import Counter
from pathlib import Path
R=Path(__file__).resolve().parent
def read(n):return json.loads((R/n).read_text())
def rows(n):
    with (R/n).open() as f:return list(csv.DictReader(f))
def table(h,rr):return '\n'.join(['| '+' | '.join(h)+' |','| '+' | '.join(['---']*len(h))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rr])
def write_csv(n,rr):
    with (R/n).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]),lineterminator='\n');w.writeheader();w.writerows(rr)
def main():
    final=rows('FINAL_METRICS.csv');effects=rows('PAIRED_EFFECTS.csv');costs=rows('SEED_DOMAIN_CLASS_COST.csv');ci=rows('UNCERTAINTY_SUMMARY.csv')
    verify=read('CLOSEOUT_VERIFICATION.json');runtime=read('TEST_AND_RUNTIME_REPORT.json');mem=read('MEMORY_AND_DEPLOYMENT.json');p1=read('P1_SELECTION.json');p2=read('P2_REPLICATION.json')
    assert verify['source']=='800cabe5e5dd69612738aaba45ad35121d63c0f2' and verify['worktree_status']==''
    assert sum(verify[p]['updates'] for p in ['P1','P2'])==sum(int(r['updates']) for r in rows('RUN_LEDGER.csv'))==106000
    assert sum(verify[p]['tasks'] for p in ['P1','P2'])==40 and sum(verify[p]['snapshots'] for p in ['P1','P2'])==200
    assert p1['selected']=='MIX_CED' and p2['status']=='VALUE_REPRODUCED' and p2['stability']=='DIRECTION_REPRODUCED_ON_TWO_NEW_SEEDS'
    domains=['RIM_ONE_r3','Drishti_GS'];qs={}
    for s in [31,32,33,41,42]:
        for a in ['SUP_CE','SUP_CED','MT_CED','MIX_CTX','MIX_CED']:
            rr=[float(r['macro_fg_dice']) for r in final if int(r['seed'])==s and r['arm']==a]
            if rr:assert len(rr)==2;qs[s,a]=st.mean(rr)
    summary=[]
    for phase,seeds in [('P1',[31,32,33]),('P2',[41,42]),('ALL5_SELECTED_DESCRIPTIVE',[31,32,33,41,42])]:
        for a in ['SUP_CE','SUP_CED','MT_CED','MIX_CTX','MIX_CED']:
            if not all((s,a) in qs for s in seeds):continue
            v=[qs[s,a] for s in seeds]
            summary.append(dict(phase=phase,arm=a,mean=st.mean(v),median=st.median(v),sample_SD=st.stdev(v),minimum=min(v),maximum=max(v),seeds=str(seeds),values=str(v)))
    write_csv('RECIPE_SEED_SUMMARY.csv',summary)
    for r in effects:
        if r['seed']=='all':
            seeds=[31,32,33] if r['phase']=='P1' else [41,42] if r['phase']=='P2' else [31,32,33,41,42]
            v=[qs[s,r['arm']]-qs[s,r['baseline']] for s in seeds]
            assert abs(st.mean(v)-float(r['mean']))<1e-14 and abs(st.stdev(v)-float(r['sample_SD']))<1e-14
    epochs=rows('EPOCH_METRICS.csv');late=[]
    for phase,s,d,a,model in sorted({(r['phase'],r['seed'],r['domain'],r['arm'],r['model']) for r in epochs}):
        rr=sorted([r for r in epochs if (r['phase'],r['seed'],r['domain'],r['arm'],r['model'])==(phase,s,d,a,model) and int(r['epoch']) in (60,80,100)],key=lambda r:int(r['epoch']))
        assert len(rr)==3
        v=[float(r['macro_fg_dice']) for r in rr]
        late.append(dict(phase=phase,seed=s,domain=d,arm=a,model=model,epochs='60/80/100',mean=st.mean(v),sample_SD=st.stdev(v),range=max(v)-min(v),values=str(v),meaning='within one trajectory, not three independent trained models'))
    write_csv('LATE_EPOCH_VARIATION.csv',late)
    counter=Counter();stream=Counter();deploy=0
    for phase in mem.values():
        v=phase['verification'];assert not phase['error'] and all(r['exit_code']==0 for r in v['exits'])
        for r in v['operations']:
            for kind in ['training','deployment']:counter.update(r[kind]['counts'])
        for r in v['receipts']:stream.update(r['counters'])
        deploy+=sum(r['cases'] for r in v['deployments'])
    assert counter['optimizer_steps']==counter['backward']==counter['ema_updates']==106000 and deploy==1300
    total=counter.copy();qualification=Counter()
    for r in runtime['qualification']['attempts']:qualification.update(r['counts'])
    total.update(qualification);total.update(runtime['smoke']['counts']);assert total['optimizer_steps']==106434
    account=dict(formal_matrix_operations=dict(counter),training_streams=dict(stream),qualification_all_attempt_operations=dict(qualification),smoke=runtime['smoke']['counts'],current_all_operations=dict(total),formal_updates=106000,synthetic_updates=426,smoke_updates=8,current_all_updates=106434,historical_formal_total=354108,diagnostic_snapshots=200,deployments=40,deployment_images=deploy,formal_wall_minutes=(verify['parent']['finished_unix']-verify['launch']['started_unix'])/60,units='model_forward counts calls; image_forward counts image records; sample/HDF5 counters count loader/file operations, not unique patients or OS read calls',old_gap='Old synthetic EMA/I/O gap remains disclosed; old qualification was not repeated.')
    (R/'ACCOUNTING_SUMMARY.json').write_text(json.dumps(account,indent=2)+'\n')
    text=['# Anchored Mix Stability V0.1 最终报告',
        '**MIX_CED 在固定患者划分上取得开发收益，并在两个预定新优化种子上复核通过。** '
        'P2 相对 SUP_CE 平均 Dice +0.036492，相对 SUP_CED +0.046829；两种子均正。'
        '相对同输入 MIX_CTX 的伪目标增量为 +0.006228，达到预定 +0.003 幅度，但患者 bootstrap 区间包含 0。'
        '主要收益已经出现在混合上下文对照中，不能把全部提高归因于伪标签。',
        '**终态：VALUE_REPRODUCED；DIRECTION_REPRODUCED_ON_TWO_NEW_SEEDS；ENGINEERING_COMPLETE。** '
        'P1 24任务/63600步，按冻结规则选择 MIX_CED 后，P2 16任务/42400步全部完成。'
        '共40任务、106000次正式更新、200个固定诊断、40次单学生部署、80个成功子进程退出。'
        '存在 seed41/Drishti cup 相对 SUP_CE −0.021771 的 amber 类别代价。未启动额外实验或 CL。',
        '## 五个比较与发现/复核分离',
        'Q 为域内每患者 rim/cup 宏平均 Dice，再将两域等权平均；固定 epoch100 学生。'
        'SUP_CE 的 P1 数值来自冻结旧 SUP 评分，未重训；P2 的 SUP_CE 为新训练的匹配对照。',
        table(['配方','P1三seed平均Q','P1 Q sample SD','P2两seed平均Q','P2 Q sample SD'],[
            [a,*[f"{next(r[k] for r in summary if r['phase']==p and r['arm']==a):.6f}" if any(r['phase']==p and r['arm']==a for r in summary) else '未准入/未训练' for p in ['P1','P2'] for k in ['mean','sample_SD']]] for a in ['SUP_CE','SUP_CED','MT_CED','MIX_CTX','MIX_CED']]),
        table(['阶段','比较','平均配对差','median','配对sample SD','范围','正/负seed数'],[
            [r['phase'],r['arm']+' − '+r['baseline'],*[f"{float(r[k]):+.6f}" for k in ['mean','median']],f"{float(r['sample_SD']):.6f}",f"[{float(r['minimum']):+.6f}, {float(r['maximum']):+.6f}]",r['positive']+'/'+r['negative']] for r in effects if r['seed']=='all' and r['phase']!='ALL5_SELECTED_DESCRIPTIVE']),
        '**真实监督基础：**CE+Dice 相对 CE，P1 +0.003326（2/3正），P2 −0.010337（1/2正），五seed描述性均值 −0.002139。'
        '因此不能声称 Dice 单独稳定改善监督基础或降低方差。P1 SUP_CED 的 Q SD 较低，但 P2 升到0.020539。',
        '**普通 MT：**MT_CED 相对 SUP_CED 为 −0.005485（仅1/3正），且 seed33/Drishti rim 下降0.072761，未准入。'
        '**混合上下文：**MIX_CTX 相对 SUP_CED 的 P1 增益 +0.030880，已覆盖 MIX_CED 总增益中的大部分。'
        '**伪目标增量：**MIX_CED−MIX_CTX 在发现三seed及复核两seed均为正，但增量幅度小于上下文对照收益，且患者不确定性尚未消除。',
        '## 每个种子的主候选配对差',
        table(['阶段','seed','MIX_CED Q','ΔSUP_CE','ΔSUP_CED','ΔMIX_CTX'],[[('P1' if s<40 else 'P2'),s,f'{qs[s,"MIX_CED"]:.6f}',*[f'{qs[s,"MIX_CED"]-qs[s,b]:+.6f}' for b in ['SUP_CE','SUP_CED','MIX_CTX']]] for s in [31,32,33,41,42]]),
        '全部五seed为经过P1选择后的描述性汇总，不代替独立列出的新seed P2结果。'
        '五seed MIX_CED−SUP_CE 平均 +0.038502，配对SD0.014230；−SUP_CED 平均 +0.040642，配对SD0.008045；'
        '−MIX_CTX 平均 +0.005873，配对SD0.004881。完整 Q 的 median/SD/range 与配对差分别见 RECIPE_SEED_SUMMARY.csv 和 PAIRED_EFFECTS.csv。',
        '## 冻结门槛与分项状态',
        'P1：MIX_CED 的平均 ΔSUP_CED +0.036517、3/3正，超过较强监督参照 +0.036517，'
        '每域/类别均值与单次严重退化保护线通过，ΔMIX_CTX +0.005636≥0.003。MT_CED不合格，因此按规则选择MIX_CED。'
        '全部候选原始门槛布尔值和逐行成本保存在 P1_SELECTION.json。',
        'P2：相对 SUP_CE/SUP_CED 的均值分别 +0.036492/+0.046829，均超过 +0.010；'
        '相对两种基线的每域均值≥−0.005，每域/类别均值≥−0.020，单seed类别差均≥−0.050。'
        'ΔMIX_CTX +0.006228≥0.003。两个新seed相对SUP_CED均正。门槛按原协议判定，没有用bootstrap或五seed混合结果修改。',
        table(['分项','状态','边界'],[
            ['MEAN_VALUE','VALUE_REPRODUCED','新seed分别战胜两种监督基线'],
            ['PAIRED_SEED_VARIATION','DIRECTION_REPRODUCED_ON_TWO_NEW_SEEDS','仅两个新优化seed，不证明普遍低方差'],
            ['DOMAIN_CLASS_COST','MEAN_GUARDS_PASSED_WITH_AMBER','保留seed41 Drishti cup下降'],
            ['PSEUDO_TARGET_INCREMENT','PRACTICAL_MARGIN_MET_COHORT_UNCERTAIN','P1/P2患者区间均跨0'],
            ['REPLICATION_STATUS','FIXED_SPLIT_DEVELOPMENT_REPLICATION','不是独立患者、临床安全或CL确认']]),
        '## 类别代价',
        table(['seed','域','基线','macro差','rim差','cup差'],[
            [s,d,b,*[f"{next(float(r['delta']) for r in costs if r['phase']=='P2' and r['arm']=='MIX_CED' and r['baseline']==b and int(r['seed'])==s and r['domain']==d and r['metric']==m):+.6f}" for m in ['macro_fg_dice','rim_dice','cup_dice']]] for s in [41,42] for d in domains for b in ['SUP_CE','SUP_CED']]),
        'P2 相对 SUP_CE 的 Drishti cup 平均下降 −0.010916；seed41 为 −0.021771，标记 amber；'
        'seed42 为 −0.000061。平均保护线通过不等于所有类别都改善或临床安全。P1主候选相对SUP_CED的最差类别为seed31 Drishti cup −0.003280。',
        table(['阶段','配方','基线','seed','域','类别','所有amber差值'],[[r['phase'],r['arm'],r['baseline'],r['seed'],r['domain'],r['metric'],f"{float(r['delta']):+.6f}"] for r in costs if r['flag'] and r['phase']!='ALL5_SELECTED_DESCRIPTIVE']),
        '## 不确定性与晚期波动',
        '下表为预定2000次配对bootstrap、分析seed2026090807的95%百分位区间。患者在每域内重采样，所有配方/训练seed共享患者权重；'
        '训练seed则整块重采样配对轨迹。两个域固定等权，不把噪声draw或像素当独立样本。',
        table(['阶段','比较','重采样','95%区间','含0'],[[r['phase'],'MIX_CED − '+r['baseline'],r['resampling'],f"[{float(r['lower_95']):+.6f}, {float(r['upper_95']):+.6f}]",r['contains_zero']] for r in ci if r['arm']=='MIX_CED' and r['seed_scope']=='all']),
        'P2 相对两种监督基线的患者区间均在0以上；相对CTX的区间 [−0.001576,+0.014222] 含0。'
        '因此伪目标组件具有预定幅度和方向信号，但不能宣称其独立患者效应已经确定。仅2/3个训练seed的bootstrap是粗略敏感性，不能声称精确覆盖或显著性。',
        'LATE_EPOCH_VARIATION.csv 独立列出每条轨迹的60/80/100阶段波动；旧参考见P0_LATE_VARIATION.csv。'
        '这些epoch不构成独立重训练，不挑最佳阶段。P2 MIX_CED Q的seed SD0.011782高于SUP_CE的0.005503，'
        '即使配对增益两次都正，也不称为全面降低原始分数方差。',
        '## 机制与计算边界',
        '详见 MECHANISM_SUMMARY.md 和 MECHANISM_METRICS.csv。MIX_CTX与MIX_CED的学生输入及L来源计分相同；'
        'CTX仍使用U图像上下文，不能称纯监督。相同原L像素在互补两视图中恰好监督一次；没有额外干净L损失。'
        '单张末尾U在两个上下文中的损失分别权重0.5，再按唯一U接受支持归一化；不平均logits，不把重复U计成独立患者。'
        'GT-Dice仅作用于真实标签，未给伪标签加Dice。',
        '## 执行、内存与来源',
        f"冻结源码 `{verify['source']}`，本次只读重验服务器checkout干净，40任务/200诊断/40部署通过。"
        f"GPU3/4/5后台总耗时 {account['formal_wall_minutes']:.2f} 分钟。正式更新106000；历史正式累计354108。"
        '426次合成更新及8次smoke单列，本轮包含资格/smoke的成功更新合计106434。旧资格计数缺项没有填0或重跑。',
        table(['实际操作','正式矩阵(含诊断/部署)','资格+smoke后本轮全部'],[[k,counter[k],total[k]] for k in ['optimizer_steps','backward','ema_updates','model_forward','image_forward','sample_access','hdf5_open']]),
        '正式训练L样本访问212000次，U图像访问108160次；诊断image-only6500次及含GT的val样本6500次；'
        '部署原图1300次且新增GT读取0。上述是记录张次/loader调用，不能称为同等数量的唯一患者。'
        '每域唯一训练人群保持RIM L16/U63、Drishti L10/U41，val40/25。SUP不读训练U；CTX不做teacher-U前向。'
        'SUPERVISION_SOURCE_ACCOUNTING.csv逐epoch逐类记录真实曝光和U接受数；global loss/record数在三类行重复，不能跨类别重复求和。',
        '每任务最多student+当前EMA两份完整模型，最终仅部署学生；无历史教师、原型或CPU full shadow。'
        '每模型参数1936064 bytes、Adam3865412 bytes，惰性sigma/grad_update各1728 bytes不入Adam。'
        f"实际峰值Torch allocated {max(r['peak_allocated'] for p in mem.values() for r in p['verification']['receipts'])/2**20:.3f} MiB，"
        f"reserved {max(r['peak_reserved'] for p in mem.values() for r in p['verification']['receipts'])/2**20:.3f} MiB；不把两模型数解释为全部显存仅两倍参数。"
        '所有200诊断保持学生/EMA/Adam/RNG状态，真实诊断额外backward=0。操作计数含尝试与成功两列；合成中有故意缺路径断言，不是正式训练失败。',
        '## 停止与交付',
        '配方按本次冻结定义保留。仅准备 NEXT_STAGE_DRAFT.md，不开展新患者划分、标签预算、更多seed、CL或外部测试。'
        '同一split多轮暴露，优化seed复核不是独立患者验证；此结果不自动构成新颖性、SOTA、BCP完整复现或临床安全主张。',
        '源代码、协议、全部聚合指标、区间、成本、测试及退出证据公开；图像、GT、患者ID、逐病例预测和权重留NAS。'
        'EXECUTOR_FINAL_REPORT.md保留运行器原报告；原P1/P2选择与终态不改。发布提交、匿名可读性和NAS归档见PUBLICATION_VERIFICATION.json。'
        '运行本目录build_final_report.py仅重建聚合叙述并交叉检查计数及配对差，无数据/模型访问。']
    (R/'FINAL_REPORT.md').write_text('\n\n'.join(text)+'\n')
    mechanism=rows('MECHANISM_METRICS.csv');pr=[];qr=[]
    for d in domains:
        for a in ['SUP_CE','SUP_CED','MIX_CTX','MIX_CED']:
            rr=[r for r in mechanism if r['phase']=='P2' and r['epoch']=='100' and r['kind']=='probability_aggregate.csv' and r['model']=='student' and r['mode']=='posterior_mean_clean' and r['domain']==d and r['arm']==a]
            pr.append([d,a,*[f'{st.mean(float(r[k]) for r in rr if r[k]):.6f}' for k in ['patient_mean_NLL','patient_mean_multiclass_Brier','patient_mean_wrong_foreground_confidence']]])
        for c in ['1','2']:
            rr=[r for r in mechanism if r['phase']=='P2' and r['epoch']=='100' and r['kind']=='quality_aggregate.csv' and r['model']=='teacher' and r['mode']=='training_like' and r['domain']==d and r['arm']=='MIX_CED' and r['filter']=='teacher' and r['class_id']==c]
            qr.append([d,c,*[f'{st.mean(float(r[k]) for r in rr if r[k]):.6f}' for k in ['patient_mean_precision','patient_mean_accepted_correct_recall']]])
    (R/'MECHANISM_SUMMARY.md').write_text('# 机制摘要\n\n混合上下文对照自身有明显收益；增加U软目标提供较小增量，其患者区间仍含0。完整训练配方差不能解释为单像素因果证明。CE+Dice单独未稳定提高基础质量，普通MT_CED未获准复核。\n\nP2 epoch100原图最终学生的两优化seed均值（每模型先患者平均）：\n\n'+table(['域','配方','NLL','Brier','错误前景confidence'],pr)+'\n\nP2 MIX_CED教师筛选的val前景质量（training-like draw先在患者内平均，再两个seed等权）：\n\n'+table(['域','类1/rim 2/cup','precision','accepted-correct recall'],qr)+'\n\n这些是隔离val描述，不是隐藏train-U真实错误率；高confidence仍不等于正确。CE、GT-Dice、soft-CE、目标熵和去熵KL实际训练项及逐类来源曝光见SUPERVISION_SOURCE_ACCOUNTING.csv；同表全局loss在class行重复，除以steps求epoch均值，不跨类累加全局loss。\n\n互补M及1−M来源计分、odd-U重复权重、q/mask停止梯度、上下文臂U目标梯度为零、共同warm-up及部署已在冻结源码CPU/CUDA资格验证。未额外计算真实参数梯度网格。\n')
    print(json.dumps(dict(status=verify['terminal']['status'],formal_updates=106000,all_updates=106434,report='FINAL_REPORT.md')))

if __name__=='__main__':main()
