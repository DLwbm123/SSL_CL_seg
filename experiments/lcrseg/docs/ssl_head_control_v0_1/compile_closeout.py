"""Post-run prose from public frozen tables only. Does not evaluate or choose a model."""
import csv,json
from pathlib import Path
D=Path(__file__).resolve().parent
read=lambda n:json.loads((D/n).read_text())
rows=lambda n:list(csv.DictReader((D/n).open()))
def table(headers,rr):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rr])
def num(x):return f'{float(x):.6f}' if x not in ('',None) else 'undefined'
metrics=rows('SINGLE_DOMAIN_METRICS.csv');gate=read('SCREEN_DECISION.json');confirm=read('CONFIRMATION.json');check=read('CLOSEOUT_VERIFICATION.json');status=read('STATUS.json');mass=rows('PAS_FIXED_MASS_AUDIT.csv')
assert len(metrics)==14 and check['formal_updates']==37100 and gate['path']=='P2_SUP' and status['ENGINEERING']=='COMPLETE'
def score(seed,arm,domain,key='macro_fg_dice'):
    return float(next(r for r in metrics if int(r['seed'])==seed and r['arm']==arm and r['domain']==domain)[key])
def Q(seed,arm):return sum(score(seed,arm,d) for d in ('RIM_ONE_r3','Drishti_GS'))/2
p1=[]
for arm in ('LIN_SUP','LIN_MT_CONF','LIN_MT_PAS'):
    p1.append([arm,num(score(11,arm,'RIM_ONE_r3')),num(score(11,arm,'Drishti_GS')),num(Q(11,arm)),num(Q(11,arm)-Q(11,'LIN_SUP'))])
p2=[[x['seed'],num(x['Q']),num(x['reference_Q']),num(x['Q_delta']),num(x['delta'][0]),num(x['delta'][1])] for x in confirm['per_seed']]
classes=[[r['seed'],r['domain'],r['arm'],num(r['rim_dice']),num(r['cup_dice']),num(r['macro_fg_dice'])] for r in metrics]
p0=[r for r in mass if r['arm'] in ('MT_PAS_G0','MT_PAS_G1')];pas=[r for r in p0 if r['method']=='PAS_NATIVE' and r['model']=='teacher' and r['class_id']!='0'];comparisons=[]
for r in pas:
    paired=[next(x for x in p0 if all(x[k]==r[k] for k in ('domain','arm','mode','model','class_id')) and x['method']==method) for method in ('PAS_NATIVE','CONF_TOPK_MATCHED','RANDOM_MATCHED')]
    comparisons.append([r['domain'],r['arm'],r['mode'],'rim' if r['class_id']=='1' else 'cup',*[num(x['patient_mean_precision']) for x in paired],num(r['patient_mean_geometry_K']),r['n_patients_defined_precision'],num(r['pooled_LR_sim_given_C'])])
assert min(float(r['patient_mean_effective_teacher_class_counts_equal']) for r in mass)==1
p0text='''# PAS 等数量审计

P0 的4个旧最终模型对已全部完成：65例×2臂×5次预测×2模型=1300次 sample-model 前向，0更新。实际原型来自各旧模型 epoch96 刷新，未重拟合原型。旧缓存缺少所需概率/特征字段，因此按授权重做固定前向；没有替换epoch或seed。

下表为教师 argmax 主口径的前景 precision。4次随机排列先平均，再平均同一患者的4次训练式预测，最后按患者平均；clean 单列。K为每患者/预测draw的平均几何接受数。LR为同一confidence集合C内、按教师正确性计算的汇总筛选比，不是假设条件独立后的保证。

'''+table(['域','旧臂','预测模式','类别','PAS','同数量 confidence top-k','同数量随机','平均K','有效患者数','pooled LR'],comparisons)+'''

全432个公开聚合行覆盖旧P0和新LIN_MT_PAS固定快照，包括background/rim/cup、teacher/student两种标签来源和两种预测模式。每组有效计分数量相等比例均为1。GT255没有参与K或排序；当前观察到计分支持相等不改变这一边界。新旧总计1950组预测掩码密封记录已核验为先密封后读GT。GT、患者标识及逐病例像素记录不公开。

旧G0以及多数cup对照几乎相同。Drishti G1 rim 的PAS相对随机有正差：clean约+0.0111、训练式约+0.0244；但均低于相同K的confidence top-k，分别约−0.0242和−0.0132。RIM G1 rim训练式PAS还低于随机对照约0.0062。因此不能把减少数量或改变类别构成后的precision变化直接解释为PAS提供了稳定、额外的正确性信息。

这是描述性结果，没有统计显著性、逐病例安全性或部署风险保证；P0未用于候选选择。PAS在少数cup行相对top-k有很小正差，同样完整保留，不能扩大为统一优越性结论。

完整 PAS_FIXED_MASS_AUDIT.csv 还保留预测覆盖率、图像覆盖率、accepted-correct recall、正确/错误计数、接受分歧比例、各指标有效患者数、LR无穷/undefined计数，以及student标签来源的对照。空分母不填1；LR不加伪计数。汇总像素计数是重复draw-pixels，不作为额外患者。新LIN_MT_PAS在epoch40/60/80/100的同数量审计同表提供，epoch20没有实际训练原型库。
'''
(D/'PAS_FIXED_MASS_AUDIT.md').write_text(p0text)
probs=rows('PROBABILITY_DIAGNOSTICS.csv');probtable=[]
for r in probs:
    if r['epoch']=='100' and r['mode']=='posterior_mean_clean' and r['model']=='student':
        probtable.append([r['seed'],r['domain'],r['arm'],*[num(r['patient_mean_'+x]) for x in ('NLL','multiclass_Brier','fraction_pmax_above_099','wrong_foreground_confidence','wrong_foreground_count')]])
final='''# 分类头对照与 PAS 等数量归因 V0.1：最终报告

完整固定队列已结束。工程完成；SSL增量未建立；线性头的监督参考改善未在两个新优化种子上通过预定复核。本轮到此停止，不继续同一val上的温度、阈值、增强、权重或种子搜索，不进入CL。

## 终态与执行范围

'''+table(['维度','终态'],[[k,status[k]] for k in ('ENGINEERING','PAS_MATCHED_COUNT_EVIDENCE','HEAD_SUPERVISED_EFFECT','SSL_INCREMENT','PAS_INCREMENT','CL_AND_SOTA')])+'''

从405337e71d22aef011a041664fbc1b4b72b60408建立独立分支 codex/ssl-classifier-head-control-v0-1。实际执行源码始终为 f478cc204a87a807858c76a95fc4246d301eeff2；后续提交仅补充公开报告。冻结原始协议、HEAD_CONTRACT、来源和门槛保持不变。旧Foundation的SSL_FOUNDATION_NOT_ESTABLISHED及更早历史证据均保留。

P0四组已完成。P1六任务15900步；两个SSL候选均未过筛查，但监督参考改善通过，按唯一获准分支完成P2_SUP八任务21200步。正式合计37100步，与manifest预算、逐步日志、训练receipt及成功退出记录一致；历史累计168608步。未启动P2_SSL或旧seed12/13。

## P1：固定seed11筛查

主结果是最终epoch100学生，Q为两域等权rim/cup宏平均Dice；表格四舍五入只为阅读，所有判断使用完整精度。

'''+table(['臂','RIM-ONE-r3 Dice','Drishti-GS Dice','Q','相对LIN_SUP的Q差'],p1)+'''

LIN_MT_CONF与LIN_MT_PAS的Q增量分别为−0.004354和−0.002511，均未达到+0.010；CONF也越过单域−0.005保护线。不能以EMA或某个更早epoch替代最终学生。

LIN_SUP相对更强旧监督参照SUP_G1的Q增量为+0.011428，且各域及rim/cup保护条件满足，因此自动进入监督分支复核。旧SUP_G0、SUP_G1、MT_PAS_G1的Q分别为0.696040805、0.705364552、0.712331688；这些是固定历史参照，未重训旧seed11任务。

## P2_SUP：新优化种子复核

'''+table(['seed','LIN_SUP Q','SUP_G1 Q','Q差','RIM域差','Drishti域差'],p2)+f'''

两个新种子的平均Q差为{num(sum(x['Q_delta'] for x in confirm['per_seed'])/2)}，通过均值+0.010门槛。但seed22的Q差为−0.002053，违反每seed必须为正；RIM域差−0.013910低于−0.005；同域cup差−0.032346低于−0.020。因此终态为SUPERVISED_HEAD_REFERENCE_NOT_REPLICATED。均值为正不能覆盖这些失败条件，也不把seed11混入确认统计。

## 全部最终rim/cup代价

'''+table(['seed','域','臂','rim Dice','cup Dice','宏平均Dice'],classes)+'''

完整阶段曲线见EPOCH_CURVES.csv，学生与EMA分别保留；完整差值及head×SSL对照见HEAD_SSL_CONTRASTS.csv。

PAS相对同头CONF的seed11两域平均差为+0.001844：RIM域+0.006975，Drishti域−0.003287，方向不一致。它仍低于同头监督参考，没有获准SSL新种子复核，故只是单seed组件对照，不能声称PAS增益可复核。

## 等数量审计、概率和实际损失

P0完整解释见PAS_FIXED_MASS_AUDIT.md。PAS并未显示相对同数量confidence top-k的一致优势；部分Drishti rim相对随机的正差不等于分割增量或安全性。

以下均为最终学生clean evaluator的患者平均；错误前景数量是平均每例像素数。

'''+table(['seed','域','臂','NLL','Brier','pmax>0.99比例','错误前景confidence','错误前景数量'],probtable)+'''

线性头仍存在很高的confidence，seed11有约91.7%–96.2%的有效像素pmax>0.99。P2中线性头NLL均低于对应SUP_G1，但seed22 RIM Dice仍退化；概率尺度/校准改善不能替代Dice或SSL门槛。未拟合温度。

实际SSL训练按教师预测类记录接受数、MSE、weighted MSE及应用真实mask/分母/lambda后的解析局部logit梯度。四个SSL任务的背景接受像素占约85.0%–91.7%，但背景weighted MSE份额约7.0%–34.3%，因此单看像素coverage不能推出损失由背景支配。TRAINING_CLASS_LOSS.csv是实际训练统计；GRADIENT_LOGIT_DIAGNOSTICS.csv及CORRECTNESS_STRATA.csv是固定evaluator分层。局部logit梯度不是全网络参数梯度，不能据此宣称encoder梯度消失。完整分位点、10等宽bin可靠性表、有效分母和概率质量表全部提供。

## 工程、成本与部署

14个训练任务、14次单学生部署验证、70次epoch20/40/60/80/100诊断以及4次P0全部完成。32个子进程exit_code均为0；70次诊断的学生/EMA/RNG/训练原型状态均保持不变。未见正式失败或重试。实际训练37100次backward、37100次Adam更新、37100次EMA更新；GAS监督导数调用10600次是SUP_G1既有算法组成，不是新增诊断backward。额外真实诊断backward为0。

训练前的精确源码CPU/CUDA各8项测试通过；共5次开发/精确源码资格运行805次合成更新，另有隔离真实smoke8次更新，均不算入37100正式步。源码中的断点恢复测试逐值核验学生/EMA/Adam/head_mode/RNG/数据顺序，实际任务无恢复重跑。

'''+f'''14个任务的完整模型数均为学生+当前EMA共2份，部署进程均为1份学生。每份参数1936064字节；惰性sigma和grad_update各1728字节，grad_update不进入Adam；线性sigma冻结，所有任务sigma无梯度。峰值CUDA allocated约{max(x['peak_allocated'] for x in check['tasks'])/2**20:.2f} MiB，reserved约{max(x['peak_reserved'] for x in check['tasks'])/2**20:.2f} MiB。整个队列墙钟约{check['wall_seconds']/60:.2f}分钟，后台进程已退出。

'''+'''训练学生L前向74200张次，学生U/EMA U各16640张次；训练原型416张次，固定诊断原型910张次，新学生/EMA诊断各11375张次，P0额外1300次sample-model；单学生部署另有455张次前向。完整计数、loader打开次数、子进程退出、状态及内存见TRAINING_AND_MEMORY_ACCOUNTING.json和CLOSEOUT_VERIFICATION.json。最终部署仅加载head_mode明确的学生checkpoint，复现所有14个任务的最终预测，无新增GT读取。

## 交付边界与停止

公开代码、协议、汇总指标、完整门槛、资格/部署/内存和退出证据。未发布GT、患者标识、逐病例像素、原始影像和checkpoint；这些保留在NAS的本轮create-only目录。SOURCE_AND_INPUT_LINEAGE.json保留启动前冻结来源，SERVER_INPUT_LINEAGE.json提供服务器输入核验。原始编译结果也保留在服务器，不被这份解释性报告覆盖。

这仅是固定data_split_seed0上的优化随机性复核，不是跨划分/外部域验证、CL或SOTA结论。本轮关闭这一固定配方的分类头参数化排查；没有启动新模块、额外种子、风险拟合、冻结层或CL。未读取test/隐藏GT或旧formal_03，未改变旧终态/锁，未合并main。
'''
(D/'FINAL_REPORT.md').write_text(final)
