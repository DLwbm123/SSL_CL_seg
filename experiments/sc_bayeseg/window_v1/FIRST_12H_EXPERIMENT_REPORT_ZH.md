# SC-BayeSeg 路线：原始实验链前12小时总结

统计窗口（上海时间）：**2026-10-20 16:34:39—2026-10-21 04:34:39**。以R-START首次实际启动为原点，含首次失败、准确续接、阶段间隔与所有正式子阶段，不因改名或修复重置。本文汇总窗口内已结束的32份阶段结果；独立审计和公开交付可以在窗口之后完成，本文是事后核验汇总，**不声称全部审计/报告在第12小时之前已经交付**。更早P2/structure/RCF仅为来源背景，不计入本窗口成本；后来的缓存清理、资源准入检查另列现状，不混入12小时实验统计。

## 一、主要结论

1. **条件G的患者分割缺口被明显缩小，但未建立超越匹配U-Net的证据。** 初始端到端U/G为68.82%/60.14%；加入前景Dice与细化R空间接口后，G约70.5%—70.9%，匹配U-Net+Dice为70.70%。点估计与四翻转变体可达约71%—73%，但相关患者差异区间与反复开发限制不允许把最高均值宣称方法胜出。
2. **复杂随机后验的额外价值目前没有证明。** R32完整KL的原始NLL较低，但简单局部均值平方正则已复现主要收益；匹配温度之后，全部均值模型也能达到相近或更低NLL。不能把原始NLL改善直接归因Bayesian分布。
3. **可靠性主资格明确为负。** 同一个K1预测、75%同预测类别接受配额下，局部分歧选择错误率比校准置信度高1.7295pp，患者95%区间[1.3731,2.2680]pp。病例聚合曾改善，但简单前景体积参考更强、失败正例极少，未显示复杂分歧增量。
4. **四翻转改善平均指标，但条件G出现真实RV结构损失。** U-Net新增两例空预测仅移除原FP；G新增一例丢失1,388个TP、相位RV Dice23.24%→0。不能以整体Dice/NLL升而抹去这种失败。
5. **两图Dice100%只证明训练拟合资格。** 它不是独立患者分割结果，更不是Bayesian、SSL或CL收益。后续主线已经转到10位训练患者和23位患者不重叠开发验证，而不是继续只看两图。

## 二、方法与数据边界

本路线是自研UNet图像特征→H→冻结G的条件生成分割接口；当前患者路径不含图像生成因子F，**不是完整/官方BayeSeg复现**。本文既不能用这些结果否定官方方法，也不把改过目标的小模型称官方模型。

- 患者主线：canonical320 seed0，Siemens20%标注10患者230切片、20个ED/ES相位；验证23患者544切片、46相位。患者不重叠，但val已被历史版本及本轮多次评估，只称开发验证集。
- test24、38位U及目标域保持封存。验证推断只输入图像，预测先封存再标签计分；不使用验证GT优化局部q、bias、prior或选视图/阈值。
- 两图八视图与GT条件oracle单独报告：训练标签直接优化局部变量的结果不能作为image-only预测，也不能传入部署状态。
- 原生128 FP32概率恢复到384处理网格，按processed_spacing计算三维ED/ES指标，再每类及患者宏平均。原扫描逆变换未完成独立核查，不称原扫描网格。空预测和HD95的NA/惩罚保留。
- 患者bootstrap10000、seed163只描述固定训练/固定预测下患者抽样变化，不覆盖训练seed或MC全部不确定性。多配置、多轮开发均为探索；CI跨0不等于等效/非劣。

## 三、从两图接口资格到患者主线

**初始接口。** R-START四臂各600步共2400更新：R0直接头35.89%、共享R2直接头4.90%，生成分支约26.81%/29.06%；断共享梯度对照的直接头与R0逐保存点精确一致。说明这个固定共享配置损害直接预测，不代表所有生成辅助都无效。后续原CE和类平均CE监督资格仍未过24项；一次Adam CPU步共享导致1000次无效成本，修复只重做无效臂，负结果和成本不抹除。

**标准U-Net。** 新初始化/freshAdam固定3000步，两图八视图24项Dice全1.0，NLL约.00263，建立该小训练任务可拟合；新旧网络/初始化/参数量不同，不把差值当纯架构因果，更不能称患者Dice100%。

**同族局部oracle及一次G偏置收口。** 原G冻结，两个初态各1000局部步约89.26%/89.80%，分别22/24、21/24，仍有类3缺口。相同500步预算的4Theta×1local不优于2×2，类平均CE反降到约87.13%且出现其他类退化；KL方向/交叉采样诊断未证明KL为唯一主因。一次G3三均值干预配合GT条件q可让两份不同G副本在两图达到稳健24/24，但**迁移到患者H训练后G3只有53.53%，低于G0的54.17%**。oracle可达性没有自动变成图像识别能力。

**患者主线起点。** 共享5000步U-Net监督预训练后，冻结特征、匹配训练两头：DIRECT68.14%、G/H54.17%，G低13.97pp且23患者均低。解除特征冻结后DIRECT68.82%、G60.14%，G改善5.97pp但仍低8.68pp，95%[−11.99,−5.86]pp；冻结特征是部分限制，主要剩余MYO/RV缺口。DIRECT冻结头不是充分端到端U-Net上限，因此端到端控制另报。

训练患者读出进一步发现U/G训练Dice87.75%/76.43%，MYO81.37%/58.59%，不是单纯验证域差异。P0分解显示G训练MYO召回47.23%、开发34.76%，大量误分背景/LV。旧R16背景主导NLL改善的观察不能沿用到后来的R32：R32复核中K1相对K0四类条件NLL均降低、RV约贡献总体降幅一半。后验尺度上界饱和和KL绝对值只能作现象，不能当完整梯度根因。

依据：[早期统一报告](SC_BAYESEG_RESEARCH_SUMMARY_20261020_ZH.md)、[患者初轮](PATIENT_VALIDATION_REPORT_ZH.md)、[G3迁移](PATIENT_G3_REPORT_ZH.md)、[端到端](PATIENT_E2E_REPORT_ZH.md)、[训练读出](PATIENT_TRAIN_READOUT_REPORT_ZH.md)、[P0补充](PATIENT_P0_SUPPLEMENT_REPORT_ZH.md)。早期报告的“下一步/禁止H”等是其写作时状态，后续授权与已完成患者结果以本文和各后续协议为准，不能用历史候选复活旧实验。

## 四、损失与空间接口：主要患者结果

G_D等为R16；R32_K0为R32去局部KL，K1为完整局部KL，KQ只将R-KL按维数比例降至1/4。所有相关G/prior/initial冻结，主训练更新UNet特征与H。Dice辅助监督与去KL/分块权重改变了原ELBO目标，不能统称纯Bayesian推断改进。

|配置|训练Dice%|开发Dice%|LV%|MYO%|RV%|原始NLL|空预测类相位|
|---|---:|---:|---:|---:|---:|---:|---:|
| G_K0 | 85.8616 | 67.2461 | 78.9052 | 63.5694 | 59.2636 | 0.0622385 | 5 |
| G_D | 85.7120 | 67.1546 | 79.0721 | 63.3459 | 59.0459 | 0.0652245 | 4 |
| G_DK0 | 85.4741 | 69.0235 | 80.2269 | 64.1026 | 62.7409 | 0.1057375 | 0 |
| U_D | 90.2017 | 70.7044 | 81.5098 | 69.4124 | 61.1910 | 0.0841448 | 1 |
| R32_K0 | 88.2608 | 70.8689 | 82.2158 | 68.5478 | 61.8433 | 0.1053005 | 2 |
| R32_K1 | 87.9762 | 70.4881 | 82.3782 | 66.9466 | 62.1395 | 0.0535232 | 2 |
| R32_KQ | 87.8384 | 70.4085 | 80.2925 | 67.9600 | 62.9730 | 0.0617482 | 1 |

损失/KL因子的原G CE+KL开发60.14%；G_D变为67.15%，G_DK0为69.02%，U_D为70.70%。G_D−U_D低3.55pp，[−5.46,−1.46]；去KL后低1.68pp，[−4.28,+1.20]，不证明非劣。有Dice时去KL的额外收益仅+1.87pp且区间跨0；Dice×去KL交互−5.24pp，不能把两种收益简单相加，更不能宣布KL普遍有害。

R16→R32保持参数数目不变、局部R维数增4倍，训练MYO改善2.89pp，[2.19,3.64]；beta0用于避免KL维数混杂，不把不同形状local噪声说成逐draw相同。恢复完整KL后K1仍70.49%，对U_D差−.2163pp，[−2.3233,+2.0323]；降低R-KL到1/4未显示优于K1。K1原始NLL比U_D低.03062，区间为负，但Brier差未决、空RV2例，后续必须匹配校准和点估计控制。

依据：[匹配损失/KL](PATIENT_LOSS_FACTOR_REPORT_ZH.md)、[R32](PATIENT_R32_REPORT_ZH.md)、[R32恢复KL](PATIENT_R32_KL_REPORT_ZH.md)。完整原CE控制、全部类别/患者、HD95、失败与NA均在原匿名JSON，未因本表聚焦后继而撤下。

## 五、概率风险：随机后验尚未显示超越简单均值的增量

K1是局部Gaussian后验采样，G权重也采样；D0/Dμ仅局部c/R取均值、G权重仍采样。Dμ加.5*(均值平方和)/128²，不是Dirac KL。MEAN_READ是原Dμ采样训练终态在G权重均值下推断；Mμ则训练和推断均用权重均值。方差专属800输出标量与Adam矩冻结，shared hidden仍训练；不声称未使用σ输出数值必然不变。点估计G调用更少，相同5000更新不等相同算力。

|配置|开发Dice%|原始NLL|同五患者折校准NLL|校准Brier|空预测类相位|
|---|---:|---:|---:|---:|---:|
| U_D | 70.7044 | 0.0841448 | 0.0520610 | 0.0203980 | 1 |
| R32_K0 | 70.8689 | 0.1053005 | 0.0578912 | 0.0208510 | 2 |
| R32_K1 | 70.4881 | 0.0535232 | 0.0488497 | 0.0206293 | 2 |
| R32_KQ | 70.4085 | 0.0617482 | 0.0493940 | 0.0208532 | 1 |
| D0 | 72.1912 | 0.1031418 | 0.0575541 | 0.0202046 | 1 |
| Dmu | 71.3869 | 0.0511182 | 0.0465918 | 0.0205600 | 1 |
| MEAN_READ | 72.0211 | 0.0526299 | 0.0458086 | 0.0202252 | 1 |
| Mmu | 70.2842 | 0.0556862 | 0.0465287 | 0.0204325 | 0 |
| U_D_LS | 71.0155 | 0.0842520 | 0.0794506 | 0.0227713 | 0 |
| U_D_TTA | 72.1949 | 0.0572239 | 0.0418533 | 0.0180111 | 3 |
| G_MEAN_TTA | 72.7563 | 0.0415527 | 0.0395691 | 0.0182631 | 2 |

- D0−U_D Dice+1.4868pp，[−.7015,+3.9877]；Dμ−U_D+.6825pp，[−1.9384,+3.2259]，不能据最高均值宣布胜出。D0原始NLL反而比U_D高；Dμ相对D0原始NLL降.05202，均值收缩已复现主要风险收益。
- 匹配校准后K1−Dμ NLL+.002258，[−.000959,+.005872]，尚无局部分布额外价值证据。Mμ−K1校准NLL−.002321，[−.004594,−.000204]，更不能把复杂采样视为当前风险改善的必要条件。
- 推断只取权重均值使Dμ Dice+0.6343pp，但原始NLL变差+.001512；匹配校准后NLL反而略好。不能从未校准混合NLL推出分布收益。
- 同权重均值推断下，Mμ较采样训练Dμ的Dice低1.7370pp，[−3.1393,−.4061]，提示该配置的训练噪声可能有结构正则作用；然而主NLL差+.003056及校准后+.000720都跨0，空预测还1→0，不是全面失败或普遍Bayesian因果证明。
- 固定U-Net标签平滑epsilon=.1的主校准NLL明显更差：对U_D差+.027390，[+.021435,+.034917]，校准Brier也差。此单探针关闭，不扫描epsilon救结果；它失败也不反证Bayesian有效，因为均值模型已复现主要风险表现。

校准采用相同患者排序mod5的五折，fit排除评分患者；固定stride64/phase等权、单全局T及预定范围/优化器，留出全像素评分，argmax0改变。拟合温度确实使用其他开发患者标签，是标量优化，不叫完全零优化；重新分折不产生未触碰最终test。TTA的原始/校准风险均为平均后概率上的风险，不与native期望CE混报。

依据：[MC复核](PATIENT_R32_MC_REPORT_ZH.md)、[RV/风险/温度](PATIENT_PRO_DIAGNOSTICS_REPORT_ZH.md)、[D0/Dμ](PATIENT_POINT_ESTIMATE_REPORT_ZH.md)、[点估计校准](PATIENT_POINT_CALIBRATION_REPORT_ZH.md)、[权重均值推断](PATIENT_GLOBAL_MEAN_REPORT_ZH.md)、[均值训练](PATIENT_GLOBAL_MEAN_TRAINING_REPORT_ZH.md)、[均值校准](PATIENT_GLOBAL_MEAN_CALIBRATION_REPORT_ZH.md)、[标签平滑](PATIENT_UNET_SMOOTHING_REPORT_ZH.md)。

## 六、可靠性与空间读出：必须保留的负结果

**像素选择主资格。** 同一K1分割预测、每预测类别相同75%配额，local分歧前景宏错误率14.0427%，calibrated confidence12.3132%，差+1.7295pp，[1.3731,2.2680]；50/90%覆盖率及global/total也更差。不同模型预测区域不同，不能拿跨模型条件精度冒充整体召回优势。随机RV draw在共同失败相位虽偶有RV区域，TP仍为0，不叫结构恢复。

**病例聚合一次收口。** 原whole top1%打分的低Dice失败检测AUROC异常，经独立两两比较和方向核验确认不是代码反向；没有翻分数追结果。另立固定前景分层top10%聚合后K1 calconfidence/local AUROC升至.7333/.8，但简单1−预测前景占比为.9833，低Dice与RV失败正例仅3/2。此为事后探索、已关闭支线，不声称复杂分歧胜过简单几何参考。

**固定四翻转。** U_D和原Dμ终态的G均值路径各复用identity，新增横/纵/双翻转逆变换、四份等权平均，0训练更新。U增益+1.4905pp，[+.4696,+2.4566]；G+.7352pp，[−.4786,+1.8609]。主增益交互G−U=−.7553pp，[−2.0884,+.6237]未决。原始和校准风险各自都改善，但U的校准NLL增益更大。两管线训练/正则历史不同，空间TV差不能解释成纯G架构因果。

**迁移诊断。** U新增两例空RV原TP均0，仅移除559/7FP；G新增一例原1388TP→0、相位RV Dice23.24%→0。全46相位G的RV净TP−18,955、FP−38,208，整体存在精度/召回权衡；体素汇总不替代患者宏Dice。此支线不再扫阈值、视图或权重。三个新视图完整概率未保存，审计未重跑网络；只能核保存ensemble、标量恒等和完整指标，不能声称逐视图输出独立复演或指认某一视图主因。

依据：[可靠性](PATIENT_RELIABILITY_REPORT_ZH.md)、[病例聚合](PATIENT_CASE_AGGREGATION_REPORT_ZH.md)、[匹配四翻转](PATIENT_FLIP_READOUT_REPORT_ZH.md)、[TP/FP迁移](PATIENT_TTA_TRANSITIONS_REPORT_ZH.md)。

## 七、实际成本、工程故障与审计口径

32个阶段结果按原始elapsed<=43200秒纳入，全部有独立PASS回执。以下是已发生的成本，不把审计通过当方法成功：

|成本项目|数量/说明|
|---|---|
|累计optimizer应用|98,400，含局部oracle优化和网络训练，不是98,400次独立实验|
|其中已知无效更新|1,000次Adam CPU step共享故障成本，已保留；其余97,400不等于全部产生科学收益|
|无参数应用的类别/KL和交叉MC反向请求|34+128=162；另原资格2次，其他工程准入/失败请求各阶段另账|
|G3 Newton优化|12次接受、12候选、16导数遍历；0Adam不等于0优化|
|温度标量优化|55 fits、680次目标函数调用；旧控制复用不重复计数|
|末次正式阶段终点|原elapsed42,464.04737秒，即首启后11小时47分44秒|
|末次正式阶段guard累计占用|四根17,910,382,592字节，新路径累计15,028,375,552字节，二者统计口径不同，不当末阶段新增量|

每阶段完整网络/局部状态、Adam/RNG、源/冻结、真实更新与调用账本、封存预测和全部指标按相应协议验证；metric误差以各自冻结容差为准，不把所有早期CPU/GPU差写成0。公开汇总只重新读取已审计数值，不重跑训练、网络、MC、GT评分或温度fit。

保留的工程故障包括：首次R-START默认CUDA mean确定性错误（0训练更新）；监督资格Adam CPU step共享（1000无效更新）；偏置20曲线序列化失败后只恢复汇总、0重复扫描；因子审计未匹配deterministic插值产生2.55e−10舍入差，修复审计设置而不重跑网络；点估计准入entrypoint/CUBLAS环境错误；均值消融审计缺统计模块；标签平滑控制入口缺旧依赖搜索路径。这些按原回执保留真实成本和失败文件，未删除失败后重新计时。

12小时是墙钟窗口，包含分析、审计、等待和阶段间隔，不等于12小时GPU忙时。本文没有按阶段耗时相加推算GPU利用率，也未把optimizer步数当严格算力匹配：Point均值、全权重均值、U-Net的G调用预算不同。

## 八、后继决定与窗口外现状

科学上优先候选是训练端跨翻转空间一致性，且G/U各自需要同双视图计算的无一致性控制；具体系数、初态、固定终点、指标和成本须另冻协议。目前没有可执行新训练启动，也不把此方向算作完成结果。不自动引入SSL/CL/teacher/F/后验尺度族、新训练seed或最终test；不以自主授权替代冻结边界。

窗口之后的资源检查表明既有18GB软限余量不足下一完整匹配实验预留1.2GB；用户后续授权清理确认无用缓存，实际清除约1.3MB，最新本地回执保守四根17,909,714,944字节、余90,285,056字节，仍不准入。清理不计入前12小时训练成本，历史模型/封存预测/失败/审计输入均保留。服务器物理磁盘有余量，当前是旧本链预算门限，不擅自删除科学证据或放宽限额。原首次24h及失败续跑共享时钟不重置；唯一半小时监测保留，未变资源阻塞静默。

当前结论应写成：**已经建立患者级image-only比较并缩小G的分割缺口，但尚无相对匹配U-Net的可靠优势，也无复杂Bayesian随机部分的额外价值证据；下一路线需要匹配控制和资源准入。**

## 附录：前12小时阶段索引

终点为首启后的elapsed，不是阶段纯计算耗时；全部32阶段有PASS回执，相关原始匿名JSON与审计均可点击。阶段内多臂不重复当作新增患者样本。

|阶段|首启后终点|optimizer应用（含已知无效）|匿名结果|审计|
|---|---:|---:|---|---|
| R_START | 00:02:50 | 2,400 | [JSON](results/R_START.json) | [PASS](results/R_START_AUDIT.json) |
| SUPERVISED_QUALIFICATION | 00:17:18 | 3,000 | [JSON](results/SUPERVISED_QUALIFICATION.json) | [PASS](results/SUPERVISED_QUALIFICATION_AUDIT.json) |
| UNET_QUALIFICATION | 00:46:21 | 3,000 | [JSON](results/UNET_QUALIFICATION.json) | [PASS](results/UNET_QUALIFICATION_AUDIT.json) |
| SAME_FAMILY_ORACLE | 00:59:27 | 2,000 | [JSON](results/SAME_FAMILY_ORACLE.json) | [PASS](results/SAME_FAMILY_ORACLE_AUDIT.json) |
| ORACLE_READOUT | 01:03:43 | 0 | [JSON](results/ORACLE_READOUT.json) | [PASS](results/ORACLE_READOUT_AUDIT.json) |
| CLASS_KL_GRADIENT | 01:10:08 | 0 | [JSON](results/CLASS_KL_GRADIENT.json) | [PASS](results/CLASS_KL_GRADIENT_AUDIT.json) |
| ORACLE_GRADIENT_MC | 01:12:59 | 0 | [JSON](results/ORACLE_GRADIENT_MC.json) | [PASS](results/ORACLE_GRADIENT_MC_AUDIT.json) |
| ORACLE_ALLOCATION | 01:16:50 | 2,000 | [JSON](results/ORACLE_ALLOCATION.json) | [PASS](results/ORACLE_ALLOCATION_AUDIT.json) |
| ORACLE_CLASS_RISK | 01:26:54 | 1,000 | [JSON](results/ORACLE_CLASS_RISK.json) | [PASS](results/ORACLE_CLASS_RISK_AUDIT.json) |
| CLASS3_BIAS_FEASIBILITY | 01:59:50 | 0 | [JSON](results/CLASS3_BIAS_FEASIBILITY.json) | [PASS](results/CLASS3_BIAS_FEASIBILITY_AUDIT.json) |
| FIXED_BIAS_MC | 02:02:42 | 0 | [JSON](results/FIXED_BIAS_MC.json) | [PASS](results/FIXED_BIAS_MC_AUDIT.json) |
| G_BIAS_CLOSURE | 02:52:58 | 0 | [JSON](results/G_BIAS_CLOSURE.json) | [PASS](results/G_BIAS_CLOSURE_AUDIT.json) |
| PATIENT_VALIDATION | 02:56:03 | 15,000 | [JSON](results/PATIENT_VALIDATION.json) | [PASS](results/PATIENT_VALIDATION_AUDIT.json) |
| PATIENT_G3 | 03:07:35 | 5,000 | [JSON](results/PATIENT_G3.json) | [PASS](results/PATIENT_G3_AUDIT.json) |
| PATIENT_E2E | 03:48:33 | 10,000 | [JSON](results/PATIENT_E2E.json) | [PASS](results/PATIENT_E2E_AUDIT.json) |
| PATIENT_TRAIN_READOUT | 04:09:48 | 0 | [JSON](results/PATIENT_TRAIN_READOUT.json) | [PASS](results/PATIENT_TRAIN_READOUT_AUDIT.json) |
| PATIENT_P0_SUPPLEMENT | 04:23:29 | 0 | [JSON](results/PATIENT_P0_SUPPLEMENT.json) | [PASS](results/PATIENT_P0_SUPPLEMENT_AUDIT.json) |
| PATIENT_LOSS_FACTOR | 04:49:03 | 20,000 | [JSON](results/PATIENT_LOSS_FACTOR.json) | [PASS](results/PATIENT_LOSS_FACTOR_AUDIT.json) |
| PATIENT_R32 | 05:23:38 | 5,000 | [JSON](results/PATIENT_R32.json) | [PASS](results/PATIENT_R32_AUDIT.json) |
| PATIENT_R32_KL | 05:55:58 | 10,000 | [JSON](results/PATIENT_R32_KL.json) | [PASS](results/PATIENT_R32_KL_AUDIT.json) |
| PATIENT_R32_MC | 06:20:07 | 0 | [JSON](results/PATIENT_R32_MC.json) | [PASS](results/PATIENT_R32_MC_AUDIT.json) |
| PATIENT_PRO_DIAGNOSTICS | 06:53:29 | 0 | [JSON](results/PATIENT_PRO_DIAGNOSTICS.json) | [PASS](results/PATIENT_PRO_DIAGNOSTICS_AUDIT.json) |
| PATIENT_POINT_ESTIMATE | 07:25:29 | 10,000 | [JSON](results/PATIENT_POINT_ESTIMATE.json) | [PASS](results/PATIENT_POINT_ESTIMATE_AUDIT.json) |
| PATIENT_POINT_CALIBRATION | 07:42:59 | 0 | [JSON](results/PATIENT_POINT_CALIBRATION.json) | [PASS](results/PATIENT_POINT_CALIBRATION_AUDIT.json) |
| PATIENT_RELIABILITY | 08:25:29 | 0 | [JSON](results/PATIENT_RELIABILITY.json) | [PASS](results/PATIENT_RELIABILITY_AUDIT.json) |
| PATIENT_CASE_AGGREGATION | 08:53:01 | 0 | [JSON](results/PATIENT_CASE_AGGREGATION.json) | [PASS](results/PATIENT_CASE_AGGREGATION_AUDIT.json) |
| PATIENT_GLOBAL_MEAN | 09:16:26 | 0 | [JSON](results/PATIENT_GLOBAL_MEAN.json) | [PASS](results/PATIENT_GLOBAL_MEAN_AUDIT.json) |
| PATIENT_GLOBAL_MEAN_TRAINING | 09:44:49 | 5,000 | [JSON](results/PATIENT_GLOBAL_MEAN_TRAINING.json) | [PASS](results/PATIENT_GLOBAL_MEAN_TRAINING_AUDIT.json) |
| PATIENT_GLOBAL_MEAN_CALIBRATION | 10:12:07 | 0 | [JSON](results/PATIENT_GLOBAL_MEAN_CALIBRATION.json) | [PASS](results/PATIENT_GLOBAL_MEAN_CALIBRATION_AUDIT.json) |
| PATIENT_UNET_SMOOTHING | 10:51:02 | 5,000 | [JSON](results/PATIENT_UNET_SMOOTHING.json) | [PASS](results/PATIENT_UNET_SMOOTHING_AUDIT.json) |
| PATIENT_FLIP_READOUT | 11:24:56 | 0 | [JSON](results/PATIENT_FLIP_READOUT.json) | [PASS](results/PATIENT_FLIP_READOUT_AUDIT.json) |
| PATIENT_TTA_TRANSITIONS | 11:47:44 | 0 | [JSON](results/PATIENT_TTA_TRANSITIONS.json) | [PASS](results/PATIENT_TTA_TRANSITIONS_AUDIT.json) |

[机器可读汇总与统计口径](results/FIRST_12H_SUMMARY.json)。源码、冻结协议、原阶段报告、匿名全患者指标与独立审计随原阶段公开；不发布患者身份/图像、私有路径、模型权重/Adam/RNG、概率/梯度/选择mask、原日志、凭据或论文PDF。本文不覆盖窗口外新实验，不启动或复活任何已退休worker。
