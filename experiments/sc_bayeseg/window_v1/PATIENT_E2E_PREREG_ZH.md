# 患者主线后继：匹配解除U-Net特征冻结

2026-10-20，运行前冻结。原冻结特征DIRECT68.1415%、G0/H54.1691%；G3/H53.5329%，G3−G0为−.6362个百分点。两图G偏置收口不再延长。当前缺少相同病例/历史下充分端到端更新的U-Net对照，且G分支只得到为直接CE预训练的冻结特征。本实验用两臂同时解除特征冻结，检验固定特征限制，并补上普通端到端U-Net继续训练对照。

## 两臂与匹配控制

两臂同源patient_validation_v1/PRETRAIN/S5000全模型与RNG，freshAdam、固定5000新增步（历史各10000步），不重复共享5000预训练。DIRECT_E2E：U-Net特征及原复制16→4直接头全可训练、原像素CE；CONDITIONAL_G_E2E：U-Net特征与原hc/hr全部可训练、原G0仍冻结、原期望像素CE+局部KL/N。H仍原R_START/R1 S0000初始化，不用G0/H或G3/H终态，更不用oracle局部q。使用原G0是预先定义的原方法对照，不继续扫描G3来源。两臂backbone中未用于输出的旧segmentation_head被冻结，避免把无梯度字段算作活跃参数；DIRECT使用复制直接头。预计DIRECT482500、条件分支484576活跃参数，实际由模块账本报告。

相对原各自冻结特征分支，唯一主动改变是允许编码器/解码器特征更新，包含随全参数梯度分配的原clip1优化行为；不是把所有差异仅归因某一层。两臂头和监督目标不同，不是等容量或纯Bayesian模块因果增益；条件G不含F，不是完整/官方BayeSeg。新DIRECT是从共同预训练起点端到端CE继续训练的标准U-Net参数化输出。

## 固定数据、优化与评价

沿用原canonical320 seed0 Siemens20% L的10患者230切片/20相位；开发val23患者46相位544切片。无U/目标/保留/test访问。两臂原患者均衡batch2/四翻转/绝对schedule5001..10000相同，128图像/标签预处理一致，freshAdam3e−4 betas(.9,.999)eps1e−8、wd0 foreachFalse、FP32确定性、全活跃参数联合clip1、无scheduler/earlystop。原H族/raw范围/KL/N及2Theta×2local/原训练噪声不变。每臂0/1000/2000/3000/4000/5000全model/Adam/RNG，共12状态。

两个终点全部46相位图像预测先封存再读val标签；原4Theta×4local、batch2/seed/384概率恢复/processed_spacing/三维评分公式不变。完整报告23匿名患者ED/ES聚合Dice、类均值、NLL/Brier、HD95成功限定及失败对角线惩罚/NA/空预测数。主配对条件G_E2E−DIRECT_E2E；同时复用原DIRECT/G0冻结分支数值而不重训/重评分，报告各自解除冻结的配对变化及两变化之差。患者bootstrap10000/seed163，无val挑步/调bias/加MC。多轮使用该val明确作开发结果，不能据此称未触碰最终测试或选择测试模型；24位test仍封存。

## 预算与收口

仅两臂各5000、共10000新optimizer/reverse，20000训练图像编码；条件臂10000Theta/20000批量G/40000图像G。验证1088编码图像、1112Theta/4448G/8704图像G、92标签相位读。预计新增<3GB，启动预留4GB；12全状态和概率/日志/source/cache/tmp全部pro5000原授权/data/bmw create-only。沿用R_START原首次时钟/全部失败成本，<=24h和各服务器新增实际<50GB；独立原23h45保存/23h59硬停，四根18GB软20GB硬及新45/49GB。新协议已提交后才启动，挂载/容量/真实读写/GPU余量/来源/剩余时钟先核，中性argv并ps+nvidia复核。不删历史/失败/他人数据。

非有限/源G变化/数据边界或硬预算失败保存partial及真实成本，准确接续不盲重跑。完成审计10000连续更新、12状态/Adam步、共同来源/RNG、匹配schedule、G全程冻结而特征确有更新、封存/46相位/全部患者指标及bootstrap、EXIT/FAILURE/worker及guard退休，中文正负报告和匿名数值代理push核SHA+200。若仍无优势，承认该条件G路线的患者缺口，不盲加相同步数或返回无限两图实验；下一建模假设须另立。唯一半小时sc-bayeseg继续。

启动前合成准入已通过：2次synthetic反向、4批量G/2Theta、0optimizer、0真实val/test数组，DIRECT/G两条特征梯度均非零、全部活跃梯度有限、G无梯度；活跃参数482500/484576。该合成成本单列，不混入正式10000训练更新。
