# 语义覆盖修复记忆的目标训练检验 v1（独立冻结）

准入：语义覆盖修复源轮恰好256outer+256functional、工程30，总542、EXIT0/worker退休/一次AUDIT PASS，质量门槛PASS（器官非空码19/26/21、同1781652字节）。这只是覆盖资格，不是功能收益。

唯一新组M=A(C0/R0)的目标配置，仅将C0记忆替换为已冻结修复memory。使用相同共同S†入口/空SGD状态、seed163、P/G/32码/frame全固定，原native全部允许参数训练、sigma/grad_update常量固定，LR.001/Nesterov.9/矩阵wd.0005/vector0/globalclip1；L2/U14/合成8，三视图72图/96行、D32/R4、lambda.1/beta1/3，R0不变。不增加新几何项，不采用EMA主终点，不挑best，不中间验证/加步数。

复用原target工程32调用，资格权重丢弃；通过后恰好2000步最后学生，EMA仅诊断，总本轮≤2032optimizer、0functional。失败/恢复全计，不偷用其他实验余额。目标训练只Philips L/U硬HDF5白名单；无Siemens历史图/mask/bank/逐图特征/在线旧teacher监督。生成图mask仅瞬时，不保存。评估另进程在训练worker退出后执行：Siemens val23/46、Philips31/62、源10L/20与目标13L/26训练口径，EMA同原协议仅两域val；test与U GT封存。

对照复用已冻结A最后学生78.2908826/83.1969906%（旧/新）及患者记录，不重跑。属于相同入口/seed/训练合同的历史匹配对照，披露非同期与开发验证限制，不称新增GYRO机制证据。成功解释预设：相对A旧≥+.5pp且新≥−.5pp；同时独立报告绝对保持旧≥78.9337143、新学习≥73.7043174。无论是否通过均报告两域、LV/MYO/RV、EDES、源/目标训练、患者配对改善下降/中位差与固定seed163/10000次bootstrap区间。不新增C1/R1/D/E，不用EMA改判主终点。

独立create-only新root，既定GPU3/UUID、授权数据盘；复用只读repair memory与旧入口数据/评估记录，不复制checkpoint/HDF5/bank。预计<150MB；800MB保护+200MB余量严格<1GB（也<50GB），每保存/每50更新检查及独立原始时钟2h硬超时，90min不新阶段、105minTERM、120min前KILL，恢复不重置。全部cache/temp/log/checkpoint置新root，启动实际mount/write-read/free GPU准入；中性主子入口与ps/nvidia-smi检查。唯一space-lora-gpu3-4每30分钟心跳，无重复监测。

NaN、坏恢复、非法访问、计数/时间/存储越界硬停并保留dirty现场；仅有依据有限工程修复，科学负结果不改本轮。完成后一次必要审计EXIT/精确步数/ledger/同入口空opt/冻结memory/finite final-latest及opt-RNG-reads/患者集合/worker退休。源码协议匿名中文报告选择性显式代理push，核remoteSHA/匿名HTTP并local/remoteDELIVERY/results_public齐，不发布患者/图像mask/码/权重/原日志/服务器配置，不建PR。
