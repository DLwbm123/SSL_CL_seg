# 当前谱配置：患者代价、模块兼容性及谱分解P0（2026-10-21，运行前冻结）

用户授权按已讨论顺序执行：先P0，再依据预声明规则决定S/G/T/GT四组；不恢复U/frame/生成记忆，不修改任何旧终点。此协议只启动P0；后续训练须在本P0完整分析后另立执行协议，最多16320实际optimizer更新，不能用P0验证分数挑教师权重、病例、入口或seed。

## 数据、状态与调用硬预算

源为已审计两目标流谱轮。A部分仅CPU读取既有患者ED/ES Dice，分别对两流S/G00学生与EMA：G00−S、EMA−学生、各自相对S†入口与原源。报告均值/中位/第10分位、下降严格>.5/1/2pp人数、留一患者均值范围、类/EDES、两流差值Pearson/Spearman与同向人数；相关性常量分母NA。不公开患者身份，不按验证患者身份选训练样本。阈值只是开发分析，非临床显著阈值。原源记录若确实缺失仅该比较NA，不新增前向。

B部分：两个目标流20261021/20261022×S/G00×两种组合(EMA encoder+student decoder；student encoder+EMA decoder)，共8新增模型，旧val544+新val660=1204切片/模型，共9632完整native切片暴露。encoder实际state块parent.native.enc1/enc2/enc3/bottleneck；decoder为up/dec/head及原native剩余读出块，核实际state互斥穷尽，禁止未知块静默落入某类。固定P/常量/copy状态保持同组学生规则，所有模型eval/无梯度。完整学生/EMA既有记录直接复用，不从组合挑新部署终点。

C部分：共同入口+两流S/G00/Aalign最后学生共7状态；固定诊断训练流20261028、step0..15，同批L2+L池仅图14、两强增强及slot位置，48encoder图/批，共5376encoder图像暴露。Current白名单仅Philips train-L，补图只image，无U accessor、无隐藏GT/val训练；B与C独立数据访问器进程以免训练HDF5白名单误拒合法独立验证。每次C均eval、不反传不更新，P固定。输出centroid能量、三对view距离、核范数、奇异值、r*=核范数平方/Frobenius平方、中心化协方差trace/offdiag/eigenvalues，分别报告未中心化与中心化。零能量比值NA。全部批次逐状态相同，仅存匿名矩阵统计，不存图像/特征/逐患者向量。

硬cap：B native9632；C encoder5376；optimizer/functional/reverse均0。CPU特征数学自检0医疗模型前向。reserved在调用前落盘、executed后落盘，异常pending保留，失败/重放不得清零或扩cap。无原患者模型额外评价、无test/U隐藏标签。

## 资源与工程

独立create-only root、只GPU3指定UUID/pro5000、解释器与共享只读资源沿既定例外/data/bmw。所有新增代码/log/cache/tmp/结果在新root；不复制数据/checkpoint。原始时钟首次运行锁定，2h硬界，90min不新状态、105min TERM、119min前KILL，独立timeout覆盖主子进程不依赖SSH/Codex。新增500MB硬界，100MB预警停止新增状态、400MB写入保护+100MB退出余量，预计<20MB，另2MBops元数据计入；每次forward或每50调用guard，保存和新状态先guard。启动核挂载/容量/实际写读/显存；中性父子argv并ps/nvidia-smi核验。所有异常保留现场不盲重评，不删除旧数据。

冻结输入checkpoint文件只读；状态分块、finite、data集合/患者EDES、模型/buffers不变、ledger精确caps/pending0、EXIT0与受管worker退休后一次CPU审计，0审计前向。不能把代码已准备或仅启动称PASS。

## 预声明进入下一步规则

先完整报告P0（含不支持原推断的结果）；若读取/状态分块/finite/样本账本/只读/资源任一失败，停止对应阶段、保留成本，仅允许有依据有限工程修复且同cap/时钟，不自动进入训练。若工程完整通过，P0不要求某个交换组合改善或谱统计“证明坍缩”；它是描述性诊断，不用验证表现选配置。继续冻结四组两新目标流S/G/T/GT的机制检验，回答EMA教师是否改善最后学生及谱项在T上的增量；不得仅凭P0相关性声称因果。

下一训练拟固定目标流20261025/20261026、共同校准流20261027、源入口/P163不变，最后学生2000主/统一EMA辅助。T为更新前EMA弱视图空间特征对两学生强视图cosine，P/位置/stopgrad固定，optimizer后更新EMA。G原.1谱目标不变；lambda_T=.25*sqrt(sum||grad(.1G_raw)||²/sum||grad(T_raw)||²)，一次16训练批校准，同两流同T/GT。低能量/非finite/权重越界停止，不换批次追通过。四组匹配48学生+16教师encoder暴露，未启用路径断梯度，S/Sdetach32步状态精确；8资格32+正式16000+detach64=16320optimizer。实际前向/反向/评价及时间存储硬预算须在启动前另文冻结，不把16320当全部成本。正式前不展开强VC对照/U/独立源种子/多域等条件后续。

## 交付与隐私

完成完整中文报告/匿名聚合/源码协议/一次审计，通过既有选择性公开树代理push；核remote SHA/匿名HTTP，本地和remote DELIVERY/results_public齐。只用现有space-lora-gpu3-4唯一心跳；健康静默，实质结果/异常通知。禁止公开患者、图像或mask、任何权重/code/P/frame、逐图特征、原始日志、服务器配置。旧终点/成本不重置。
