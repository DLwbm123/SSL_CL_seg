# 分类别决策预算与SGD实际位移软修正 V1（冻结）

2026-10-23，用户明确授权先冻结协议、修复末头检查，仅新增D/DP单种子，复用B/W不重跑，结束先完整交付后等待决定。独立DECISION_BUDGET_V1，不修订已结束邻域实验；原邻域A错误及FAIL_PROTOCOL_DEVIATION永久保留。本轮检验现有保护信号以实际位移软控制使用是否改善旧尾部且维持新域非劣，不作患者身份/属性/逐病例诊断，也不构建历史记忆。所有保护样本仍仅来自当前13名train-L。

## 固定数据、模型与复用

源FixMatch43600 EMA163，固定真实native3×3末头、encoder/preheaddecoder训练，普通原图CE；SGD lr .001/momentum .9/Nesterov，矩阵WD .0005其他0，全梯度clip norm1（原engine）；2000步；final EMA .999主终点，student全部报告。无U/旧图/新测试/参数锚/DS/定向邻域。源eval且requires_gradFalse，无参数grad/更新。末头在建optimizer前显式冻结，不以D/DP字母定义冻结：setup断言requires_gradFalse、optimizer全参数ID与可训练清单精确相等且不含末头；每步前后断言头无grad及权重不变；每EMA更新后同时断言学生/EMA末头与源精确一致。源其余冻结sigma/grad_update保持原样。

原邻域种子20271401_B与20271401_W及原源患者行只读复用，报告提交32edb2a00c412057a6df8a8d390b3243e0c874cd。审计逐项确认B/W初始source/EMA、固定头、optimizer/增强/步数/患者键/曝光匹配。相同seed20271401、同Current提供器及私有RNGkey保持原主批、弱光度与随机邻域序列；不读取新验证性能选择参数。原16native中定向分支及其源输入grad在新轮删除，保留的前向对源/student在eval native路径；预先以相同输入验证去除分支前后保留W项、主状态与全局RNG一致，计入P0且无更新。若不一致P0_STOP，不重跑B/W。历史源训练集辅助评估仅复用既有评估管线的只读输出，不进入训练。

## 新风险与两组

原R双弱视图同时源正确支持、r=1−JS/ln2、.5 minmargin单侧hinge、至多512/图uniform、存在类均值、原源尺度均保持；λR=.8791842259819201及四尺度直接继承父P0，不重新校准。随机邻域w=clip(x+.05 ξ,[-1,1])，ξ私有key neighbor_random，支持=原可靠支持且源w仍预测GT；采样与W完全一致，r沿原detach。

在同一批随机邻域可靠采样位置，四类分别风险Rc=mean(r×CE(student(w),GT))，参考R0c=mean(r×CE(source(w),GT))，均以位置数为分母而非sumr。T=1。空类别不建约束且不计入存在类平均；绝不用源错误伪标签覆盖GT。源与student风险在完全同一位置计算。Bc=R0c+.05 max(R0c,.1)，参考模型全程固定；预算随训练输入/支持变化，不声称是全局固定数据集风险上界。

D：CE+λR R+λD mean_present Rc。λD在只读继承的父Bprobe256终态上，用独立16批seed20271592一次.25 sqrt(CE梯度平方和/D风险梯度平方和)校准。无新probe优化，无扫描。D的CE风险不是原W的零中心margin，源初始化也可有非零梯度，明确披露目标形状改变。

DP：原图CE+λR R先产生真实SGD候选位移dopt（在原engine clip后执行正常SGD，含Nesterov及WD），随后在optimizer post hook且EMA之前修正。每类ac=grad Rc detach，Ac=ac/(||ac||+1e−12)，bc=(Bc−Rc)/(||ac||+1e−12)，零能量/空类不建立有效约束。

解 min_d,s>=0 .5||d−dopt||²+.5||s||²，s.t. Ad<=b+s（μ=1）。对偶max_v>=0 vᵀ(Adopt−b)−.5 vᵀ(AAᵀ+I)v，以最多四类的16个活跃集穷举小线性系统求唯一解，KKT残差<=1e−8 max(1,||Adopt−b||)。dP=dopt−Aᵀv；alpha=min(1,.25||dopt||/(||dP−dopt||+1e−12))；最终d=dopt+alpha(dP−dopt)。κ=.25固定，无二阶、无追加forward、无整步拒绝。参数float32实际写回后记录舍入后的位移/修正幅度与线性预算残差；允许<=.25001修正比例。修正与松弛会留下违反，非线性风险未逐步复测，不宣称预算满足/CE下降/无遗忘。

动量规则：正常SGD已生成的新momentum_buffer保留，不按修正反推或清空；下一步再以真实候选位移处理。checkpoint同时保存修正后参数和候选SGD动量，EMA仅在修正后更新。D也计算四风险梯度（但不改位移）以统一额外调用；这不意味着同FLOPs或同有效保护强度。两组共享所有采样、风险、参考、λR/尺度、初始化与曝光；D用λD加权风险，DP以相同风险预算控制，因此是两种使用方式的比较，不声称只改一个等强正则系数。

## P0与实现资格

只读父probe256（原seed20271490）和原尺度，不延长探针。源/测量student/optimizer/RNG前后精确不变；16独立训练批测量（seed20271592）。P0：CE/R/D梯度能量finite>=1e−12；λD finite[1e−6,1e4]；各FG随机邻域支持>=.05且>=3训练患者；D实际FG标量份额>=.05；D及有效风险梯度encoder/decoder非零；预测D与DP候选位移归一距>=.05；DP修正至少一批非零且每批有限/不超κ，KKT全通过，固定头/源无grad。预测候选用父probe动量和原SGD公式（零step、不改状态），只用于可辨识检查，不保证正式全程活跃。失败P0_STOP，不扩probe/换批seed/门槛/λ/μ/κ/预算宽度。

CPU/CUDA合成QP活跃/非活跃/空类/零梯度、修正上限、SGD真实位移与候选动量/恢复一致性、无修正恒等；原geometry数值检查复用。先做父W兼容读取检查一次（26native/1reverse上限，零opt），再各32步D/DP零新增机制等价共64optimizer（原CE+R，无D、DP控制；全状态/EMA/optimizer/RNG一致）。两份×32真实资格：golden/反向、非空optimizer+EMA+RNG恢复、两例overfit、最后真实D/DP一步；仅成功才两组各2000正式。P0和资格丢弃，正式均从原源空optimizer开始。formal 0/999/1999额外CE/R/D梯度；每步记录风险/预算/四梯度能量/预测违反/修正比例/松弛/幅度/动量保留策略，无额外forward。

## 调用账、时钟与停止

复用同engine，不另写训练loop。每正常loss10native=原R6+随机source/student4，四类风险autograd每组均4次，每步backward1。反向是调用口径非FLOPs。P0 compatibility26/1、measure160/112；detach640native/320reverse；两份资格204native/92reverse（纯CE overfit与三次完整loss）；formal40000native/20018reverse；evaluation5784native/0。预计总46814native/20543reverse；合成/数值SVD与reverse另列（合成reverse<=32，SVD<=16）。optimizer4128=64detach+64资格+4000formal。caps分别P0400native/256reverse、detach640/320、engineering512/160、formal40000/20100、evaluation5784/0，不借bucket、不增加源评估或重评估。资格额外非空DP状态恢复数值自检只用合成，不增加医疗调用。

仅pro5000 GPU3既定UUID；create-only /data/bmw/SSL_CL_seg/runs/decision_budget_v1_20261023，中性entry。8h原钟、25200不新stage/27900stop/独立timeout28600TERM+30KILL；root严格<4GB/3.5GB保护，含ops预留4GB+local50MB，累计新增已完成4076305728加4050000000=8126305728<27GB，30GB账不清零。实际挂载/写读/GPU显存准入；原数据及代码祖先只读，不下载/复制模型数据。

## 科学门槛与交付

主候选DP，报告D、DP及复用B/W全部student/EMA与配对类别/尾部。DP旧均值>=80.36740663770262；源配对患者95%CI下界>=−.30；下降>1pp人数<=2、>2=0、p10>=−.50且人数不劣B/W。新绝对>=83、DP−B新均值与CI下界>=−.20，DP对W/D两域均值及CI下界>=−.20。DP−B旧>=.30，DP−W旧>=.15且>0，DP−D旧>=.15且>0。全部通过才单种子综合PASS；另明确联合旧尾部/新非劣判定，不以均值改善覆盖失败。D同样报告全部部署门槛，不把D表现选为替代主终点。共享源163与开发旧23/新31，患者EDES共同bootstrap10000seed163；不做逐患者定向训练、不扩大访问，不宣称独立测试或跨seed稳定性。

全部两组完成、EXIT0/worker退休且AUDIT不存在才一次decision_budget_audit零模型forward。P0_STOP同样收尾。完整公开协议/源码/审计、匿名报告/summary/status；local+remoteDELIVERY、remoteSHA与匿名HTTP核实。仅一个30min完成监测，完成交付后等待用户决定，不自动下一轮。历史A错误不回填。参数/患者行/权重/私有日志配置不公开。
