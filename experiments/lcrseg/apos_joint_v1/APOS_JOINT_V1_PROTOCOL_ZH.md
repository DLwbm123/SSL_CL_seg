# APOS 联合求解完整冻结协议 单种子 N Q R

2026-10-24用户明确授权。新独立联合求解修订，保留V3科学能量、对照及门槛。旧V3已终态交付，31优化/2328native/33reverse等全部保留，不重启旧根/重做旧资格、不伪称第31步状态可恢复。新轮仅原共同入口开始，禁止机制/种子/尺度/迭代扫描。主R，单seed20271611。

## 1 数据和科学能量保持

只读复用V2 prior.private.pt与source/CAL/final.pt校准EMA。源原生FixMatch43600 EMA163、真实3×3末头/padding0/原插值永久固定eval；尺度/模式头也永久固定。源新优化/校准/先验拟合/源训练标签读取均0。学生唯一可训练16→8→16无bias残差B tanh(Ah)，B源严格0，A沿原seed。冻结源q0只读取同一当前输入，学生qθ走第二条完整native路径，包括N每输入2native，不声称同FLOPs。

源Q/R各完整Siemens验证23患者46EDES/544切片；N原生保存行复用不重评。源函数保持与能力门禁前，禁止任何目标资格或训练。适配只Philips既定13患者26体积254切片train-L、每批2不同患者，同seed/同draw，无旧图/旧标签HDF5/U/test。目标固定末步评估旧23/新31重复开发患者，验证不用于选择或调参。不得改HDF5/manifests/splits。

A384→32平均占据，a0=Aq0FG；固定pi来自h0源模式头，W=s²/(s²+b²)来自冻结源尺度头且0<W<=1。Fx拼接sqrt(pi)的8份rank8因子与8模式均值差，至多72列；d=Σpi sigma²，floor1e−6。r=u−a0−Fxz。Q势=.5Σr²/d+.5||z||²；R势=Σ(sqrt(1+r²/d)−1)+.5||z||²，二者零点曲率匹配，lambda=beta=1、标准化尺度1。能量E=mean KL(P||qθ)+.5/M||APFG−u||²_W+H/M，M1024；P四类单纯形、u三前景>=0,sum<=1。qθ=q0时P=q0/u=a0/z0给零非负能量，源函数保持；不由此宣称适配后旧域无遗忘或新域达标。

N=同adapter的原生输出；Q=同能量二次结构；R=同能量等中心曲率pseudo-Huber结构，固定主候选。只对求解层作本轮修订，不改变任务损失/训练条件，不同时引入隐式梯度。

## 2 八个联合方向与完整矩阵投影

每次预测内部先把logq/logq0在FP64作数学等价归一化，求解全FP64，最后返回原输出dtype；这属于数值精度绑定，不改温度。初值u=AqθFG/z0。像素条件块仍固定16次不动点，最后另16次刷新，res<=1e−4；每次完整求解须恰8个联合方向，无提前恒等return删除结构梯度，不增加方向或使用另一个fallback求解器。

对当前像素条件解，v=W(AP−u)，每占据格C=mean(diag(PFG)−PFG PFG')，T=(W^-1+C)^-1。有限16次使方向属于inexact Newton，不能冒称精确像素最优；最终原残差和对偶间隙都须通过。

s_Q=r/d；s_R=r/[d sqrt(1+r²/d)]。真实Hessian D_Q=1/d，D_R=(1/d)(1+r²/d)^(-3/2)，禁止把旧IRLS权重当真实Hessian。梯度gu=−v+s，gz=z−F's；联合Hessian=[T+D,−DF;−F'D,I+F'DF]。先按局部完整H=T+D投影u−H^-1gu到占据单纯形，枚举15个非空/零/质量面，KKT固定first tie，投影数值原始可行容差1e−12、相对dual等式容差1e−10；只清理KKT后微小负零，不用对角clamp替代完整矩阵投影。

该投影的活跃面给出切空间B。固定活跃面联合Newton方向通过Schur S=I+F'KF求<=72维Cholesky；内部面K=(D^-1+W^-1+C)^-1，活跃面另外加正半定约束修正。采用并联和+约束正项构造，禁止以D−DBD的大数相减形成K。逐格小KKT消元，活跃集随投影的梯度/乘子重选，不把曾经为0的组织永久锁死。记录线性相对残差<=1e−8。

每方向仅4个预声明候选alpha=1,.5,.25,.125，依次取第一个通过者。候选u沿联合方向后用同一完整H投影到真实可行域；z沿同alpha。重新16次像素条件求解计算原E；Armijo用实际投影位移的梯度内积，要求非上升方向（数值1e−12）和c=1e−4充分下降，FP64能量容差1e−12max(1,|E|)。所有候选失败STOP，不试更小步长、不加方向、不切solver。原总能量容差1e−6max(1,|初值|)仍保留。全显式8方向代数图一次反向；不另做二阶autograd调用。

## 3 原残差和原始对偶证书同时门禁

原驻点门禁1e−4原样保留：最后用当前r的原IRLS/梯度精度（Q1/d、R1/[d sqrt(1+r²/d)]）计算z条件解变化/(1+||z||inf)，u原对角预条件投影梯度差，以及最后像素不动点差，max<=1e−4。方向Hessian与此一阶残差使用的精度不同，不能混用。

额外构造合法dual v=W(AP−u)、s=原残差势梯度。Q共轭phi*=.5d s²；R共轭phi*=1−sqrt(1−d s²)，约束|s|<=1/sqrt(d)，用有理式评价，浮点触边仅作固定机器精度向内映射并计数。C支持函数hC(v−s)=Σcell max(0,三前景v−s)。

Dual=−mean_pixel logsumexp(logq−lift(v))−[.5Σv²/W+a0's+Σphi*(s)+.5||F's||²+hC(v−s)]/M，背景lift系数0。逐样本保存E和Dual，gap=E−Dual，必须gap>=−1e−10（FP64弱对偶数值检查），gap/max(1,|E|,|Dual|)<=1e−6。不能通过夹为0隐藏负gap。该证书控制数值次优程度，不是Dice/临床安全保证。

## 4 首次资格与故障现场

先CPU/CUDA零医疗测试：加载已有真实非零先验参数，固定4类数值场景(identity/低秩对齐漂移/非零F与1e−6方差同时压力/近缺失组织边界)，每类Q/R各一次backward。另两份aligned目标各两个中心差分调用；梯度差绝对<=2e−5且相对max(1e−5,|两导数|)<=.02。完整非对角占据投影KKT和独立小维联合二次dense解对Schur解，error<=1e−9。用原始对偶证书检查完整求解，不只finite。

真实384×384合成输入检查实际非零prior、入口函数，再固定B权重1e−3制造非零漂移，对N/Q/R完整目标各一次backward；不得再用零因子或零漂移替代尺寸资格。合成cap128native/64reverse，预计26native/19reverse；求解器合成调用另外计场景，不混医疗账。预估剩余整轮含1.7裕量及评估/保存缓冲若>=20000秒则不读医疗。

每方向记录能量/步长/线性残差/活跃面。所有solver异常在re-raise前保存第一份私有logq/logq0/W/pi/F/d/u/z与最终证书、轨迹，无图像/标签/患者ID，绝不公开或训练回放，snapshot及atomic副本预留100MB。资格每实际optimizer之后另存完整qualification_latest.pt，记录逻辑step与资格实际calls，精确回放消耗另计，失败不再只剩step0。原始现场/退出/计数永久保留；非finite也保留诊断，不伪造可恢复性。

## 5 源门禁 资格 正式与最终科学判定

合成PASS后Q/R各544输入，两native路径共2176native。每张源输出对q0概率max<=5e−7且argmax精确一致；23患者相对复用N的均值/CI两端/p10绝对值<=1e−10、>1/>2人数0，全过才目标阶段，否则STOP。

之后N32资格固定原流程：golden/空非空状态恢复/固定两例30次+一次精确回放+恢复初始完整1步共32。N拟合终态一次同图前向复用qθ/q0求Q/R结构任务梯度，能量各>=1e−12、归一方向距>=.05才P0 PASS。不扩probe、不换seed。再Q/R各32资格，全过才正式N/Q/R各2000，各自从原共同源入口/空optimizer开始。source或P0失败不是目标性能阴性。

适配只当前Philips L：普通像素GTCE+按图存在前景Dice，空图Dice0但CE罚假阳性。没有校准辅助训练；仅adapter clip1，SGD.001/.9/Nesterov/matrixWD.0005、EMA.999固定。0/999/1999在原图作CE/Dice两额外grad诊断，无额外native。全部3正式和固定末评估完成才读目标性能，不选checkpoint/主终点。主R旧绝对>=80.36740663770262、新>=83；own及original旧CI下界>=−.30、>1<=2/>2=0/p10>=−.50且人数不劣N；R−N新mean/CI>=−.20，R−N旧>=.30/R−Q旧>=.15，R−Q两域mean/CI>=−.20，全部同时才科学PASS。两源参照因源恒等应相同，仍明示。EDES共同bootstrap10000seed163，旧23新31重复开发、单seed非独立测试/稳定性保证。

## 6 完整调用预算及资源

Optimizer6096=3*32资格+3*2000正式，源新优化0。医疗native cap34000=源2176+资格600+正式24000+终评7224；预计资格460但失败完整计账。医疗reverse cap6168=资格150+正式6018（含3组各6诊断grad）；源/终评0。无新增训练probe。原代码引擎复用，不复制训练循环。

医疗求解primitive分桶：joint方向sample<=120000；merit候选sample<=480000；pixel固定点sample<=10000000；逐cell逐face投影/小线性解计数<=10000000000；dual证书sample<=20000。一次solve最坏8方向、32候选、初始及末pixel刷新：最多(2+32)*16=544pixel sample更新（当前候选通过则缓存为下方向P，无额外当前P重算）；每方向与候选各15face×1024cell。以上是明显更高的代数预算，不能借旧V3的250万pixel或20万latent桶。网络forward/reverse在调用前记账，optimizer pre预留与post实际完成分开，失败调用不抹去。

合成另cap128native/64reverse、512joint sample方向/2048merit sample候选/25000pixel sample更新/25000000cell-face测试/128dual证书/4独立dense参考解；每固定任务进度预留并记录成功/失败调用，不把未捕获退出码写0。medical阶段不可借合成桶。真实尺寸速度资格还须核最坏方向/候选工作估算，不能只按原native速度沿用旧估计。

首次数值/资格起同一新原始8h钟，25200不新stage、27900stop、独立timeout28600TERM/30KILL。3.5GB写保护、严格新增<4GB（含ops、缓存、snapshot/atomic），remote+local4.05GB峰值预留，累计30GB不清零/准入<27GB留3GB恢复余量。不删历史。仅pro5000 GPU3既定UUID，/data/bmw项目例外、中性父子argv，实际挂载容量写读及显存准入并核ps/nvidia。

终态全退出且AUDIT不存在，只一次本轮audit.py零forward；完整发布命名协议/源码/匿名结果/停止失败/审计/status/report，患者行/权重/先验/诊断张量/配置日志私有。GitHub显式7897代理/NO_PROXY不绕过，核remoteSHAeq/匿名HTTP200/localremoteDELIVERY/results_public齐才完整交付。唯一30min健康静默监测，实质完成/故障/需用户动作才通知；交付后删除监测等待用户，不自动后继。

## 零医疗算术修复记录

首次CPU身份测试在精确类别同分时触发argmax不一致，1次合成reverse、160pixel/8joint/8merit/245760cell-face/1dual，0native/0medical/0optimizer。保留原log/exit1/counters/launch。像素块改成logq+[logsoftmax(logq−corr)−logsoftmax(logq)]的等价中心化表达；完整内点投影面采用解析恒等x=c，避免H^-1 Hc舍入。所有8方向仍执行、结构导数不切断；不改能量/门槛/迭代/候选/数据。合成剩余资格共享原钟和原硬cap，最终累计调用包括该失败，不重置计数。
