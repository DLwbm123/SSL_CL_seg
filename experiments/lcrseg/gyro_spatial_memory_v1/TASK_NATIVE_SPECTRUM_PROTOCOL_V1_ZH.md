# 任务相关的原生瓶颈谱保持：最小辨别协议 v1

状态：DESIGN_FROZEN_AWAITING_USER_REVIEW。2026-10-21本地冻结；尚未实现新训练分支、形成权重、访问服务器或启动任何模型调用。用户要求“先给我看看再启动”，本文件不构成启动授权；现有唯一心跳不得据此自动开跑。修改须形成明确版本差异，启动前重新确认适用版本。

## 1. 问题与可证伪主张

既有G在两个新目标流相对S的旧域收益复现；GT的教师增量未过冻结门槛。保持所有旧学生/EMA终点和负结果，不继续扫描T。当前G约束源训练P后的32维辅助表征，分割读取的是原生多尺度特征，因此P空间谱好不等于分割任务方向被保留。

本轮只检验：在相同128维原生瓶颈、同谱算子和入口梯度尺度校准下，合法源任务敏感性加权，是否优于等权和保持权重分布的通道打乱；并与已复现G比较。不是完整分割功能保持定理，不是旧特征精确锚定；不声称无遗忘。权重与源标签的对应关系有效是主要新增主张，通用各向异性加权有效不足以支持它。

思想来源：Alignment, Not Isotropy（https://openreview.net/forum?id=cnxMfTs9PW）的任务方向；λ-JEPA（https://arxiv.org/html/2609.35288v1）的正则位置；Stability Where It Matters（https://openreview.net/forum?id=32qKx7U7cG）的决策敏感性。以下权重/密集谱组合是本项目待验证假设，不继承其定理。四类类均值至多三条独立对比，故本轮不使用类均值构造“满秩任务子空间”，不复制FIFO或冻结旧teacher。

## 2. 固定共同合同与五组

新目标流20261028、20261029；共同源163、S†原69步student入口不变。新流只改变目标采样/增强，不能称独立源训练重复。使用既有slot独立流规则，监督L2、补图L池14仅image，三视图48encoder、每图4位置共64行；无U accessor。空间增强/位置/监督CE归一化沿上一轮，不加入目标额外标签。

|组|目标训练损失|用途|
|---|---|---|
|S|原L2 CE|同流监督基线|
|G|CE + .1 原P32谱|原方法复现锚点|
|NI|CE + λI 原生128维等权谱|原生谱对照|
|NP|CE + λP 原生128维打乱任务权重谱|同权重分布、失去通道任务对应|
|NT|CE + λT 原生128维任务权重谱|唯一新候选|

十组各正式2000步最后student为主，EMA .999统一辅助，不选best/补步/中途val。空SGD .001/Nesterov .9/矩阵wd .0005/向量0/clip1；原native允许encoder/decoder/head训练，sigma和grad_update固定。复用现有引擎，不另写正式训练loop。

所有组执行同L2 native前向及48encoder辅助前向，并计算四个辅助标量；只将本组项接入loss。G的P只读；其余不通过P构造几何。无G/code、frame损失、历史图/逐图特征bank、在线旧teacher、T项。所有新缓冲固定，不进入优化器；前向暴露相同不等于反向/FLOPs完全匹配。

## 3. 任务敏感性的一次合法形成（0更新）

独立源统计阶段只读共同入口S†、既有Siemens source train-L 10患者/20 EDES/230切片；不用source U、源val/test、Philips或其他标签。S†已经包含既定接入训练，称“共同入口上的源训练敏感性”，不是纯源初始化统计。原G没有此次额外源梯度统计，明确披露信息/计算差异；NP和NT使用同一统计以控制该差异。该阶段必须与目标训练进程分离，目标配置/数据白名单不能加载源图像。

逐切片一次eval模式真实完整native分割前向，捕获实际送入decoder路径的128通道bottleneck H。对该切片存在的类别K，定义 Lbal=(1/|K|)Σ[k∈K] mean[p:y=k](-log softmax(logit)[k])。无该类时不补造像素；全背景切片K={BG}，如实计数。保留模型原native读出规则和固定RNG；不得用辅助P输出代替真实读出路径。

只求 g=∂Lbal/∂H，autograd.grad止于H，不更新任何模型/optimizer/EMA，不要求梯度穿过辅助池化。逐患者先平均其切片/空间的(H_c*g_c)^2，再对10患者等权平均得到s_c。该统计是局部梯度×激活敏感性近似，包含当前decoder的下游作用，但不保护绕过瓶颈的skip路径，不等同Fisher矩阵或全网络Jacobian保证。

v_c=clip(s_c/mean(s), .25, 4)；u_c=.5+.5*v_c；w_c=sqrt(u_c/mean(u))。因此mean(w²)=1、有正下界、不会删除任何通道。固定mix .5、clip [.25,4]，不扫描。NP用CPU generator seed20261030的一次randperm(128)打乱同一w，不换排列；NI令w=1。s均值<1e-12、任意非finite、CV(w²)<.05，或NP与NT的w²余弦>.99，均判辨别信号不足并在正式前科学停止，不改阈值或重抽排列。

只保存128维聚合s/w、聚合类别出现数、精确数据角色和成本；不保存逐图激活/梯度/图像或mask。源输入和模型参数/buffer/RNG前后恢复必须精确，0更新。权重文件属私有参数，不公开。

## 4. 原生谱的精确定义与位置边界

H为实际bottleneck输出；在该输出上复用既有8×8不重叠确定性池化（不是新STE），先池化，再逐通道乘固定w，沿通道unit-normalize eps1e-6，按同一组4位置取行。分别得到弱/两强视图z，C=.5*zweak+.25*zstrong1+.25*zstrong2。

对NI/NP/NT：n=64、d=128、r=min(n,d)=64；Rnative=1-||(2/3)C||*/sqrt(n*r)。保留原2/3常数，分母64是按unit-row谱上界维度归一化；G仍原n64/d32和sqrt(64*32)，不改旧算式。三原生组完全同维度/算子，仅w对应不同。

原生组对G的差异同时包含P移除、维度32→128和分母变化；NT-G只能说明整个新配置收益，不能单独归因“位置”。本最小轮不增加随机P32等额外消融，也不证明32→128本身有效。NT-NI及NT-NP才是任务对应关系的关键辨别。池化及稀疏位置仍是读出路径的近似采样，不是对所有原生/skip表征完整约束。

## 5. 单次尺度校准与工程资格

共同入口、固定目标训练校准stream20261031 step0..15，同16批L2+补L14；0更新、无val。每批复用48encoder前向，对encoder允许参数分别求grad(.1RG)、grad(RI)、grad(RP)、grad(RT)。Ej=Σ||grad(Rj)||²，EG已包含.1；λj=sqrt(EG/Ej)，j=I/P/T。三值一次固定，两流共享，不按患者/结果/流重标定。所有能量>=1e-12、λ∈[1e-4,10]且模型buffer/RNG恢复精确才通过，否则停，不换校准批或追调权重。这只匹配入口RMS，不是全轨迹尺度匹配。

CPU张量自检不调用医疗模型；核池化梯度、unit/centroid、128维谱归一化、w=1恒等、置换权重分布、四组slot前缀一致、源统计0更新、非法数据访问硬停。另S最小/Sdetach各32更新（全部辅助前向但不连loss），逐步model/opt/EMA/RNG/监督输入exact，终态丢弃。十组各32资格（固定两例30步+恢复重放1步+全目标1步），全部丢弃；沿已冻结first>=1e-6下降>=min(1e-4,.01first)，低floor项NA且非增，不更换病例。golden/backward和非空optimizer恢复必须通过才正式。

固定顺序：源统计→校准→detach→十资格→流28 S/G/NI/NP/NT→流29同序。每组train退出后独立eval。正式每组step0/499/999/1499/1999共50点，复用forward记录CE/aux encoder梯度范数内积、合并梯度/clip和实际SGD位移，额外100次autograd单列，不据此调整训练。

## 6. 硬预算（不得以失败或恢复重置）

optimizer上限20384 = 正式20000 + 十资格320 + detach64；functional/toy均0。所有失败/临时/恢复重放实际更新计同账。下表native为完整分割图像暴露；encoder为额外encoder-only图像，不能将native内部encoder再重复计入。reverse按一次backward/autograd.grad调用计；SVD/CPU统计不冒充训练更新。

|阶段|native图像硬cap|额外encoder图像硬cap|reverse硬cap|
|---|---:|---:|---:|
|一次源敏感性|230|0|230|
|一次16批校准|0|768|64|
|detach两组|160|1632|64|
|十资格（含golden/恢复）|960|1920|400|
|十正式及入口preflight|40200|962000|20040|
|固定50点诊断|0|0|100|
|十终点评价|28920|0|0|

总native上限70470、额外encoder上限966320、reverse上限20898、teacher/generator调用0。formal本体40000 native+960000 encoder+20000反向；其余明确留作既有有限preflight，不代表可以新增研究阶段。期望评价严格28920=10×2892：student旧val544/新val660/源train230/目标train254+EMA两域544/660，不评旧模型或新混合模型。资格预期800native/1440encoder/350reverse；实际计数为准，不因cap余量新增调用。

reserved先持久化、executed完成后写，异常pending保留；更新ledger与调用ledger独立。代码实现前按此表静态核对，若既有helper无法在cap内完成则先修订协议供用户看，不能静默扩cap。正式失败只在clean checkpoint且所有成本可追溯时有限恢复；dirty/pending不明确不盲重放。

仅pro5000指定GPU3 UUID GPU-9ea1d4ae-e403-cb3b-29b3-c63b083bf8ff，解释器/data/bmw/envs/x20261014/bin/python；pro5000已确认/data/bmw存储例外，不与jiangsuiyang NAS混用。未来批准后新create-only root拟为/data/bmw/SSL_CL_seg/runs/gyro_task_native_spectrum_v1_20261021（已存在则停止，不复用覆盖）。本次不创建远端root。

同一原始时钟自第一次模型调用起至退出<=10小时；9小时不新stage，35099秒TERM+899秒后KILL，进程组独立timeout不依赖SSH/Codex。新增实际严格<4,000,000,000bytes：3.5GB写保护+.5GB退出余量，预计<800MB，所有code/ops/log/cache/tmp/checkpoint计入。启动核真实mount/free/write-read/GPU余量，中性父子argv并ps/nvidia-smi核身份；不干扰他人，不删历史数据。NaN/非法读/恢复不一致/预算/资源超限硬停并保现场。只用现有唯一30分钟监测，健康静默。

## 7. 预声明分析与停止

主student，逐流两域报告，不按双域均值选组。主要门槛：NT-NI、NT-NP分别旧域两流平均>=+.3pp、每流旧>0、新每流>=-.3pp；NT-G同一门槛作为实用增量。三项均过才支持“当前任务加权配置值得继续”；任一不满足，不追加权重/clip/排列/维度扫描。若NT与NP相近，只能说未证明任务对应必要。NI优于G不能排除维度/算子变化；NT单纯高分不能证明功能保持。

G-S旧平均>=+.5pp、每流旧>0、新各>=-.3pp作为新流复现检查，不以失败为理由替换seed；未复现时降低整体结论。所有主要比较均报告固定seed163/10000患者配对bootstrap95区间；点估计门槛通过且CI跨0时只能称开发门槛通过，不能称稳定人群增益。

绝对保持NT相对S†79.433714每流>=-.5pp另列；原源80.367407到入口-.933692pp仍计。患者相对入口下降>.5/1/2pp人数、p10/中位/改善下降、各器官/EDES/train-vs-val完整列；旧域>1pp人数每流不得超过G作为患者代价门槛。NT-NP/NI/G均列逐器官损益，不称人数阈值有临床意义。

EMA统一辅助，不能替换最后学生或事后选择部署口径；无T/GT组合故不虚构两因素交互。本轮检验局部敏感性权重的功能效用，不证明全decoder/skip保持、独立源种子、多域长序列或临床泛化。共享源/反复开发/多重比较限制全部保留。

全阶段COMPLETE/EXIT0、全部worker/timeout退休后仅一次本轮新CPU审计，0模型forward；核精确20384更新、calls无pending、源统计0更新、冻结w/P、共同入口空opt、无Gcode/teacher、finite final=latest opt/RNG/reads、各患者EDES/数据角色与退出资源。预算内未完成则报告部分/失败，不能伪称通过。只有用户后续允许启动并实际完成后，才整理匿名中文报告/源码/聚合/必要审计，按既定显式代理选择性push及remoteSHA/匿名HTTP/local+remote收据；本次不push、不改旧STATE、不启动后继。

## 8. 本轮交付与待确认状态

此次交付仅本协议供用户审阅，无新实验结果。审阅后若用户授权启动，先实现复用引擎的分支与新审计，再执行上述同一个有限预算；不得将实现完成或CPU张量自检当作医学实验成功。原已退休轮次不重启，旧报告/终点/失败现场全部保留。
