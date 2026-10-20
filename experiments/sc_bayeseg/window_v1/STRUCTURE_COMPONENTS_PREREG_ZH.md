# Structure只读分解：均值、方差与类2响应

2026-10-20。持续自主授权下的下一判别步骤。前提：structure-only和四格/固定数值init1已完成并交付80edfd4d；坐标目标收益约91.7%为KL下降的点估计加法份额、GT B改善MC未决，四相对指标的F效应及总变化均负。不能盲延长同训练，也不能据此认定7×7上下文或m是根因。

## 固定边界

只原两canonical训练L，128²，不翻转、不读开发保留/U/目标/test，不增加训练seed。全100模型、Adam/scheduler/RNG/prior只读，0optimizer、0反向、0局部更新。唯一旧状态为spatial_generator/S0600(cursor8675)，唯一新状态为structure_only/S0200(cursor8875)；old P2的十个候选×初始化200步qxm固定，禁止重推断或选更好q。

## 均值/方差二乘二

构造四份正规Gaussian qTheta，只交叉F的六均值张量和六logsd张量；其他88模型及所有prior/initial保持旧源精确相同：

- 00=旧mean/旧logsd，复用P2终点评分。
- 10=新mean/旧logsd，新增评分。
- 01=旧mean/新logsd，新增评分。
- 11=新mean/新logsd，复用structure fixed_q评分。

两个混合格是只读参数交换，不是分别训练过的mean-only/sd-only优化器，也不称等价训练轨迹。先回放00和11的首流首Theta十状态×16local，两次160密度，误差各≤1e-6，否则停止保留现场。

新增两格各十状态×4已有流×32Theta×16local，40960密度；同P2 Theta/local分流键及共同噪声。加回放合计**41280批量密度、258完整Theta draw、1290 structure批量forward**。局部q不变，image/appearance/entropy应逐Theta相同，只shape直接改变。原始全模型/RNG/历史账本/源局部状态必须起终不变。

报告四格Gamma、GT−BG/SHIFT/DELETE2的每图/四流/配对MC；均值直接效应10−00、方差直接效应01−00、交互11−10−01+00、两因素对总变化的对称Shapley加法分配。闭合误差≤1e-12；MC为有限两图下数值不确定性，不是患者CI/精确条件证据。仍用原1e-5+四流同向规则，不为过门槛加draw。

解析KL用FP64逐张量分解mean Mahalanobis项与variance项，正规KL/(2N)系数不改；四格KL应无交互。结合固定训练GT q的B变化报告各格坐标负目标。Shapley只是端点参数交换加法分配，不是训练原因占比。

## 类2响应与残差

额外仅4批量**均值权重**F前向：00/11×GT/DELETE2；不作Θ采样、不进入MC池。计算D_F=F(GT)−F(DELETE2)，保存匿名区域聚合，不公开字段。固定qGT/init0/200下，计算sar(mx−F(Z))与Student点shape项（省略常数），4张量图计算单列计数；这是均值点诊断，不是q期望或ELBO，也不改写前述MC评分。

区域预固定为类2、按图外非类2做radius3腐蚀所得内部、其边界、radius3膨胀减类2的邻域、膨胀外部。每图报告count/mean/RMS/maxabs；空区域记null，不修改半径。检查F删除响应在radius3支持外≤1e-6；经周期SAR后在周期radius4支持外≤1e-6。约束源于三层3×3 stride1局部均值映射，不将其说成全概率模型只有7×7依赖。报告F0→F1输出变化和qGT的x标准差作为描述，不从响应大小推导类别证据充足性。

## 预算、停止和后续

总41280密度/258Theta/1290采样Fforward+4均值Fforward/4Student点图，0反向/optimizer/local；无额外候选优化。预计新增<30MB，所有运行输出/缓存/日志在既定数据盘、create-only中性入口。启动前挂载容量/du/真实探针、GPU/ps，独立原始时钟守护23h45保存/23h59硬停，新增软45GB硬49GB且保留旧四根18GB保守限制。失败保留所有成本/现场，不自动重跑，不能改容差刷通过。历史文件不删不覆写。

若方差交换解释多数伤害而均值交换有针对性收益，才更支持另立mean-only训练控制；若均值本身也损害相对证据，优先分析训练目标/固定q与类别支持冲突，不立即放大同类训练。有限响应只是条件表达候选证据，不能证明架构不可能；没有无GT预测输出，不称分割/SSLCL/泛化成功。完成后中文正负报告、源码/协议和匿名标量push，核验远端SHA及匿名HTTP200，唯一心跳统一推进。
