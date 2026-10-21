# 两域六臂真实 SSL + CL：v1（运行前冻结）

授权：2026-10-21 用户要求按 Pro 第二轮执行，pro5000 GPU 2/3/4/5。
这是新的独立实验，不续跑已关闭的监督/合成诊断。旧负结果保留。

## 数据和问题

M&Ms canonical320，既有 seed0 患者划分，Siemens→Philips，20% 标注。
Siemens 患者 L/U/val/test=10/38/23/24；Philips=13/50/31/31。
两期 ED/ES 同一患者同一角色；审计远端清单与仓库冻结患者表角色一致、所有角色及域间患者不交叉。
训练只访问当前域 L 图像/GT 和启用 U 的臂的 U 图像。U 清单必须无标签路径。
验证仅两域 val，测试图像和 GT 均不打开。不得把 val 报为 test。

## 模型与冻结简化

B 为 image→共享 U-Net 特征→独立高斯 q(c,R,S,M)，c16、R16×32×32、S/M各1×128×128；各方差用 logsd=-4+3tanh(raw/3)。kappa=0 指独立后验。
G 是可学习 MAP 网络，16→256→16×4×4，上采样后加 R，3×3 conv16→16→3，以零背景 logit 得四类概率。F 为可学习 MAP conv4→16→16→1；输出空间去均值。
保留 S/M 图像路径：p(I|S,M) 为 scale .1 的归一化 Student（Gamma shape2/rate .01），p(S|P) 为 D(S−F(P)) 的 scale .2 Student，D=I−.9B 为周期四邻域 SAR，包含 logdetD；p(M)=N(0,.5²)。c/R 为标准高斯先验。
局部 q 采用重参数样本，4 次；训练期监督损失为各样本 CE 的均值 + 后验预测均值上的 GT-present foreground Dice（复用既有CE/Dice定义）；无前景时 Dice 项0。图像项为 E[logq(S,M)−logp(I|S,M)−logp(S|P)−logp(M)] 加解析 KL(c,R)，除以128²；总权重 .01。这个带 Dice、归一化加权的损失是混合训练目标，不宣称严格整体 ELBO。F(P) 用软概率，是本轮明确的连续语义条件简化；不称原版离散 BayeSeg。G/F 均 MAP，不声称全权重 Bayesian。
U 为相同 base16 U-Net，原四类输出；监督 CE+同一 Dice。所有模型从同一已有 Siemens L-only 5000步预训练 backbone 开始（该继承成本额外声明），B 的 G 从已有两标注病例 G checkpoint 的均值初始化，H(c,R) 从既有初始识别器；新增 S/M/F 随机初始化。来源只含源域训练；不选取目标域表现初始化。B 与 U 初始化非完全参数匹配，因此主要因果比较是模型内。

## 源阶段和六臂

源域 B、U 各5000 SSL更新；EMA teacher .99，源终点模型/teacher/随机状态/域统计为各家族唯一目标入口。
目标四 B 臂：B00无U无history，B10有U无history，B01无U有history，B11有U有history；U10/U11均启用U，分别无/有history。每臂5000更新，总新增40000正式更新。
各臂每步2个当前 L；启用U时另2个当前 U。患者均匀→期均匀→切片均匀；L抽样和几何翻转由step确定，各臂一致。无U臂不构造U dataset，不从U计算统计或BN。
训练模式由各家族同一checkpoint复制；目标优化器统一新建 Adam lr3e-4, betas(.9,.999), eps1e-8, wd0, gradclip1；源同设置。端到端训练，不冻结 backbone/G/F。
每步L监督两视图均值：原当前视图及一张几何不变强度视图。U使用EMA弱视图的后验预测均值作为软目标，与学生强视图做 pixel-mean KL，系数 min(step/1000,1)。不加置信度阈值/entropy filtering/历史G KL。teacher不回传；每步更新参数及buffer；GT优先于伪标签。
两种强视图同一运算/次数：从当前图像减全图均值并按标准差归一化，使用域统计均值/标准差还原，加随机对比(.8,1.2)、offset±.05、低频二维余弦bias幅度±.05（相对std）。50%选择历史风格（history臂），其余当前风格；无history臂100%当前风格。历史统计仅源域训练图像均值/标准差两维的患者均衡均值，源B/U同一预先统计；不保存旧图像/特征/患者latent/概率图。不声称这是解剖生成回放；它是当前解剖的低维域强度统计迁移。
目标当前统计仅取该臂可见L（所有臂相同），源历史统计取源L（所有家族相同），不从验证或U暗中取统计。几何只做同步翻转，强度视图不改GT坐标。

## 评价

源终点立即评价两域val；目标step0复用同一入口评价，避免把到达性能当适应；终点5000固定，不挑最佳epoch。报告患者等权宏Dice（期内按GT-present类→两期→患者，另报告各类/空预测）、NLL/Brier与HD95惩罚版。图像预测先于该期GT读取，GT不参与预测；不保存大概率缓存。
主要对比 B10−B00（SSL），B01−B00（历史），B11−B10（有SSL时历史），交互(B11−B10)−(B01−B00)，U11−U10；家族间比较为辅助。
报告旧/新域终点、两域等权均值、arrival gain、新域适应；旧域 forgetting=源终点旧域−目标终点旧域，BWT反号。单seed163为pilot，不作多seed显著性结论；配对患者bootstrap95% CI可用于描述。

## 集成、预算、停止

正式运行前检查：划分/隐藏标签边界、真实L/U forward/backward有限、G/F/S/M梯度、teacher隔离与EMA、四B数据开关、几何对齐、checkpoint+optimizer+RNG可恢复。集成少量更新计入单次实验总时钟并单列，正式参数从初始状态重建。
存储/data/bmw/SSL_CL_seg/runs/sslcl_v1_20261021，create-only；中性运行目录/data/bmw/.ops/x20261021_s1。预计<3GB；本实验新增运行+源码+临时全部<50,000,000,000 bytes，49GB硬停，预留恢复；已有只读数据/初始checkpoint不重复计。
从首次集成GPU执行计时，子任务/重试共用原时钟，后台控制器23h45停止自有进程组，外层timeout23h55兜底；不终止他人作业。失败/非有限停止受影响臂，其他独立臂保留；若入口失败，不启动其后继。
GPU2/3运行源B/U；目标最多4臂并行使用2/3/4/5。GPU以启动时UUID绑定。保留一个30分钟监测。结果出来自行分析继续，协议变更另立记录不追改本轮。
