# 固定oracle的Theta/local交叉梯度采样诊断

2026-10-20。原A独立复核方向出现流间反号，故先检查采样波动，不改训练目标。使用已完成两臂S1000；G唯一cursor8675、原族/尺度/两训练图八视图不变，0optimizer/参数应用。不执行历史structure MC草案。

每臂参考风险向量r固定为前阶段四个独立Q_eval梯度的算术均值，只读已封存梯度；它仍是有限MC参考，非精确总体梯度。主标量h=−r·(gD(theta,epsilon)+gK)，负值代表此固定参考Q的一阶下降，不是新独立验证或Dice。

每臂固定8个Theta×8组局部epsilon交叉网格。Theta种子1220163，local种子1320163；每个Theta重置同一local种子，保证八组epsilon在Theta间共享；两臂完全配对。同一格先算8视图期望多类像素CE的单样本梯度gD（一次反向），KL梯度只读上一阶段的解析核验向量，不新增反向。保存每格gD私有数组和h标量；共16Theta/128批量G前向/1024图像G前向/128梯度请求，0optimizer、0新局部训练、0旧评分重跑。固定网格不按读数追加。

描述性分解h的行均值、列均值与交互残差，报告平衡网格SS_theta/SS_local/SS_interaction及其和。方法矩估计vTheta=(MS_theta−MS_interaction)/8，vLocal=(MS_local−MS_interaction)/8，vInteraction=MS_interaction，保留负估计。报告解释限制：单网格、有限参考r、无重复cell，不作为精确方差/置信区间。

为原每Theta独立local的嵌套采样，探索性代理方差V(T,L)=vTheta/T+(vLocal+vInteraction)/(T*L)。报告原2×2及同4前向成本4×1、1×4；计算用负分量截0的保守描述代理，同时保留原估计供审计，不称最优分配已获证明。交叉网格本身不同于原嵌套训练，不能直接把格子相关性当原训练独立性。两个终态若对4×1相对2×2都有至少20%代理降低，可优先另立预算匹配的采样分配控制实验；否则不凭估计直接改训练。此阈值在计算前冻结，后继仍先注册。

源G/raw/Adam与参考梯度文件metadata不变，G无参数梯度；私有样本梯度、参考r、KL与完整标量封存，独立CPU审计复算投影及ANOVA。预算/停止/隐私/交付与原协议一致：指定pro5000数据盘create-only、GPU2/原UUID、原R_START首启时钟含全部失败≤24h、每服务器新增<50GB、独立原时钟23h45/23h59guard与四根18/20GB更严格累计限制，预计<100MB，启动前挂载/探针/GPU检查，不删除历史。源码/中文报告/匿名标量经原代理交付、唯一心跳原位更新，不公开数组/权重/患者信息。不启动H/共享R2/教师/TaskRel/U/目标/holdout/test/旧dilation。
