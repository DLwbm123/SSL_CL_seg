# 方法与实际张量绑定

## 底座和坐标

新增辅助项的唯一入口是 `decoder.dec1.merge.block.3` 的 forward 输入；实际为 [B,16,H,W]，卷积权重 [16,16,3,3]，stride=1、padding=1、dilation=1、groups=1。`binding()` 核对真实 LowRankConv、LR_SRC_A、无 left/B 侧约束，以及 `parent.V64_13` [144,8] 与层 `right` 的 dtype 转换后逐值一致。

每个 patch 的 144 项按 channel → kernel_y → kernel_x 展平，与 Conv2d.weight.flatten(1) 和 unfold 的坐标一致。只 gather 已选中心的 patch，训练辅助项不创建整图 unfold。CPU 坐标测试用 unfold 作为独立 oracle。

该层输入与图像标签处于同一空间网格；这里不使用最终无 padding 读出头专用的 output_to_feature 映射。形状不一致直接拒绝，不静默 resize。3×3 patch 任一位置越界、L ignore=255 或 U 几何无效，均排除该中心。U PAS 仅用于中心的接受与类别，不把整 patch 的 PAS 一致性作为额外门槛。

## 类别条件 SWD

teacher-L 捕获来自原 B2 当前 clean-L 原型估计前向，不增加 teacher 前向。student-U 使用去重后的原 B2 同一当前 U batch，额外一次 clean feature forward，保持到上游 A/B 的计算图。参考和坐标基均 detach。

C0 执行相同 clean-U feature forward，但不保留无用反向图。C1 使用原始 144 维 patch；C2 使用一次固定随机正交 144×8 基；C3 使用入口现有真实 V。所有臂的原生保护仍是原 V，不采用 LR_RAND；新增模块不计算 SVD。

向量 L2 normalize eps=1e-6；每类 n=min(64,n_L,n_U)，n<8 不参加等权平均。rim=1、cup=2。方向32，沿用已有 sliced_wasserstein_equal 的单位方向、排序和平方差平均。所有有效类别平均后乘 D/8。无有效类别时返回 student.sum()*0 并计数，不伪造类别/support。

固定随机基使用 random_key_entry；位置使用 pixels/L/class 与 pixels/U/class；投影方向使用 directions/D。它们是独立私有 generator，均不包含 arm，位置 namespace 不包含 D，也不消耗共同训练 RNG。随机基不择优。

## 目标与梯度

`L = L_supervised_original + ramp * (KL_B2 + 0.05 * L_align)`。

父约束原函数保持；lambda_KL=1、B2 PAS=.6/.7，warmup/ramp=.2，A/B 初始lr=.0005、Adam及调度均原样继承。完整设置以 FROZEN_OPTIONS.json 为准。C0 系数0，C1–C3系数.05；API额外只允许资格用的零系数等价检查，不提供调参列表。

复用 StageTrainer.update/_update、split_gradients、EMA、约束检查、scheduler 和原子提交逻辑；新类只提供 B2 loss 路径的适配及成本统计。KL 保持对全部原 A/B 的权限。辅助项位于所选层输入，所选层自身 A/B 无此项梯度是预期行为；新 B=0 时 A 的辅助梯度为0也是预期行为。测试要求至少可达上游 B 非零，不要求全部 A/B 非零。

没有 R/F、投影 MLP、优化器新组、动态门控、梯度裁剪、B侧正交或历史状态。部署仍使用原 B2 parent.deploy。

## 诊断

每步记录实际 PAS coverage、n_L/n_U/n、有效类/无效步、原始 SWD、D/8 后损失、最终权重后损失、patch 投影范数和有双类支持时的归一化均值距离（类别分离指标，否则null）。有效步比例由 active_steps 和 invalid_steps 得到。

固定诊断位于每阶段25/50/75/100%更新前损失图：Drishti 525/1050/1575/2100；RIM 800/1600/2400/3200。分别记录监督、ramp*KL、ramp*lambda*align 的梯度范数以及辅助项与前两项的余弦/角度；零范数角度为null。额外 autograd VJP 单列，retain_graph 后原 optimizer 正常更新，不依诊断改权重。
