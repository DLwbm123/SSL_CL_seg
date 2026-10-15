# MAIN_STYLE_SSL_V2 方法规范

2026-10-14。独立于 V0.1 和运行中 MAIN_STYLE_SSL_V1 的下一版。实现与工程检查获授权，训练未获授权。单一完整方法，无 ablation、方法搜索或独立 test 访问。结果改善尚未验证。

## 1. 保留的路径

沿用 V1 的 native UNet、固定线性 readout、现有分割标签与父类映射、当前 EMA、唯一 preceding-entry 教师、A/B 右投影、Q/R 子空间、LCTX、同域风格扰动、当前域 bank 和 clean R-only SWD。styled-U 对父网络仍可反传；clean-SWD 才 detach 父网络。新历史摘要不是第三个教师。V0.1 方程和 V1 实现不变。

当前目标：`CE+Dice+.1 hierarchy+.1 memory+ramp*.25*(weighted styled KL+.2 clean SWD)`。新增 memory 权重 .1 是预先固定设计值，未经验证集搜索。V1 原始梯度 halfspace 被 V2 实际位移约束替代，不叠加执行。

## 2. 特征变换与 MRI 首层

`G=I+QRQ^T`，`F=F_prev G`。每次已约束的学生写入、EMA R 更新和阶段 seal，G 与 F 的全部奇异值均须处于 [0.5,2]。源 F 为恒等矩阵；界限相对固定源坐标，不逐阶段重置累计界限。检查用 detach 后 FP64 SVD，不对 SVD 反传。超界的 inherited F 拒绝导入，不用裁剪已崩塌 V1 target 充当恢复。

Prostate/M&Ms 目标阶段首层卷积 A/B 冻结且从 optimizer groups 排除，保留零增量，所以 effective 首层保持入口权重。后续层 rank8、Q/R rank4 均保留。眼底首层沿用原保护。首层冻结不是对 MRI 收益的已验证结论。

## 3. 训练 L 历史摘要

每域独立保存：类别 post-F feature mean、raw second moment、present-image support、观察批次数，以及 native 参数/F anchor 和类别平衡的对角敏感度。不得保存患者图像、标签数组或身份，不用验证/测试建立摘要，不用 U 伪标签建立历史 bank。

目标阶段每100个 committed-cursor 位置（从0开始）观察一次**已读取的当前 L**，不新增 L loader 读取；统计仅在整个 update 成功后提交。每个有 L 和 feature-grid 支持的类，计算该类平均像素 CE 的 native 权重/F 梯度。在 LCTX 的多视图有效权重上，同名梯度先求和再平方。平均这些平方构成 `I[d,c,name]`，是梯度二阶敏感度代理，不是精确 Fisher。MRI 冻结首层没有 target 敏感度更新。source 观察包含可训练 dense native 参数及临时恒等 F 坐标，无 optimizer 参数新增。

每张有支持的 L 图像先求类别 feature mean，再按 present-image 次数累计 mean 和 second moment。坐标为 fixed source-readout 前的 post-F 坐标；摘要来自阶段轨迹，不宣称是精确最终特征重算或跨域精确 transport。prototype 只作软证据，不将不同 scanner 中心平均为单一训练目标。

每个历史域、每个有支持类和每个有效层：`memory_layer = sum(I*(W-W_anchor)^2)/sum(I)/max(mean(W_anchor^2),1e-8)`；按层、类、域分别平均。W 使用 effective native 参数和累计 F，禁止直接继承重新初始化的 A/B 坐标。所有 anchor/I 均 detach、无梯度、非 optimizer 参数。当前 L bank 继续服务当前域 PAS，不替代历史摘要。

最终 seal 将当前域摘要附加到 immutable inherited history。任意前景类缺少训练-L观察/feature支持时 seal 失败，保留缺失，不伪造支持或额外采样补齐。历史域重复、负敏感度、非有限值、坐标/shape/lineage 不符均拒绝。

## 4. 独立入口保护

U 伪标签接受逻辑保持 V1 confidence .7、前景无绝对 cosine 硬拒绝、背景 PAS/entry conflict veto、前景冲突权重 .5、class cap5、confidence taper。历史保护 mask 则只要求原 U geometry 与 entry confidence >=.7，完全独立于 EMA class、EMA acceptance 和 PAS；入口前景/EMA背景或 cup/rim 分歧保留。

每100步的当前训练 L 观察，同时只读 entry 弱图预测，累计每个 predicted class 的正确数与预测数；可靠性为 `(correct+1)/(predicted+2)`，无证据时 .5。不读未来 L，不在验证集校准。可靠性下降不构成支持区域硬删除。

历史 prototype 证据：各域有支持的 normalized prototype 与 entry post-F feature 的 cosine，按类别取域内最大值，再以温度 .1 计算类别 softmax；保护权重为 `.5+.5*p_historical(entry_class)`。没有历史时为1；不支持类的相似度填-1e4；权重 detach。

入口保护 loss 为 entry 对 student styled-U 的类别平衡 KL。每类先按其支持像素数平均，乘该类训练-L可靠性和 soft evidence，再平均 present classes；这些系数不在分母中抵消。空支持为可微0，显式记录 no-foreground-reference，不能称为保护成功。

## 5. Adam 后约束

仍使用唯一 StageTrainer `_update` 和一次 Adam 调用。先存 before，Adam 计算提案/更新 moment 后，V2 `transform_update` 检查实际 displacement，再进入既有 parent constraints、scheduler、finite commit、EMA、bank 和 cursor 提交顺序。

参考向量为 `(grad(entry_protect+.1 memory), grad(combined L objective))`，范围覆盖全部 trainable A/B/R，source bootstrap 不执行这个 optimizer 路径。约束 `g^T displacement <= 0`，相对内积容差1e-7。最多16轮 cyclic halfspace 投影；未收敛则显式零位移。随后所有参数统一按1、1/2、…、1/65536检查实际 FP32舍入后内积及G/F谱界；仍不合法则恢复 before。Adam moments保留本次梯度观察，不伪造回滚、不额外 step、不重放。零位移仍记一次 optimizer 调用并推进既有成功 cursor。

EMA R 从原 EMA R 沿本次 .99/.01 候选插值，使用相同有限回溯，确保 EMA G/F 也合法。权重 EMA仍沿用已有 dense-effective实现。记录提案范数、实际范数、内积、缩放、零位移、谱值、保护空前景/冲突统计。该约束是局部一阶保护，不保证有限步 loss 或旧域 Dice 单调。

## 6. 生命周期与边界

复用每种子既有健康 V1 source sealed；源 native/F 不改、不重训。显式 source import 观察固定32个训练-L批次，加上既有当前bank初始化的1批（共33次 L batch reads），bootstrap U/val/test reads及 optimizer calls为0。32批仍缺前景支持则失败，不自动延长。额外 L exposure 单独报告，不称为完全相同数据曝光。

source memory在源节点构建一次、眼底两个顺序共享其 immutable copy。target只接受精确 predecessor/version/updates/domain/order/seed；跨阶段记忆与当前可靠性/观察状态进 checkpoint。没有未来新域记忆或 old-image replay。仅 final fixed checkpoint 做既有 seen-domain val；主指标/患者级MRI体积指标不改。

仍有风险：谱界不能防止所有语义旋转；参数重要性与 prototype 摘要近似；entry新域错误仍可能受保护；MRI stem冻结可能降低Incoming；多重保护可能造成零位移和吞吐开销。必须报告这些负结果，不宣称工程通过等于改善指标。
