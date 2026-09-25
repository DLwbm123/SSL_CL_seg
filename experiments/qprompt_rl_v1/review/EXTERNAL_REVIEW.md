# QPrompt R0 外部代码审阅

审阅基线：`b100df67bee1db0f7c0ed1819126901b0f47390d`  
分支：`codex/qprompt-rl-v1-r0`  
审阅日期：2026-09-25  
审阅意见：`REQUEST_CHANGES_BEFORE_R1_EXECUTION`  
本意见不修改远端状态，不构成用户对任何阶段的执行授权。

## 范围与证据边界

已读取提交固定版本的审阅入口、来源记录、关键协议、R1 预算、CPU 报告，以及 qprompt/models.py、losses.py、state.py、data.py、metrics.py、tests/test_r0.py、CPU 测试入口。另对公开 ICLR 2026 论文第 4、5、6、10 页做了图像核对。没有逐个审完所有迁移工具与第三方文件。

没有访问私有迁移回执、服务器上的数据、官方权重或用户确认的私有 PDF。本次能核对公开声明和实现，不能独立背书目标服务器的全部文件 SHA、GPU 资格或训练状态。公开论文不能替代对指定私有 PDF SHA 的验证。

本包为审阅者环境中的独立 CPU 最小复现，不是目标端 R0 套件的第三次运行。PyTorch 为 2.10.0+cpu；目标公开报告记录的是 2.6.0+cu124。优化器调用 0，真实数据读取 0，CUDA 调用 0。losses.py 的完整字节 SHA 与提交内 CODE_MANIFEST 相符；semantic_probabilities 是从 models.py 原样摘出的函数，不声称完整 models.py 已在本地逐字节校验。

## 结论

R0 的已交付工作和记录应保留。没有理由据此否定 QPrompt/GRQA 方向，也没有科学收益证据可供判断。建议修复两处基础接口边界后再审阅新的提交；R1、R2、R3 继续禁用。

R0 有意不包含正式科学训练入口。因此，“R1 尚无端到端资格”属于阶段边界，不应冒充本轮新发现的算法错误。

## F1：全 ignore 图像仍被施加 no-object 分类监督

位置：`qprompt/losses.py::supervised_query_loss`；对应覆盖不足位于 `tests/test_r0.py::test_query_semantics_and_matching`。

标签全为 255 时，循环跳过 mask 匹配，但 targets 保持为 3，随后 cross_entropy 仍使用该图像。对 B=1、K=12、class logits=0、mask logits=0 的合成输入：

- matched = 0。
- total = 1.3862941265。
- class gradient L1 = 1.4999998808。
- mask gradient L1 = 0。

在一张有效背景图像与一张全 ignore 图像的混合 batch 中，只把 ignore 图像的 no-object logit 从 0 改成 10，总损失从 6.5015149117 变为 5.9974575043。

这是“标签未知”与“确认不存在匹配目标”的混淆。现有测试只断言 matched=0，没有断言 loss/gradient 语义。尚未证明真实 M1 数据会触发这个分支。

建议修复：全 ignore 图像排除于分类损失和其归一化分母；整个 batch 全 ignore 时返回同时连接 class/mask logits 的零损失。合法的全背景图像仍须正常监督背景，不能一并跳过。

验收：全 ignore 时 loss=0、两类 logits 梯度为零；混合 batch 对 ignore 图像 logits 的修改不敏感；全背景有效图像 matched=1；常规输入结果不变。

## F2：极小分数下 semantic 输出不满足概率契约

位置：`qprompt/models.py::semantic_probabilities`。

代码以 denominator>0 选择正常分支，却将实际分母 clamp 到 1e-12。于是 0<denominator<1e-12 时，三类输出之和小于 1。

复现：B=1、K=12，四类 logits=[0,0,0,35]，mask logits=0，FP32。每类输出约 0.0037830707，三类和为 0.0113492124，而非 1。

在本地 CPU FP16 的另一个边界测试中，前向均匀回退为有限值，但反向 class/mask 梯度均出现非有限值。原因是 1e-12 在该精度下不能充当有效安全分母；torch.where 不会阻止未选分支被计算。这是本地测试结果，不是目标 CUDA/AMP 故障记录。

正的公共缩放不改变该像素的 argmax，因此不能据此声称 R1 Dice 已被污染；但非归一化输出不符合公开接口，并会影响后续以 semantic 概率计算的 confidence/soft KL 等量。

建议修复：至少在 FP32 中计算聚合及归一化；在除法前建立安全分母，避免零分母分支；明确零或极小质量的统一回退规则。非有限输入应报错，而不是悄悄伪装成正常预测。

验收：普通、极小、零质量情况下概率有限、非负、三类和近似 1；需反传的路径梯度有限；FP32 与正式拟用的精度分别验证。不要改变正常区间的 query 聚合定义。

## 已核对的正面结果

1. 公开分支 HEAD 与提供的完整 SHA 一致。
2. 真实背景与 no-object 分开；ViT query 仅在最后 block 注入；query head 属于推理结构，bank/reference 是额外训练状态。
3. 选定式 (13) 的 expm1(delta)-delta 与公开论文所印公式一致，完整 categorical KL 没有被静默拿来替换它。独立测试还确认：reference 与 current 数值相同时，group loss 接近零而 query 梯度非零，reference 不获梯度。
4. 公开 CPU 报告区分了真实合成优化器调用与临时假优化器的失败记账测试，不能将二者混为未记录的真实训练。
5. CODE_MANIFEST 的 42 个成员不包括自身；加自身后为 43 个代码包文件。公开数据总字节 22,326,425，加 194,701 个成员字节和 10,392 字节清单，恰为用户报告的 22,531,518。这是公开数字的一致性检查，不是对私有回执的独立复核。

## R1 正式运行前应单独审阅的闭环

- D0/Q0/Q1/Q2/Q3/Q4 显式分发、损失权重、prefix/suffix 边界、scheduler 连续性和 Q4 直接对齐应有真实调用路径，不只是预算条目。
- 同组合共享 2000 步 prefix，应恢复学生、AdamW、scheduler、计数、数据游标，以及实际使用的 RNG。capture/restore 当前覆盖 Torch 全局 RNG；若使用独立 torch.Generator、Python/NumPy 随机数、sampler 或 AMP scaler，必须另外保存或证明其无状态派生。不应把未来可能的缺项写成已经观测到的漂移。
- reference/bank 的初始化与更新顺序应按协议执行；失败或跳过的 optimizer 调用如何影响 scheduler/reference/bank/计数，必须明确。跨域 reset 暂不作为已验证的 R2 能力。
- 数据资格测试应真正经 CanonicalM1.__getitem__ 读取允许的样本，检查几何、值域、类别、有效像素及图像标签对应关系；val 由独立评价路径处理。字节和顶层键检查不替代这些检查。
- 当前共享依赖可用于 R0；R1 要固定并核验依赖指纹、精度和资源预算。未知用户 quota 不能用文件系统总可用空间代替。

原 R1 预算不变：4 × (3000 + 2000 + 5 × 1000) = 40,000 次正式更新，28 个任务，24 个最终学生。CUDA 合成资格最多 32 次、L-only smoke 共 16 次为额外的未来分项，均不是本审阅的授权。

## 建议的下一步

只做 `R0_REVIEW_FIX`：修复 F1/F2、补充定向回归、更新清单和审阅入口，提交新 SHA；不要在同一次交付中启动 R1。

目标端原 R0 套件 2/2 配额已经用完。保留原 ledger，不删除、不改目录伪装新计数。新增回归必须单列并获得明确预算确认；上述数学回归可做到 optimizer 0。完整 R1 执行器准备与其端到端测试应另列范围，获批后再做，不默认新增训练授权。

## 复现方法

在安装 PyTorch 的独立 CPU 环境中执行：

```bash
python review_probes.py
```

脚本不会读取医学数据、不会调用 optimizer.step、不使用 CUDA。结果写入 `review_probe_results.json`。脚本用可执行断言核对 losses.py 的 SHA。

## 来源定位

主仓库：DLwbm123/SSL_CL_seg；以上文件均按完整审阅 SHA 固定。算法交叉核对来源为 ICLR 2026 公开会议论文《QPrompt-R1: Real-Time Reasoning for Domain Generalized Semantic Segmentation via Group-Relative Query Alignment》，第 4、5、6、10 页。私有 PDF 的同一性不在本包的已验证范围内。
