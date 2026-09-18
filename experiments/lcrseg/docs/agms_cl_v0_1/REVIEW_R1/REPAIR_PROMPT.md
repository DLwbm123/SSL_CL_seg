# 给Codex：AGMS外部审阅R1局部修复，零训练更新

请处理 AGMS_CL_V0_1 对提交 ac01ab6de6250e11877fe9000a30665c2245292d 的外部审阅。

结论是CHANGES_REQUESTED，approved_phases=[]。原CPU模型资格有价值，但P0入口尚有确定的嵌套计量问题。

我转发本段只授权局部代码/报告修复、公开元数据读取、最多两次新的零optimizer回归调用，以及提交推送。不授权原生训练测试、任何optimizer.step、真实权重/患者读取、P0生产审计、CUDA、smoke、正式训练或监测。所有fixture使用生成张量/元数据，不能创建生产approval或launch receipt。

## 1. 冻结科学方案

保留：NATIVE_LR_SRC_A_3DOMAIN_V1/B2_C06、A-only、H/M/risk公式、294个辅助参数、LCTX来源mask、所有阈值与学习率、风险滞后/EMA/梯度白名单、两个顺序、seed163、A1-A5。

P1仍10个新stage2/26500更新，A0两条历史导入；新增source/第一目标0；CUDA36、smoke24、P0原图像上限不变。不得添加SCNP/SWD、额外seed、剂量或性能剪枝。

原80/96次、4/4次尝试和旧134次完整保留。不能重跑原28次套件，不能把剩余16次当第五次尝试授权。本次新增optimizer调用严格为0。

## 2. 修复生产阻塞R1

当前p0.run在P0的cost_session内部调用accept_prefix；accept_prefix内又进入cost_session。共享Operations.__enter__明确拒绝嵌套，会报nested counter scope。

修复为：先分别完成O1/O2前缀验收及独立prefix cost sessions，保存已验收(path, receipt)；所有这些上下文退出后，再开启P0_read_only成本会话，执行原固定L审计。

不要删除prefix验收、不要放宽共享计数器的禁止嵌套规则、不要隐藏失败会话、不要通过去掉成本记录绕过错误。保留P0模型加载处的identity/hash复核及只读权限。分别统计prefix和P0成本，不重复相加嵌套时间。

检查同类实际调用链，避免另一个路径仍嵌套；不为此重构原训练器。

## 3. 修正R2报告口径，优先只改报告层

core.coverage现有ignore是无效geometry数，不是合法但未用于伪监督的像素；fine/coarse按合法geometry归一，ignore_fraction按全图归一。

保留原始诊断，不改变mask/loss。报告中明确：
- invalid_geometry_pixels=原ignore，比例分母总像素；
- unselected_valid_pixels=geometry-fine-coarse，比例分母geometry；
- geometry=0时合法域比例null；
- 校验fine+coarse+unselected_valid=geometry；
- 校验geometry+invalid_geometry_pixels=全图像素数。

为原字段标legacy含义，不能将其解释为全拒绝率。优先在reporting规范化与schema说明中实现，不改core数学函数、teacher或风险更新。

## 4. 定向零更新回归

新增隔离的REVIEW_R1_REGRESSION目录与尝试记录，最多2次调用，每次0 optimizer。不要调用现有run_tests或为了验证P0启动真实访问。

测试至少覆盖：
1) 通过真实计数器/成本上下文及stub前缀/生成provider测试编排，两个prefix会话全部退出后才开始P0；未使用原生方法可stub，不能伪装成生产原生验证。
2) 旧嵌套结构确实触发nested counter scope，修复结构不触发。
3) prefix失败时没有患者/当前L读取；成本证据保留，不产生P0 PASS。
4) 成功/已封存P0幂等检查；不再次消耗图像审计额度。
5) 全合法但无fine/coarse、全部fine、全部coarse、混合、全无效geometry的报告分区和null语义。
6) 只读断言模型、风险、梯度、RNG与optimizer状态未受这些报告/编排测试改写。任何oracle也不得调用optimizer.step。
7) 禁止旧批准或缺失授权进入真实前缀/P0入口的前置检查保持。

附带的外部复现脚本是定向证据，不是新的生产测试许可；可参考其counter fixture，不能声称它完成了原生CUDA或模型验收。

## 5. 新提交的资格绑定必须诚实

不要为了让authority.preflight通过，简单将旧CPU TEST_REPORT.code_tree_sha256改为新值并假称完整原生suite已复跑。

保留原CPU ATTEMPT_1..4、ATTEMPTS、PHYSICAL及attempt4报告原字节；将其作为原生训练资格的明确来源。新的组合验证报告单独标注SCOPED_ZERO_UPDATE_REVALIDATION，绑定：
- 被审基线ac01ab6...及其完整代码清单/attempt4报告摘要；
- 当前代码树/计划/执行摘要；
- 精确的局部差异清单；
- 当前零更新回归结果与覆盖范围；
- 原模型/损失/训练/恢复/数据/随机流实现未变的逐文件hash证明。

允许为生产预检新增狭窄的组合证据校验，但不能删除current-code条件，也不能接受任意旧PASS。只允许P0编排、报告规范化、相关回归及证据验证的声明变化；未知差异应失败。不得把旧报告说成当前提交独立重跑的全部native结果。

原生训练核心若必须改动，先停止并报告需要的新测试范围，不在本次零更新授权中自行扩大。生产仍须下一轮真实外部批准与独立用户Prompt B。

## 6. 交付

提交REVIEW_RESPONSE_R1.md，逐项给出R1/R2修改位置和新零更新证据。更新清单、RUNBOOK与审阅请求；明确保留原生CPU80次来源与未执行生产状态。旧审核文件不能覆盖，不能生成APPROVED。

提交推送后给出完整新SHA、修改文件、零更新回归结果、组合证据绑定、真实数据更新0和所有PENDING项。

停止于STOP_AWAITING_EXTERNAL_CODE_REVIEW，不执行Prompt B。
