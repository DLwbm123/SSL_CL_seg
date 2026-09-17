# NATIVE_KEY_ALIGNMENT_V0_1 — STOP_AWAITING_EXTERNAL_CODE_REVIEW

R1 代码修复与生产接入已实现并推送供复审。本次授权只有代码、元数据和有限 CPU 生成数据验证；没有真实权重/患者读取、CUDA、真实 smoke、D1 或监测，也没有创建生产批准或用户启动确认。

基于被审提交 `9915168cc3707fb70afdc2c719c8781cdf76e3fb`。原 R1 **CHANGES_REQUESTED** 原文保留在 [external_review_R1](external_review_R1/EXTERNAL_REVIEW_R1.md)，未改写审阅结论。新代码需新的外部批准和独立用户启动确认。

## 审阅入口

- [逐项修复与证据](REVIEW_RESPONSE_R1.md)、[运行手册](RUNBOOK.md)、[代码差异](CODE_DIFF.txt)
- [科学字段未变](SCIENTIFIC_DIFF_R1.json)、[代码清单](CODE_MANIFEST.json)
- [D1 科学矩阵](D1_MATRIX.json)、[完整 options](FROZEN_OPTIONS.json)、[四份共同前缀](PREFIX_BINDINGS.json)
- [新增执行与资格定义](EXECUTION_PLAN.json)、[生命周期](LIFECYCLE.md)、[数据权限](DATA_ACCESS.md)
- [当前 CPU 报告](CPU_INTEGRATION_R1/TEST_REPORT.json)、[两次尝试](CPU_INTEGRATION_R1/ATTEMPTS.json)、[累计成本](COST_SUMMARY_R1.json)
- [原方法](METHOD_SPEC.md)、[待审风险](REVIEW_CHECKLIST.md)

## 不变的科学范围

NATIVE_LR_SRC_A_3DOMAIN_V1 + B2_C06，LR_SRC_A/A-only；C0 no-op、C1 全144维 patch、C2 随机8维辅助键、C3 原生入口V的8维键。层、.05、D/8、PAS .6/.7、全部 B2 options 与四份历史前缀不变。

D1 仍是四臂 × 163/164 × 两顺序：16个第二目标阶段，42,400次正式更新；source和第一目标新增训练均为0。D2/D3无可执行节点。它是共同前缀诊断，非完整新方法两阶段轨迹或独立患者验证。

`D1_MATRIX.json` 保留原科学文件及历史代码准备状态；`EXECUTION_PLAN.json` 单独记录新执行定义。矩阵标志不授予运行权限，CPU PASS 也不授予权限。

## 本次 CPU 结果

最终6/6组PASS。新增CPU补充两次各17调用，累计34/48、2/2次尝试；包含两次预设 after_optimizer 失败调用，均记账且没有提交错误状态。旧D0仍30/32、2/2且未重跑。合计64/78 CPU调用；补充诊断62次VJP单列。剩余数值额度不授权第三次suite。

两次报告各自绑定当时代码树；第二次覆盖最终提交代码。旧根目录 TEST_REPORT.json/COST_SUMMARY.json 是历史D0证据，保留原样，不能冒充当前代码验证。原F5和共享模块均未修改。
