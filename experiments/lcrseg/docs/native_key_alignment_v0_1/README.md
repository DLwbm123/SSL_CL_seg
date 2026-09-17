# NATIVE_KEY_ALIGNMENT_V0_1 — CODE_READY_FOR_REVIEW

本轮交付为独立的新研究代码准备，不是 F5 P2。旧 F5 的 STOP_NO_ADDITIONAL_EXPERIMENTS、结果、代码和批准文件均未修改。

## 实现和冻结范围

- 新包：`experiments/lcrseg/native_key_alignment_v0_1/`，6 个 Python 文件。
- 基于代码 `667f178c3b80183fd80809760ff31ec5f9a14e25`；已封存结果来源 `3034b2199aa7d6e54ea67499f391a0dd4ea3d21e`。
- D1：C0/C1/C2/C3 × seed163/164 × O1/O2，共 16 个第二目标阶段、42,400 次计划更新。source 与第一目标新增更新均为 0。当前所有节点 executable=false。
- 仅计划 D2/D3，没有可执行节点，没有候选筛选、调参或自动推进。
- 真实底座：NATIVE_LR_SRC_A_3DOMAIN_V1 + B2_C06。保持 14 层 LR_SRC_A、A/B 更新权限、全部 B2 options 和部署结构。

入口审阅：[方法](METHOD_SPEC.md)、[数据权限](DATA_ACCESS.md)、[生命周期](LIFECYCLE.md)、[差异与风险](REVIEW_CHECKLIST.md)、[D1 矩阵](D1_MATRIX.json)、[完整 options](FROZEN_OPTIONS.json)、[前缀绑定](PREFIX_BINDINGS.json)、[CPU 证据](TEST_REPORT.json)。附件原文单独保留。

## 当前边界

CLI 仅有 freeze / plan / test。没有 CUDA、smoke、run 或监测入口。新 trainer 只接受带本研究 CPU 合成 scope 的能力与生成数据提供器，旧 F5 能力被拒绝。

生产调度、真实前缀重新验收、真实恢复能力、CUDA 正确性与真实 smoke 均未授权，也不声明已通过。这个提交供外部代码审阅；任何后续执行需要新的本研究批准和启动授权。

## 复核方式

`plan` 仅验证冻结 JSON，不读取真实张量。CPU 测试使用已安装 Python/PyTorch 和固定 JASCL 源码；通过 NAS 包装器和中性环境入口运行，CUDA_VISIBLE_DEVICES 为空。不要以 python -O 执行，不改变依赖。

本轮有限 CPU 上限为 32 次 optimizer 调用和 2 次测试调用；账本追加且不可重置。既有测试证据不等于新一轮执行批准。审阅人独立测试的环境和成本应另行声明。

## 本轮验证结果

最终7/7组PASS；两次合成测试累计30/32次CPU更新、2/2次测试调用，零失败。两次共约9.26秒CPU测试墙钟时间。新增clean-U前向和诊断VJP分列在COST_SUMMARY.json；真实/CUDA更新为0。
