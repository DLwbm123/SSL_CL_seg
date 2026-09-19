# 当前可交给 Codex 的指令：冻结范围、补齐审阅材料，不启动训练

请按照附件 01_EXPERIMENT_PLAN.md 整理 AGMS_DS_HALF_V1 的定向审阅交付。
这是研究规划和审阅准备，不是外部批准。外部审阅由用户将提交交给ChatGPT/审阅者完成；你不能自行签署APPROVED。

固定来源：
- 原AGMS完整结果 ef6888dc81519ce9e9ca13ea497bbf6397747d70。
- HALF候选 44c1da8a5a8a75021a0298d42bd8f9088e9b059a。
- 原AGMS执行 89dc7657c4f0b7c6f51de339d0f6bc2a621da55f。

1. 保留原10/10、26500更新和P_perf/P_joint失败结论，不删负结果、不补训练。
2. HALF科学矩阵保持2节点5300，lambda_DS=.125，其他不变；保留已经执行的CPU21次，不重跑旧suite、不清空账本。
3. 只读核对精确SHA下的模型、训练、风险、EMA、数据、RNG、恢复和权限差异清单。说明DS减半同时影响头和主网络，不能宣称只缩小主网络干预。
4. 确认科学输入、原B2两前缀、A0/原A5历史导入、当前环境、代码树、CPU证据和报告定义都绑定到本候选。真实权重/数据验收仍PENDING。
5. HALF原注册成功门槛不得改变。预先在解释文档中区分“恢复旧A5”“恢复到B2附近”“真正超过B2”，每个顺序都报告Final/Old/Incoming及每旧域rim/cup。
6. 当前操作限代码/聚合元数据读取和静态比较；不调用optimizer、不执行P0/CUDA/smoke、不开真实前缀或患者数据，不启动监测。
7. 不实现附件的OBS0/ROUTE_HALF，也不添加SCNP、风险温度调整、H增强或fine降级；这些全部PLAN_ONLY，等待HALF结果后的新决策。
8. 不为保存分析而更改被审执行checkout。分析可放独立文档位置。若发现确需代码修补，先列真实阻塞项与窄修补计划，提交新SHA后重新交外部审阅；不能沿用旧SHA批准。
9. 给出精确候选SHA、科学/执行/前缀/导入/环境/代码绑定、已保存CPU证据及可执行RUNBOOK入口，停止在STOP_AWAITING_EXTERNAL_CODE_REVIEW。

仅当外部审阅者实际给出绑定该SHA的APPROVED，且用户另行给出启动确认后，才可按既有HALF合同执行CUDA11、smoke8与正式5300。不得把本文当作Prompt B，不伪造launch receipt。
