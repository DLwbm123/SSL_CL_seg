# V2 实现合同与状态生命周期

- 独立包，不修改 V1/five_frameworks/lctx engine；V2 Parent/Model/Trainer继承已有实现。ObservedConv只是实例局部 subclass，保留 state keys、effective投影、merge和dense EMA；不额外计算卷积。
- `protocol.py` 固定一个方法/预算。`model.py` 使用 losses/labeled_addition/gradient_audit/transform_update hooks。`protection.py` 约束提案/EMA/seal。`memory.py` 持有私有纯摘要。
- `runtime.setup` 只接受 V2 sealed predecessor，不重训 source。`bootstrap_source` 明确读取已有完整 V1 source native，固定训练L统计，零 optimizer 调用。未在本交付实际执行 bootstrap。
- `runtime.worker` 先拒绝缺失 future authority receipt，再检查 protocol digest、节点和 create-only V2 root；V1 root禁止。通过 context-local patch 复用 V1 worker编排，仅在未来独立进程中执行；不得在运行中的 V1 worker 调用。`entry.py` import无调度，无 controller/自动训练入口。
- 一个 Adam调用即一个 physical ledger事件；发生异常后 `requires_restore` 保持，既有失败成本/后代BLOCKED契约不变。有限投影/回溯不调用 optimizer。zero displacement不复活已用步数，不增加预算。
- 观察 draft在L反向图上产生，梯度 detach 后暂存；skip 不提交 draft，成功 update 后提交。若该提交失败，设 requires_restore并停止，禁止 checkpoint未提交状态。teacher/anchor始终eval、无 requires_grad/梯度。
- save先使用既有语义绑定产生私有 staging payload，再扩展 V2状态并原子提交最终文件；失败保留staging/历史最终文件。extras包含历史/当前摘要、可靠性、guard totals和physical count。恢复前验证版本、不可变历史完全一致、各 native key/shape、谱界和counter；再调用原engine恢复模型/EMA/Adam/scheduler/scaler/RNG/cursor/provider_reads。不把V1 checkpoint当V2 resume。
- Checkpoint/patient数据/工程临时tensor/原始日志只在获授权NAS新工程根；源码/匿名汇总在Git。当前测试只CPU生成数据，CUDA屏蔽、单线程且nice10，进程中性，禁止Adam/SGD实际step。
- 固定 native upstream与FP32/noAMP/noTF32沿用V1。跨backend不声称逐位同轨迹。当前检查只针对运行主机现有Torch2.6 CPU后端，不访问旧服务器。
- FREEZE.json绑定protocol、V2 Python源码与三个直接继承包的源码digest；未来authority还须匹配protocol/implementation identity。当前无authority、无启动。真实6个V1 source的native权重已只读加载，在生成输入上验证零增量target入口argmax/ logits与source一致；没有建立真实source L记忆或读取患者影像。
