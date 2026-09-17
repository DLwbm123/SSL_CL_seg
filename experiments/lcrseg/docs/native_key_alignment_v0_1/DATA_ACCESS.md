# 数据、前缀与发布权限

当前只使用生成 CPU 张量及已发布聚合元数据，不打开真实 source/prefix 权重、L/U 图像标签或 val/test。CPU 原生构建只读取固定的 JASCL 源码。

未来 D1 训练输入只能是本第二目标当前 L（含真值）和 U（图像、几何、source id；无真值接口）。U 类别与集合来自原 B2 当前 EMA 的预测/PAS。参考特征只在当前 batch 活跃，不入队列，不跨 batch/domain 保留。旧域 val 只能交给隔离 evaluator，不能成为训练器输入。

四份真实前缀保持原 B2_C06 方法身份：seed163/164 × O1/O2 的第一目标学生。PREFIX_BINDINGS.json 从已发布真实完整性证据绑定原 identity、receipt canonical digest、student tensor hash、transform hash 与实际文件 hash。

每个 D1 节点以自己的 prefix_binding_sha256 明确声明 diagnostic_common_prefix 权限。validate_prefix 检查 seed/order/stage/phase/arm、原身份及 receipt digest；不会伪造或改写旧 receipt，也不会放开跨轨迹继承。元数据校验返回 PENDING tensor acceptance，不能据此启动训练。

未来正式接入还须在本研究新授权后重新验收实际文件身份/hash/schema/有限值、原B2 identity F，并独立初始化每个臂。当前不读取真实文件，也不把历史 VERIFIED 视为当前生产启动验收。

16个第二目标阶段共同前缀属于诊断，不是完整新方法两阶段轨迹。早期分数固定，DeltaForget=-DeltaOld，不能计为两个独立证据。163/164已经用于开发，不能称为未见优化seed或独立患者确认。

公开范围仅代码、冻结计划、聚合前缀绑定和生成测试/成本证据；不上传权重、图像/标签、逐患者结果、患者标识、私有路径或凭据。旧F5结果与批准文件不可作为新任务启动权限。
