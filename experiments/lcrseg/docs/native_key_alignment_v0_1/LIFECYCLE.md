# 生命周期与当前运行边界

状态：CODE_READY_FOR_REVIEW。生产训练调用数为0。

阶段入口从原生合并权重建立新的 LR_SRC_A A/B 和父规则 V；C2 在此时产生一次固定随机辅助基，C3只引用已存在V，teacher读取同一个基。A/B、EMA、原型、optimizer、warmup均按独立新阶段初始化。stage exit仅保留合并后的父权重；辅助坐标、样本特征、teacher、原型、optimizer均不部署。

新包复用训练器的单次optimizer更新/有限性检查/硬投影/EMA提交实现。C0 no-op额外前向无buffer或RNG副作用已经用实际原生模型状态等价检查，不仅依赖GroupNorm推断。

当前 CLI 无真实执行、CUDA资格、smoke或monitor子命令，trainer同时检查当前研究CPU合成scope、CPU设备、生成provider和B2无sidecar模型。旧任务能力会在构造前被拒绝。

本次不创建真实前缀加载器/生产队列，不将新arm伪装成旧B2任务运行。未来审阅通过后的接入必须把 study/arm/plan/prefix/辅助basis绑定加入独立新协议 checkpoint/receipt；不得直接用旧B2 checkpoint身份来恢复不同alignment臂。该生产恢复接入目前PENDING，未声称验证通过。

D1若获未来授权，必须按冻结16节点完整运行后统一判定；科学负结果不触发调参、seed追加、剪枝或自动D2/D3。工程故障保存尝试、物理调用、失败/诊断成本再停止，不自动重放耗掉预算。

当前合成成本与未来正式成本分开。本轮有限测试最多两次调用、32次CPU optimizer调用；两次实际测试的全部账本保留，禁止删除历史来重试。真实更新、真实smoke、CUDA调用均为0。
