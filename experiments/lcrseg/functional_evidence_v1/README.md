# Functional evidence pilot

本轮五组工程资格均通过；两组400步正式L-only触发预设可塑性停止门槛，后三组正式未启动。见[结果报告](results/REPORT_ZH.md)。

原生U-Net上的固定坐标功能证据资格试验，独立于CARE-Seg。完整科学设置、成本、失败保留和停止条件见[PROTOCOL_ZH.md](PROTOCOL_ZH.md)。

- 复用 `philips_transfer_v1.run` 唯一训练循环，不另写优化循环。
- `state.py`：Gaussian/Laplace站点和状态合同；标准代数不作为创新声明。
- `bank.py`：仅合法训练L/U输入的一次性私有特征记忆。
- `core.py`：原生网络接入、密集一致性、固定/可修订图及共享总预算。
- `qualification.py`：计费工程检查；`controller.py`：资格和可塑性停止门槛。
- 私有银行、查询身份、权重、原始日志均不得公开。

本轮为400步短程资格，不能作为收敛性能、真实新增标签修订或TMI新颖性的证据。
