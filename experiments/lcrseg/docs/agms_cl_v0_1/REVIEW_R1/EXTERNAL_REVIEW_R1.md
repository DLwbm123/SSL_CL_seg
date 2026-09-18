# AGMS_CL_V0_1 外部代码审阅 R1

审阅日期：2026-09-18。对象：`ac01ab6de6250e11877fe9000a30665c2245292d`。

## 决定

**CHANGES_REQUESTED；approved_phases=[]。本文件不授权 P0、CUDA、smoke 或正式训练。**

一项确认的生产入口阻塞：P0 在自己的 cost_session 内调用 accept_prefix，而 accept_prefix 再创建 cost_session。共享 Operations 明确拒绝嵌套。修复的是编排，不是科学方案。另建议在同一提交修正覆盖统计命名；该项不会改变训练。

## 审阅范围与已有正向证据

读取了本次 REVIEW_REQUEST、trainer/core/model/authority/protocol/state/execution/qualification/reporting/p0/tests、CPU TEST_REPORT、RUNBOOK，以及共享 model、assurance、native_operations、telemetry。与本对话已有 AGMS 实现规格对照。

静态核对未发现需要改变 H、M 或风险门控数学设计的问题：fine PAS/KL 仍独立保留；coarse 位于 fine 补集；parent loss 按合法几何归约；两辅助头共294参数、仅L训练；U参数白名单为父A/B；辅助EMA显式更新；风险使用已提交的旧值，pending在成功步骤后提交；恢复重建AGMSTrainer；部署继承共享主网络deploy，不保留辅助头；P1仍为10新阶段/26500更新、A0两条历史结果。

仓库的完整原生CPU结果是提交方报告：attempt4 PASS，累计80次/96次上限，4/4次修订后尝试，旧研究134次单列。报告环境为Python3.12.9、PyTorch2.6.0、CPU。没有在本次审阅重跑它，也没有据此声称冻结生产环境通过。第4次测试授权由仓库CPU_REPAIR_1记录；本次审阅没有独立访问对话外的授权系统。

## R1 — 阻塞：P0嵌套成本会话

直接代码链：

```
p0.run
  with cost_session(.../P0, 'P0_read_only'):
    accept_prefix(...)
      with cost_session(.../PREFIX_<id>, 'prefix_integrity'):
        verify_file(...)
```

`f5_confirmation_v1.assurance.cost_session` 每次创建继承 `NativeOperations` 的记录器；`ssl_anchored_mix_v0_1.telemetry.Operations.__enter__` 在 `ACTIVE is not None` 时抛出 `RuntimeError('nested counter scope')`。

P0是RUNBOOK的第一个真实步骤，也是formal run的前置报告。因此在合法环境与元数据前置检查通过后，该嵌套会在第一个前缀计量入口失败；并不是训练到后面才出现的可选日志问题。内层模型验收主体尚未进入，优化器更新为0。

### 独立反例

在本地CPU上，使用真实提取的cost_session实现和完整、Git blob匹配的两个计数器类；只对未使用的原生模型/数据 instrumentation targets提供stub。用cuda=False排除真实CUDA调用。

结果：`nested counter scope`；内层主体没有执行；P0和PREFIX会话均记录FAILED；退出后ACTIVE和optimizer方法包装正常清理；optimizer调用0，真实权重/患者读取0。

这是实际计数器/上下文行为的反例，不是完整生产p0.run、原生模型或服务器测试。

### 必要修复

在进入P0计量会话前，先分别完成O1/O2的前缀验收并留存各自成本；随后开启单个P0推理计量会话复用这些已验收的(path, receipt)。仍在payload加载处检查identity/hash，不删除验收，不放松底层禁止嵌套的规则，不将成本子集重复相加。

添加零optimizer的编排回归：两前缀先验收、P0随后进入、无嵌套；故障前缀时不访问当前L；失败与幂等检查保留；真实P0/CUDA/smoke/formal保持未执行。

## R2 — 诊断语义：ignore不是未使用的合法像素

`core.coverage` 当前把 `ignore` 定义为 `(~geometry).sum()`，把 `ignore_fraction` 定义为全图无效几何比例；fine/coarse比例的分母则是合法几何数。

生成反例：4个全合法像素、fine=0、coarse=0时，返回ignore=0、ignore_fraction=0；真正未用于伪监督的合法像素是4个。

这不是mask或梯度错误，不要求改变训练。建议在报告层保留原始数据并明确转换：
- invalid_geometry=现有ignore，分母为全图；
- unselected_valid=geometry-fine-coarse，比例分母为geometry；
- geometry=0时合法域比例为null；
- 验证fine+coarse+unselected_valid=geometry，以及geometry+invalid_geometry=总像素。

不要用旧ignore字段解释全拒绝率。可以仅修改报告规范化和文档，无需改变core.py中的选择函数、损失、risk或optimizer路径。

## 为何CPU PASS没有覆盖R1

当前math_metadata仅调用P0的opportunities/count数学辅助；资格测试覆盖训练、恢复、模型和报告，但不执行实际P0编排到accept_prefix的嵌套路径。因此完整CPU PASS与本次发现并不矛盾。

## 独立测试证据

`TARGETED_CHECKS.json`：19个定向核验通过，其中包含两个发现的观测和顺序会话替代设计验证。优化器0，原生网络运行0，CUDA0，真实前缀0，患者0，生产permit0。

三个完整源文件通过Git blob匹配：
- core.py：90a644ebb26a8eb28a0d062018eb867b863f34ac
- telemetry.py：8c0c3545175f37308be046d42cf34daa0b40e293
- native_operations.py：2829fc1b330869e0b7b372b4e58c256103bff271

本地Python3.13.5/PyTorch2.10.0+cpu，只运行生成张量与计数器验证，不冒充提交方CPU环境或生产PyTorch验证。未克隆/重新哈希整仓库；公开源文件经GitHub连接器读取，完整三文件手工转存后哈希验证。

## 下一步与额度

保持科学矩阵、全部方法公式、学习率、阈值和后续停止条件不变。建议只做P0编排修复、报告语义修正及零更新回归，再交外部复审。

不重新跑28-call CPU suite，不清空80次/4次历史。旧134次单列，总已记录CPU214次。剩余16次物理额度不表示有新的测试尝试权限。

若需要新代码树资格绑定，使用诚实的组合证据：保留原始attempt4报告和旧完整清单；新提交只读验证受保护训练代码未变、明确列出修复文件及新零更新回归。不得把旧报告哈希简单改成新哈希并声称完整原生测试重新通过。发现超出预定局部修复的变更则停止并申请独立测试范围。

## 主要源码入口

- https://github.com/DLwbm123/SSL_CL_seg/blob/ac01ab6de6250e11877fe9000a30665c2245292d/experiments/lcrseg/agms_cl_v0_1/p0.py
- https://github.com/DLwbm123/SSL_CL_seg/blob/ac01ab6de6250e11877fe9000a30665c2245292d/experiments/lcrseg/agms_cl_v0_1/execution.py
- https://github.com/DLwbm123/SSL_CL_seg/blob/ac01ab6de6250e11877fe9000a30665c2245292d/experiments/lcrseg/f5_confirmation_v1/assurance.py
- https://github.com/DLwbm123/SSL_CL_seg/blob/ac01ab6de6250e11877fe9000a30665c2245292d/experiments/lcrseg/ssl_anchored_mix_v0_1/telemetry.py
- https://github.com/DLwbm123/SSL_CL_seg/blob/ac01ab6de6250e11877fe9000a30665c2245292d/experiments/lcrseg/agms_cl_v0_1/core.py
- https://github.com/DLwbm123/SSL_CL_seg/blob/ac01ab6de6250e11877fe9000a30665c2245292d/experiments/lcrseg/docs/agms_cl_v0_1/CPU/TEST_REPORT.json
