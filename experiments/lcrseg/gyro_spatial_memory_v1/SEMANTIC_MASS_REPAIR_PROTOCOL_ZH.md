# 源记忆语义覆盖修复 v1（独立冻结实验）

依据：初始32码RV非空数0且码间余弦0.9957；凝缩后C0全背景、C1前景0.0528%。既有主轮/诊断原样保留。本候选只修复C0记忆形成，不加强几何、增加码量/数据/seed，不复活原图回放。

固定共同P/G、32×8×8×32初始化与源入口、query/码采样序列、原256outer Adam(.01)及256functional SGD(.001)不变。在原utility+.1maskKL上增加1.0 KL(p_source||q_generated)，其中p_source只由原Siemens源train-L10患者/20EDES全部合法label像素流式累加四类比例得到，255忽略；q_generated是本批8码生成soft mask的四类像素平均。无U GT、验证或测试参与。p仅匿名四类统计，不保存逐图特征/索引。增加项系数固定1，非同一空间GYRO新机制声明。计算该项额外生成8样本、不增加native前向；全部实际optimizer和functional照账。

本轮只做一个repair候选，与已冻结C0及A参考对照，明确历史匹配对照、非同期、探索性限制。工程上限32调用/等价更新，正式恰好256outer+256functional；总≤544，0既有toy，不重置失败/恢复成本。复用原真实超梯度、golden、恢复、固定query下降工程资格，工程权重丢弃；不过关则不正式，无盲重试。

形成后全部32码一次生成统计，最多32个新增生成样本、原native前向0。冻结资格：每器官soft质量占比在[.25p_c,4p_c]内，hard非空码各≥8/32，所有数值有限，持久方法≤8MiB。达到才可另立目标训练协议；本轮不自动目标训练，不通过也不改系数/补步数。通过只代表避免前景覆盖不足，不代表生成真实性、任务效用或真实旧域保持已证实。

输出create-only授权数据盘新root，仅既定GPU3/UUID；旧codec与initial_codes只读引用（可用symlink），不复制原checkpoint/data/bank。预计新增<250MB，新增800MB保护+200MB原子余量，严格<1GB，更严格于全局50GB。cache/temp/log/checkpoint均新root；原始时钟统一2h，90min不新阶段、105minTERM、120min前KILL，重试共用原时钟。挂载写读探针/显存检查、中性主子argv/ps/nvidia-smi、独立timeout；仅原space-lora-gpu3-4心跳跟进。失败保留现场，不动历史原数据及结果。

完成后核计数/EXIT/worker退休与固定资格；选择性显式代理公开源码、完整协议、匿名结果和中文报告，remote SHA/匿名HTTP及local/remote DELIVERY齐。不公开图像、生成mask/码/权重、患者记录、原日志/服务器配置，不建PR。
