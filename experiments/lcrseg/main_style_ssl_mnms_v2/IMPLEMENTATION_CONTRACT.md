# M&Ms trainer 与训练就绪合同

本次是在既有 f90a8e9 核心候选上补齐绑定，不改变 METHOD_SPEC 的算法、历史摘要定义、常数或正式预算。历史核心工程回执在 core_engineering_history，旧的“没有runner/GPU资格”只描述当时状态。新增资格结果以 TRAIN_READY/ENGINEERING_REPORT 为准；训练就绪不构成正式训练授权。

## 单一引擎与状态顺序

MainTrainer 继承既有 main_style_ssl_v1.MainTrainer / StageTrainer，只实现 labeled_addition、独立 entry reference 和实际位移 hooks。combined-L 加 .1 的逐域类历史代理；一次真实 Adam 后，在提案 native/F 坐标逐项求导，按历史 .5、combined-L .25、entry .25 的权重调用既有弹性算子。无谱回溯和 V1 半空间梯度硬投影。保留 parent 参数约束、scheduler、dense effective EMA、R EMA、bank 和 cursor 的原提交顺序；MRI 首层和 entry teacher 不更新。

Precision 使用 Beta(1,1) 初始值，正确数/总数初始为1/2；训练L每100步更新。历史和当前校准不混为一个在线记忆。

## 历史摘要

collector 在最终 deployment 的独立冻结副本上 functional_call，对 native effective参数和累计 F 的同一 final 锚点计算四类 CE 梯度。恰好32批当前域 train_labeled，每批两位不同患者；使用独立 history_final stateless RNG，弱几何，不做 LCTX。无历史影像、U标签、val/test。没有类支持则失败，不额外采样；只保留 J、I、均值CE、支持和计数，不保存图像或患者身份到摘要。

采集不推进训练cursor、provider读取数、Adam/scheduler/scaler、bank、EMA或训练RNG。source另一次原生bank初始化单列2图像曝光；每目标正常bank初始化和训练读取仍独立记账。所有summary与既有checkpoint只写隔离NAS路径。

源只允许精确已完成V1 Siemens sealed、原native/F和seed/steps，源不训练。正式target seal要求成功提交且达到原固定final；采集失败保留训练账本并阻止seal/后代。工程短端点使用 engineering_only=true 的不同标记，严禁成为正式前代。

## Checkpoint 与恢复

继承原子 payload 的 student/EMA、native/A/B、Q/R/F_prev、Adam、scheduler、scaler、torch/CUDA/Python RNG、cursor、bank、provider计数和telemetry，扩展不可变逐域类history、precision计数、guard累计与物理调用数。恢复先验证schema、预算/计数、正交性、固定前代F、entry/projection、history前缀及与外部前代副本逐tensor一致；无效状态必须在写入live trainer前失败。GPU资格比较恢复后的下一步全部状态和读取数，而不只比较loss。

## Controller 和授权

固定8名义节点、两源87200继承、六目标209800新调用；两seed各一固定GPU，串行Philips→GE→Canon。一次尝试，失败阻断后代，不重试或补步。worker复用既有训练循环和既定seen-domain final评价；controller汇总完整矩阵/Final/Old/Incoming/Forgetting与消耗。源摘要单独记账。

formal worker/controller先验证 TRAIN_READY源码身份和另行明确的M&Ms正式授权（version、datasets、seeds、209800预算），然后才创建输出。当前没有该正式授权。formal配置必须提供storage_root，实际NFS挂载及写读探针通过后输出必须位于其中；NAS上的cache/scratch显式传给worker。入口使用中性n.py和环境配置，不将私有路径写入可见参数；设置FP32/noAMP/noTF32与CuBLAS确定性配置。

## GPU工程范围

工程资格与正式根隔离，只访问my-gpu当前存储。预定成功资格110次Adam：两seed各7次有限绑定/恢复/三目标历史携带，seed163另96次两病例训练L拟合；全部单列，formal=0。短工程阶段不冒充完整53,300/25,100/26,500步轨迹。full-U图通过隔离虚拟phase cursor覆盖，不声称正式走过warmup/ramp。损坏history拒绝、恢复下一步、固定final采集不变性与历史传递均需实际通过。失败尝试与实际成本保留；不因工程失败改算法阈值或正式预算。
