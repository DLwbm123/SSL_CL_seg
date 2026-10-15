# V2 工程交付：冻结，未启动训练

2026-10-15 00:08（Asia/Shanghai）。状态 `PASS_WITH_DECLARED_NOT_RUN` / `FROZEN_NOT_LAUNCHED`。当前主实验配置、代码快照和预算未变。

已实现：G及累计F谱界、Adam实际位移保护、有界EMA/seal；MRI目标首层冻结；训练L的域/类原型和native/F敏感度记忆；与EMA/PAS接受独立的入口保护及训练L可靠性；原子V2 checkpoint、跨域记忆携带、协议/源码/依赖冻结、未来启动授权和隔离路径检查。复用现有StageTrainer与worker编排，无新训练循环。

## 正结果

- 最终16项CPU生成输入检查全部通过，覆盖C2/C3/C4实际锁定native架构、完整U图和256/384 L几何、MRI首层optimizer排除、A/B/R图、谱退化/累计约束、实际提案内积、EMA/seal、历史记忆、恢复/篡改拒绝、原子写入故障及启动/隔离路径拒绝。
- 6个真实已完成V1 source checkpoint均可只读加载；在生成输入上的V2零增量target入口与source logits最大差0、argmax相同。包含曾有rank问题的Prostate164，无源模型重训。
- 每次尝试工程/正式optimizer调用均0，实际患者影像读取0、独立test访问0、V2正式训练启动0。测试将Adam/SGD step拦截；固定合成提案和虚构phase cursor只是图/guard用例。
- qualification与本地protocol、implementation、直接继承依赖三个digest完全一致。旧V1/five_frameworks/lctx源码无工作树改动；当前两个M&Ms PID+birth匹配并继续增长。V1 Prostate12节点177400更新完成回执已核对，已告知用户。
- 名义全链511600；健康源复用130400、新目标预算381200，额外source记忆bootstrap L曝光单列。现在没有执行这381200次新调用或32批真实bootstrap。

## 负记录与修复

| 尝试 | 结果 | 原因/处理 |
|---|---|---|
| 01 | 8 PASS / 3 FAIL | 错误要求B=0入口A梯度非零；固定M&Ms测试cursor未过warmup。使用固定合成提案覆盖非零A图，按每域预算选择测试phase。 |
| 02 | 11 PASS | 修正初始用例。 |
| 03 | 14 PASS | 增加实际trainer提案hook、full-size geometry、history/F篡改和写入故障检查。 |
| 04 | 15 PASS | 增加冻结源码变化拒绝。 |
| 05 | 15 PASS / 1 FAIL | 负路径测试本身已有V2工程祖先，不是外部路径；改成不创建的外部路径。 |
| 06 | 16 PASS，真实源导入FAIL | 前3源对照通过，旧满秩依赖阻断Prostate164。停止该检查，保留全部结果；修复依赖绑定，不重训源、不调整SVD容差或rank。 |
| 07 | 16 PASS + 6源PASS | 独立快照加入已提交c234bcd retained-rank支持，冻结三组继承源码。 |

另保留4个setup负项：本地Python无Torch；远程系统Python的Torch namespace不可用；默认copytree在NFS复制元数据报Errno5（改为源码字节复制）；setup失败后的attempt02重定向目录不存在（Python未启动）。正式主实验没有因这些检查失败而重启/重试。JUnit、完整测试日志、各尝试源码及可用setup错误摘要保留在私有工程根。

## 未执行与风险

未执行GPU实际optimizer更新、真实source训练L的32批bootstrap、迭代两病例overfit、真实验证指标比较、ablation或test。用户要求先不训练，所以未将overfit标记PASS。实际Adam moment轨迹未新实测，恢复沿用原引擎；当前检查包括状态一致性和新记忆/guard字段，不能冒充训练恢复资格。

谱界与一阶位移保护不能保证旧域Dice；摘要近似、entry新域错误、MRI首层冻结和过强保护可能损害Incoming或吞吐。没有证据声称V2提升指标。冻结后不能通过半小时监测自动改阈值、预算或启动训练；新的方法设计变更需要另行记录新协议。

半小时监测继续跟踪V1与眼底/Prostate改进工程，正常安静、实质变化通知。完整参数和边界见PREREG/PROTOCOL，完整匿名字段见ENGINEERING_REPORT.json；私有权重、生成tensor checkpoint、原始日志、患者材料未放入Git。按项目禁止自动push的规则，本次工程源码仅作本地提交；正式实验完成的公开交付规则保留。
