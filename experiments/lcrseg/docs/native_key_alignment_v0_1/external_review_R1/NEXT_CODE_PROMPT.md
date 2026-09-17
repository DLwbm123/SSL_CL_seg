请基于 9915168cc3707fb70afdc2c719c8781cdf76e3fb 处理 NATIVE_KEY_ALIGNMENT_V0_1 外部代码审阅，并完成下一步生产接入代码。

本轮结论是 CHANGES_REQUESTED：核心方法组件可保留，但尚未批准 D1 运行。生产接入未完成是原交付明确的边界，不需要重做方法。不要生成 APPROVED_FOR_EXPERIMENTS、复用 F5 批准或伪造用户启动确认。

我转发本段仅授权代码、元数据读取及下述有限 CPU 合成验证；不授权读取真实模型张量或患者 payload，不授权 CUDA、真实 smoke、正式训练和监测。

一、唯一已复现的必修缺口

修复 protocol.py::validate_plan（被审版本第71–86行）。它当前只验证内部哈希和总数；重算哈希后，更改 lambda_U、交换 O1/O2 末域、跨臂挪一步、修改 C3 声明坐标/权重以及错误前缀仍能通过 plan 校验。

1. 把 canonical plan 构造与文件写入分离。独立绑定不可默改的 B2_C06 全量options、四份历史前缀证据和其结果/运行锚点内容哈希。
2. 每个节点逐字段核验 id、arm、seed、order、stage、domain、updates、prefix和options；O1末域必须Drishti_GS/2100，O2末域必须RIM_ONE_r3/3200，updates必须等于total_steps，不能用总和抵消单节点错误。
3. 四臂定义、层、144/8维、.05辅助权重、D/8、PAS .6/.7、B2 lambda_U=1、warmup/ramp、诊断步、分析门槛及无D2/D3都必须与canonical语义一致。
4. 不仅核对plan文件，还在生产payload读取前检查实际解析options、trainer类型/arm/loss/basis绑定；声明与代码不能分离。
5. 添加上述重算哈希反例及改PAS、改底层证据等零optimizer回归测试。单独validate_prefix已有的跨seed拒绝保留，不把它误写成完全缺失的保护。
6. 当前冻结计划没有被证实遭修改；这是完善验证器，不得把反例当成实际训练事故。

二、科学设计全部不变

NATIVE_LR_SRC_A_3DOMAIN_V1 + B2_C06；A-only保护V不变，不增加B侧约束。
C0 no-op，C1完整144维patch，C2随机8维辅助键，C3真实入口V的8维辅助键。
层仍decoder.dec1.merge.block.3输入；teacher/V detach，学生上游梯度保留。
loss仍 Lsup + ramp*(KL_B2 + .05*(D/8)*SWD)。
不改变原生A/B的KL权限，不加入R/F、MLP、历史数据/特征库或新优化器规则。
不调系数、PAS、层、rank、seed或mask定义。

D1仍为163/164×O1/O2×四臂，16个第二目标阶段、42,400正式科学与物理optimizer调用上限。
新增source和第一目标训练均为0。C3唯一预指定主候选，D2/D3不展开、不自动运行。

三、生产接入代码：实现但本轮保持关闭

A. 独立新研究权限、有限控制器、运行身份和真实当前域provider。
不得简单删除CPU/SyntheticCurrentDomain保护后用旧F5能力开跑；正式路径要求本研究的真实外部批准和独立用户启动确认，绑定最终提交、完整代码树、科学plan和prefix证据。
当前代码任务不生成这些批准。

B. 四个共同B2前缀的窄权限适配。
保留历史receipt原identity及hash；native基础recipe仍为B2，新实验study/arm是独立字段。不能把旧receipt的family改成C3或全面放松旧require_predecessor。
前缀真实文件/student/F hash与schema验收实现为待运行路径；四臂均从同一合法前缀独立新建adapter、EMA、原型、optimizer和warmup，不共享这些可变状态。
缺失或不匹配停止，不补训前缀。

C. 新协议checkpoint/resume。
恢复必须构造KeyAlignmentTrainer，不能调用返回原StageTrainer的旧恢复函数后悄悄继续纯B2。
在完整原生状态之外保存/核验study、arm、完整options、layer、prefix绑定、辅助basis及hash、采样namespace、alignment_cost和已提交诊断记录。
C2精确恢复随机basis，不能重抽；C3恢复同一入口V；跨臂/跨prefix/改loss恢复必须拒绝。
不换类monkeypatch，不把权重加载测试冒充训练恢复。

D. 成本与诊断。
复用已审物理账本、单次optimizer语义、实际模型完整性验收、操作与显存计数、封存跳过/失败停止等工具；不调用旧F5有限DAG或旧授权。
正式42,400、未来synthetic CUDA资格、未来真实smoke、当前CPU追加测试分别记账。
先冻结有限CUDA/真实smoke用例与明确预算，仅提交代码，不执行。
固定25/50/75/100%诊断必须持久化；不能只留会被下一步覆盖的last，不按诊断在线调权。
任何失去已提交状态的tail不得默默重放或超预算；保留失败尝试。

E. D1报告代码。
所有16阶段结束后统一计算C3-C0/C3-C2/C3-C1；先seed/order配对，再seed内平均顺序，最后跨seed。
单列每域rim/cup与第一目标刚学完/最终成绩；共同前缀下验证DeltaForget=-DeltaOld，不计作两份独立证据。
标注开发seed和共同前缀诊断，不能写成完整新方法或独立患者验证。
报告正负结果、样本支持、固定诊断和全部成本；不自动D2/D3。

四、CPU新增验证的单列预算

旧D0账本已30/32次、2/2尝试，完全保留，不删除、不重置，也不要直接第三次调用旧有限suite。
本段另行授权一个可追溯的 CPU_INTEGRATION_R1 补充：最多48次生成数据optimizer调用、最多2次suite尝试。每次开跑前先列出用例和计划调用数，单次最多24。旧30加新预算合计最多78次；失败调用也计入。
这是新增、显式披露的代码验证补充，不是改写旧上限。元数据/索引/梯度-only测试列明0optimizer，额外VJP单列。

至少覆盖：
- canonical plan重算哈希反例全部拒绝，实际options与对象绑定一致；
- 原B2与C0及C3-zero的关闭等价；
- C2/C3有限连续与恢复继续的model/EMA/optimizer/prototype/RNG/basis一致；
- 错arm、prefix、basis或lambda的恢复拒绝；
- 16节点/42,400预算、窄前缀权限、越界phase拒绝；
- 合成损坏模型/失效证明、失败账本、不完整诊断、报告样例；
- 合成测试不创建或伪造生产批准，不访问患者数据/真实权重。

若两次或总预算不足，保留记录停下报告，不自行扩大上限。无需修改科学配方来让合成测试通过。

五、交付

保留原方法组件和旧F5全部文件。尽量在独立新包接入，不重写共享网络、B2数学核或数据语义。
更新CODE_DIFF、CODE_MANIFEST、D1_MATRIX（或将科学子计划与执行元数据明确分层）、冻结options、PREFIX_BINDINGS、测试/成本、RUNBOOK及REVIEW_RESPONSE_R1。
提供科学字段未变的diff，具体说明新哈希仅来自校验/执行/资格定义等必要变化。
真实prefix张量、CUDA、真实smoke均标PENDING，不能借CPU PASS伪报通过。

推送新提交，给出完整SHA、逐项修复证据、原始及补充CPU累计成本、实际命令与仍待运行检查。
停在STOP_AWAITING_EXTERNAL_CODE_REVIEW，不执行CUDA、真实smoke、D1或监测。
