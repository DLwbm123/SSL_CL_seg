# Codex 修补 prompt：F5_CONFIRMATION_V1 外部审阅 R1

本轮外部审阅对提交 `4ac9a27b52a9d381dce6bdfaf96bba23b17245f8` 的结论是 **CHANGES_REQUESTED**，不是 APPROVED_FOR_EXPERIMENTS。

请在 `codex/f5-confirmation-v1` 上完成以下有限修补，提交并推送新 commit 后停在 **STOP_AWAITING_EXTERNAL_CODE_REVIEW**。本段只授权代码、文档及原限额内 CPU 合成测试；不授权真实 source 张量读取、CUDA 资格测试、真实 smoke、正式训练或监测。不要发送/执行旧 Prompt B，不生成批准文书或用户启动确认。

## 一、保持科学协议不变

保持父方法 NATIVE_LR_SRC_A_3DOMAIN_V1、history-free、参数记忆、A-only 输入侧约束及当前冻结 B0_P1 / B2_C06 / F5_C02；不改网络、损失、梯度权限、EMA、Q/F、数据划分、随机流、学习率、候选或指标。

正式矩阵仍为 14 条轨迹 / 28 个目标阶段 / 74,200 次正式科学更新及正式物理调用上限；新增 source=0；真实 L-only smoke<=24；真实数据调用合计<=74,224；合成 CUDA 调用<=60。B0/F5 seed162 只导入，B2 seed162 新执行。P2 不生成执行节点。

结果锚点仍为 96ba0e29f601589a069bcbfdf50b2f4aafc1d883；原训练实现仍为 cc21871a43e3cd105cddaf831f5c3ea2fef59b9e。保护历史记录和当前用户工作区。

## 二、R1：区分元数据封存与模型完整性验收

问题：execution.stage_state 的 SEALED 路径仅检查 student.pt 存在且非空，并不检查 student_hash / transform_hash 或实际 payload；run 的 SEALED_SKIP 和 report 的 COMPLETE 又直接信任这个结果。

1. 可以保留一个严格只读元数据状态函数，但它不能单独代表“已验证模型完整”。另实现明确的生产模型完整性验收函数。当前只实现和做合成测试，不加载真实模型。
2. 在正式写出最终完成状态前、复用/跳过已封存模型时，必须有真实文件完整性检查。至少覆盖 identity、node_id、step、student tensor schema、有限性、student_hash、transform_hash，以及 transform 的形状/状态。
3. 计算值必须来自实际保存的文件，不只比较 receipt 内两个字符串。哈希/schema 都匹配后才可进入 VERIFIED/SEALED_SKIP；读取真实张量必须仍在用户授权的生产路径内。
4. 对28个新目标阶段执行同一标准，尤其不能漏掉没有后继训练读取它的最终阶段。不要因为第一阶段可能被 target_task 读到而省略其可审计证据。
5. 正式 model preflight 与 metadata-only report 分开。离线 report 不打开患者图像，不重新训练或评估患者；缺失/过期的模型验收证据应报告 PENDING_INTEGRITY，而不是 VERIFIED_COMPLETE。
6. 验收输出绑定 code/plan/node、实际文件身份与实测哈希。文件发生变化须让旧验收记录失效。若添加合成输入前向，单列成本并保全 RNG；不是额外患者评估。
7. 不以给伪模型文件补一个非空 hash 字段解决问题；合成故障测试要覆盖截断文件、可加载但错权重、错F、错身份、缺失hash、合法幂等跳过。保留 lost-tail 和账本断号拒绝测试。

## 三、R2：补齐真正的原生 CUDA 资格测试定义

问题：当前 qualify(cuda) 对每个方法仅连续 update 4 次，没有调用原生 native_state.resume、检查全模型合并等价或进入第二目标阶段。CPU merge/resume 测试用的是 SyntheticParentBridge，不能替代这些 native 路径。

1. 新增有限、显式列举的 native synthetic CUDA 用例，仍使用原 NativeLRParent / StageTrainer / native_state.resume / checkpoint 路径，不用玩具桥接代替待验证路径。
2. B0/B2/F5 至少覆盖 warmup 与活跃U、允许/禁止的梯度、F5额外clean-U路径、有限损失。确保有确定的非退化支持案例；全拒绝mask或零SWD不能被当成SWD路径通过。
3. 覆盖训练—保存—连续继续与同一检查点的原生恢复继续，比较最终学生、EMA、optimizer/scheduler、原型/support、数据游标和对应RNG状态。比较基于同一受控CUDA环境，而不是要求CPU与GPU逐bit一致。
4. 覆盖阶段出口合并前后的全模型合成输入预测一致性；构建真实第二阶段，核验自身前驱、合并参数/F继承、新adapter与零R、当前EMA/原型/优化器重置及新Q来源。无source训练、无旧teacher继承。
5. 覆盖失败调用计账与失效后禁止继续提交，不把资格fixture当科学训练结果。恢复对照分支的所有optimizer调用也计入合成CUDA累计预算。
6. 将用例ID、预期物理调用数、逐用例结果字段、资格专用选项写入plan；planned_calls必须<=60。可采用紧凑的连续/恢复/阶段切换方案，不需要扩大总上限。
7. require_qualification 必须检查完整用例集合、逐项PASS、对应计数与实际账本，不只看顶层PASS和12/24。修改资格计划时同步取消硬编码CUDA=12，但smoke仍为24，科学矩阵仍74200。
8. 冻结并记录实际目标运行环境指纹（Python/PyTorch/CUDA/cuDNN/必要依赖、设备类型和确定性设置），在资格、smoke、run之间核验；不能把仅code/plan相同当作运行环境相同。历史环境缺失项明确标UNVERIFIED，不伪造。
9. 当前只提交这些测试的代码、有限计划和CPU可验证部分；真实source、native CUDA和真实smoke仍写PENDING，不执行、不伪报PASS。

## 四、R3：补回直接调用target_task时遗漏的成本记录

原 NativeRunner.worker 在 target_task 外使用 NativeOperations，并记录每阶段CUDA峰值；新控制器直接调用target_task，遗漏这部分数据。

1. 复用已有非侵入式 NativeOperations 或经过核验的等价只读采集方式，并恢复每阶段 peak_cuda_allocated / peak_cuda_reserved。
2. 保留训练器已有telemetry和operation_counts；明确二者口径，不相加冒充独立调用数。对阶段训练、资格、smoke、完整性检查分别记成本。
3. 不增加诊断VJP、额外训练forward或改变随机流。新增哈希/合成输入验证独立计账。
4. 成本必须从首次执行累计；中断后恢复不能只保留最后一次会话的wall time/操作数。以attempt/session记录累计，保持科学步和物理调用分别核对。
5. report明确分开新执行和历史导入。历史source更新及历史seed162结果不能再次计为本轮新更新。不要把共享GPU下worker时间称作纯GPU计算时间或速度优势。
6. 追加合成的采集开关不改变状态/RNG、计数对应调用、恢复后成本不丢失的检查。不能用NA代替本轮本可采集的显存/操作指标。

## 五、R4：实现原先约定的报告交付与完成状态

PUBLIC_RESULTS.json可保留为规范化中间产物，但还必须实际生成：
- FINAL_REPORT.md；
- FINAL_METRICS.csv：三方法×三个seed×两顺序=18条最终轨迹，显式historical_import/new_execution；
- STAGE_METRICS.csv：逐阶段、逐已见域的rim/cup/disc_union/macro_Dice与来源标记；
- PAIRED_COMPARISONS.csv：逐seed/逐顺序、seed内双顺序平均、主要163/164与补充162–164分开；
- COST_AND_COMPLETION.json：28新阶段、8导入阶段、3复用source、真实/合成/失败/恢复成本和模型完整性验收。

沿用现有Final/Old/Incoming/Forget与G1–G3定义，不重新选择checkpoint、不提前判断科研门槛。两顺序不是两个独立seed。

只在28阶段完成、正式74200、smoke/资格通过、模型完整性核验通过且要求文件全部生成并通过schema检查后，写：
COMPLETE_P1_AWAITING_SCIENTIFIC_REVIEW

若要区分执行完成和公开发布完成，使用不同状态。不要把仅PUBLIC_RESULTS.json存在写成“已公开交付”；公开提交/推送及可读性要有真实证据。没有发布权限时说明LOCAL_REPORT_READY，不自动扩大授权。

元数据report在未完成矩阵或验收缺失时返回清晰PENDING，不启动训练、source验证、患者评估或新实验。保持正负结果与逐顺序退化可见。

使用纯合成汇总记录测试18条最终轨迹、导入标记、文件schema、指标/门槛一致性和不完整证据拒绝；不能仅用伪student.pt把整个完整性验收测成PASS。

## 六、提交前检查与交付

保持原CPU累计账本：当前已记录15/40次合成optimizer调用、3/8次调用尝试。不得清空或将失败尝试挪出成本；新CPU检查限于剩余额度，超限前停止并报告。标准库元数据fixture不得计为实际optimizer执行。

尽量只修改f5_confirmation_v1及其新文档/测试。不改五框架原训练数学；若原生测试暴露共享实现的问题，单独列出并停止，不能悄悄修复再宣称复验相同代码。

更新PLAN、CODE_MANIFEST、TEST_REPORT、RUNBOOK，并保留原始审阅输入和历史测试尝试。科学字段不变；资格用例/验收schema变更要显式列diff，生成新哈希等待重审。

增加REVIEW_RESPONSE_R1.md，逐项列R1–R4的修复位置、测试证据与尚PENDING的生产验收。不得把本次CHANGES_REQUESTED改写成APPROVED，也不得生成批准或启动确认。

提交并推送后报告：完整新SHA、修改文件、CPU新增/累计实际调用、每条问题对应证据、source/CUDA/smoke仍PENDING、真实数据optimizer更新0，并停在STOP_AWAITING_EXTERNAL_CODE_REVIEW。不要恢复监测或启动任何正式节点。
