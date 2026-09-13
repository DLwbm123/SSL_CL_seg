# 给Codex｜第一轮：实现全部五框架，推送后停在代码审阅点

你在 `DLwbm123/SSL_CL_seg` 项目工作。请一次性完成五个候选框架的统一代码实现、合成测试与可执行实验计划，推送独立GitHub分支，交给外部ChatGPT审阅。**当前任务不是启动实验。**

## 0. 先读取

本交付目录中的：
1. README.md。
2. docs/01_IMPLEMENTATION_SPEC.md。
3. docs/02_EXPERIMENT_PLAN.md。
4. docs/03_TEST_AND_REVIEW_CHECKLIST.md。
5. docs/04_DESIGN_DECISIONS_AND_SOURCES.md。
6. docs/05_OPERATOR_HANDOFF.md。
7. configs/study_plan.json、operator_inputs.template.json、review_approval.template.json。
8. reference_impl/ 及其中测试（它们是参考，不是已经集成的生产训练器）。

阅读真实仓库AGENTS和相关父实现。用户上传综述在sources目录，不要改写其事实；本包框架是我们设计的改造，不宣称原论文已验证该组合。

## 1. 当前授权范围

允许：源码/配置/manifest元数据读取、文献/作者源码核对、CPU合成测试、本轮配置与DAG计划展开、独立分支创建与push。

禁止：真实图像/标签payload、hidden-U-GT/test、真实checkpoint tensor读取、真实smoke、正式训练、服务器GPU排队/作业提交、main合并、改写旧结果/终态、杀他人进程。默认所有run入口保持CODE_ONLY。

可以验证文件存在/权限等元数据，但不能把读取完整checkpoint伪装成元数据审计。CPU toy tensor不是真实模型tensor。不要创建automations/后台监控。

## 2. 父实现绑定与缺项处理

优先从用户给的一个原始启动命令/run_id/源码路径或已有workspace配置绑定真实单侧隔离父方法，列清ΔW乘法顺序、A/B、rank、约束、冻结集合、阶段转移、数据权限。旧参考commit `8a6e93a8bc39554c878cffe0dedf7fbfefe7d16d` 仅为代码参考，不能自动当作原始KI。

历史成绩缺失不阻止写代码，不要索要SOTA表。原父身份不明确时，继续实现五框架独立模块、正式接口、合成ParentBridge和审阅文件；真实ParentBridge标 `PARENT_BINDING_REQUIRED`。不要无限考古、不要换成F_CONV或后来的SVD模块冒充原KI。最后只报告最少一条待补线索，不因局部缺项放弃整个任务。

## 3. 代码交付

创建独立worktree/branch，建议 `codex/sslcl-five-frameworks-v1-review`。真实base commit以实际可用工作树与父源码为准，记录差异；保护用户未提交修改，不用reset --hard清掉其他工作。

按Implementation Spec建立统一ParentBridge、model、losses、reliability、gradients、checkpoint、planner/controller/evaluator/analyze与registry。不得只交五个配置或伪代码。每个F1–F5都需真实forward/backward实现；缺外部依赖用显式异常和状态，不许loss返回0假装完成。

- F1：classifier前 `F_prev(I+QRQᵀ)h`；PAS-KL+JML U只直接更新R。
- F2：当前L结构梯度选父合法A方向；CWMI结构项不重复CE；不重置旧adapter；U只直接更新B。
- F3：同前向，在低维残差上施加逐位置各向异性custom backward；明确它不是普通标量loss导数。
- F4：真实D-Convexity结构后端；冻结teacher/readout，有限步只优化当前d，原q PAS集合不变，target全detach。
- F5：类别条件经验SWD，当前L teacher作为参照，extra clean-U student特征路径与计算明确。

包含五基线与未来消融注册；不叠加五框架。公共L/U输入、父预算、随机key和损失归约保持统一。

特别核验Q坐标、F矩阵乘法顺序、实际3×3读出/插值、阶段EMA和合并。新F是明确增加的d²参数，不假装原方法已有；训练/部署内存和capacity控制必须给出。

不要使用production monkeypatch、import *、更改旧函数全局行为。若为兼容不得不修旧代码，最小diff、单独报告并加原回归，不能改方法科学含义。

## 4. 文献后端与依赖

固定JML、CWMI、D-Convexity的目标与文件版本，登记author repo/commit/file sha/license/API/改造点。UnCL与SWSEG只借鉴本规格指定部分，不导入整套FeCL/uniformity模块。GOLD原来的历史prototype/AGOP库不引入。

CWMI不能换成FFT L2，D-Convexity不能换成TV，JML不能换成softDice，SWD不能换成均值MSE。论文与作者实现不一致时在SOURCE_MAP报告并固定所采用的本轮定义，不静默混合。

如果作者源码没有可用再分发许可，不把整个第三方repo镜像推到公开仓库；锁定可获得的本地依赖或明确独立实现来源与公式测试。

## 5. 测试与实验计划

运行CPU合成单元+集成测试，覆盖Checklist全部关键项，包括5个实际call graph、梯度权限、empty/ignore、source collection、当前状态生命周期、完整模型合并、故障恢复和运行锁。

参考包的本地测试不是你的实际集成测试。不要复制本包passed数量冒充自己的结果。记录你实际环境/测试命令/通过失败skip与未执行CUDA情况。

用机器可读study_plan生成父4配置、基线各8配置、框架各8配置及C/D全DAG。C0/C候选胜出属于未来事件，计划里保留明确待解析节点，不填假SHA或分数。预算从真实父步数绑定，source另计，F2/F4 VJP与F5额外前向单列。

禁止启动所有真实任务来“验证计划”。CPU合成toy可以跑统一controller的小矩阵，以证明每框架可执行、恢复与失败记录存在；不能称正式smoke或真实实验完成。

## 6. 生成外部审阅包

按Checklist写全review/目录：REVIEW_INDEX、PARENT_BINDING、坐标/梯度/数据/状态契约、SOURCE_MAP、DEPENDENCY_LOCK、FRAMEWORK_COMPLETENESS、参数清单、TEST_REPORT、实际日志摘要、RESOLVED_PROTOCOL、完整候选/DAG、预算、REVIEW_LOCK和CODE_MANIFEST。

REVIEW_LOCK保持未批准；review_approval模板is_template=true。所有CLI run在第一次真实数据或checkpoint打开前拒绝模板/空hash/不匹配代码/越界phase。Codex不能自填外部审批。

私有运行路径、患者ID、密钥、原图、标签、checkpoint和逐患者指标不push。PUBLIC审阅包使用sanitized路径；实际私有绑定存授权位置。

## 7. GitHub推送与停止

提交源代码、测试、公开小配置和审阅包，push独立分支。可创建draft PR但不merge。核验remote分支头SHA等于本地提交；push失败如实报告，不给不存在链接。不得声称本地commit已经在GitHub。

结束状态必须是：

`STOP_AWAITING_EXTERNAL_CODE_REVIEW`

最终回复以下真实信息：repo、branch、base_commit、review_commit完整SHA、真实PR或NOT_CREATED、review_index路径、五框架完成表、parent绑定状态、实际测试结果（CPU/CUDA分开）、新增文件数、明确未完成项、push核验。

并报告这轮真实图像读取、真实标签读取、真实checkpoint tensor读取、真实smoke、formal optimizer updates均为0；若实际不是0必须如实披露，不编零。

**push后立即停止，不以测试绿色或文档齐全为由开始实验。用户会把该SHA带回外部ChatGPT审阅；只有独立批准且用户确认启动，才执行Prompt02。**
