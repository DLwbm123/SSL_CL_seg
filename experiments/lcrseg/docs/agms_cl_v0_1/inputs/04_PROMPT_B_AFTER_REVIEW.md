# Prompt B：新代码审阅通过后才发送

本文件当前不是启动授权。必须先完成Prompt A、提交新完整SHA，并取得对此SHA的真实外部审阅通过；不能借用旧NKA/DOSE的APPROVED。用户在该条件满足后转发下面正文，才表示本范围启动确认。

---

请执行已通过外部代码审阅的 AGMS_CL_V0_1 P0/P1。
我现在转发本段，授权已审版本的合成CUDA资格、真实前缀验收、当前L机会审计、L-only smoke、固定P1及相应聚合报告发布。不授权其他实验。

## 启动绑定

使用本轮真正的REVIEW_APPROVAL和完整审阅证据，核对其reviewed commit、code tree、science/execution/prefix/import/environment digests。
实际checkout必须为该完整SHA且干净。审阅、用户确认与私有配置放在checkout之外，不为了记录批准再产生新代码commit。

另行记录我真实转发本段的用户launch confirmation及审阅canonical digest；不得自行创造外部批准，不复用旧许可。新review未通过、绑定不符或环境变化时停止。

不要再跑已耗尽的CPU suite，不清空旧134或本轮新CPU账本，不运行prepare来刷新状态。

## 顺序与范围

1. 原生合成CUDA资格：按已审定义36次，其中两个是预设after_optimizer失败注入；全部计账。验证原B2/A0关闭等价、所有六种模型分支、活跃父类/多尺度学习、连续/恢复及部署丢弃等价。不得读真实前缀或患者数据。
2. 真实前缀验收：只允许B2_C06 seed163/O1与O2的原第一目标模型；核对实际文件/身份/schema/finite/student及F hash。缺失或不符不补训。
3. P0：各前缀最多16张当前域L，合计最多32图像的只读机会审计，0optimizer；不读取hidden U-GT/val/test，不调阈值、不择图。
4. L-only smoke：A0-A5各4步，共24，正式warmup语义；不读取U/val/test，所有状态丢弃。
5. P1：仅A1-A5 × seed163 × O1/O2的10个新stage2节点。O1 Drishti2100，O2 RIM3200，共26500次正式科学和物理调用。每臂从对应原B2前缀全新独立初始化；新增source/第一目标更新为0。

A0两条B2终点仅历史导入；不新增A0训练、高剂量C1、NKA/SWD、seed164、SCNP或任何完整轨迹。
本轮真实optimizer总上限26524；CUDA36、新CPU、P0/评估成本另列。
正常通过各门槛后继续完成同一冻结P1，无需反复申请已授权范围。

## 科学语义

保持已审完整方法规格：B2细KL规则不变，M两个训练期头仅L监督，H只回收细集合外可信disc父类；A4等权与A5当前L风险权重的区别清楚。风险滞后一成功步并在阶段内使用，不能读val/U标签；teacher/q/mask/R/alpha无梯度。

原A-only保护、原主读出、学习率、数据/随机流、阈值和lambda均固定。不按中途Dice更改权重、risk温度、接收比例或训练步数。
辅助无支持按合法0记录，不伪造样本支持。负结果不提前剪枝。若用户再次缩减，登记独立amendment和门槛NOT_ASSESSED，不改原计划历史。

## 恢复与验收

每次optimizer调用包括失败都记账；原模型、teacher heads、R、优化器、prototypes、cursor、RNG、诊断/成本的状态必须精确恢复。跨arm/prefix拒绝，不重新初始化当前阶段head/R冒充resume。
封存节点只能合法验收后skip，不能覆盖。真实模型验收与metadata_sealed分开。
丢失tail、非预设CUDA失败、OOM、非有限值、环境/前缀/权限/账本不符时保存证据并ENGINEERING_STOP；不在线修改已审代码、放宽容差、借smoke预算或自动重跑。
只保留原主网络作为部署学生；严禁报多尺度融合推理为单学生。

## 结果与停止

完成10/10后导出：
P0_REPORT、FINAL_REPORT、12行FINAL_METRICS、36行DOMAIN_METRICS、27行PAIRED_COMPARISONS、40点GRADIENT_DIAGNOSTICS、覆盖/风险汇总、COST_AND_COMPLETION、PUBLIC_RESULTS及实际模型完整性证明。

A0两历史+10新执行明确标记。只比较seed163同order的配对，再平均两order；不拿跨seed旧均值代替基线。不声称两order是两个独立seed，DeltaForget=-DeltaOld不重复计证据。

按冻结P_perf/P_joint逐项判断，不声称TMI标准或显著性。即使全过也不自动进入seed164、结构模块或完整轨迹。

在独立结果发布checkout中只发布聚合和必要复现资料，并核验公开可读。禁止上传权重、患者/逐样本结果、图像标签、private_val、凭据/私有路径；本地完成和公开发布分开记录。

最终停止在COMPLETE_AGMS_P1_AWAITING_SCIENTIFIC_REVIEW。
报告完整执行SHA、36CUDA/24smoke/P0/10阶段情况、科学与物理更新、失败成本、A5-A0及所有简单对照差值与门槛、公开入口。
不恢复监测、不追加节点、不运行P1R/P2/P3。
