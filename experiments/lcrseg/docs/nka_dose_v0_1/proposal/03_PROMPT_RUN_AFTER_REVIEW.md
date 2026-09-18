# Prompt B — 仅在新代码通过真实外部审阅后发送

请执行已经完成代码审阅的 NKA_DOSE_V0_1。转发本段表示我授权下述限定生产范围，不授权改方法、扩矩阵或后续研究。
本文件本身不是外部审批。必须另有新研究真实审批文件；旧NKA/F5批准不能使用。

1. 启动前校验
读取真实外部APPROVED_FOR_EXPERIMENTS、review evidence、被批准的完整SHA及代码/计划/前缀/历史导入/环境/执行定义hash，确认批准只含本轮16节点dose阶段与明确资格预算。
与实际干净checkout和已审CPU证据一致才执行。若没有新审批或代码变化，保持STOP_AWAITING_EXTERNAL_CODE_REVIEW；不自己签发审批。
审批/本次用户确认/私有config在checkout之外保存，用户确认绑定真实review digest。不要为保存批准改HEAD，不重跑旧/新CPU suite或prepare以“更新状态”。
沿用已知NAS wrapper/受管资源环境，不升级依赖，不使用python -O，不重写HOME/CODEX_HOME。旧基线环境复用条件不满足时停止，不偷偷增跑C0。

2. 先资格再正式
先执行已审原生合成CUDA资格40次：四个new dose cells连续/恢复/失败24次，旧/新低剂量计量开关对照12次，生成参数Adam oracle4次。四个预设failure调用计入24；其他异常不是获准重试。
CUDA不得读取真实前缀；0.05/0只出现在生成等价测试，不进正式节点。
通过后验收规定B2前缀，进行新四cells各8次、共32次当前L-only smoke，正式warmup语义，禁止U/val/test，状态全部丢弃。
每个其余前缀进入正式前验收实际identity/schema/finite/file+student+F hash；缺失/不匹配停止，不补训。
计量实现必须通过不扰动验证及Adam oracle，不能为通过换正式optimizer或放宽容差。
正常通过以上门槛后直接完成本轮16个正式节点，不需要每节点重新请求同一范围确认。

3. 唯一正式矩阵
C2/C3 × lambda_align=.5/2.0 × seed163/164 × O1/O2，仅stage2。
O1 Drishti2100；O2 RIM3200；共42400正式科学/物理更新。
每个node从自身指定原B2 stage1前缀全新初始化，不能用低剂量/其他arm终点或warmup。
新source/stage1训练0；旧C0四条、旧C2/C3 .05各四条共12条只导入，标historical_import，不重训。
不运行高剂量C1、新seed、新顺序、外部数据、D2/D3。保持完整科学options、随机流和剂量绑定，不据中途结果调参。

4. 记录与停止
正式每节点4个更新前诊断，共64点：额外4VJP/点共256，两个只读Adam候选/点共128，无额外真实optimizer.step。清洁U前向及读只公式/有效W计算算子与时间单列。
原梯度更新的计算顺序不被诊断代替；实际参数、Adam m/v、EMA、随机流不被计量修改。
记录支持、原始和加权损失、全量/上游/分层A-B梯度、rho_theta/rho_W分子分母/无定义原因、实际lr及预测与唯一真实步误差。历史没有的Adam诊断标NOT_MEASURED_HISTORICAL。
真实总更新<=42432，合成CUDA40、旧CPU64和本轮CPU单列。失败尝试必须计费；未闭合成本、错身份、丢失tail、OOM/非有限值/公式明显不匹配时工程停止保留记录，不自动改代码重试或扩大预算。
科学负结果不是工程故障，16节点均须按计划完成后统一分析。没有类别支持的合法graph-connected零如实记录，不降低PAS凑支持。

5. 交付
生成28行FINAL_METRICS、84行DOMAIN_METRICS、63行主PAIRED、64点新的ADAM_COUNTERFACTUAL，以及FINAL_REPORT/GRADIENT_DIAGNOSTICS/COST_AND_COMPLETION/PUBLIC_RESULTS/验收证明。
分别报告每固定lambda的G_perf/G_key，不按seed/order择优；两剂量都满足则选较小0.5。开发门槛不是显著性。不能外推独立患者、完整新方法轨迹、原始KI复现或SOTA，也不能声称同剂量优于未运行的C1。
所有执行/资格/完整性/成本/schema通过后停在COMPLETE_DOSE_AWAITING_SCIENTIFIC_REVIEW。
本段授权另开独立结果发布位置上传聚合结果与复现元数据，并实际核验公开可读；不改变被审运行checkout。禁止公开权重、患者/逐样本记录、private_val、图像/标签、私有路径或凭据。未完成发布不得声称已公开。

最终报告真实commit、各资格与前缀结果、16/16及分口径科学/物理/失败成本、每lambda配对差、G_perf/G_key及发布入口。即使通过也不自动进入完整轨迹/D2/D3；未过不再搜索更大lambda。关闭有限控制器，不新增监测或后台轮询。
