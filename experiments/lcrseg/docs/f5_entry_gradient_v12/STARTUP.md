# V12资格通过并正式后台启动

冻结执行 `2a64df4441f7725d62804973c274aacd89fb1ce9`；NAS根 `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/protocols/f5_entry_gradient_20261011T040723Z`。继承用户明确自主科学改进授权，V11完整正负结果已先公开验证[f864bdf](https://github.com/DLwbm123/SSL_CL_seg/blob/f864bdf0a3feee6b534644a9fa7ed0153c1feb19/experiments/lcrseg/docs/f5_anchor_restore_v11/COMPLETION.md)。本轮为独立create-only有限协议。

V11 Final67.5628%，相对V10+0.6075pp、F5+2.1518pp、历史B2−0.3383pp，F5门槛通过、B2门槛失败。164 Old平均较V10改善0.9628pp，但仍比B2低4.7100pp；164/order2 Drishti cup最终47.9215%，较V10下降5.0065pp、阶段1→2下降27.7202pp。均值改善并未解决该旧目标域遗忘，不能把入口目标正确性或伪标签错误当成已证实因果。

V12回到V10正向底座，保留原KL冲突拒绝，不叠加V11软目标恢复。唯一模块在原A/B/R合并L+U梯度与当前U入口知识参考梯度冲突时投影到参考半空间。借鉴[A-GEM Eq10/11](https://arxiv.org/html/1812.00420v2)的投影形式；参考来自当前U冻结入口teacher，未使用论文的历史episodic memory，因此不是完整A-GEM复现。原lambdaU0.25、SWD0.2、PAS0.7/0.5、硬输入投影、日程/原型/Adam/数据均不变。完整公式与数据边界见[PREREG.md](PREREG.md)。

资格PASS/QUALIFICATION_EXIT0，实际32CPU/10生成CUDA/2真实当前L，包括故意失败优化调用；另5资格参考VJP、3生成梯度VJP、1生成loss调用、2直接anchor前向，无额外患者探针。disabled与归档V10完整native状态/原loss路径一致；启用的实际A/B/R合并冲突梯度投影、原主KL拒绝与非零参考、teacher冻结共享、线性R EMA、SWD R-only、真实checkpoint续训及错误flag拒绝、部署/own stage2和zeroU当前L smoke均通过。

正式控制器已启动，一次启动核对164物理调用、0封存、无EXIT；joint_kl/anchor_guard/entry_gradient/SWD parent detach均true、lambdaU0.25，neutral父子argv/GPU名称PASS。启动前GPU4–7各24124MiB可用，未改系统或终止其他进程。仍处工程启动阶段，无V12性能结果。

四轨迹两seed163/164两order，8目标阶段21200正式/source0；先S163/O1两阶段工程成功后其余3，不按效果剪枝，全8封存/4EXIT0后统一20已见域评价。额外正式参考VJP上限16960，沿用固定R诊断VJP上限64；复用原V10anchor完整前向，不新增正式完整前向。参考不作为额外训练loss或新系数。跨轮正式启动核对实际297637，上限318673；资格累计768CPU/181生成CUDA/38真实L。历史负结果、失败与673回滚重复/丢失及5条NA保留。

只约束原始梯度的一阶参考方向；Adam动量、权重衰减、视图差异及当前U覆盖不足仍可能造成旧域退化，不能宣称实际更新或旧域保证。固定双成功门槛同时vs匹配F5和历史B2，不放宽；完整paired V12−V10和背景V12−V11、B0、全部负细胞及实际计算成本均报告。开发患者反复使用，不称独立泛化/SOTA/临床获益。

代码、协议、匿名资格/启动证据公开，患者/逐患者指标/权重/private配置/原始log留NAS。小时f5继续ACTIVE，RL保持PAUSED；仅原双门槛通过并完整公开交付验证后停止。
