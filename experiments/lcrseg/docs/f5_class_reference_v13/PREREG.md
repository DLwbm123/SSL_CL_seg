# F5_CLASS_REFERENCE_V13 — 入口参考KL的类别均衡模块

用户授权未达标时自主方法模块改进，明确V10正向底座、不只调系数、小时监测至原双门槛。V12完整正负结果已先公开验证4501d5ce2967f24759a90af898611fc4df8ddb17，本轮独立create-only有限协议，不改V12旧配置/权重/台账。

## 有限证据和一个主改动
V12 Final67.3290%、vsV10+0.3737pp、F5+1.9181pp、历史B2−0.5721pp，两原门槛均失败。164 Old vsV10+3.5617pp、Drishti cup57.9907%（V10+5.0627pp），但163 Old−3.0061pp、163/order2 Drishti cup42.4088%（V10−9.3613pp）。16960参考VJP、8062次投影，合并梯度保护真实生效而收益跨seed不一致。原参考对所有支持像素取均值，类权重数学上等于像素支持占比；V12未记录类支持，不能声称背景实测主导或cup损害由此造成。假设：均衡参考的类权重可能减少小类在整体保护方向中的弱化；也可能放大少量错误入口预测而恶化适应。

保留V10正向底座全部主KL高置信冲突拒绝/kept均值、joint KL->A/B/R、clean SWD->R-only和线性R；原lambdaU0.25/SWD0.2/PAS0.7,0.5/Qk8D16/硬输入投影/Adam/lr/日程/原型/数据不变。沿用V12的全A/B/R合并L+U半空间投影，仅将参考r的像素均值替换为present-class均衡：

原支持m=原PASvalid & (入口maxprob>=原0.7)，c=argmax(q_entry)全部stop-gradient。对每个当前batch支持中出现的c∈{background0,rim1,cup2}，r_c=mean_{m且入口类别c} KL(q_entry||p_UL)；r=mean_{出现的类别} r_c，缺失类不补零或强制纳入，空支持返回graph-connected零。全部类同规则，无cup专属权重/阈值/温度/系数搜索。类别来自冻结入口预测，不是患者真标签；不改变主KL、不把拒绝像素强制当训练软目标。参考不加入优化loss，仅h=grad(r)。相同g·h<0且||h||²>0时g*=g−(g·h/||h||²)h；其他原样。一个参考VJP即可，最多3个类内KL归约，相比V12增加KL归约而不是native完整forward或VJP。新单一模块是类参考聚合，不叠加V11目标恢复/Cayley/V7/V2/B cap/V1/RL。

单训练引擎不复制loop。V12新增默认entry_reference_objective hook保持原fine_kl；V13子类仅override该参考归约，实际flag/公式/源码绑定loss_contract。关闭类均衡但保留entry_gradient必须与归档冻结V12完整native state/EMA/optimizer/cursor/prototypes及旧loss路径一致；原V10关闭梯度模块的已验证资格复用，未变检查不重跑独立旧轮。

同两个teacher当前EMA+immutable共享本阶段入口anchor；首阶段source+identity、次阶段own封存学生及线性prefix。只有当前U/已有collected UL student view包含L donor，不使用历史图像/标签/replay/第三teacher/新辅助loss/患者额外探针。teacher不进optimizer或EMA，部署丢弃，推理参数不增加；参考loss/grads单步临时，checkpoint保留flag/anchor/telemetry，resume按cursor重算。原始梯度投影不保证Adam实际更新、有限步损失或旧域保持；同一总体参考仍不能分别保证每类，当前U可能不覆盖旧域、少数类及入口teacher可能错误。

## 先资格再正式有限预算
优化前冻结PLAN/USER_AUTHORIZATION/CODE_COMMIT/private CONFIG/本文；资格上限96CPU/32生成CUDA/4真实L，预计32/10/2含故意after_optimizer失败永久计费，不重置额度。相关资格：present/absent/empty支持、复制仅背景不改变balanced标量及聚合梯度、冻结target无梯度；CPU两case32优化，关闭与归档V12全native/原loss parity，启用A/B/R梯度与实际投影、原主KL拒绝不变、teacher冻结共享、SWD R-only/线性R EMA、真实checkpoint续训/telemetry及错误actual flag拒绝、部署/own stage2、zeroU当前L smoke。预计5资格reference VJP、3生成额外梯度VJP、4额外CPU数学VJP、1生成loss调用、2直接anchor前向，无额外患者探针；实际据台账记录。

正式4轨迹seed163/164×order1/2、8目标阶段21200/source0。先S163/O1两目标仅工程成功封存/EXIT0后其余3，不按性能剪枝；全8封存/4EXIT0后原20已见域统一评价。16960活跃参考VJP上限，与原固定R诊断64分列；沿用V10原16960anchor完整前向，无额外正式完整forward。记录reference_background/rim/cup_pixels、balanced_calls、present_class_calls和固定点reference_class_pixels；前者总体占比是支持，不是保护正确性。保留entry_reference_vjps/guard/projected及dot_before/after/reference_norm，不能据激活推断成功。不改运行中配置、扩预算、加seed/data、搜索PAS/系数/组合、选checkpoint或自动正式失败重试。

此前正式实际318673，本轮累计正式上限339873；actual=318673+本轮物理调用，source0。资格此前768CPU/181生成CUDA/38真实L，预计800/191/40，资格累计上限864/213/42。历史正负/失败、中断673回滚重复丢失和5缺失诊断NA保留，复制封存5300不双计。NFS真实挂载/容量/write-read/GPU4–7>=4GiB余量、既定storage wrapper/neutral父子argv、匹配580.178.04默认库，旧580.173.02不prepend，不改冻结HDF5/manifests/splits/系统或终止健康无关进程。

## 对照、成功、交付和停止
完整paired V13−V12（唯一参考归约差）/V10（正向底座）/V11背景，匹配F5、历史B0/B2来源明确；协议不可比补必要匹配对照。原双门槛同时vsF5和B2：每seed平均两order Final增益>0、总均值>=0.005，各order平均Old/Incoming>=−0.005，每seed/order/domain/rim或cup>=−0.05。全部负细胞/轮次报告，不事后放宽、选checkpoint或sealed test调参。重复开发患者不是独立泛化/SOTA/临床获益/创新证明。

完成先复用审计、一次必要核对，公开源代码/冻结协议/完整匿名结果/激活正负发现/实际优化、forward/VJP和累计成本/报告；经本机7897代理Git HTTPS显式http.proxy、HTTP/gh代理防NO_PROXY绕过，验证远端SHA/匿名访问后再独立下一科学轮，不直连/强推。患者/逐患者读出/权重/private配置/原始私有log/密钥NAS不进Git。工程中断不伪装新科学轮绕过不可重试，新增恢复预算须明确许可。f5小时ACTIVE、RL PAUSED；只有原双门槛达标且完整公开验证后暂停，或明确用户停止。

## 工程资格失败保留
首次资格因注册器使用非标准USER_AUTHORIZATION结构被原引擎拒绝，优化0CPU/0CUDA/0真实L、EXIT1，另此前大环境载荷注册失败消耗0。原目录及失败永久保留，不覆盖或重置预算。新create-only绑定目录修正为原引擎严格instruction/plan_id/execution_commit三字段并保存代码快照CODE_COMMIT；同科学计划、总资格上限96/32/4、正式21200不变，资格实际跨attempt累计。
