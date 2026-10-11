# F5_DUAL_REFERENCE_V14 — 类别均衡入口与当前监督的双参考约束

用户明确要求以V13为底座，分析164 RIM rim和Incoming后直接开展最有依据的方法模块，小时监测至超过B2并达原门槛。V13完整正负结果已公开1730f9ac8a24787e35951244a3e4cd4bbcde569f。独立create-only NAS目录，优化前冻结PLAN/USER_AUTHORIZATION/执行commit/private CONFIG/本文。ANALYSIS.md是有限依据，不预认因果。

唯一主改动：原V13类别均衡入口参考h不变；取同一步split_gradients已经计算的所有实际可训练A/B/R监督梯度gL。对原合并L+U梯度g，解min_d 0.5||d−g||²，约束d·h>=0且d·gL>=0。归一化非零参考，枚举0/单一/双活动约束的两半空间QP并选最近可行候选；零参考忽略，double计算、参数dtype回写。数值可行容差1e−10*max(||g||,1)，不是科学调参；cast后记录实际双dot。只修改p.grad，原Adam/权衰/硬约束/EMA/原型提交不变，不能宣称实际Adam更新约束或有限步loss/旧域保证。

全部V13主损失与参考归约保持：线性R、joint KL->A/B/R，clean SWD->R-only；入口高置信EMA类别冲突拒绝、kept主KL均值；入口参考在原PASvalid且入口confidence>=原0.7的present背景/rim/cup类内均值再类间等权，参考不加训练loss。lambdaU0.25/SWD0.2/PAS0.7,0.5/Qk8D16/低秩硬输入投影/Adam/lr/日程/数据/原型不变。无V11恢复/旋转/其他模块/阈值或系数搜索/RL。两teacher仅EMA+共享冻结own入口，source+identity或own封存学生+own线性前缀，无旧训练图像标签/replay/第三teacher/额外推理参数。

保持单训练引擎：V12新增默认project_entry_gradients hook仍原halfspace；V14仅override投影，实际dual_reference/公式/源码/入口fingerprint绑定loss_contract，错误实际flag checkpoint恢复拒绝。关闭模块对归档V13完整native model/EMA/optimizer/scheduler/cursor/prototypes及主loss路径一致。临时梯度按cursor重算，anchor/telemetry保存，部署丢弃anchor。当前L包含原监督约束，并非只task CE；不按seed用不同模块。

资格总有限上限96CPU/32生成CUDA/4真实L，预计32/10/2含故意after_optimizer失败永久计费。新增双约束解析解、双活动/零/同向/反向/shape/nonfinite/overflow、128随机KKT与最近性样本；两case32CPU过拟合保留入口；disabled归档V13完整native状态、真正全A/B/R双约束梯度、原主KL/均衡参考/clean SWD/教师冻结与原始R EMA、真实checkpoint续训/错误flag、部署/own nextstage、当前L smoke零U读取。未变检查复用，不重置失败成本。相关资格PASS并绑定commit后才正式切状态。

正式4轨迹seed163/164×order1/2、8阶段21200优化/source0；先S163/O1两目标仅工程成功封存/EXIT0后其余3，不按效果剪枝；全8封存/4EXIT0后原20已见域统一评价。参考VJP上限16960、固定R诊断64分列，沿用V13的16960anchor前向和最多50880类内KL归约，新增正式完整forward/VJP均0。双dot/QP归约及向量操作有计算成本，不能当免费模块。dual_guard/projected/both_active、V13_negative_L_direction_calls记录原V13投影在同一V14状态下的反事实raw L冲突（不是归档V13全步实测），固定点entry_L_dot_before/after、entry_V13_L_dot_after和入口dot记录。激活不等于正确；当前U覆盖/伪标签错/零方向/Adam限制报告。

此前跨轮实际正式339873，本轮正式授权累计上限361073，实时actual=339873+V14物理调用；source0。资格此前800CPU/191生成CUDA/40真实L，预计832/201/42、本轮总上限累计896/223/44，以实际台账为准。V1–V13所有正负/失败/中断保留；V5的673回滚重复丢失永久计费、复制封存5300不双计、5NA不补算。不得改运行中配置、扩预算/seed/data、组合或系数搜索、选checkpoint、部分患者读出或正式失败重试。工程恢复额外预算须用户明确许可。

完整paired V14−V13（唯一投影变更）、V12/V10背景，以及匹配F5和历史B0/B2来源明确。原双门槛同时对照F5/B2：每seed平均两order Final增益>0，总均值>=0.005；各order平均Old/Incoming>=−0.005；各seed/order/domain/rim或cup>=−0.05。不事后放宽或sealed test调参；全部负细胞/负轮/成本公开，重复开发患者不是独立泛化/SOTA/临床获益。

NFS真实挂载/容量/写读与GPU4–7>=4GiB准入、既定NAS storage wrapper/中性父子argv，默认580.178.04匹配库不prepend旧580.173.02，不改数据/HDF5/manifests/splits/系统或终止其他健康进程。大文件/权重/log/cache/tmp NAS，患者/逐患者/权重/private CONFIG/rawlog/secrets不进Git。完成复用审计一次必要核对，源代码/冻结协议/匿名正负结果/诊断与累计成本公开经本机7897代理推送、验证SHA及匿名访问。先完整交付再下一有限轮，f5小时ACTIVE、RL PAUSED，仅原双门槛达标并交付验证后暂停或明确停止。
