# 给 Codex：父实现已指定，使用现有 LR_SRC_A 接通并运行五框架

## 0. 先读结论：不再查找“原始 KI 父身份”

用户要求从其 GitHub 直接指定五框架共用底座。本次指定为：

- repository: `DLwbm123/SSL_CL_seg`
- working branch: `codex/sslcl-five-frameworks-v1-review`
- inspected immutable source: `ab69b6b3cd7e0102cc929edfda8a0f4035f635dd`
- new reference identity: `NATIVE_LR_SRC_A_3DOMAIN_V1`
- native network: `experiments/lcrseg/di_dmpa_jascl/modeling.py::LCRSegUNet2DJASCL`
- native factory: 同文件 `build_lcrseg_unet_jascl_model`
- head adapter: `experiments/lcrseg/ssl_head_control_v0_1/core.py::set_head`，模式 `raw_linear3_same_geometry`
- low-rank mechanism: `experiments/lcrseg/lctx_weight_memory_v0_1/core.py` 的 `LowRankConv`、`basis`、`initial_A`、`configure`、`LAYERS`，选臂 **LR_SRC_A**。
- historical configuration reference: `experiments/lcrseg/docs/lctx_weight_memory_v0_1/PROTOCOL.json` 与 `experiments/lcrseg/lctx_weight_memory_v0_1/engine.py`。

**身份区别必须保留：这是明确选用仓库现有单侧源权重投影实现作为新一轮共同参考，不是找回原始 KI，不继承原始 KI 的 SOTA 声明。** 历史两域 LR_SRC_A 也不是本次三域扩展的既有结果。

把本文件交给 Codex 表示采用上述明确指定，并继续此前“接入、自检、smoke 后自动训练”的用户授权。这是对原 prompt 第 2 节的明确修订：撤销“必须找到未定位的原始 KI，否则禁止训练”，也撤销将现有 SVD 实现**明确选作新参考**的限制。仍禁止把它冒称为原 KI、用 toy 或 F_CONV 偷换低秩基线。

不要再返回“请提供原始 KI launch command/run_id”。原 KI 恢复记录可以保持 NOT_RECOVERED；它不再是本次实验前提。身份已指定不代表生产代码已经实现或新代码通过了外部审阅。

## 1. 指定的真实模型与参数机制

复用上述真实 U-Net：3 输入通道，3 输出类别，通道 16/32/64/128，GroupNorm。保留原转置卷积、skip connections、线性 3x3 head 的实际 padding=0 和最终 bilinear align_corners=True。不要拿 SyntheticParentBridge 替换。

上游分类头源码固定为仓库现有锁定依赖：JASCL commit `3c93ca70784fc3a1d2a887f8d7dce5af6bc75f53`，路径 `Semi-Supervised_Natural-FoSSIL/inc/deeplab_gaps_meanT`。复用已存在 reference checkout；不因为旧单域 build() 不接受 REFUGE 而换网络。直接调用已列出的 native factory 并显式设置线性头，新增三域 wrapper。

每个增量阶段，对既有 `LAYERS` 中十四个 Conv2d 权重，使用阶段入口已训练的合并权重 W0：

```
rank r = 8
保护维数 k = min(out_channels // 2, flattened_in_width - r)
V = W0 展平后 SVD 的前 k 个右奇异向量
A_eff = A - (A @ V) @ V.T
W_eff = W0 + reshape(B @ A_eff)
```

仅输入侧约束；B 无输出侧正交约束。A/B 都可由真实标签训练，B 初始为零，A 采用现有投影、Frobenius 范数匹配的初始化。不要改成 LR_SRC_AB，不增加新正交 loss，不重新设计一种“原 KI”。

冻结范围严格沿用 `configure(..., 'LR_SRC_A', ...)`：W0、归一化、转置卷积与线性读出等非白名单参数冻结；sigma/grad_update 为惰性冻结状态，不启动 GAS。五框架附加参数按已审阅定义处理。

### 本次明确新增的三域转移规则

历史实现只有一个源→目标适配。本次将同一单侧机制应用于两个增量阶段：
1. 首域 REFUGE 用真实标注预训练 native 模型，得到每个优化种子自己的源学生。
2. 每个后续域开始时，从该轨迹紧邻前驱的合并权重重新导出 V，创建新的零增量 A/B；禁止从其他 arm 借用权重。
3. 阶段结束，把 B A_eff 合并进 W0。当前 A/B 与 V 不作为额外历史特征/统计库传往下一阶段。
4. 五框架已定义的 F_prev→F_next 按原设计处理；F2 和父-only 基线不继承别的框架的 F。
5. 不保留旧数据、旧伪标签、历史原型、累计 AGOP 或旧教师；当前 EMA、PAS 原型、优化器每域重建。

这是一项预先声明的三域协议扩展，不宣称复现历史 KI，也不宣称该投影保证零遗忘。rank 与保护维数本轮固定；父学习率仍有既定 4 组调参机会。

## 2. native 桥接必须实际完成，不能只把 JSON 改为 BOUND

复用并扩展 five_frameworks_v1 的真实桥接与 runner，而不是直接运行老 `lctx_weight_memory_v0_1/execute.py`。旧 execute/engine 绑定了旧两域、旧源码哈希和仅 L 权限；保持这些历史文件与运行记录不变，在新 runner 中导入/适配数学组件。

具体要求：
- native adapter descriptor 必须表达 W0、A、B、V 以及真实 effective_weight()；禁止用 toy 的 `base+B@A` 遗漏右投影。
- F2 的当前有效权重梯度探针通过 native override 实现，提供 FP64 正交 P_free 或等价算子；不能因 FP32 投影舍入套用不合理精度门槛，也不能吞掉非法投影错误。
- F2 只允许改当前新建零增量 adapter 的 A；保留本轮结构项单独生成方向的定义。
- geometry bridge 实际对应 native 3x3 输出裁切及 resize，不能使用 toy 的 identity geometry 假设。
- 注册 native reader、trainer、恢复、封存、student-only evaluator；保留 R01–R07 所有回归。
- AMP/scaler、optimizer/scheduler、stage callbacks 使用实际 native 对象，不删除 synthetic 判定后沿用 toy Adam/defaults。

### EMA 的明确约定

对 native 父卷积部分，复用现有 `dense_items` / `dense_ema` 的语义：教师更新的是**有效稠密权重的 EMA**。分别 EMA(A) 和 EMA(B) 一般不等于 EMA(BA)，不能混称。

生产教师可以是 dense parent 加相同 feature-sidecar；按真实命名对应更新，不用 zip 将不同结构的参数盲目配对。对固定 Q/F_prev 下的 R 进行 EMA 对应当前线性 sidecar 的有效矩阵 EMA。两个完整模型预算分别统计父参数、F/Q/R 与优化器缓冲。不要构造独立历史教师或偷偷增加第三个完整模型。

## 3. 已指定的基础配置与训练预算

不要继续索取“未定义原 KI 的配置”。本次基础配置以下列**现有 native 训练设置＋明确三域扩展**为准：

- target images: 384x384；batch_size=2；固定末尾学生，不挑最佳 epoch。
- target supervised loss: `ssl_anchored_mix_v0_1/core.py::supervised_parts` 的 CE＋Dice；LCTX 的 source collection 仍用原定义。
- source REFUGE: 同一 native 线性头，标注数据 CE 预训练；正常网络可训练参数参与，sigma/grad_update 除外；不启用五框架 sidecar 或 target PAS。
- Adam 基础 lr=0.001，betas=(0.9,0.999)，eps=1e-8，weight_decay=4e-5；每个参数组保留自己的 multiplier，lr 按 `(1-step/total_steps)^0.9` 衰减。
- 源和目标都采用 100 epoch。当前已冻结数据的 L/U 元数据计数是 REFUGE 40/160、RIM 16/63、Drishti 10/41；监督组只使用 U 数量计算匹配步数，不因此读取 U 图像。
- steps_per_epoch = max(ceil(nL/2), ceil(nU/2))，分别是80、32、21。
- 因此 S_REFUGE=8000，S_RIM=3200，S_DRISHTI=2100；若合法元数据与这些已核对计数不同，报告真实的数据冲突，不静默换 split。
- source / target 首20%阶段 warmup 与后20% ramp 依既定 study_plan；不要改变 native epoch 与 successful-step 对应。

父配置 P0/P1/P2/P3 保持 delivery/configs/study_plan.json 的 4 组：lr multiplier 1、0.5、2，以及 multiplier1 + B/A lr 比4。注意原表字段 `lr_multiplier` 到 runner 的 `parent_lr_multiplier` 的显式映射，不能配置写了却未生效。baseline/family 的搜索网格维持现有定义。

源学生按 seeds161/162/163/164 各一个，无合法匹配可复用时新训，至多4个；这些是本轮优化种子，不是新的患者划分。所有候选在同一种子共享相同合法源学生，后续每条轨迹独立。

预算：212条目标轨迹/424个目标阶段上限 = 1,123,600 次目标科学更新；至多4个新源学生 = 32,000次源科学更新；两者总上限1,155,600。合法别名去重/合法源复用可减少，不补新候选。资格、smoke、F2探针、F4内循环和失败恢复另外计数，不能藏进科学更新。

## 4. 数据与依赖的确定入口

native数据来源：`experiments/lcrseg/single_teacher_scd_v0_1/data.py` 的 CurrentData/metadata/geometry，以及其调用的 canonical path/hash loader。

- manifest: `manifests/training/lcrseg_v1_seed0.csv`
- manifest SHA256: `0622f54f42f05d6ef87f9dc89ee9435cf8da03c6c30cd970db6ea167e00dd8a3`
- split: `splits/fundus_seed0.json`
- split SHA256: `f250d97aea1f36f21899f5dd40bb6c9a819e7755aee458c8ee27506496b46a88`

两种序列仍为 REFUGE→RIM→Drishti 与 REFUGE→Drishti→RIM。旧 CurrentData 把物理域索引与阶段号耦合，不能把反向顺序的 stage=1 当作 RIM。新数据 adapter 用明确 domain 和 seen-domain 集合管理训练/评价权限，复用读取 primitive；不改写旧历史协议。

复用用户已提供的本地路径、NAS、Python 环境、SSH 和GPU权限。GPU 5/6/7 是本次沿用的设备范围，不等于承诺这些卡空闲；启动前检查负载，不杀其他任务，不租额外资源。私有路径和凭据不上GitHub。

## 5. 必要数值与生产自检后直接执行

旧权重记忆版本的恢复说明在 `lctx_weight_memory_v0_1_1`，应复用其对参数化残差和 FP32 权重相加舍入的区分。不要直接复活旧版“很小delta作分母”的苛刻数值闸门；也不能关闭有限性与真正投影检查。

继续此前用户授权的生产流程：CPU回归→已授权GPU的CUDA合成资格（上限256次合成optimizer调用）→当前L smoke最多24步（F1=4,F2=8,F3=4,F4=4,F5=4）→正式B–D矩阵。smoke与资格不要求Dice提高，不继承权重入正式实验。所有失败开销计数。

为新native集成记录新commit/源码指纹，推送后不等新的外审。批准记录使用 USER_DELEGATED_EXECUTION 和这份明确指定，不伪造外部reviewer或“新native源码已外审”。

PARENT_BINDING可先记录 PARENT_DESIGNATED_PENDING_NATIVE_INTEGRATION；实际接通后才设置实现验证状态。用新 `real_parent_identity=NATIVE_LR_SRC_A_3DOMAIN_V1` 注册runner。历史KI恢复文件保留，不能将其NOT_RECOVERED改成已找到。

## 6. 选择与论文解释

保持父/基线合理调参、五框架每个8组配置、全部五框架及五基线进入三种子复核。排名按已定指标，不要求每个seed/order/class都正、患者区间排除0或均值超过0.005。

本次选择现有LR_SRC_A是为复用真实单侧参数机制，不代表它已是最优：原固定两域实验中它低于F_CONV。新结果只支持在本轮共同底座上的增量筛选，不得把胜出排名直接当作原KI上的增益或SOTA证据。旧F_CONV分数不能与新三域数值直接混排。最终论文仍需同协议强持续学习对照；不在本轮自动追加Phase E或额外搜索。

## 7. 完成形式

你应交付真实native绑定、生产runner、必要资格结果及真实启动回执。仅在真实运行产生第一步成功更新后才报告RUNNING。父实现已经由本文件明确指定，不能再次以“没有原始KI命令/run_id”为理由返回同一阻塞。无法访问合法数据/设备/依赖、资格失败无法在预算内修复，仍应报告实际工程问题，不谎称开训。

推送代码、配置与聚合结果；不公开图像、标签、患者标识、权重、token或私有日志。不得承诺会话外无人监督的继续工作或未来必然回报。
