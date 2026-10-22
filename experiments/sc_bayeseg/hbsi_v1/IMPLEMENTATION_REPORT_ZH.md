# HBSI-Seg v1 实施交付

**独立实现与数学/状态单测完成；神经合成资格失败，真实工作未启动。** 当前状态 UNIT_PASS，不能称 SYNTHETIC_QUALIFIED，更不能宣称医学收益。B11保留不变。

## 实际完成

- 新目录包含完整最小G/F全局权重高斯后验、固定先验白化稀疏投影、条件域记忆及解析期望KL、归一化SAR/Student/标签密度、结构化局部高斯、可微K步展开、唯一顺序自由能、volume联合IS预测、独立proposal/importance随机流、阶段封存与恢复。
- 父类别配置通过 `label_groups` 提供；叶码0..K−1，父码K+j，−1为缺失观测。每像素只有一个观测事件，无父子双计。U loader没有标签能力。
- 当前域均匀切片抽样及N/b目标缩放；未复用旧患者均衡抽样后直接乘患者数。已有MRI指标通过隔离适配器调用，预测封存后才打开评价GT。
- 18项T01–T18实际执行，最终汇总全部PASS。首次T15失败保留，修复的是独立参考中float32常数，不改生成模型或阈值。补充映射检查只重跑受影响T01/T02/T04/T16，全部通过。
- 两域各40步训练、t1封存、t2继续、两个域无域ID/image-only预测、K=0/3与三次独立MC重复均实际完成。生成数据、checkpoint与逐步日志保留NAS，没有旧图像回放。
- 七篇主文献的实际版本、阅读范围与借鉴边界已核验，见REFERENCES.md。数学规格先于合成运行落盘；未把Pro给出的解析数值充作本次结果。

## 关键数值

| 检查 | 实测 | 门槛/解释 |
|---|---|---|
| Normal/Student/Categorical/SAR及DCT | 最大绝对误差4.55e-13 | float64，atol1e-8/rtol1e-6 |
| 条件KL vs MC | 误差5.45e-5，MC SE6.26e-4 | 32768样本，6SE+1e-6 |
| 全量 vs 顺序高斯 | mean3.33e-15，cov1.94e-16 | <=1e-8；三个错误递推负对照均被辨识 |
| 半摊销超梯度 | 最大相对误差2.21e-10 | <=1e-4；phi、global mean/scale、domain m/C/scale全部覆盖 |
| 故意detach | 梯度最大差约2.209 | 负对照被辨识 |
| 低维主预测 | 最大概率误差0.002918 | 65536样本/域×4重复，阈值0.015 |
| checkpoint恢复、volume分块、预测状态 | 测得完全一致 | 包含optimizer、已有grad、buffers与外部RNG |
| 神经演示 | OPTIMIZATION_FAIL / INFERENCE_UNQUALIFIED | 预测前景99.6%，global ESS约1/8，MC稳定性失败 |

完整测试含seed、耗时、误差、门槛与失败见TEST_RESULTS.json及evidence目录。合成Dice约0.717来自近乎全前景预测，不应作为正结果。

## 原代码审计与保持边界

起点HEAD `df521f351621a2938a4d3aca274f5b96956989fd`，分支 `codex/sc-bayeseg-pilot-20261019`；未checkout/reset。工作区原有未跟踪文档和structure_mc_allocation代码均保留。本次只新增hbsi_v1，已有tracked文件无diff；指定7个B11/公共参考文件在本机和测试副本的hash一致，见BASELINE_SNAPSHOT.json/T18。

已读旧sslcl_v1 model/train/data、旧概率模块、patient_validation及B11预注册/阶段报告。旧step_loss含监督、图像权重与EMA一致性，不是本次自由能；未移植。旧Student helper以Gamma混合参数定义scale，且channels参数影响归一化；sslcl_v1的直接多通道调用需要另外审查其调用契约，不能拿它当本次独立逐通道StudentT密度。HBSI重新实现并用C=1/3交叉验证，本次未修改或追改旧B11结果。

## 运行与资源

实际环境：Python3.10.16、PyTorch2.6.0+cu124，Linux CPU、float64、单线程，CUDA设备被隐藏。NAS挂载为NFS且真实写读探针通过；通过项目with_nas_storage包装器启动，中性argv经ps核验。实验实施阶段没有GPU占用、旧任务重启或监控创建；GitHub交付见下文补记。

实测新包：global variational参数476（238个真实生成随机维度），amortizer1096，当前域变分参数18。每个封存域3952字节，包含当前实现重复保存的固定投影元数据；不声称常数内存。参数和DCT/projection指纹见PARAMETER_AUDIT.json，活动域注册表另见evidence/PARAMETER_AUDIT_ACTIVE_STAGE.json。

累计外层更新89=合成80+三轮恢复测试各3；global/domain/phi分别89。计账测试与演示体约29.97秒，含首次失败，未重置预算。实际NAS运行目录抽样统计<1MiB，远低于2GiB；源码/审计摘要后续同步只增加小文件。预算硬停、失败保留、create-only均由T17验证。

资源口径必须保留：RESOURCE_LEDGER里的joint_forwards是Refiner内逐样本联合密度评估，非全部底层卷积调用；单独的代数MC测试样本数在测试结果中。首次测试的Predictor外层draw计数、前两轮与神经演示的CPU峰值没有埋点，标为NOT_RECORDED，不回填估计值冒充实测。后续测试进程观测峰值约558MiB；GPU峰值0。账本计算体计时不包括交互开发、文献阅读、导入和仅schema/拒绝路径的CLI开销。所有运行仍明显低于给定自动计算上限。

## 实际命令与退出码

真实中性启动命令：`bash /tmp/.q20261022a/a.sh /tmp/.q20261022a/python /tmp/.q20261022a/run.py`；RUN_ROOT/RUN_ID/TASK_COMMAND通过环境传入，worker内部构造CLI参数，私有NAS输出路径不写入公开命令。补充检查使用同目录check.py。每条的有效子命令见COMMAND_RECEIPTS.json。

| 子命令 | 实际退出码 | 含义 |
|---|---|---|
| validate-spec | 0 | schema与参数注册通过 |
| unit-tests attempt01 | 1 | 17PASS，T15参考精度失败 |
| unit-tests attempt02、attempt03 | 0 | 各18PASS |
| unit-tests mapping04（4项相关测试） | 0 | 配置映射/RGB/HDF5/序列化通过 |
| synthetic-qualification demo01 | 2 | 已完成计算，但优化/推断不合格 |
| real-smoke（无协议） | 0 | 成功记录NOT_RUN，非真实smoke通过 |
| train-source、train-stream（缺授权/协议/预算） | 2 | 已验证拒绝路径，未训练 |
| evaluate（缺协议） | 1 | 已验证拒绝路径，未读数据 |

## 部分完成与未运行

真实数据适配器经过人工HDF5 fixture检查；真实MRI smoke、native-space几何、患者聚合和HD95尚未运行。灰度正式训练/评价入口已实现但未执行；RGB只过密度/前向shape资格，尚无完整真实RGB评价流程。源训练、多域医学训练、多seed和封存test均未运行。

识别器采用新的小型无BN U形编码器，支持4×4数值测试和配置尺寸；没有宣称复用了完整既有U-Net或完成了真实backbone公平性验证。相应神经容量、初始化、local proposal和训练预算都需要下一阶段独立资格。

当前局部ESS比例门槛0.25在L=4时恰等于理论最小比例，因此不能单独证明局部推断好；本轮整体仍因外层ESS和重复稳定性明确失败。未来正式门槛需要重新独立校准并冻结，不能沿用这个局部下界声称合格。没有为了改变本轮判定而改阈值或追加实验。

下一步草案聚焦局部proposal与生成模型优化的可行性，见NEXT_STAGE_PREREG_DRAFT_ZH.md。本轮到此完成已授权实施与有界验证；不把工程通过升级为方法有效。

## 公开交付补记（2026-10-22）

用户在实施完成后明确要求推送代码与实验报告。此次提交范围仅为 `experiments/sc_bayeseg/hbsi_v1/`，包含源码、配置示例、规格、报告和小型合成结果/测试摘要；保留失败尝试和未资格结论。真实数据、合成原始数组、checkpoint、运行日志和私有路径不进入Git。B11和其他工作区文件保持原样。

目标为公开仓库 `DLwbm123/SSL_CL_seg` 的 `codex/sc-bayeseg-pilot-20261019` 分支。文内测试与资源统计是本轮实验的历史快照；提交和推送状态由仓库提交记录及交付回执确认。发布不改变科学资格，也不代表启动下一轮实验。
