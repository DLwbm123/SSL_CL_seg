# HBSI-Seg v1

独立的层级贝叶斯半监督顺序推断实现。B11 文件未修改。原始提交5fe9b7d8的两域合成优化/推断失败及18项测试证据原样保留。当前修订的接口、四项修复和固定真参数诊断见 [revision_v2/README_ZH.md](revision_v2/README_ZH.md) 与 [修订报告](revision_v2/REPORT_ZH.md)。本实现尚无正式医学分割资格。

从仓库根运行。环境需要 Python>=3.10、PyTorch>=2.6、NumPy；HDF5 接口需 h5py，既有医学指标需 SciPy 及仓库依赖。实际验证环境为 Linux、Python 3.10、PyTorch 2.6.0+cu124，CPU、float64、单线程，CUDA_VISIBLE_DEVICES 为空。没有安装新环境。

```bash
export PYTHONPATH="$PWD:$PWD/experiments/lcrseg"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m experiments.sc_bayeseg.hbsi_v1.cli validate-spec
```

计算运行使用项目 `with_nas_storage.sh`。`OUT` 必须由操作者设置为 NAS 下本次独立目录；同一任务的重试复用 OUT，改变 run-id，不重置账本。实际执行时通过中性 python/脚本入口读取环境配置，避免在进程 argv 中显示身份或方法名称；以下是等价可复现的 CLI 参数。

```bash
python -m experiments.sc_bayeseg.hbsi_v1.cli unit-tests --output "$OUT" --run-id NEW_UNIT_ID
python -m experiments.sc_bayeseg.hbsi_v1.cli synthetic-qualification --output "$OUT" --run-id NEW_DEMO_ID --unit-receipt "$OUT/NEW_UNIT_ID_UNIT_RECEIPT.json"
python -m experiments.sc_bayeseg.hbsi_v1.cli real-smoke --output "$OUT" --run-id NEW_SMOKE_ID
```

以下退出码描述原始v1实施：`unit-tests` 实际退出码 0，18/18；首次退出码 1 的记录保留在 evidence。`synthetic-qualification` 实际退出码 2，表示完成计算但科学/推断资格不通过；不应被包装为成功。`real-smoke` 未提供真实协议时退出码 0，仅生成 NOT_RUN 记录，不读取数据。当前命令要求匹配源码/配置的资格回执，旧目录和旧回执不可用新ID重新解释为新资格。

```bash
python -m experiments.sc_bayeseg.hbsi_v1.cli train-source
python -m experiments.sc_bayeseg.hbsi_v1.cli train-stream
python -m experiments.sc_bayeseg.hbsi_v1.cli evaluate --output "$OUT"
```

这三条拒绝路径已验证：缺少真实训练显式授权/协议/预算时，前两者退出 2；evaluate 缺少具体协议时退出 1。它们不是已执行的真实训练/评价。

未来正式命令必须提供 `--authorize-real-training --protocol APPROVED.json --budget-seconds N --config configs/source_train.example.yaml --output "$OUT"`。example 协议默认拒绝；需独立冻结实际路径、当前域、stage、划分与预算，并另行授权。源训练需要SYNTHETIC_LEARNING回执，stream需要与输入checkpoint绑定的MODEL_LEARNING回执；单独的PASS字符串不再有效。活动阶段通过 `--resume-checkpoint` 恢复，保留原run-id、协议、账本目录、总步数和原始时间预算。本入口保留<=3600秒、<=1000外层更新的更严格界限；扩大真实训练预算需另立版本。

真实 smoke 需模型配置 `configs/real_smoke.example.yaml` 与真实协议，最多2批次、global MC=1、K<=3、外层更新0。数据路径不得出现在公开模板。`configs/*.yaml` 使用 JSON 子集，可直接由标准库读取。

主要接口：`GlobalPosterior`、`ConditionalDomainMemory`、`GenerativeModel.log_joint`、`Recognition.rsample_and_log_prob`、`Refiner.refine`、`SequentialObjective.estimate`、`Predictor.predict_image_only`、`StreamState.begin_task/finalize_task/save/load`。真实 MRI 接口保留已有 processed volume 网格与 spacing 顺序，预测回插至该原生存储网格后调用既有指标；不声称重新恢复至原扫描仪未预处理空间。

详细证据见 MODEL_SPEC_ZH.md、TEST_RESULTS.json、SYNTHETIC_REPORT_ZH.md、IMPLEMENTATION_REPORT_ZH.md。私有 checkpoint、HDF5 测试 fixture、逐步原始日志留在 NAS；本目录仅源码、模板及小型审计摘要。
