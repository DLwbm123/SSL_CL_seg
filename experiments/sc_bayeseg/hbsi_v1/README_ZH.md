# HBSI-Seg v1

独立的层级贝叶斯半监督顺序推断实现。B11 文件未修改。本轮达到 UNIT_PASS；两域合成演示完成，但优化与重要性推断均未通过资格，不能用于正式医学分割比较。

从仓库根运行。环境需要 Python>=3.10、PyTorch>=2.6、NumPy；HDF5 接口需 h5py，既有医学指标需 SciPy 及仓库依赖。实际验证环境为 Linux、Python 3.10、PyTorch 2.6.0+cu124，CPU、float64、单线程，CUDA_VISIBLE_DEVICES 为空。没有安装新环境。

```bash
export PYTHONPATH="$PWD/experiments:$PWD/experiments/lcrseg"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m sc_bayeseg.hbsi_v1.cli validate-spec
```

计算运行使用项目 `with_nas_storage.sh`。`OUT` 必须由操作者设置为 NAS 下本次独立目录；同一任务的重试复用 OUT，改变 run-id，不重置账本。实际执行时通过中性 python/脚本入口读取环境配置，避免在进程 argv 中显示身份或方法名称；以下是等价可复现的 CLI 参数。

```bash
python -m sc_bayeseg.hbsi_v1.cli unit-tests --output "$OUT" --run-id attempt03
python -m sc_bayeseg.hbsi_v1.cli synthetic-qualification --output "$OUT" --run-id demo01
python -m sc_bayeseg.hbsi_v1.cli real-smoke --output "$OUT" --run-id cli_real-smoke
```

`unit-tests` 实际退出码 0，18/18；首次退出码 1 的记录保留在 evidence。`synthetic-qualification` 实际退出码 2，表示完成计算但科学/推断资格不通过；不应被包装为成功。`real-smoke` 未提供真实协议时退出码 0，仅生成 NOT_RUN 记录，不读取数据。

```bash
python -m sc_bayeseg.hbsi_v1.cli train-source
python -m sc_bayeseg.hbsi_v1.cli train-stream
python -m sc_bayeseg.hbsi_v1.cli evaluate --output "$OUT"
```

这三条拒绝路径已验证：缺少真实训练显式授权/协议/预算时，前两者退出 2；evaluate 缺少具体协议时退出 1。它们不是已执行的真实训练/评价。

未来正式命令必须提供 `--authorize-real-training --protocol APPROVED.json --budget-seconds N --config configs/source_train.example.yaml --output "$OUT"`。example 协议默认拒绝；需独立冻结实际路径、当前域、stage、划分与预算，并另行授权。stream 还需要已封存源 checkpoint 与 source_qualification=PASS。本入口保留<=3600秒、<=1000外层更新的更严格界限；扩大真实训练预算需另立版本。

真实 smoke 需模型配置 `configs/real_smoke.example.yaml` 与真实协议，最多2批次、global MC=1、K<=3、外层更新0。数据路径不得出现在公开模板。`configs/*.yaml` 使用 JSON 子集，可直接由标准库读取。

主要接口：`GlobalPosterior`、`ConditionalDomainMemory`、`GenerativeModel.log_joint`、`Recognition.rsample_and_log_prob`、`Refiner.refine`、`SequentialObjective.estimate`、`Predictor.predict_image_only`、`StreamState.begin_task/finalize_task/save/load`。真实 MRI 接口保留已有 processed volume 网格与 spacing 顺序，预测回插至该原生存储网格后调用既有指标；不声称重新恢复至原扫描仪未预处理空间。

详细证据见 MODEL_SPEC_ZH.md、TEST_RESULTS.json、SYNTHETIC_REPORT_ZH.md、IMPLEMENTATION_REPORT_ZH.md。私有 checkpoint、HDF5 测试 fixture、逐步原始日志留在 NAS；本目录仅源码、模板及小型审计摘要。
