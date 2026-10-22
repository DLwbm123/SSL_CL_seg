# v4 整块与分块更新匹配对照

范围、预算及门槛见PREREG_ZH.md。只能使用全新NAS输出目录；OLD指向v3只读原始链目录。采用NAS包装器与中性入口，所有命令共享独立3600秒父进程timeout，CUDA隐藏、CPU单线程。前序非零立即停止，不能跳过P0/P1直接跑P2。

```bash
python -m experiments.sc_bayeseg.hbsi_v1.cli unit-tests --output "$OUT" --run-id unit --max-outer-updates 24
python -m experiments.sc_bayeseg.hbsi_v1.cli revision-checks --output "$OUT" --run-id revision --max-outer-updates 24
python -m experiments.sc_bayeseg.hbsi_v1.kernel_comparison p0 --output "$OUT" --old "$OLD"
python -m experiments.sc_bayeseg.hbsi_v1.kernel_comparison p1 --output "$OUT" --old "$OLD"
python -m experiments.sc_bayeseg.hbsi_v1.kernel_comparison p2 --output "$OUT" --old "$OLD"
```

P0复用资源账本，扣除前序回归时间后使用剩余300秒；P1/P2各自1200/2100秒，总账本3600秒。P1/2检查当前完整UNIT/REVISION回执和P0状态。exit2表示无合格核/资格失败；异常或超时表示失败/未完成。部分链与PROGRESS保存NAS，不能只报告已完成的好结果。

源码仅新增kernel_comparison.py、kernel_checks.py，复用v3 Gaussian条件及诊断实现但不修改其文件。数学检查用独立NumPy路径交叉核对诊断。发布聚合JSON/CSV，全部原始图像、链数组和checkpoint留NAS。源码和报告不包含私有服务器路径。
