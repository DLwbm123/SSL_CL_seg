# 固定生成模型的独立后验参照

先读 [PREREG_ZH.md](PREREG_ZH.md) 和 [METHOD_ZH.md](METHOD_ZH.md)。运行范围限原16×16灰度合成模型，生成器/域/识别器更新均0。真实数据入口不在本模块中。

在仓库根目录运行下列命令，OUT必须是经过挂载和写读验证的全新NAS输出目录；用项目 `experiments/lcrseg/scripts/with_nas_storage.sh` 包装每次Python调用。服务器进程通过中性入口传递环境配置，不能把项目标识写进可见argv。所有命令外还需一个共享起点的3600秒timeout，不能为每个命令重置总墙钟预算。

```bash
python -m experiments.sc_bayeseg.hbsi_v1.cli unit-tests --output "$OUT" --run-id unit --max-outer-updates 24
python -m experiments.sc_bayeseg.hbsi_v1.cli revision-checks --output "$OUT" --run-id revision --max-outer-updates 24
python -m experiments.sc_bayeseg.hbsi_v1.reference_checks "$OUT"
python -m experiments.sc_bayeseg.hbsi_v1.posterior_reference --output "$OUT" --unit-receipt "$OUT/unit_UNIT_RECEIPT.json" --revision-receipt "$OUT/revision_REVISION_RECEIPT.json"
```

前序任一步非零则停止，不启动参照。参照CLI验证当前代码的完整UNIT/REVISION回执，不能用旧提交回执。参照exit0只表示预注册链诊断通过，信息判断另看RESULTS的information_status；exit2表示运行完成但参照不合格；异常/预算耗尽为失败。当前模块不会训练快速识别器，不会开真实数据，不会自动延长链。

原始合成数据、每链概率/latent样本及checkpoint仅留NAS；发布聚合RESULTS、检查回执及资源摘要。参照MCMC的bulk/tail ESS与旧重要性权重ESS不同，不能直接比较数字。全量回归中的fixture更新单列，不冒称本轮没有任何数值优化；科学参照自身没有学习更新。

预注册固定的8幅新图不是独立医学病例证据。即使标签指标好，只要多链可信度失败，就不能把该预测称作正确后验，也不能以此证明或否定生成模型的信息能力。
