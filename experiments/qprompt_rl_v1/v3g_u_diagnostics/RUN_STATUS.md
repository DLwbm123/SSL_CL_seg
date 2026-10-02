# V3G 启动记录

- 执行提交：9c9d3872a1966987efc81a1206266a5ff9583b61。
- run_id：v3g_u_diagnostics_20261002T005000Z，NAS protocols同名create-only目录。
- 2026-10-02 01:02 UTC启动，GPU4；guardian 2185539/ticks290557136，worker 2185547/ticks290557148。
- 源码语法及CPU梯度正向/反向/零范数、KL方向检查通过。
- NAS NFS挂载、19TB余量及新目录读写探针通过；GPU4启动前空闲显存24124MiB。
- with_nas_storage.sh后台启动，ps确认主子argv均为中性Python入口，nvidia-smi显示同一worker，约2166MiB。
- 首次检查16/400探针、2/50状态完成，无日志错误；这是启动证据，不是完成声明。
- 每小时监测已绑定本轮；硬截止01:50 UTC，禁止优化器更新和自动重试。完整完成状态以NAS报告与后续公开报告为准。
- 前轮V3F发布提交782ff874010d5e986ef1d4d81b2e5501fbd29a5c：时间假设失败，继续只读机制诊断的理由见冻结计划。
