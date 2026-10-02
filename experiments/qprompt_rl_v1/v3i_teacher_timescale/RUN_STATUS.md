# V3I 已核验启动记录

- run_id：v3i_teacher_timescale_20261002T045400Z，NAS protocols同名create-only目录。
- 执行提交2acdec5fd217bf31585f343befa525198be8f71a；准备起点2026-10-02 04:54UTC，optimizer截止18:54UTC，硬截止20:54UTC。
- 05:01:32UTC启动；guardian2263800/start ticks291994406，worker2263807/start ticks291994414，GPU4。
- NFS挂载、19TB余量、新目录读写探针通过，GPU4启动前空闲显存24124MiB；with_nas_storage.sh/guardian中性主子argv及nvidia-smi已核验，初始约1802MiB。
- CPU实际共享EMA默认/慢方程、缓冲规则、validate_only只读、冻结门控、非法beta及固定矩阵联合判读检查通过。
- 80/80工程smoke成功，8条件Adam预演/只读/精确恢复通过；FIXED教师不变、其余教师有更新，计数匹配。
- 05:02:05UTC进入MAIN_MATRIX；05:03:05UTC确认272main成功，零失败，首cell四条件均完成65步。
- 串行保守估计15544.10秒（4.32小时，含1.25安全系数和1800秒评估余量），可用optimizer窗口49914.92秒，通过资源门槛。
- 每小时自动化保持ACTIVE并绑定本轮；完整矩阵尚未完成。旧source/checkpoint/报告保留，固定硬截止不变。
- V3H完整结果已发布e56ba5692a9a566eb191df82f67c95723af22406，远端SHA和匿名HTTP200已验证。
