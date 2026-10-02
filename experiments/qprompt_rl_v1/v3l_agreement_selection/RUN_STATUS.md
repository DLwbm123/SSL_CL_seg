# V3L 已核验启动记录

- run_id：v3l_agreement_selection_20261002T110500Z；执行4065ea04a8b410d620440d6fb24259577dab42a3，NAS protocols下同名create-only目录。
- 2026-10-02 11:14:51UTC启动，GPU4；guardian2409627/start ticks294234209，worker2409634/start ticks294234253。
- 准备起点11:05UTC，optimizer截止次日01:05UTC、硬截止03:05UTC。50轨迹×1200=60000main，100计划smoke；完整预算见冻结计划。
- NFS挂载、19TB余量、真实读写探针通过；GPU4启动前可用24124MiB。源码初稿42c34bb在首次启动前修正只读观察器位置为4065ea0，避免buffer恢复触及已建梯度图；没有旧执行训练、失败或预算重置。
- CPU可选mask原路径精确一致、valid分母、拒绝像素零梯度/零mask、预测类数量匹配、确定性随机/RNG不变和非法数量拒绝检查通过；矩阵/二维判读自检通过。
- 11:15:33UTC进入MAIN_MATRIX。首次检查100smoke/10条件集成PASS，实际Adam预演、只读完整状态、快照所有权与精确重复续步通过；主矩阵5次成功、零失败。
- 全矩阵保守估计19710.19秒（约5.48h），optimizer剩余49766.93秒，通过准入。该数为启动时估计，后续ETA须读最新账本。
- with_nas_storage.sh与既有guardian启动；ps主子argv中性，nvidia-smi确认worker约994MiB。日志仅既有scheduler包装提示，无异常堆栈；完整终结以最终receipt为准。
- 每小时监测保持ACTIVE并绑定V3L。V3K完整报告与五份匿名聚合已发布789be9dd01d98179e9eaf85076271e39bb6e0058，远端SHA和匿名HTTP200已验证。
