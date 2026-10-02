# V3J 已核验启动记录

- run_id：v3j_teacher_calibration_20261002T085800Z；执行ec58627f095f416c5eac427ce094535fb2c85f56，NAS protocols下同名create-only目录。
- 2026-10-02 09:08:21UTC启动，GPU4；guardian2364642/start ticks293475238，worker2364649/start ticks293475250。
- 准备起点08:58UTC，硬截止09:58UTC（含准备1小时），optimizer严格0。完整诊断325批次、50状态、100匿名汇总行。
- NFS挂载、19TB余量、真实读写探针通过；GPU4启动前空闲显存24124MiB。源码归档传输成功，CPU混淆/255/类平衡接纳错误/ECE/零接纳检查通过。
- with_nas_storage.sh与既有guardian启动，ps主子argv中性，nvidia-smi确认worker，约582MiB。
- 首查09:08:43UTC完成146批次、22完整状态不变检查，日志无错误；账本随后150success/151attempt对应一个在途批次。这是启动确认，完整完成以最终报告为准。
- 每小时监测保持ACTIVE并绑定V3J；此短诊断可能在下一次监测前完成，须读最终receipt与exit，不能把正常退出当作故障。
- 前轮V3I完整结果已发布529662ba35507ce5bb224bfae51db4b423642439，远端SHA和匿名HTTP200已验证。
