# V3K 已核验启动记录

- run_id：v3k_agreement_diagnostic_20261002T100400Z；执行1980aebbb9fa87851e0fa8ac084c1bda7f5cea10，NAS protocols下同名create-only目录。
- 2026-10-02 10:10:19UTC启动，GPU4；guardian2386835/start ticks293847053，worker2386842/start ticks293847062。
- 准备起点10:04UTC，硬截止11:04UTC（含准备1小时），optimizer严格0；完整覆盖1650批次、50状态、400聚合行。
- NFS挂载、19TB余量、真实读写探针通过；GPU4启动前可用24124MiB。源归档传输成功；CPU手工配对分歧/共有错误、255、缺失类、零接纳、U无标签和随机期望检查通过。
- with_nas_storage.sh与既有guardian启动，ps主子argv中性，nvidia-smi确认worker约602MiB。
- 首查10:10:46UTC450批次、13状态完成，日志无错误；随后账本460attempt/459success对应一个在途批次。完整完成以最终receipt为准。
- 每小时监测保持ACTIVE并绑定V3K；此短诊断可能在下次唤醒前完成，不将已退出PID当作故障。
- V3J完整报告与五份匿名聚合已发布faa2b339927dd0df2e1aff2cb1ee46d63a5731cf，远端SHA和匿名HTTP200已验证。
