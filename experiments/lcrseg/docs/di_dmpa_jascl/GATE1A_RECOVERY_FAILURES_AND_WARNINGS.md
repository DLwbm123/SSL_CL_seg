# Gate 1A recovery：全部失败和警告

## 正式数值失败（唯一 attempt2）

B0-EMA / seed2 / stage0 REFUGE / train_labeled，case REFUGE_test_n0128，class1，注册坐标(y=125,x=212)的16-D向量范数为0。当前case注册无效数1；完整图有限零向量3，无NaN/Inf。原始异常包含panel/baseline/source/seed/stage/domain/role/checkpoint ID和SHA/case/class/坐标/无效数/最小范数/unit SHA/plan SHA。

状态保持BLOCKED_NUMERICAL_FAILURE。34/72 feature units；0/432几何任务；A1–A6未计算；selected_K=null。未删除、替换、重采样、eps归一化或重跑。

shard退出[1,0]，父进程退出2。协作取消在另一shard当前checkpoint guard完成后生效，未触发600秒强制SIGTERM。9份内存不变性审计全部PASS；不声称未运行的9个checkpoint完成提取。18个磁盘checkpoint事后哈希均一致。

## 原始失败仍有效

Attempt1的BLOCKED_NUMERICAL_FAILURE和全部证据永久保留。已知seed1定位PASS是新的完整注册坐标审计，不将attempt1重标为PASS。该定位仅发现2个未采样零向量，不能挽救attempt2的真正注册零向量。

## 测试和运行警告

- 开发期 synthetic attempt1：64 passed，无失败。
- 开发期 synthetic attempt2：64 passed，无失败；保留两轮日志，不覆盖。
- exact-code unit/full-coordinate integration：65 passed，0failed/skipped/warnings，21.14秒。不是正式准入PASS。
- 正式attempt2日志中的数值异常完整保留；未做机制结果试跑或第三次attempt。

## 同步与发布过程（不涉及模型结果）

1. 云端直接GitHub fetch长时间无返回，改用本地已发布提交的增量Git bundle（9.5MiB显示值）。原fetch进程在读回PID/命令后取消；原调用退出143，记录为 `fatal: early EOF / Terminated / git-remote-https died of signal 15`。这不是诊断进程取消。
2. bundle成功导入，并创建独立detached worktree后，第一次创建JASCL_REFERENCE符号链接因third_party目录缺失退出1；补建目录、链接原始未修改reference后，核验执行HEAD和clean状态。发生于任何真实forward前。
3. 一次只读远端状态命令发现rg不可用（`rg: command not found`）；随后用grep读取进程状态，未安装软件。
4. 一次本地汇总输出过长被工具截断，JSON解析失败；缩减汇总字段后成功。未改写证据或重新计算feature。
5. 独立云端git ls-remote成功返回精确recovery code SHA，真实定位和attempt2启动时也各自验证了远端SHA。不把早先取消的fetch描述为成功。

原始失败attempt、partial caches、transcripts、STOP_REQUESTED和取消证据全部保留。归档仅复制文件/校验字节，停止后没有model forward或tensor load；没有清理或覆盖旧目录。
