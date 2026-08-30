# Gate 1A 失败与警告（完整保留）

## 正式 attempt：唯一一次

- 执行提交：`8f4a71a5ea8d145183a3007ccd398ab79387478e`。
- 目录：`/root/LCRSeg/runs/di_dmpa_gate1/cfb62554f1e6a2a36850547485b1857dc9a28a20/gate1a_formal_8f4a71a_attempt1`。
- B0-EMA / seed1 / REFUGE / val 完整 `decoder.dec1` 特征图的最小范数为0，
  触发冻结 `<=1e-12` hard stop。不是已完成的多模态拒绝。
- GPU shard1 退出1；shard0 被协调器 SIGTERM（-15）停止；主进程退出2。
- 无聚类结果、无 A1–A6、无 K 选择、无控制救援、无重跑。
- 原异常没有 case/像素坐标；不推断零向量位于注册采样点。
- 4个已审计 checkpoint task 内存态未变；被取消的在途 task 缺少 after-state
  审计；18个磁盘 checkpoint 事后重哈希均一致。
- 原始日志、traceback、manifest、48份远端原始 feature cache 均保留；
  没有清理 partial outputs。

## 开发期（非正式数据结果）

| 记录 | 结果 | 说明 |
| --- | --- | --- |
| rsync 传输 | 未启动 | 远端无 rsync；改用 tar，未安装软件 |
| gate1a_unit_dev_attempt1 | 44 passed, 1 failed, 1 warning | macOS AppleDouble `._*.py` 被源扫描当作UTF-8源码；另有 CUDA error804 库路径警告 |
| gate1a_unit_dev_attempt2 | 45 passed, 1 skipped | 新干净 sandbox；禁止复制 xattr/AppleDouble，使用既有正确 LD_LIBRARY_PATH；真实 checkpoint 测试尚未授权启用 |
| gate1a_unit_dev_attempt3 | 46 passed, 1 skipped | 增加数值失败时仍保存不变性证据的测试；真实集成仍未启用 |
| exact-code tests / 8f4a71a | 47 passed, 0 failed/skipped/warnings | 已发布精确源码 + 明确启用的只读真实 checkpoint 集成 |

开发期失败环境目录保留为 `/root/sslcl_gate1a_dev.k2R7Hv`，
干净开发目录为 `/root/sslcl_gate1a_dev.hE7w4K`。
日志在 `gate1a_results/publication_evidence/gate1a_unit_dev_attempt*/`。
第一次 tar 还输出了不识别 macOS xattr header 的警告；使用
`COPYFILE_DISABLE=1`、`--no-xattrs`、排除 `._*` 后消失。
CUDA库路径使用历史 Gate 0 已有的 `/lib/x86_64-linux-gnu`，
未重装 Torch、驱动或任何包。

这些修正发生在正式执行前，属于传输/环境和测试基础设施；
不改变预注册字节、阈值、K grid、sampling、seed 或 metric。
正式零范数失败后未修改执行算法或进行第二次 attempt。

## 归档期网络复核

正式退出后的一次额外远端 `git ls-remote` 遇到 GnuTLS recv error (-110)，
退出128；这不是机制诊断重跑。随后只做当前云端 checkout、预注册/授权字节、
config/protocol/manifest/split/checkpoint 的本地哈希核验，全部通过。
执行开始时已成功完成的 remote SHA 验证仍保留在原始 RUN_METADATA；
不把失败的事后在线查询冒充成功。另有两次取文件操作先于该记录生成或使用了
错误路径，返回文件不存在；更正后重新复制报告，不触及正式 attempt。
最终报告 push 后另从本地核对 GitHub 远端 SHA。

## 发布时的格式检查

`git diff --cached --check` 在原始开发期 attempt1 的 `pytest.xml` 和
`pytest_output.txt` 中各发现3行 pytest traceback 自带的尾随空格。
这6条格式警告只属于原始失败证据；为保持已登记的原始字节与 SHA，
不对日志做格式化。新增报告正文及其他交付文件的格式检查通过。
