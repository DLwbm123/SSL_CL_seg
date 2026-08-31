# SSL_CL_seg NAS 存储迁移记录

状态：`PASS_MIGRATED_VERIFIED_HOME_RELEASED`；最终观测时间：2026-08-31T20:31:39.170421+08:00。

此次只迁移当前 SSL_CL_seg 项目的实验产物，不包含其他项目、冻结输入数据、环境或本地已验证备份。未启动任何新实验，模型前向和优化器更新均为 0；既有科学结论与硬停止要求不变。

## 结果与空间

- NAS 根目录：`/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg`。
- 完整验证 57,582 个文件路径、37,785 个目录和 17 个符号链接；42,982 个独立文件 inode 的硬链接关系保留。
- 迁移前这些产物在 home 实占 23,327,932,416 字节（23.33 GB）。NAS 上同一批产物实占 23,172,706,304 字节；不同文件系统的目录/块分配不同。
- 切换期间 home 可用空间实际增加 23,331,319,808 字节（23.33 GB）。共享盘其他活动会使该值与本项目目录占用略有差别。
- 最终 home 可用空间：80,719,065,088 字节（80.72 GB），使用率 98%。
- 所有临时 home 副本均已移除；下表中的旧路径全部保留为指向 NAS 的软链接。

## 路径映射

| 旧路径（现在是软链接） | NAS 上的实际位置 |
|---|---|
| `/home/jiangsuiyang/SSL_CL/gate1c_v3_clean_regeneration_20260831` | `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/gate1c_v3_clean_regeneration_20260831` |
| `/home/jiangsuiyang/SSL_CL/launch_logs` | `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/launch_logs` |
| `/home/jiangsuiyang/SSL_CL/reports` | `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/reports` |
| `/home/jiangsuiyang/SSL_CL/runs` | `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/runs` |
| `/home/jiangsuiyang/SSL_CL/srgas_pilot_a1.log` | `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/srgas_pilot_a1.log` |
| `/home/jiangsuiyang/SSL_CL/srgas_pilot_a2.log` | `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/srgas_pilot_a2.log` |
| `/home/jiangsuiyang/SSL_CL/srgas_pilot_a3.log` | `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/srgas_pilot_a3.log` |
| `/home/jiangsuiyang/SSL_CL/srgas_pilot_a4.log` | `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/srgas_pilot_a4.log` |
| `/home/jiangsuiyang/SSL_CL/srgas_pilot_a5.log` | `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/srgas_pilot_a5.log` |
| `/home/jiangsuiyang/SSL_CL/srgas_pilot_a6.log` | `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/srgas_pilot_a6.log` |
| `/home/jiangsuiyang/SSL_CL/srgas_v01a_parent_launcher.log` | `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/srgas_v01a_parent_launcher.log` |

## 验证与保护

复制保留权限模式、文件字节及链接关系。复制前对 home 原件生成完整 SHA-256 清单，复制后对 NAS 全量读回验证；切换后再次核对保留的 home 原件及 NAS 文件。每个 home 临时副本移除前又检查了路径覆盖、设备/inode、模式、修改时间和 ctime；ctime 变化时重新核对文件哈希。新增文件或内容变化会阻止移除。

4,007 个受保护文件通过迁移前后完整哈希核对，包括冻结 HDF5、清单、划分、校验文件和原代码文件。原始输入、环境和本地已验证归档均保持原位。既有 NAS 历史归档未覆盖。

formal 与完整 Gate1C v3 bundle 的原 manifest 字节哈希保持不变，全部 manifest 条目与本次已全量验证的文件哈希索引一致。

- formal manifest SHA-256：`a80f6175f7f8e3fd2bd9a46a709495f3e11a81129b90717722b7bce435a7db20`。
- 完整 v3 bundle manifest SHA-256：`480b627e0f63839ff5430d980020ca026c45838cf5eeb345f2b4cf7c4d578bb2`。
- 原精确代码 `db4af88eca0dca48025f8884bf7f85e068eabf2a` 通过旧路径访问，Git 工作树干净。
- 代表性检查点通过 CPU 加载；代表性 NPZ 的 11 个数组正常读取。
- 复制校验、路径切换、配置验证三个任务均由服务器本地父进程记录实际退出码 0。

## 后续保存规则

服务器启动入口：

```bash
bash /home/jiangsuiyang/SSL_CL/with_nas_storage.sh COMMAND [ARG ...]
```

仓库对应脚本：`experiments/lcrseg/scripts/with_nas_storage.sh`；规则已保存到本地仓库 `AGENTS.md`、服务器 `/home/jiangsuiyang/SSL_CL/AGENTS.md` 和 NAS 根目录 `AGENTS.md`。

- 结果默认路径：`/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/runs`。
- 临时文件：NAS 根目录下的 `tmp`。
- Torch、Hugging Face、XDG、Triton 和 CUDA 缓存：NAS 根目录下的 `cache`。
- 新协议/运行必须创建新的输出目录；显式输出参数也必须指向 NAS。
- 启动脚本验证 NFS 挂载并做实际写入/读回探测；失败会停止，不会回退 home。该服务器的 `os.access(..., os.W_OK)` 返回值不能单独用于判断 NAS 是否可写。
- 全局 shell 配置未修改；此脚本不授权新实验，也不解除科学硬停止。

配置和本记录新增于当前本地工作区；未自动推送 GitHub。服务器上的存储路径与入口已生效。

## 迁移凭证

私有完整迁移凭证位于：`/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/.storage_migration_20260831_01`，共 74 个文件，已封存并复验。

- 凭证 manifest：`STORAGE_MIGRATION_BUNDLE_MANIFEST.json`。
- Manifest SHA-256：`1a1021aa4371abe7a12eb0eb5fa0328d9c89e32e650724fa1fc1490954948aaa`。
- 凭证内容 SHA-256：`e74574880b1ce98605b961194805c4cdbe8a3ed0410e64cb2bc46a53ff4f6cf6`。
- 少量已核验凭证的本地副本：`/Users/bominwang/.codex/private_artifacts/SSL_CL_seg/nas_migration_20260831_01`。
