# G0 快速筛查启动回执

前瞻协议与源码提交 `a03dc39aa94f23561fbcfe71125a45c5b5202697` 已推送并核匿名可读，先于 G0 真实训练读取。旧 U/B 历史对照的 `run.py` 本地与服务器内容一致，两个旧臂与本次使用的 `all_train` 私有清单 SHA-256 同为 `72c5186559b23d650738ac1131f856c708d3bc3eaf4aaab9b5f5ae04111897a3`。新臂在 GIN 合成 256×256 GPU 检查和内置 P0 通过后启动；启动初查两臂源阶段各 100/8000 步，无 `FAILURE` 或 `EXIT`。这仅证明运行正常开始，**不代表可行性门通过**。

统一原始时钟 `1792988376.999927`（2026-10-26 12:19:37 CST）；内部 23 小时预停、外部 23 小时 15 分钟独立硬超时，严格小于 24 小时。真实 `/data_nas` NFS 挂载、每个新根的创建／写读探针、清单一致性与剩余约 10.70 TB 均通过。GPU0/GPU3 启动前分别至少 18 GiB 可用，且未终止已有进程。每根在 22×10⁹ B 前预停，合计严格小于 50×10⁹ B；全部私有权重、患者级验证、日志与缓存留 NAS。

| 臂 | create-only NAS 根 | 后台控制器 PID / start ticks | GPU |
| --- | --- | --- | --- |
| U+GIN | `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/protocols/supervised_cl_g0_u_20261026_97` | `574118 / 43183890` | `0 / GPU-bd9449f7-3c43-2a76-b00a-5c276557d14e` |
| B+GIN | `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg/protocols/supervised_cl_g0_b_20261026_98` | `574121 / 43183891` | `3 / GPU-13c88441-ea74-d6f6-ba95-4db09824e5d5` |

控制器及初见 worker 的完整可见命令行均为中性入口；`nvidia-smi` 已看到两 worker。唯一 30 分钟监测任务 `g0` 已创建，健康进展静默；终点只按冻结门分析，不自动启动第二种子或新机制。
