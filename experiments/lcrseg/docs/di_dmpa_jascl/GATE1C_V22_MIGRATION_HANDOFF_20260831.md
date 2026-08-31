# 当前实验报告与 jiangsuiyang 迁移交接

日期：2026-08-31（Asia/Shanghai）。仅限当前 JASCL / DI-DMPA 医学半监督持续分割；B0/C0 是对照。

用户明确报告原算力不可用，要求同步代码与已有结果至 GitHub，准备迁往 jiangsuiyang。
本次是**代码、脱敏结果和证据缺口交接**，不是新实验、旧 full 成功关闭或方法复现完成。
原节点定时跟进已暂停；没有探测、修改或部署目标服务器，没有启动新 worker。

## 1. 目前可以下的实验结论

| 阶段 | 已有证据与结论 | 不能推出什么 |
| --- | --- | --- |
| Gate0 v2 对照 | C0/B0 各三种子完成；半监督梯度/工程门槛通过，但 B0 无平均提升 | 不是 DI-DMPA 最终性能或论文复现成功 |
| Gate1A v2 | 多模态门槛通过，B0-EMA / K=2 已冻结 | 不重新选 K，不据此认可 transport |
| Gate1B v2 / 原总体 Gate1 | `FAIL_TRANSPORT_NOT_SUPPORTED`，B3/B4/B5 未通过 | C 面板、数值修复或迁移不能挽救旧失败 |
| Gate1C v2.1 attempt1 | `BLOCKED_INCOMPLETE_EVIDENCE`；9/9 验证单元，draw0 仅 25/72，指定 R2/class-balanced 梯度分解失败 | 不是完整 Gate1C 通过 |
| 旧 v2.2 精度 pilot | 旧代码 `7fdd431` 上 75 次前向通过，已封存 | 不替代后来的新 exact-code integration |
| 新 v2.2 exact-code integration | `1cfd823` 上 210/210 合成测试及独立的新 75 次真实前向通过 | 仅三对、三个阶段、两个种子；无 C1-C8 科学准入结论 |
| 新 v2.2 full attempt1 | 确实已启动；SSH 观测中断后状态未知 | 未确认实际退出、完整计数、产物或 C1-C8；不判成败 |

Gate0 的最终阶段、三个已见域等权平均（百分数；± 为三种子样本 SD）：

| 指标 | C0 | B0 | 配对 B0−C0（百分点，均值±SD） |
| --- | ---: | ---: | ---: |
| Mean Dice（含背景） | 74.28 ± 3.87 | 73.80 ± 4.95 | −0.48 ± 1.60 |
| Foreground Dice | 62.40 ± 5.72 | 61.78 ± 7.23 | −0.62 ± 2.34 |
| Mean IoU | 63.63 ± 3.83 | 62.97 ± 4.79 | −0.66 ± 1.27 |

这是已冻结对照结果的汇总，未重新评价或调参。完整来源：
[Gate0 报告](GATE0_V2_FINAL_REPORT.md)、[配对结果](C0_VS_B0_PAIRED_COMPARISON.json)、
[Gate1A](GATE1A_V2_FINAL_REPORT.md)、[Gate1B](GATE1B_V2_FINAL_REPORT.md)、
[旧 v2.1](GATE1C_V21_EXECUTION_PROGRESS.md)、[旧 pilot](GATE1C_V22_PRECISION_PILOT_REPORT.md)。

Gate1B 的主要失败仍是：历史前景 angular reduction 6.92505% < 10%；
2/12 单元 angular worsening 超过 5%（最大 35.19047%）；
2/9 单元 prototype-only macro accuracy 下降超过 0.005（最大 0.02501461988304088）。
这不是分割 Dice/IoU 指标，且不是工程中断。

## 2. 精确代码和不可变血缘

| 身份 | Commit |
| --- | --- |
| 新 v2.2 实际执行代码 | `1cfd8235293e157afd6b40f0f091ce6bc6df9f9f` |
| 完整执行预注册 | `9593908bd36f7f833e385a70b2b772b7a8c84d22` |
| 独立有条件授权 | `aabef38c473f281bef7717e77fa326a542266d76` |
| 原数值引擎 / 旧 pilot 代码 | `7fdd4312278eb64dbfb471107bb47e6b897c6859` |
| 官方 JASCL | `3c93ca70784fc3a1d2a887f8d7dce5af6bc75f53` |

[执行预注册](DI_DMPA_GATE1C_V22_EXECUTION_PREREGISTRATION.md) MD / JSON SHA256：

```text
01a24ebfbb92db87a00263f1fd9e84262f730a8d64e128a7cd4e1cb72227246a
a7ed480aac09cbf9fdf5fe723f4d236ec128e4619eb490a9005c32b81beba6f0
```

[独立授权](GATE1C_V22_EXECUTION_AUTHORIZATION.md) MD / JSON SHA256：

```text
c59123607189f5ff2bad5206bf5b9fc792f93e045f318f223457b8b4e6c99715
ff1cb6b4175f176457b9f60cd12d5d62c74daaa84b31bf7e4f42c31d72734b60
```

报告提交推进 `codex/sslcl-long-running-reproduction`，不把报告 HEAD 当作被测代码。
这是用户本次同步请求所授权的发布，不伪造旧 run 关闭证据。
如果旧 controller 在观测之外仍运行，分支 HEAD 变化可能使其后续精确发布守卫拒绝；
不能据此补写旧 run 的实际退出码或科学结论。
不更改 `main`、旧诊断分支、三个数值引擎、五个科学判定函数或重复 CLI 拒绝保护。
所有历史 `launch_ready=false` 字段保持原样。

## 3. 新 exact-code integration：已完成的真实结果

精确代码合成测试：210/210 PASS，0 failure/error/skip，约 25.63 秒；实际 SSH 命令 exit 0。
排除了旧 opt-in `test_real.py`，不把合成测试称为真实集成。
JUnit 原始 SHA256 为 `edfb4f6b103c3a3b77a34e26bdb111bd3b585ba6c2c68c41688e1891893871db`。

| 阶段 | 新原生前向 | 新 FP64 前向 | 原生 autograd | FP64 autograd |
| --- | ---: | ---: | ---: | ---: |
| draw0 | 9 | 6 | 27 | 99 |
| noise | 27 | 6 | 171 | 171 |
| posterior | 9 | 6 | 27 | 27 |
| poe | 6 | 6 | 51 | 69 |
| 合计 | 51 | 24 | 276 | 366 |

新前向合计 75，12 个新模型守卫、288 个 global 比较、630 个 class-component 记录、
12 个监督 global 比较、30 次 PAS 全部通过；零 optimizer 更新。
目标相对 L2 最大值 `0.00012594988782098875`，目标 cosine 最小值 `0.999999992160753`；
监督相对 L2 最大值 `4.2237250813564015e-06`；component-sum 残差最大值 `4.440892098500626e-15`。
所有未舍入逐项结果、阶段屏障、模型守卫和输入审计见
[集成公开副本](gate1c_v22_results/9593908/integration_attempt1/)。不做平均救援或新的科学判定。

495-cache 数值/来源审计通过：9 单元、72,990,720 pixels、4,856,574,421 bytes；
保留 `seed2/stage0/REFUGE_train_n0038` 的单个 null `[[185,180]]`。
`cache_reuse_approved=true` 是原实例此次审计的历史结论，不是新服务器的就绪状态。
9 个 validation guards 和 990 次 validation forward 明确属于旧 v2.1 复用，不能加成新的前向数。

集成实际 SSH exit 0 后，另行核验 controller 退出、全部新文件及 1,686 个外部引用
（5,089,788,401 bytes）；本机又完成了独立归档核验。
新集成原始 manifest SHA256：`1dcbf3199afa0df92a4d5ffcc69940c7b4ef235e7b1dd3d425a31952eff48c0c`。
102 个索引文件 / 126,389,395 bytes；包括 manifest 及其 SHA 文件共 104 文件 / 126,403,893 bytes。
本次迁移发布前再次只读核验全部本地文件 SHA/bytes，没有加载数组或 checkpoint tensor。

## 4. 正式 full：已用过的 attempt，结果仍未知

原 full 于远端 `2026-08-31T03:55:48.613808Z` 启动。
最后成功远端快照为 `04:01:15.835003Z`：输入与缓存审计 PASS，两个 CPU worker
正在重算 validation metrics；当时尚无阶段屏障、draw0 或终态。
随后原 SSH session 返回 **255（传输连接断开）**，不是 controller 的实际退出码。
后来重连也未成功；用户现报告原算力不可用，不再据旧快照声称 GPU/进程/存储现状。

正式完成的 forward 数、C1-C8、实际退出码与最终 manifest 均未知。
原计划的 72 对、三种子、八 draw、四阶段、1,800 新前向只是预算，不能登记为完成量。
两个真实 attempt（integration、full）机会都已使用，不能在新主机复跑同一 attempt，
也不能以旧目录不存在、SSH 255 或内部 completion 代替独立退出证据。

[启动和独立观测收据](gate1c_v22_results/9593908/records/)保留原始事实。
其中旧 `next_action`、机器路径和执行命令是历史记录，不是迁移后的启动许可。

## 5. GitHub 有什么，仍缺什么

| 内容 | 本次交接状态 |
| --- | --- |
| 当前源码、配置、tests、冻结协议、历次脱敏结果 | 在指定分支保留；本次不改实验实现 |
| 新集成输出 | 本机完整 104 文件；公开 89 个文本副本（85 JSON、2 MD、1 XML、1 SHA 文件） |
| 新集成 6 个 NPZ / 118,181,696 bytes、9 个 log / 1,740 bytes | 仅留本机私有归档，不进入 GitHub |
| 旧精度 pilot | 本机已有经核验的 66 文件 / 122,825,857 bytes，原公开报告保留；不重复归档 |
| 必需外部引用 | 需另行私有保留 567 文件 / 4,932,630,373 bytes，含 495-cache；**尚无完整本地副本** |
| full 原始输出与真实退出证据 | **尚未取回 / 未核验**，不在此次备份完整性声明内 |
| 其他历史 Gate 的原始数据/权重 | 不因公开报告存在就视为已备份；本次没有声称全部原始历史产物齐备 |
| jiangsuiyang 身份、环境、GPU、磁盘与数据 | 尚未检查或部署；不宣称 ready |

[公开副本清单](gate1c_v22_results/9593908/PUBLIC_COPY_MANIFEST.json)逐项保存原文件 SHA/bytes 与公开副本 SHA/bytes。
脱敏只替换实例地址、hostname、本机用户目录；数值、数据角色和所有旧失败保持不变。
原 manifest 内的哈希仍指原文件；检查脱敏副本须用公开副本清单，不能混用两套哈希。
缺失私有引用的逐项路径/bytes/SHA 已在
[CACHE_REUSE_AUDIT.json](gate1c_v22_results/9593908/integration_attempt1/CACHE_REUSE_AUDIT.json)
的 `references` 中，按 `retain_private_copy=true` 选取；清单存在不等于文件已取回。
机器可读最新状态见 [MIGRATION_STATE.json](gate1c_v22_results/9593908/MIGRATION_STATE.json)。

本次公开校验已通过：89 个集成文本副本、5 个独立收据副本、2 个新交接/校验文件，
合计 96 个索引文件 / 8,257,417 bytes，另附公开清单自身。94 个副本逐项与原件比较，
仅 82 个文件发生声明的字符串脱敏，其余逐字节相同；没有改变原始私有归档。
公开校验脚本开发首轮因监督比较没有 `precision_comparable` 字段而报 KeyError；
已按原 schema 分开检查标志和原数值门槛，最终校验通过。该修正仅影响新增的只读发布校验器，
不改原结果或实验实现，亦不计作重跑 210 项 Torch 测试、集成或 full。

## 6. 在 jiangsuiyang 接续前的有限步骤

先只取得源码和公开报告。以下 clone 使用新的目录名，不覆盖已有 checkout；若目录已存在先核对，勿 reset/clean：

```sh
git clone --single-branch --branch codex/sslcl-long-running-reproduction https://github.com/DLwbm123/SSL_CL_seg.git SSL_CL_seg_jascl_handoff
cd SSL_CL_seg_jascl_handoff
git log -1 --oneline
python3 experiments/lcrseg/docs/di_dmpa_jascl/gate1c_v22_results/9593908/verify_public_snapshot.py
```

该校验只读公开文件，不访问 GPU、数据、GT、checkpoint 或网络，不启动实验。
历史路径候选 `/home/jiangsuiyang/SSL_CL/code/SSL_CL_seg` 与 `/home/jiangsuiyang/SSL_CL/runs`
仅来自旧项目记录，**不是本次已确认的目标服务器或可覆盖目录**。

后续须先只读确认目标主机/用户、完整进程、空闲 GPU、挂载与容量、现有解释器，以及
冻结数据/split、9 个 B0 checkpoints、bank 和缓存来源是否齐备；不抢占任务或重装已有环境。
原运行环境为 Python 3.10.21、Torch 2.2.1+cu121、两张 RTX3090；实际 metadata 保留在公开副本。
官方 [JASCL 源码审计](CODE_AUDIT.md)绑定上表 upstream commit，完整 third-party mirror 不在本仓库。
现有 `experiments/lcrseg/scripts/sync_to_jiangsuiyang.sh` 仅用于 data bundle，不负责这些私有 run/cache/checkpoint 的完整迁移。

原协议绑定 `/root/LCRSeg`、原资源预算、一次 integration/full 和精确发布守卫。
**不能只改路径、GPU、环境或关闭守卫后照旧 run**；本次报告 HEAD 推进后也不能冒充旧执行绑定。
如私有输入能从用户既有备份恢复，先逐项 SHA/bytes 验证；若无法恢复，明确保留缺口。
任何新真实执行、重算缓存或替换输入都必须另作前瞻版本/有限预算与执行授权，不能追溯修改本次注册或旧失败。

所有 method/transport/prototype 训练注册开关继续 false，R4 unavailable；
legacy PAS 仍仅 `RECONSTRUCTION_SUPPORTED_NOT_HISTORICAL_HASH_VERIFIED`，其 400 次 baseline 恢复更新不算新方法训练。
未来性能实验还须前瞻确定论文/commit、UNet 医学适配差异、benchmark/split、主指标、数值容差和多种子预算。
工程通过、同步成功和迁移就绪均不等于方法复现成功；Goal 不标 complete，不新建任务或实验 worker。
