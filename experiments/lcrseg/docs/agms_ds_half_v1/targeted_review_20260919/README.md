# AGMS DS HALF 定向审阅交付

状态：**STOP_AWAITING_EXTERNAL_CODE_REVIEW**。本目录是独立文档交付，**不是外部批准，不是启动确认，也不是新的生产候选**。

唯一候选：`44c1da8a5a8a75021a0298d42bd8f9088e9b059a`。原执行：`89dc7657c4f0b7c6f51de339d0f6bc2a621da55f`；原完整结果：`ef6888dc81519ce9e9ca13ea497bbf6397747d70`。本目录所在文档分支/提交只提供审阅补充，生产仍须绑定上述候选，不能在此文档 checkout 运行。

## 阅读入口

1. [当前授权和 RUNBOOK 补充](CURRENT_SCOPE_AND_RUNBOOK.md)：优先于旧 RUNBOOK 的自主启动措辞。
2. [静态差异与证据清单](STATIC_DIFF.md)：模型、训练、risk、EMA、数据、随机流、恢复、权限。
3. [已有结果与预先固定的解释](INTERPRETATION.md)：原阴性结果、逐域归因、HALF 三层判读。
4. [机器可读绑定](STATIC_BINDINGS.json)与[聚合重算](RECOMPUTED_ANALYSIS.json)。状态 STATIC_METADATA_MATCH_ONLY 不能替代原生 CUDA 或真实验收。
5. [用户附件 prompt](attachment/02_PROMPT_SCOPE_AND_REVIEW.md)、[实验计划](attachment/01_EXPERIMENT_PLAN.md)及[仅提案的路由机制](attachment/03_ROUTING_FUTURE_IMPLEMENTATION.md)。附件 `.py` 保存为 `.py.txt`，未执行。

## 不变的科学矩阵

| 节点 | 起点 | 当前域 | 正式更新 |
|---|---|---|---:|
| A5 DS=.125 / seed163 / O1 / stage2 | 原 B2 已学 RIM 的 stage1 | Drishti_GS | 2100 |
| A5 DS=.125 / seed163 / O2 / stage2 | 原 B2 已学 Drishti 的 stage1 | RIM_ONE_r3 | 3200 |

共2节点5300；A0和原A5共4终点历史导入。source/stage1新增0。辅助头学习率仍 .001，只有 DS 系数 .25→.125；这会同时影响头与主网络。不得从原 A5 终点续训或借用别臂 warmup。

## 精确绑定（canonical JSON SHA256）

| 字段 | 值 |
|---|---|
| `reviewed_code_commit` | `44c1da8a5a8a75021a0298d42bd8f9088e9b059a` |
| `code_tree_sha256` | `4a21f513279168da5c4a634a22a3e5042cd1fa3befb19d99e298084862486ff9` |
| `plan_sha256` | `ef14fc04f97d70fd8d11f38afd517c4875f7e6479674c6327469b80b581c94b8` |
| `prefix_sha256` | `40bb9a388261eaf5bcd937bb396e6d187eb28f9ccb436473cb8d7ae571fa5f96` |
| `import_sha256` | `82e3ba89e0cff483b074555bf4c48f2676f2f78073725df10d8d3afb083eb1a7` |
| `environment_sha256` | `c2def192e642a075e8bd9f3a53fa8bfc2911dcff1050e43430e4218ad7e9f039` |
| `execution_sha256` | `100e31461acec01663ec1e75549a789bb59573d992fd9b3568d56e777b1586d9` |

环境绑定是完整 wrapper 的 canonical digest；其中 fingerprint 自身的 digest 是 `2791c0dccdb576084d42eb4001ab3067294f51f87777f6b681d940f9f2c72757`，两者不可混用。本轮核对的是保存的期望环境和既有 CPU 记录，没有远程实时环境探针；环境及真实前缀张量验收仍 PENDING。

候选的 [PLAN](https://github.com/DLwbm123/SSL_CL_seg/blob/44c1da8a5a8a75021a0298d42bd8f9088e9b059a/experiments/lcrseg/docs/agms_ds_half_v1/PLAN.json)、[EXECUTION_PLAN](https://github.com/DLwbm123/SSL_CL_seg/blob/44c1da8a5a8a75021a0298d42bd8f9088e9b059a/experiments/lcrseg/docs/agms_ds_half_v1/EXECUTION_PLAN.json)、[PREFIX_BINDINGS](https://github.com/DLwbm123/SSL_CL_seg/blob/44c1da8a5a8a75021a0298d42bd8f9088e9b059a/experiments/lcrseg/docs/agms_ds_half_v1/PREFIX_BINDINGS.json)、[IMPORT_BINDINGS](https://github.com/DLwbm123/SSL_CL_seg/blob/44c1da8a5a8a75021a0298d42bd8f9088e9b059a/experiments/lcrseg/docs/agms_ds_half_v1/IMPORT_BINDINGS.json)、[ENVIRONMENT_BINDING](https://github.com/DLwbm123/SSL_CL_seg/blob/44c1da8a5a8a75021a0298d42bd8f9088e9b059a/experiments/lcrseg/docs/agms_ds_half_v1/ENVIRONMENT_BINDING.json)、[CODE_MANIFEST](https://github.com/DLwbm123/SSL_CL_seg/blob/44c1da8a5a8a75021a0298d42bd8f9088e9b059a/experiments/lcrseg/docs/agms_ds_half_v1/CODE_MANIFEST.json) 均以精确 SHA 定位。

## 已有测试和本轮成本

已有 HALF CPU：21物理调用、1次尝试 PASS，包含2次预设 after_optimizer 注入。上限24不是额外重试许可。既有 AGMS CPU80、更早134单列保留。

[CPU报告](https://github.com/DLwbm123/SSL_CL_seg/blob/44c1da8a5a8a75021a0298d42bd8f9088e9b059a/experiments/lcrseg/docs/agms_ds_half_v1/CPU/TEST_REPORT.json)、[物理账本](https://github.com/DLwbm123/SSL_CL_seg/blob/44c1da8a5a8a75021a0298d42bd8f9088e9b059a/experiments/lcrseg/docs/agms_ds_half_v1/CPU/PHYSICAL.jsonl)、[尝试记录](https://github.com/DLwbm123/SSL_CL_seg/blob/44c1da8a5a8a75021a0298d42bd8f9088e9b059a/experiments/lcrseg/docs/agms_ds_half_v1/CPU/ATTEMPTS.json)、[闭合成本 session](https://github.com/DLwbm123/SSL_CL_seg/blob/44c1da8a5a8a75021a0298d42bd8f9088e9b059a/experiments/lcrseg/docs/agms_ds_half_v1/CPU/costs/0001/session.json) 已通过静态绑定、计数和一致性核对。本轮没有重新调用测试 suite，不能称为重新通过原生测试。

本轮新增 optimizer=0、VJP=0、真实患者/权重读取=0；没有启动训练或新监测。未来 CUDA11、L-only smoke8、正式5300、真实总上限5308、诊断32 VJP 均 PENDING。

## 导出覆盖与复算

| 文件 | 行数 | 含义 |
|---|---:|---|
| [ENDPOINTS.csv](ENDPOINTS.csv) | 12 | 原12终点、逐顺序指标及对A0差值 |
| [DOMAIN_DETAIL.csv](DOMAIN_DETAIL.csv) | 36 | source/首目标/当前域，rim/cup/disc_union及对A0差值 |
| [SUPPORT.csv](SUPPORT.csv) | 12 | 已记录 active-U 全程累计支持；A0为NA |
| [GRADIENT_POINTS.csv](GRADIENT_POINTS.csv) | 480 | 40既有点 × 4项 × 3参数组，无新增VJP |
| [RISK_POINTS.csv](RISK_POINTS.csv) | 40 | 原固定点risk/alpha、训练L分歧；同状态U对照NA |

这些是对原公开结果的派生分析，不替代原12/36/27/40报告。HALF未来原生报告仍应6 Final /18 domain /9 paired /8诊断点；完成后另按解释文档逐域对比，不能将本次旧数据表当作 HALF 结果。

附 [STATIC_RECOMPUTE.py.txt](STATIC_RECOMPUTE.py.txt) 为 stdlib 静态复算工具，只读精确候选 checkout 和附件，输出到另一个目录。它不导入项目或 PyTorch、不运行 optimizer、不访问服务器；使用方法见文件首部。本轮一次静态复算所有断言通过。文本后缀用于保持生产代码树定义不变。
