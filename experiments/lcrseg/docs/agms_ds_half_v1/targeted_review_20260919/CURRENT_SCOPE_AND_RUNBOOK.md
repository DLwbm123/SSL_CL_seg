# 当前授权与原 RUNBOOK 的必要补充

当前用户转发 `agms_next_experiment_plan.zip` 的 prompt，只授权定向审阅准备。先前10小时自主研究委托不再可替代本候选的新启动确认。

候选 `44c1da8a5a8a75021a0298d42bd8f9088e9b059a` 保持不变。其[原 RUNBOOK](https://github.com/DLwbm123/SSL_CL_seg/blob/44c1da8a5a8a75021a0298d42bd8f9088e9b059a/experiments/lcrseg/docs/agms_ds_half_v1/RUNBOOK.md)与[原审阅请求](https://github.com/DLwbm123/SSL_CL_seg/blob/44c1da8a5a8a75021a0298d42bd8f9088e9b059a/experiments/lcrseg/docs/agms_ds_half_v1/REVIEW_REQUEST.md)中的“收到独立批准后沿用现有委托启动”属于旧授权表述，**本补充明确覆盖这部分操作指引**。不改写旧文件，不把本文当外部批准。

## 进入生产前必须满足

1. 用户将精确候选和本目录交给独立外部审阅者；收到真实、非模板、绑定精确候选及全部 canonical digest 的批准和审阅证据。当前缺此批准。
2. 用户随后另行明确确认启动该已批准提交。当前附件不是 Prompt B；旧委托不构成这项新确认。
3. 才能依据真实新用户消息生成 checkout 外的 launch receipt，保留消息来源、时间和原文，绑定真实 review digest 与全部实际字段。`AUTONOMOUS_DS_HALF_V1` 只是冻结解析器要求的协议字符串，不能用它证明用户本次已确认，也不能据此复用旧 receipt。
4. 干净的精确候选 checkout、冻结代码/计划/环境一致；审批和日志放 checkout 外；环境不符停下，不改指纹。运行位置与私有路径通过既有私有配置解析，不能上传公开 Git。

已有心跳的授权条件已同步收紧；没有创建或恢复新监测，本轮没有 SSH 轮询。收到外部批准但没有第2项时仍不得执行生产步骤。未来条件具备时可使用原 RUNBOOK 的实际入口，无需为了保存本文件修改生产代码。

## 冻结生产顺序（本轮不执行）

| 顺序 | 原模块入口 | 预算与权限 |
|---|---|---|
| 1 | `qualify --mode cuda --config <private config>` | 原生合成11次，包括2次预设失败；不读取真实前缀 |
| 2 | `qualify --mode smoke --config <private config>` | 先验收两份原B2真实前缀，每顺序4次L-only，共8；不读U/val/test，丢弃状态 |
| 3 | `run --config <private config>` | O1/O2两个stage2，共5300；原生实际模型验收与完整成本闭合 |

保持已有 NAS 包装器和中性 dispatcher，GPU仅5/6/7，先核对实际显存、挂载与写读探针。不得 `python -O`、PYTHONOPTIMIZE、升级依赖或放宽容差。

不要再运行 `test`、`prepare` 或 `p0`。CPU尝试已用完；P0是先前26张L的历史导入，本轮新图像0、新更新0。

每次非预设失败均保留证据并停止；不复跑尾段、不扩预算、不借旧额度。若真实审阅发现需代码修复，先给窄修补方案，生成新SHA后重审；本材料不预先批准任何修补。

HALF完整结果后停止科研审阅。OBS0/ROUTE_HALF、fine降级、SCNP及其他机制均 PLAN_ONLY，不为规避审阅阻塞另开候选。
