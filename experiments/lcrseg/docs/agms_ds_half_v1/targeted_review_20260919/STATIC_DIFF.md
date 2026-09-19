# 精确候选的静态差异

比较 `89dc7657c4f0b7c6f51de339d0f6bc2a621da55f` → `44c1da8a5a8a75021a0298d42bd8f9088e9b059a`。这是一份实施者的只读差异核对，不冒充独立审阅。原结果提交 `ef6888dc81519ce9e9ca13ea497bbf6397747d70` 只是对照来源。

## 科学与运行变化

| 范围 | 差异和证据 |
|---|---|
| 模型 | `model.py` Git blob不变；dec3/dec2 1×1三类头共294参数；部署仍只有主网络。原生父模型/线性头、A-only保护不变。 |
| 损失 | [trainer.py:65](https://github.com/DLwbm123/SSL_CL_seg/blob/44c1da8a5a8a75021a0298d42bd8f9088e9b059a/experiments/lcrseg/agms_cl_v0_1/trainer.py#L65)、106、128三处把原常数.25替换成冻结METHOD.lambda_DS=.125，分别用于实际loss、日志及诊断；无其它trainer差异。主L、fine KL、H及constraint定义未改。 |
| 头与主网络 | DS位于同一L目标，向aux及合法主A/B同时反传；没有detach/routing。降低DS不是只缩小主网络干预。后续头参数、EMA、risk和mask也可能随轨迹改变，Adam参数更新不保证按1/2缩放。 |
| 优化器 | 辅助组初始lr .001、weight_decay4e-5和poly调度不变；主组原B2配置相同。 |
| risk/门控 | `core.py` blob不变；当前L balanced disc/bg Brier，R初始.25、EMA.9、读上一成功步R；alpha=.1+.7softmax(-R/.1)，detached。主disc≥.9、fused≥.9、方差≤.01和fine互斥不变。父类loss仍-logsumexp(rim,cup)，按合法geometry归约。 |
| EMA/提交 | [trainer.py:162](https://github.com/DLwbm123/SSL_CL_seg/blob/44c1da8a5a8a75021a0298d42bd8f9088e9b059a/experiments/lcrseg/agms_cl_v0_1/trainer.py#L162)更新流程不变：物理调用先计数，候选状态验收后提交EMA/prototypes/risk；当前teacher .99。故障账本/no-tail-replay不改。 |
| 数据/LCTX | `native_data.py`、`recipes.py`、`kernels.py` blob不变。辅助L来自同一互补mask来源收集；不能混合图直接配原标签。主头valid卷积几何不变，U辅助头无梯度；不增加骨干前向或历史memory。 |
| 随机流 | `model.py`原fork_rng和head种子namespace `AGMS_CL_V0_1/aux/seed/order/stage`保持不变，arm排除；新study名不代入headRNG。训练数据/采样RNG实现不变；参数轨迹当然不要求相同。 |
| 恢复 | `state.py` blob不变，但动态导入新STUDY/METHOD，故语义绑定包含.125和新study；恢复头/EMA/R/prototypes/optimizer/torch-Python-CUDA RNG，拒绝错身份/前缀/dose。既有CPU continuation/identity PASS作为历史证据，本轮未加载checkpoint。 |
| 协议 | `protocol.py`只保留A5两节点、重命名DS_HALF namespace，原B2两前缀/原数据划分/主options不变；独立预算24 CPU/11 CUDA/8smoke/5300formal；原A0+旧A5导入。 |
| 权限 | `authority.py`要求新study外部review、独立receipt、HEAD干净、全绑定一致、已存CPU证据。旧R2不覆盖本候选。最新用户要求另行启动确认，见授权补充；不从代码字符串推断权限。 |
| 资格 | `__main__.py`的test改为独立followup_tests，p0拒绝；`qualification.py`按arm和order选各自smoke前缀；CUDA5 continuation+4 baseline+2预设失败=11，smoke4×2=8。资格短步数和fixture偏置不进入正式配置。 |
| 执行 | `execution.py`更换有限计数与历史P0读取方式；目标循环/真实前缀检查/完整模型验收规则不变。P0只导入旧26张L结果，不新读图。 |
| 报告 | `reporting.py`加入旧A5历史对照并固定6/18/9/8覆盖，32诊断VJP；门槛只对旧A5，仍分开报告对A0差值。 |

模型/core/state和6个底座模块的精确 Git blob、全部8个源文件差异路径见 [STATIC_BINDINGS.json](STATIC_BINDINGS.json)。所有生产Python源文件按候选 CODE_MANIFEST 逐项核对，代码树digest与CPU报告一致。候选科学输入ANCHORS也逐项匹配。此类hash为本次明确要求的协议绑定验证，不是患者/模型全量重读。

## 绑定链核对

- 两节点seed163/stage2、顺序/域/步数与原A5相同，method比较只有lambda_DS变化；geometry、B2 options、manifest/split、prefixes、A0 imports、诊断时间点与原冻结计划相等。
- 两份B2 prefix metadata与固定B2公开row identity、VERIFIED proof/student/receipt匹配；每节点prefix_binding/options/method digest一致。
- A0两条及原A5两条的row/proof canonical digest与已绑定公开JSON一致；原完整结果10个VERIFIED及26500/24/36物理计数未变。
- CPU物理invocation恰为1…21，一次PASS、成本session闭合；plan/execution/code_tree绑定与候选一致。原80和更早134保留单列。这里验证保存证据，不运行测试。
- 原输入FINAL_METRICS CSV与附件字节相同，Git blob `2ab7f8ae387eadfde4d141404ff967f1449e1a45`匹配。六臂均值、附件配对百分点重算一致。

## 未作的断言

真实前缀权重的文件hash/schema/合成前向、当前CUDA机器指纹、P0新筛查和真实smoke均未执行。本轮只确认期望绑定；不把公开VERIFIED元数据冒称本轮实际模型验收。CPU记录的Python3.10.6/PyTorch2.2.1+cu121/optimize0为既有记录。

此次核对未识别出必须立即更改生产代码才能交审的新增阻塞；这不等于代码已获独立批准。已识别的旧RUNBOOK授权措辞冲突由外部文档补充明确覆盖，不能据旧委托自动启动。真实外部审阅和新用户启动确认仍缺。
