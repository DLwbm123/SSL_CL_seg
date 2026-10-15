# MAIN_STYLE_SSL_V2 冻结主协议

2026-10-14建立，2026-10-15 00:08（Asia/Shanghai）冻结。状态：实现/工程检查/协议冻结；**NOT LAUNCHED**。只注册这一个整套方法，不做ablation。眼底和Prostate为本次遗忘修复重点，M&Ms采用同一保护框架并报告其结果，不能假定已有相同病因。

方法、tensor/detach与生命周期以本目录 METHOD_SPEC、IMPLEMENTATION_CONTRACT、ACCEPTANCE 和 machine-readable PROTOCOL.json 为准；常数由 protocol.py唯一提供。正在运行的V1所有快照、PLAN、数据、预算、过程不变。

FREEZE.json同时绑定本包和main_style_ssl_v1/five_frameworks_v1/lctx_weight_memory_v0_1三组继承源码。V2统一使用已认证c234bcd retained-rank准入修复，SVD容差/保护维度不变；不能退回原满秩依赖再运行seed164。资格检查与本地源码三个digest完全匹配。当前V1各分支原快照不受此下一版依赖冻结影响。

## 数据与预算

沿用冻结split seed0、optimizer seeds163/164、原manifest/label map；U无标签路径/hash。MRI患者先划分、phase/slice后采样，图像-only均匀切片，当前paired L/U batch2。原flip/90°/LCTX几何和style .2保留。风格上下限眼底[0,1]、MRI[-1,1]。20%warmup/20%U-ramp、EMA.99、Adam/poly、原LR/weight decay不变；新增方法明确改变目标MRI首层自由度与保护逻辑。

| 数据集 | 源→目标顺序 | 源复用更新 | 新目标调用预算 | 名义全链 |
|---|---|---:|---:|---:|
| 眼底 | REFUGE→RIM→Drishti；REFUGE→Drishti→RIM | 16000 | 21200 | 37200 |
| Prostate | RUNMC→BMC→I2CVB→HK→UCL→BIDMC | 27200 | 150200 | 177400 |
| M&Ms | Siemens→Philips→GE→Canon | 87200 | 209800 | 297000 |
| 合计 | 30名义节点：6复用source、24新target | 130400 | 381200 | 511600 |

6个source native/F精确复用健康V1 sealed，不重训；已有更新不能计为V2新增物理调用。每个source一次32批训练L记忆bootstrap+1批current bank初始化，optimizer/U/val/test调用0；额外66 L image exposures另记，两个眼底顺序共享source统计，不重复bootstrap。

每个target固定原更新上限，一次正式尝试；失败停止该分支、后代BLOCKED，其他独立分支可继续。没有自动retry、延长、额外seed、改变序列、挑best checkpoint或参数扫描。zero displacement计一次调用，不补跑。

## 评价与判定

每阶段固定final在既定seen-domain val报告全域/全类/两个种子及两个眼底顺序。眼底primary rim/cup macro Dice，aux cup/disc；另报告cup recall/precision、cup→rim/background错误。Prostate全患者3D Dice和各域前景错误；重点I2CVB后BMC、UCL后I2CVB/HK。M&Ms LV/MYO/RV、患者内ED/ES先平均，canonical320，aux25排除。距离指标仍以processed ZYX mm、empty/Infinity保留。若需新增recall/类别错误汇总，使用冻结验证输出，不能反馈调方法。

Old/Incoming/Final/Forgetting、signed site change、完整负域/缺失/失败均报告。改善只有在完整新主实验与同一source的V1链比较后才评估，不预设增益、不把减少遗忘代替总体改善，不声称因果模块贡献。独立test不读取、不调方法；开发验证结果不能冒充独立临床泛化。

当前冻结只是研究候选；缺少真实source bootstrap/GPU实际更新/overfit资格，并非训练启动授权。以后启动需要新授权回执匹配协议digest、新create-only根、当时资源/存储检查，不能覆盖V1输出。

## 结果与工程记录

保存所有提案缩小/零位移比例、G/F谱值、history support/类观察、entry foreground reference、conflict/precision、旧域和Incoming变化。新增VJP、CPU/GPU内存、L读数及真实墙钟/吞吐/物理调用另记，不声称保持相同算力消耗。高保护强度损害Incoming、MRI冻结负收益或summary失效必须保留。

项目的科学实验完成后按全局规则交付匿名结果和源码；当前只有工程交付，项目AGENTS禁止自动推送，未把私有工程文件发布到GitHub。
