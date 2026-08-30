# Gate 1B v2 独立审阅报告

日期：2026-08-30。分支：`codex/gate1b-v2-null-aware-transport`。

## 结论

**`FAIL_TRANSPORT_NOT_SUPPORTED`**。B1、B2、B6、B7 通过；**B3、B4、B5 未通过**。
这是完整、数值有效的机制诊断失败，不是运行中断或工程阻塞。

`selected_transport=T0_identity`，角色为 `DOWNSTREAM_FALLBACK_ONLY_NOT_EXECUTED`。
T0 已作为本次诊断 comparator 评价；没有运行任何后续 fallback 实验。T1 不用于挽救 T2。
Gate1A v2 的 `PASS_MULTI_MODALITY_SUPPORTED` / `selected_K=2` 原样保留。

本轮结果支持的有限结论是：固定 T2 能降低当前域 held-out 成对特征误差，
但没有满足历史原型改善及分类保持的全部预注册要求。不能据此宣称 transport 主机制准入，
也不能将 prototype-only accuracy 当成分割 Dice/IoU 或方法最终性能。

## 主路径与执行覆盖

本轮仅Fundus seeds0/1/2，固定3类，沿用冻结域序REFUGE→RIM_ONE_r3→Drishti_GS及原case split。
固定 B0 previous-stage EMA → B0 current-stage EMA、UNet `decoder.dec1` post-ReLU16-D、K=2、官方3×3 classifier。
没有切换 C0/student，没有重跑 Gate1A、重选 K 或延长 K5 迭代。Operational prototypes 仅取冻结的
27 个 B0-EMA K2 original-fit records /54 个 centers；全部 converged、两中心 active、有限且单位范数。

| 项目 | 实际结果 |
| --- | --- |
| exact-code tests | **77/77 PASS**，0 failed /0 skipped；覆盖44类要求及真实只读集成 |
| 真实只读集成 | seed0 的预定 RIM /Drishti case，各2048坐标；null均为0；模型和checkpoint不变 |
| transport plan | 312 case records，638,976 paired rows，12/12 fit/holdout units |
| paired support | AA=638,974；A_NULL=0；NULL_A=2；NULL_NULL=0；没有删除或替换null |
| 拟合前屏障 | 全部cache/input/coordinate/model/checkpoint核验通过；屏障前transport更新=0 |
| transport | 6/6完成；T2各1000次更新，合计6000；T0/T1更新=0 |
| trajectory | 每组step0+1000 post-update rows；合并trace共6006行 |
| evaluator | 9/9 historical-val immediate/chain units完成；27个K2 oracle fits、135个restarts |
| numerical / oracle warnings | 无NaN/Inf、无invalid output；oracle convergence/inactive-slot警告0 |
| immutability | 21次完整model/classifier/GAS/buffer守卫通过；9个B0 checkpoint磁盘SHA不变 |
| segmentation optimizer | **0更新**；实际提取/evaluator阶段禁止构造optimizer；T2只优化272个W/b参数 |
| 运行 | 正式进程exit0；21:13:47–21:17:38（Asia/Shanghai），约230.58秒 |

两张RTX3090承担只读paired/oracle提取；六组T2按预注册使用CPU float64 /单线程BLAS并行拟合。
运行结束后已核实两卡均0%利用率、1MiB显存，没有残留诊断/训练进程。

完整坐标计划SHA-256、每个case的support、cache shape/dtype/SHA、原始权重hash、checkpoint身份、
逐步loss和最终W/b/R均保存在原始产物中。2个NULL_A保留完整罚项2；B1/B2始终用full support error。

## B1–B7

表内百分比仅为展示舍入；判定使用完整float64值，未添加准入容差。B1先对同一transition的
三个seed的error取均值，再计算相对reduction；不是先计算逐seed百分比再平均。

| Gate | 预注册要求 | 实际值 | 判定 |
| --- | --- | --- | --- |
| B1 | 两个transition各自的full-support mean error reduction≥15% | 01：48.2649%；12：44.3384% | PASS |
| B2 | 各transition至少2/3 seeds严格改善 | 01：2/3（seed0、2）；12：3/3 | PASS |
| B3 | 12个immediate foreground units等权angular mean reduction≥10% | **6.92505%**；0.2495489872953954→0.232267583726251 radians | **FAIL** |
| B4 | 每个immediate foreground unit相对angular worsening≤5% | 2/12超标；最大**35.19047%** | **FAIL** |
| B5 | 每个immediate/chain unit三类macro accuracy绝对下降≤0.005 | 2/9超标；最大**0.02501461988304088**，即2.50146个百分点 | **FAIL** |
| B6 | 每seed/foreground-class chain相对identity angular worsening≤5% | 6/6满足 | PASS |
| B7 | 全部defined features/parameters/gradients/losses/outputs/SVD/metrics有限 | PASS；独立复核一致 | PASS |

### Full-support held-out error：三种子均值

| Transition | T0 identity | T1 Procrustes | T2 residual full linear |
| --- | --- | --- | --- |
| 01：REFUGE→RIM_ONE_r3 | 0.021627976823300834 | 0.016757159275826064 | 0.011189261679320597 |
| 12：RIM_ONE_r3→Drishti_GS | 0.006605478088795815 | 0.005406301714846927 | 0.003676713349901207 |

特别披露：seed1 /transition01 的T0 error为`3.911884838494807e-17`，T2为
`7.833799162820215e-09`，因此该seed不满足严格改善。该接近identity的单元没有被剔除、
没有使用epsilon制造相对增益、没有追加训练或替换checkpoint。

### B4的两个失败单元

均为seed0 /transition12，即RIM历史原型映射到stage2空间；class IDs沿用冻结协议。

| Foreground class | T0 angular error | T2 angular error | 相对worsening |
| --- | --- | --- | --- |
| 1 | 0.08279651325128082 | 0.11193299843682009 | 35.19047365812659% |
| 2 | 0.331088701953901 | 0.3663022663717634 | 10.635688928692388% |

### B5的两个失败单元

accuracy是原始case-equal /class-equal的三类prototype-only macro；null query计incorrect。

| 单元 | T0 accuracy | T2 accuracy | 绝对下降 |
| --- | --- | --- | --- |
| seed0 /REFUGE→stage1 immediate | 0.7024937343358397 | 0.6873391812865496 | 0.015154553049290143（1.51546个百分点） |
| seed0 /REFUGE→stage2 chain | 0.7017314118629908 | 0.6767167919799499 | 0.02501461988304088（2.50146个百分点） |

完整T0/T1/T2的9-unit accuracy、per-class/foreground angular metrics、directional-conditional statistics、
chain结果及18组单map spectra均在原始JSON/CSV中，未选择性删除失败单元。

## 泄漏与独立复核

transport fit API只接受current-domain `train_unlabeled` paired features、原始weights和support masks。
不接受image/label/val/test对象。原型sentinels只检查输出有效性，不进入loss或梯度。
Historical-val coordinates/class membership/weights仅用于所有六个map冻结之后的
`gt_consumer=diagnostic_evaluator_only` oracle。没有用oracle选择step、超参数、checkpoint或operational bank。

`hidden_gt_training_usage=none`；`test_gt_usage=none`。
独立postrun复核逐一验证**258个正式产物 /620,745,586 bytes**，重算36组fit/holdout×T0/T1/T2
的feature metrics，最大绝对差**0.0**；独立B1–B7不等式与正式判定完全一致。
没有新增model forward、transform fit或oracle fit。

正式manifest SHA-256：
`26e69d13935133b1cfa4e3176ff5555ba8bef73755fd8fe3c0157505a92e0ea2`。
公开副本94个正式文件全部逐字节核验；165个原始npy和既有冻结geometry plan副本留在服务器，
公开清单保留其完整描述与SHA。新的shared transport coordinate plan已纳入公开产物。

开发期首次新增端到端mock测试有1次失败，已保留原始JUnit并在
[开发验证记录](GATE1B_V2_DEVELOPMENT_VALIDATION.md)说明；正式attempt无重试。
Gate1A v2原有K5警告与全部历史文件保持不变。本轮oracle无收敛警告。

## 身份、文件与停止点

| 身份 | Commit SHA |
| --- | --- |
| 分支基点 /Gate1A v2报告 | `9b2ffd04c7a8e9da73f08edb0760be3f269065d8` |
| Gate1A v2 freeze | `58f19e968700bd7708ec00e44a11759b48ce756f` |
| Gate1B v2 preregistration | `b20f186deff287843f3c9f18bf4ab5633908f441` |
| 独立execution authorization | `c6f72b86fdfa3683a6e2c7dbf593f73cab74c592` |
| exact diagnostic code | `f2a3ed7476323119b1a4fa22481b44038bc4148c` |

以上每一步均独立push并核验远端；真实集成发生在exact code核验之后。
报告commit为首次加入本报告与正式字节产物的commit，单独记录在
`GATE1B_V2_PUBLICATION_RECEIPT.json`，避免把自身commit SHA写入自身内容的循环。

- [正式状态](GATE1B_V2_STATUS.json)
- [完整原始产物目录](gate1b_v2_results/gate1b_v2_f2a3ed7476323119b1a4fa22481b44038bc4148c_attempt1/)
- [原始诊断JSON](gate1b_v2_results/gate1b_v2_f2a3ed7476323119b1a4fa22481b44038bc4148c_attempt1/TRANSPORT_FEASIBILITY_DIAGNOSTIC_V2.json)
- [独立postrun复核](gate1b_v2_results/postrun_f2a3ed7_attempt1/GATE1B_V2_POSTRUN_AUDIT.json)
- [公开副本及远端保留文件清单](gate1b_v2_results/postrun_f2a3ed7_attempt1/GATE1B_V2_PUBLIC_COPY_MANIFEST.json)
- [Exact commands](GATE1B_V2_EXACT_COMMANDS.md)

远端正式目录：
`/root/LCRSeg/runs/di_dmpa_gate1b_v2/b20f186deff287843f3c9f18bf4ab5633908f441/gate1b_v2_f2a3ed7476323119b1a4fa22481b44038bc4148c_attempt1`。

**`STOP_FOR_INDEPENDENT_REVIEW`**。
`method_registered=false`；`di_dmpa_training_launched=false`；`Gate1C=false`。
没有执行reliability、gradient-conflict、teacher-noise、theory final、Gate2、Prostate、MnMS、full sweep或main merge。
