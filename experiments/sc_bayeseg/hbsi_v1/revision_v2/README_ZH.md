# 审阅修订：运行与解释

本轮仅授权源码修复和合成局部推断诊断。真实训练、真实smoke、真实验证和封存test均未执行；oracle推断结果不能作为正式方法成绩。原48文件版本可在提交5fe9b7d8查看，原80步失败报告与evidence目录未重写。

## 入口与依赖

使用仓库根包名 `experiments.sc_bayeseg.hbsi_v1.cli`；Python>=3.10、PyTorch>=2.6、NumPy，HDF5 fixture检查需h5py。无需新依赖。所有运行文件位于NAS新目录，通过既有with_nas_storage包装器启动。实际worker入口使用中性/tmp路径，输出目录、方法入口与选择由环境/脚本内容传入，不出现在进程argv。

下面列出worker内部构造的CLI参数（不是建议直接把私有路径放在进程argv）：

```text
revision-checks --output OUT --run-id revision02 --max-outer-updates 24
unit-tests --output OUT --run-id unit02 --max-outer-updates 24
fixed-parameter-qualification --output OUT --run-id oracle01 --max-outer-updates 24
  --unit-receipt OUT/unit02_UNIT_RECEIPT.json
  --revision-receipt OUT/revision02_REVISION_RECEIPT.json
```

`OUT`始终为同一轮账本目录；失败重试换attempt ID，不重置账本。源码指纹包含当前包全部.py，任何源码变化都会使旧准入回执失效。运行汇总保存退出码；计算完成与科学资格分开。不得反复运行合成诊断直到出现通过。

若复现本轮，应新建独立NAS目录并遵守同样预算；不要覆盖本轮目录。交付版本先运行18项旧回归与9项新增检查，再在冻结预算下运行一次诊断矩阵。原始 `synthetic-qualification` 是不同的80步学习实验，本轮未重跑。

中性worker可使用下面的最小入口，保存到项目目录之外的中性路径；SOURCE_ROOT和WORKER_ARGS由环境提供，后者为上面参数列表的JSON。使用既有NAS包装器的中性副本启动该入口，实际Python和入口路径也应为中性名称。这样私有输出路径不会出现在操作系统进程argv中。

```python
import json, os, sys
sys.path.insert(0, os.environ['SOURCE_ROOT'])
from experiments.sc_bayeseg.hbsi_v1.cli import main
raise SystemExit(main(json.loads(os.environ['WORKER_ARGS'])))
```

公开evidence中的unit02/revision02回执对应原执行源码。发布检查随后修正了把AppleDouble元数据计入源码指纹的问题，并补做R01/R09零更新检查。数学、优化器、预测器源码没有变化，但完整代码身份变化，所以旧回执按设计不能为交付版本的新运行提供准入。不要手改回执或补写PASS；新执行应按上述命令生成当前源码的完整回执。SOURCE_PROVENANCE.json保留了这一区别。

## 四项修复的接口

1. `qualification.write_receipt/require_receipt`：绑定当前代码、完整配置、证据集合指纹、报告内容及必要输入checkpoint。UNIT/REVISION只授予对应检查的资格；SYNTHETIC_LEARNING才能准入源训练，MODEL_LEARNING才能准入持续训练。oracle报告不能生成学习PASS。资格失败、缺失、过期或不匹配时，在训练数据manifest/HDF5读取之前拒绝。回执是本地研究流程的完整性约束，不是防恶意伪造者的数字签名系统。
2. `CurrentTrainingData.inspect_manifest`与`evidence_set`：实际观测事件与stage名称分离。图像身份不因L/U角色改变；标签拥有独立版本。记录重排、stage改名或输出目录改变不能重复吸收证据；同一全量manifest的不同域可合法进入。包含任何已消费事件的任务拒绝；新增标注但重复旧图像似然同样拒绝。增量label-only目标尚未实现，不伪装成支持。
3. `InferenceBudget`与预测诊断：local ESS>1、ESS/L>=.1、最大权重<=.5；外层ESS>1、ESS/S>=.25、最大权重<=.5；至少3次独立重复及冻结的概率/log证据波动检查。全部域保留诊断，低权重域失败也不静默丢弃。评价汇总显式DIAGNOSTIC_ONLY，过程退出0不等于推断合格。
4. `training.run_stage`与`--resume-checkpoint`：每一步完成后保存create-only活动checkpoint，异常写partial与失败回执；从最后提交的步骤恢复，而非从半个batch中途状态恢复。恢复须使用相同run-id、原协议、账本目录、步数目标、原预算秒数和最新checkpoint；不得延长原deadline或改名继续。数据采样RNG、MC RNG、optimizer和固定prior恢复。失败成本保留；更新预算按预约量保守计账，可能高于实际完成更新。

旧schema1 checkpoint仍能读取，但若缺少历史观测事件身份，不能直接进入新的正式持续训练；需要独立审计迁移。本轮没有迁移或修改旧checkpoint。

## 白化与积分

默认局部proposal仍为原对角族；`prior_aware_init=true`只改变尺度输出初始化，均值头保持相同初始化。`local_parameterization="sar_whitened"`将S输出解释为u，通过S=F(Pi)+sigma A^-1 u构造场，并在log q中加入C log|A|−CHW log sigma。生成密度、类别、噪声和训练目标不改变。

默认整卷积分改为factorized：对每个固定域/Theta/a样本，先分别平均各切片重要性权重，再相乘得到整卷证据；各切片局部加权概率通过整卷外层权重组合。`integrator="paired"`保留旧配对积分供比较。两者均保留域证据常数；因子化通过全Cartesian参考不等于神经重要性采样已合格。

新数学/工程结果、失败记录、固定参数诊断和限制见REPORT_ZH.md及同目录JSON。公开摘要不包含真实数据、原始预测数组、checkpoint或私有运行路径。
