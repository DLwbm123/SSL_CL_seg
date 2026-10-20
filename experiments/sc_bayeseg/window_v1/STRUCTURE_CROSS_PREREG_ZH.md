# Structure-only补充：完整四格与固定旧数值init1

2026-10-20，用户明确要求在原实验完成后执行。此协议追加，不修改structure-only已冻结的12参数干预、同规则init1及其结果。先核验原实验COMPLETE/EXIT0及独立审计通过，才启动此补充。

## 目标与来源

旧F0为P2/spatial_generator S0600，cursor8675；新F1为structure_only S0200，cursor8875。所有100全局模型及Adam/prior只读，0optimizer/0模型应用，只原两canonical训练L，128²。不读开发保留、U、目标、隐藏test，不加训练seed，不训练第二份F。原P1/P2/structure-only成本保持原样，不重复计为新增。

现有q0是P2各候选各初始化的200步终态，q1是原structure-only在F1下重推断所得终态。保留原三格B00=F0q0、B10=F1q0、B11=F1q1；新增只读B01=F0q1，补全两初始化×五候选。原init1的q1来自F1(Z)规则，标为动态／同规则初始化，不称数值相同。

## 固定数值init1

五候选GT/BG/CROSS/SHIFT/DELETE2，各直接加载P1/P2对应候选的init1/S0000完整五字段qxm及RNG；不重新用F1(Z)构造，不从旧200终态接着优化。每候选在F1下200次新尝试，共1000批量局部反向（两图工作量2000），最多8000候选，0全局更新。init0已数值匹配，直接复用，不重跑。

完全复用原Cholesky变分族、正规B下界、2Theta×4local、每步fresh原RNG、候选回放同一步、loss负逐图和、空间梯度×128²、lr.05、八值alpha、逐图不降1e-6且均值升>1e-5门限。每25完整局部q/RNG/cost，共45状态，初态逐张量相等，45完整RNG与旧轨迹配对。拒绝不等于新接受，所有尝试及失败成本保留，不盲重跑。

新增q1_numeric在F1和F0下均读出。init0四格复用现有q1；init1同时给出dynamic与numeric两种四格，不能选择较好者替代原报告。

## 独立配对评价与账本

同P2四流×32Theta×16local、各模型共享标准噪声，Theta/local分流。新增dynamic_old_F 10批量状态、numeric_new_F及numeric_old_F各5状态，共40960密度。注意每状态2048调用，20状态为40960，不重复旧三格评分。初始仅回放旧F旧q首Theta的10×16=160调用检查≤1e-6。

预算：1000梯度×8=8000，最坏8000候选×8=64000，终点评价40960，回放160，合计113120≤150000密度。真实draw/candidate/reverse/accepted/density各账本独立连续；实际成本报告，不把上限当实际。0额外诊断反向。初终模型、历史账本、源q逐项不变，所有局部状态有限且Cholesky对角正、原始/终态RNG和EXIT核验。

对Gamma和GT−BG/SHIFT/DELETE2报告四格原分数、F固定q效应M10−M00、局部效应M01−M00、交互M11−M10−M01+M00、总效应M11−M00及重推断M11−M10。Theta内先平均local，再对128组逐draw差值算配对MC SE/95%区间和四流；不是患者置信区间。两初始化及两图完整报告，保留原1e-5判别门限。数值init1与dynamic差异同样配对。固定q时image/appearance/entropy应相同，只有shape直接变化。均值与logsd参数干预量另外从已有checkpoint只读统计。

## 运行和结论边界

create-only新目录和中性入口，所有运行文件在既定数据盘。预计新增<150MB；启动检查挂载/du/容量/真实读写探针和GPU，独立身份守护软45GB/硬49GB并保持旧四根18GB检查。整个补充从首次运行含续跑≤24h，独立23h45保存/23h59硬停。故障保留q/RNG/cost，只准确接续，不重置成本或绕过时钟。

不因负结果延长训练或增加draws，不调KL/类权重、不择优终点。四格分解仍受变分族与有限推断间隙影响，不等于精确p(I|Z)；识别器冻结，不能称图像分割/SSL/CL/泛化改进。报告正/负/未决以及原init1混杂，源码协议、中文报告和匿名标量结果push，远端SHA和匿名访问核验。唯一sc-bayeseg任务统一跟进，不新增心跳。
