# P0补充：错分、分类风险和后验统计（零更新）

2026-10-20，启动前冻结。四模型来源为已完成训练读出的FROZEN_DIRECT/G0、E2E_DIRECT/G0，全部固定终点。先从原训练20相位和开发val46相位的封存384概率补算混淆4×4（行GT列预测）、逐类precision/recall/预测GT体积比、GT类NLL/各类对总NLL贡献、前景三类等权NLL；相位→EDES→患者，不把缺类替换为0或1。训练和val只读原预测及其原标签，不新跑val模型。分解数据不用于局部q/偏置拟合。

再只对原10训练患者230切片、两个G终点各一次固定4Theta×4local回放，批2和原seed620163+1000phase+batch完全不变。保存原生128网格的MC平均概率、平均log概率及H的c/R mean/logsd，均在读取对应训练标签前生成；期望CE与混合NLL用同一128网格及最近邻标签，不能与原384 NLL混报。逐像素检验Jensen（期望CE>=混合NLL，容差2e−6）。后验分别报告KLc/R的均值项与方差项除128²、均值RMS、logsd范围/分位数、上下界.01内比例和映射导数<.01比例；KL非零下限不等于梯度主导，减常数不改变优化。

成本0参数更新/0反向，264次原封存相位概率/标签分解；新增460编码图像、3840G/960Theta/7360图像G，另40训练相位标签用于128风险评分。无U/目标/test，无新val模型前向，不改原P0/原结果。所有源state和metadata只读，恢复RNG。新增native概率/meanlog/q约<.3GB，预留.6GB，原首启clock87614098/original_started_unix1792485279.2578013及失败成本不重置。原独立23h45保存/23h59硬停、四根18GB软20GB硬与新45/49GB、<=24h/新增实际<50GB保持；pro5000原GPU2/授权/data/bmw create-only，中性入口。源/容量真实写读/GPU余量核验后启动，不删历史。

完成覆盖/账本/源冻结/Jensen/保存数组风险及后验公式复算/EXIT FAILURE及worker和guard退休审计，匿名中文报告和全部数值代理push核SHA+200。随后据诊断冻结匹配CE+Dice×KL0/1因子，并保留U-Net+Dice；软Dice定义/缺类规则/初态freshAdam/固定训练预算及存储须先明确，beta0仅诊断边界而非Bayesian收益。
