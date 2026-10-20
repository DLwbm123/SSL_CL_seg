# 全10训练患者纯图像读出：拟合缺口与开发落差

2026-10-20，运行前冻结。端到端患者开发验证已交付cf700767：DIRECT68.8174%、G0条件60.1380%，MYO差20.62个百分点。先区分训练拟合与跨患者落差，不盲延长相同训练，不回到两图偏置。

只读四个固定终点：patient_validation_v1的DIRECT/S5000与CONDITIONAL_G/S5000（冻结特征），patient_e2e_validation_v1的DIRECT_E2E/S5000与CONDITIONAL_G_E2E/S5000。所有模型/来源/原Adam/RNG文件不改，加载后eval且全参数冻结。G只原G0，不用G3/oracleq。接口纯图像，先封存四模型全训练图像预测再读取训练标签评分；0optimizer/反向/局部推断优化，不以GT拟合q/偏置。

仅canonical320 seed0 Siemens20%L的10训练患者/20EDES相位/230切片，原384处理网格→128模型→384概率归一化。无增强。批2，每相位独立固定seed620163+1000×相位序号+批序号，条件模型4Theta×4local，两个G模型同流比较；种子规则与原val相同但病例不同，不称逐样本配对val噪声。固定一次评价，不按读数追加MC/改后处理。四模型20相位全封存后才开训练label数组。无val/test图像或标签重新推断/评分，无U/目标。

相位三维Dice/TPFPFN/面积/HD95物理spacing/空预测NA和对角线惩罚/NLL/Brier复用原固定phase_metrics；只将患者聚合覆盖断言从23改为10，仍先EDES再类/患者等权。报告全部匿名训练患者及每类、四模型训练均值与已有val均值的描述性差（训练/val患者不同，不做伪配对CI），以及训练内G−DIRECT配对bootstrap10000次seed163。旧val数值原样复用不重复评分。模型在这些病例训练过，这是拟合读出，不是新增泛化证据；MC固定一次不声称精确风险。

最多920图像编码前向，两个G合7360图像G前向；批量G=32×各相位ceil(深度/2)之和<=4000，Theta=8×该和<=1000。80训练相位label读取，0更新/反向/旧val网络前向。4源checkpoint只读复用、不复制权重；新增私有概率预计<3GB，预留4GB，所有文件/cache/tmp/source/log在pro5000授权/data/bmw新create-only。原R_START首启clock/全部失败成本、<=24h/新增实际<50GB、独立原23h45保存23h59硬停与四根18/20GB及新45/49GB保持，不拆名重置、不删历史。

源/剩余时钟/实际挂载容量真实写读/GPU余量核实后启动，中性主子argv并ps+nvidia复核。失败保存已封存预测/进度/真实成本，不盲重跑已完成图像；完成核验80封存预测、10患者20相位230切片覆盖、实际G/Theta账本、源状态及metadata冻结、完整指标/复用val一致、EXIT/FAILURE及worker和guard退休，独立复算后中文正负报告匿名指标代理push核SHA+200。结果后按训练拟合vs跨患者差距自主选择下一单变量假设，不能直接删KL/扩族或进入SSL/CL/test。唯一sc-bayeseg继续。
