# 冻结U-Net与固定G路径的四翻转读出与空间一致性

2026-10-21，独立假设，运行前冻结。标签平滑探针完成审计交付后启动；不改旧主终点、不改epsilon或病例风险分数方向。训练曾用四翻转，但G含全局c到固定4×4空间模板的映射，未显式保证空间等变；尚需一次直接空间诊断，而非继续加MC或调校准。

## 模型、输入与唯一读出规则

两模型固定：原U_D/S5000，以及原Dμ/S5000在G权重均值和局部c/R均值下的MEAN_READ。都不接新训练、不选择最好终点/MC流；不选U_D_LS作为新基线，始终保留原匹配U-Net+Dice。来源必须已AUDIT PASS，MEAN_READ缓存CONTEXT须对应同Dμ源和同患者映射。

只23开发患者46ED/ES相位544切片。对每模型复用原identity native128 FP32概率，新增训练时已使用的三种翻转k1横/k2纵/k3两者，输出native概率逆翻转回identity坐标，再四份概率等权1/4平均。固定相同transform集合和权重，不按GT/类别/患者选择视图、不强制RV非空。输入翻转后仍按原网络128预处理，最后原deterministic双线性恢复384归一化/processed_spacing。G仅取保存mean_w/b与局部means，Theta/local抽样0；这是测试时数据增强，不是测试时参数适应或Bayesian采样。

源全参数/prior/initial严格冻结、model.eval/gradFalse，清除临时draw且恢复入口RNG。旧identity概率直接复用，不重复identity网络或旧指标评分。全部92份ensemble预测及无GT空间统计先封存，才读取本轮GT标签。

## 空间统计、主次终点与校准

每模型/相位/三个视图记录native128逆变换概率相对identity的平均像素TV=.5*sum_class|p_view−p_id|、argmax不一致率及soft类体积分数，均不读GT。患者ED/ES后平均，跨模型只作固定管线描述；两模型训练/正则历史不同，不把交互当纯G架构因果证明。

主终点：23患者Dice增益交互(四视图G−identityG)−(四视图U−identityU)，配对bootstrap10000seed163；两模型各自增益/全部LV/MYO/RV Dice、HD95 NA/空预测惩罚及失败数完整。完整原始NLL/Brier、四类条件及贡献/正确错误/前景误BG风险；不以单个有利指标宣称方法胜出。

预先包含匹配患者五折温度：同旧PRIVATE_FOLDS23患者排序index mod5，stride64/phase等权/logT[-3,3]/bounded xatol1e-5 maxiter64，每模型5fits共10<=640实际函数。每fold只其余患者，源final384平均概率温度变换；全T先封存，每输出manifest先于该次heldout标签重读，argmax变化0。原始风险阶段可已读开发标签，不能称完全未见；fit严格排除评分患者，重复开发不是独立test。两模型原始与校准风险各对同模型identity控制配对，NLL/Brier增益交互次要完整报告，其他旧控制仅复用。原identity温度和结果不refit，不把旧T直接当新平均规则的最优T。

## 保存范围、预算与审计边界

92ensemble native概率理论285212672字节，预计全部新产物<.30GB，启动预留.32GB；旧四根实际+预留必须<18GB。只保存ensemble FP32和三个视图的标量统计，不保存各视图完整概率以满足既有更严存储边界，源identity不复制。独立审计不重跑网络，不能声称逐视图网络回放或逐像素平均独立复演；核源元数据和参数冻结断言/调用账本，保存ensemble/identity归一化、类概率体积分数四份平均恒等、TV Jensen上界（2e-6容差）、全部ensemble指标/fit目标/患者交互。worker固定顺序直接求四份均值，过程概率有限/shape/simplex检查；缺少逐视图缓存的审计限制须如实报告。

正式0optimizer/反向、3264编码图像（2模型×3新视图×544），834 U直接logit前向、834批量G/1632图像G，0Theta/local。92原identity缓存读取、276新增相位视图、92ensemble缓存。评分92raw+92cal=184恢复/184标签，10温度fits实际nfev单列，非完全0优化。无新train/test/U/目标/训练seed。独立审计92ensemble恢复/92identity原生读取/92标签/92温度变换、10fit目标复核，0新网络/refit。

准入只合成：双翻转逆变换精确、统一权重/概率归一化、1 U直接前向+1 G均值前向（2编码图像/0反向更新/0真实数据）、源参数不变。所有成本单列。完成联合EXIT0/无FAILURE/worker与guard退休、独立AUDIT/匿名统计通过后报告push核SHA+匿名200。

保留原clock_start_token87614098/original_started_unix1792485279.2578013与所有历史失败/占用，独立23h45保存23h59硬停/四根18GB软20GB硬/每服务器新增<50GB/原首启总<=24h，不拆名重置、不删除旧数据腾空间。启动核实际挂载容量/真实写读/GPU/create-only/中性主子argv；后续保存/结束仍须used+50MB<18GB，为预定<.30GB产物留出写入余量，不改变18/20GB边界。异常保留partial/成本，不能为门槛更换视图或重跑完成网络。仅一次四视图读出，不扫描transform集合/权重，不自动新训练seed/教师SSLCL/F/尺度族/test/U/目标。当前G无F非完整/官方BayeSeg；病例聚合修补支线保持关闭。
