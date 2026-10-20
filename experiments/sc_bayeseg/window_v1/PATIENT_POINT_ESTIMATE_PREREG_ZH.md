# 同结构局部点估计 D0 / Dμ 匹配实验

2026-10-20，RV/风险/温度诊断已COMPLETE/EXIT0/独立AUDIT PASS，本协议在PASS后正式冻结，按用户明确授权直接执行。诊断审计结果表明：校准缩小K1与U-Net NLL差距，K1两个空RV相位中有RV像素的随机draw均无GT命中。应直接检验局部分布价值，不能把概率改善自动归因Bayesian分布，不能用GT切换读出或调阈值。本轮不扩尺度、不扫描KL、不加F/teacher/SSL/CL/U/目标/新训练seed。

## 两臂与目标

D0：c=mu_c(x), R=mu_R(x)，损失为两Theta平均像素CE+同定义前景Dice(两Theta平均概率)。Dμ：上述损失+0.5*(||mu_c||²+||mu_R||²)/128²，batch取均值，匹配完整K1的均值先验项，系数固定1不按val扫描。点质量相对Gaussian的KL不可直接如此计算，因此这不是零方差KL/纯ELBO；G权重Theta仍按原规则采样，不是整个模型确定性。Dμ对D0只改均值惩罚；K1对Dμ同时改变局部分布、尺度正则、采样及优化，不能声称完全拆净方差单独因果贡献。

固定顺序D0、Dmu各恰5000正式更新/反向，共10000。不接任一终态，均从原patient_r32/S0000完整参数/RNG、step0空Adam开始，freshAdam lr3e-4 betas(.9,.999)eps1e-8 wd0 foreachFalse，clip1、FP32确定性/TF32False/CUBLAS_WORKSPACE_CONFIG=:4096:8。同batch2患者均衡/四翻转/原schedule5001..10000，原训练Theta种子720163+step，独立Theta流不因取消local而改变；schedule与增强由原规则生成，不消耗额外local噪声。每步2Theta各1G（无重复local），节省调用真实报告，更新数匹配不称算力完全相同。

G0所有参数/prior/initial/variance冻结，UNet特征和H均值路径训练，unused segmentation_head冻结。沿用原head完整输出形状并仅使用前16均值通道，因此原S0000均值数值准入误差0。hc.2及hr权重/bias后16输出行共800标量不进入损失，梯度/Adam一二阶矩须精确0、参数逐步不变；共享特征和hc隐藏层不冻结。优化器包含484576张量标量槽位，有效可更新483776标量，不把行冻结误报整tensor requires_grad=False。记录每步冻结行/梯度/Adam检查，每0/1000/...5000保存完整modelAdamRNG，共12份。原raw/logsd映射参数保留原值但局部点预测不使用，不称调整后验尺度。

## 数据、读出和比较

只canonical320seed0 Siemens20%L10训练患者230切片，原23患者开发val46ED/ES相位544切片，无test24/U38/目标。GT仅训练损失与封存后评分，不优化部署局部q/bias或按GT选样本。每臂训练20相位和val46相位，132 native128 FP32预测全部封存后读评分标签；每批4Theta各1均值G，seed620163+1000phase+batch。GPU双线性恢复384归一化，原spacing/三维Dice/HD95含NA及空预测惩罚/EDES到患者规则；非原扫描网格。旧K0/K1/U_D完整匹配控制直接复用，不重复旧训练/推断/评分。

全部报告训练和val患者宏与LV/MYO/RV、NLL/Brier/HD95及新增修复空预测，不只最好项。D0-K0、Dmu-D0、K1-Dmu、各新臂-U_D的患者配对差，bootstrap10000seed163，NLL/Brier也报告。旧模型温度结果属于先前交叉拟合诊断；本轮不自动为新臂调温度或把原始与校准评分混比。没有可预注册的等效证明门槛，不以CI跨零宣布相当；val多轮开发、单训练入口seed，无最终test/临床或Bayesian总体增益声明。

## 成本、准入与停止

正式训练10000optimizer/反向、20000编码图像、20000Theta/20000批量G/40000图像G；评价1548编码/3184Theta/3184批量G/6192图像G，132评分标签相位及132插值恢复（非网络前向）。12完整状态+native预测预计约.6GB，启动预留.9GB。原R_START首启clock和全部失败/成本不重置，原23h45保存23h59硬停guard、四根18GB软20GB硬/新45/49GB保持；每服务器新增实际<50GB、原链<=24h，不能删历史腾空间或拆名绕限。pro5000固定GPU2原UUID、所有大文件source/tmp/cache/log/model/probability在授权/data/bmw create-only，先核源/预算/剩余clock/挂载容量真实写读/GPU余量，中性主子argv并ps+nvidia确认。

工程准入：同原R32/S0000均值误差0；Dμ-D0损失差等于均值惩罚，FP32残差1.94e-7<=1e-6；有效梯度有限、800方差行零梯度、G grad None。成功准入2合成编码/4G/4Theta/1反向/0optimizer/0真实数据。两次准入脚本失败保留：首次入口guard误执行（0网络/0反向），第二次合成脚本漏CUBLAS环境（2编码4G4Theta/1失败反向请求/0更新），实际训练入口始终显式配置该环境；修复后不重跑正式实验。工程合计4编码8G8Theta、2反向请求（1完成1失败）0更新，单列不得抹除。

非有限、冻结项变化、空间/时钟或数据边界异常立即保留partial/model/Adam/RNG与实际成本，不盲重跑/从头重启/按中间val改步数或目标。完成独立审计10000真实连续更新/反向、12完整状态/Adam步、同初态/RNG/freshAdam/同schedule、方差800行与其Adam矩0/G冻结、实际目标权重/132封存与384重算全指标/对比、EXIT/FAILURE及worker guard退休，中文正负报告匿名全数值源码协议proxy push核SHA+匿名200。完成后自主分析选择有判别力后继，暂不自动新训练seed/隐藏test/SSL CL/教师/恢复F或尺度族修复。
