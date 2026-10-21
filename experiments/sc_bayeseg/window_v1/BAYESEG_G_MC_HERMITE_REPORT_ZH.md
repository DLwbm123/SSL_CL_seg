# 独立校准Hermite梯度CV结果

结论：多项式实现检查通过，但不能推广为一般非线性生成器修复。sin目标中同预算Sobol明显优于本次Hermite64+64，所有MC方法均未通过逐次严格梯度误差检查。未启动正式训练。

固定原K/L的起点终点、固定隐藏和均值；128draw预算，CV64校准+64估计，IID/Sobol128。每组128独立重复。没有解析梯度用于拟合，没有同样本拟合与校正。局部输出梯度经固定特征映射共享参数，链式及精确矩方向差分检查全部通过原2e-6门槛。

|目标/坐标/点|方法|归一化RMSE|相对IID方差|严格通过/128|
|---|---|---:|---:|---:|
|poly/K/initial|IID|0.401637|1|0|
|poly/K/initial|Sobol|0.133096|0.110169|0|
|poly/K/initial|Hermite64+64|1.46103e-15|1.17765e-29|128|
|poly/K/endpoint|IID|0.401337|1|0|
|poly/K/endpoint|Sobol|0.132256|0.108841|0|
|poly/K/endpoint|Hermite64+64|1.51263e-15|1.40254e-29|128|
|poly/L/initial|IID|0.867038|1|0|
|poly/L/initial|Sobol|0.242982|0.0799097|0|
|poly/L/initial|Hermite64+64|3.21803e-15|1.33567e-29|128|
|poly/L/endpoint|IID|0.869983|1|0|
|poly/L/endpoint|Sobol|0.24221|0.0788788|0|
|poly/L/endpoint|Hermite64+64|3.24042e-15|1.39752e-29|128|
|sin/K/initial|IID|0.0978614|1|0|
|sin/K/initial|Sobol|0.00581541|0.00352748|0|
|sin/K/initial|Hermite64+64|0.0256823|0.0689671|0|
|sin/K/endpoint|IID|0.0978418|1|0|
|sin/K/endpoint|Sobol|0.00576263|0.00346428|0|
|sin/K/endpoint|Hermite64+64|0.0253856|0.0672191|0|
|sin/L/initial|IID|0.0965317|1|0|
|sin/L/initial|Sobol|0.00488007|0.00256239|0|
|sin/L/initial|Hermite64+64|0.0171307|0.0315449|0|
|sin/L/endpoint|IID|0.0964959|1|0|
|sin/L/endpoint|Sobol|0.0048414|0.00251988|0|
|sin/L/endpoint|Hermite64+64|0.0168543|0.0304425|0|

四次多项式恰在所选基空间内，接近零方差只验证实现，不能宣称通用收益。正弦终点K/L，CV方差为IID约6.72%/3.04%，Sobol约0.346%/0.252%；CV约比Sobol差19.4/12.1倍。全组5SE无异常仅说明未发现统计偏差；严格单次精度未过。单次随机估计精度与最终优化收敛是不同问题，不能由此直接断言优化必败。

成本：2056局部批量梯度请求、8参考梯度、8共享链式梯度、16FD目标、12288个小型最小二乘；按臂4718592潜变量对，复用及链式检查实际3158016。预注册成本总数少乘2的笔误已保留并注释更正，无改循环/重跑。
墙钟1.307秒，600秒硬限；0optimizer更新/真实数据/GPU，输出小于2MB。

后继选择：不继续升Hermite阶数或扫校准比例。另立固定sin目标下同起点/同Adam5000步及128draw的IID与Sobol优化对照，加入精确矩梯度Adam控制及有界精确矩L-BFGS参考，报告终点相对参考目标差及梯度残差；原Gaussian KL及数值门槛继续保留，非Gaussian目标差不能冒充原KL。后续还需学习G/完整目标准入。
