# MC采样与非线性控制变量验证

同样draw预算下，Sobol在128 draws有效降方差，但未解决严格收敛；对偶采样在原Gaussian二次目标反而约增至两倍。随后固定非线性G(c)=c+0.1c²的独立验证中，解析二次控制变量几乎无收益。未启动正式医学训练；用户已授权问题解决后正式实验，当前尚未满足。

## 原Gaussian等预算结果

|坐标/点|draw|方法|归一化RMSE|相对IID方差|>5SE坐标|
|---|---:|---|---:|---:|---:|
|K/initial|8|IID|3.24755|1|0|
|K/initial|8|antithetic|4.90424|2.26175|0|
|K/initial|8|Sobol|5.08265|2.44052|0|
|K/initial|32|IID|1.59756|1|0|
|K/initial|32|antithetic|2.21901|1.93887|0|
|K/initial|32|Sobol|0.772285|0.234546|0|
|K/initial|128|IID|0.801199|1|0|
|K/initial|128|antithetic|1.18186|2.16488|0|
|K/initial|128|Sobol|0.209245|0.0678176|0|
|K/endpoint|8|IID|3.65028|1|0|
|K/endpoint|8|antithetic|5.51298|2.26196|0|
|K/endpoint|8|Sobol|5.6202|2.3618|0|
|K/endpoint|32|IID|1.79265|1|0|
|K/endpoint|32|antithetic|2.49403|1.94521|0|
|K/endpoint|32|Sobol|0.858379|0.230096|0|
|K/endpoint|128|IID|0.898913|1|0|
|K/endpoint|128|antithetic|1.32707|2.16839|0|
|K/endpoint|128|Sobol|0.231685|0.066048|0|
|L/initial|8|IID|8.37686|1|0|
|L/initial|8|antithetic|12.6629|2.26652|0|
|L/initial|8|Sobol|13.2154|2.47916|0|
|L/initial|32|IID|4.12152|1|0|
|L/initial|32|antithetic|5.72059|1.93568|0|
|L/initial|32|Sobol|2.0041|0.237216|0|
|L/initial|128|IID|2.06281|1|0|
|L/initial|128|antithetic|3.04965|2.17445|0|
|L/initial|128|Sobol|0.539775|0.0680539|0|
|L/endpoint|8|IID|11.5042|1|0|
|L/endpoint|8|antithetic|17.3911|2.26636|0|
|L/endpoint|8|Sobol|17.9249|2.4183|0|
|L/endpoint|32|IID|5.65232|1|0|
|L/endpoint|32|antithetic|7.85684|1.9414|0|
|L/endpoint|32|Sobol|2.72764|0.233623|0|
|L/endpoint|128|IID|2.82867|1|0|
|L/endpoint|128|antithetic|4.18503|2.17767|0|
|L/endpoint|128|Sobol|0.732309|0.0666134|0|

8draw Sobol亦可能差于IID，不能只报128draw优势。128draw K/L终点真实梯度范数2.20e-6/9.33e-6，Sobol RMSE .232/.732，仍远大于真实梯度。零5SE异常不等于通过原2e-6梯度门槛。

## 固定非线性生成器

目标已改变，因此原Gaussian终点不是非线性最优点，不把非线性梯度约62当数值误差。通过Gaussian多项式矩独立精确参考核对，原混合/方向梯度2e-6门槛全部通过。控制系数固定1，不拟合。

|坐标/点|采样|估计器|归一化RMSE|相对同采样full方差|>5SE坐标|
|---|---|---|---:|---:|---:|
|K/initial|IID|full|0.400992|1|0|
|K/initial|IID|CV|0.399796|0.994289|0|
|K/initial|antithetic|full|0.417344|1|0|
|K/initial|antithetic|CV|0.415861|0.992944|0|
|K/initial|Sobol|full|0.133441|1|0|
|K/initial|Sobol|CV|0.133359|0.998759|0|
|K/endpoint|IID|full|0.400733|1|0|
|K/endpoint|IID|CV|0.39953|0.994244|0|
|K/endpoint|antithetic|full|0.417294|1|0|
|K/endpoint|antithetic|CV|0.415806|0.992918|0|
|K/endpoint|Sobol|full|0.132557|1|0|
|K/endpoint|Sobol|CV|0.132479|0.998819|0|
|L/initial|IID|full|0.865111|1|0|
|L/initial|IID|CV|0.861982|0.993228|0|
|L/initial|antithetic|full|0.419376|1|0|
|L/initial|antithetic|CV|0.414297|0.974623|0|
|L/initial|Sobol|full|0.24508|1|0|
|L/initial|Sobol|CV|0.244513|0.995213|0|
|L/endpoint|IID|full|0.868235|1|0|
|L/endpoint|IID|CV|0.865205|0.993464|0|
|L/endpoint|antithetic|full|0.420572|1|0|
|L/endpoint|antithetic|CV|0.415806|0.976119|0|
|L/endpoint|Sobol|full|0.244221|1|0|
|L/endpoint|Sobol|CV|0.243715|0.995694|0|

CV/Sobol方差比约.995–.999，二次解析化不是当前非线性残差的有效解法；不直接扩大draw或追加训练步骤。对偶在L非线性目标有效而K无优势，符合奇次项贡献依赖坐标的现象，不宣称已证明全部因果。

## 成本与后继

采样比较4608 MC梯度+4精确梯度，3096576潜变量对，1.419秒。非线性3072 MC梯度+8参考梯度+8差分目标，4718592潜变量对，1.722秒。两阶段均600秒硬限，0参数更新/真实数据/GPU；全部新文件小于2MB。所有历史失败保留。

下一步按用户补充检索本地ICLR2027全投稿目录，优先非线性重参数化梯度/控制变量/随机准Monte Carlo的可复用实现；核全文和假设后冻结独立有界实验。不得以统计无偏筛查或有限12点通过替代学习G/F、完整原CE/Dice和真实训练的准入。
