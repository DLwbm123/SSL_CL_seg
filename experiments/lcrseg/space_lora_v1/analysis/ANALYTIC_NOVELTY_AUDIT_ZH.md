# ACIL、半监督解析增量、STSD：全文与代码核查

2026-10-18。只读研究；不启动实验，不修改训练实现。结论：**放弃上一版“固定语义坐标的解析残差读出＋一致性”作为论文主方法**，保留文档作为淘汰记录。能运行、能给出代数等价性，不等于有足够方法贡献。

## 取得的材料及核查边界

|方法|全文|代码与版本|本轮可确认范围|
|---|---|---|---|
|ACIL，NeurIPS2022|官方会议PDF已下载，核对§3、Theorem3.1、Algorithm1及实验/限制|作者统一仓库`ZHUANGHP/Analytic-continual-learning`，`0afd6bf87622e808ce7f730f8556e78229b9e758`|静态读`analytic/ACIL.py`、`AnalyticLinear.py`、`Buffer.py`。仓库注明当前核心为G-ACIL通用实现，传统设置等价ACIL；不冒充2022原始版本逐行一致|
|ILR-ELM，Applied Soft Computing2017|用户完成出版社CAPTCHA后，页面显示Full text access；已保存48,651字符完整HTML正文，重点核对§2 Eq3–20与实验设置|未找到可核实作者ILR-ELM实现；论文只给R-ELM/OS-ELM/SS-ELM基线网站|PDF下载两次超时，但完整正文可读，不再把它写成摘要级。不能声称实现复现已通过|
|PRMROSE，Sensors2022|作者机构仓储32页PDF，核对§3–4、Algorithm3及限制/数据声明|未找到作者发布的可核实代码；伪代码可读|作为半监督在线解析方法的补充证据，不代替ILR-ELM的源码核对|
|STSD，ICLR27投稿|已有官方全文，核对§3、Eq4–11、Algorithm1、六项源→目标任务和消融|`Sky-byte-box/STSD_Code`，`9ec79d7749b27c34fc617a4bdd77575de00d29a8`；仓库方法ZIP|静态读model/geometry/training/config/run及测试结构；没有完整真实基准数据预处理和原始运行记录|
|Optimistic Semi-supervised Least Squares Classification，ICPR2016|取得6页作者预印本，核对§IV软/硬标签交替最优化|作者RSSL库`36dd3980877ae7a51f8cd96a823a5fb855ca7bc9`，`R/EMLeastSquaresClassifier.R`|静态核查重复估计软标签和解析权重；不是持续学习实现，不声称已重跑|

作者代码均未执行。对ACIL/STSD上述7个Python文件只做AST解析；没有安装环境、跑作者测试、加载数据/权重或产生优化器更新。所有GitHub下载均显式使用既定代理；第三方全文/代码包仅留本地文献库，不进入项目公开文件。

## 1. ACIL已经覆盖哪些核心内容

ACIL先训练源特征，再固定特征与随机扩展映射，用岭回归校准分类头；后续保存正则化相关矩阵的逆与读出权重，通过递归最小二乘更新。§3 Theorem3.1证明固定表示/标签条件下，递归结果与同一联合解析问题一致。不是旧域准确率永不下降，也不是全网联合训练等价。

作者代码`RandomBuffer`把扩展权重注册为buffer；`RecursiveLinear.fit`更新逆相关矩阵与`weight`，无反向传播；`ACIL.forward/fit`处于no_grad，`ACILLearner.make_model`选择FP64。分类头输出原始分数，`fit`中的标签来自真实y的one-hot，ACIL核心没有U一致性。

我们的上一版A/B只是同类充分统计的另一表示；把目标改成`y-p0`、用RBF替代随机ReLU、最后投影到单纯形，都没有改变其解析骨架。用固定原头提供恒等入口是工程适配差异，不能独立承担TMI贡献。

## 2. ILR-ELM已经把L/U和解析增量结合

§2 Eq3/4的目标含监督平方误差、岭项以及`tr((HW)^T L(HW))`图平滑；解析解含`H^T C H + λ H^T L H`。§2.2分别处理新块为纯U或混合L/U的情况。

一个重要限制：Eq7先写出旧/新块的跨块图连接；Eq8/14随后采用近似，丢弃涉及跨块连接的B_t/E_t，再递归求解。因此不能把其结果说成对不断重建的全历史图无条件精确等价。实验使用MATLAB R2010a，基线代码来自论文列出的ELM网站；没有据此找到作者ILR-ELM源码。

PRMROSE进一步使用持续更新的ESOINN原型构建数据—原型图，并结合SOS-ELM解析更新。已有“原型＋U图结构＋在线解析”的完整先例；不能把“保存少数语义原型”当成新颖性。

## 3. 上一版是怎样退化为已有框架的

上一版目标为：

`||Φ_L R-(Y-p0_L)||² + λ||DΦ_U R + Dp0_U||² + η||R||²`。

构造

`X̃ = [Φ_L; sqrt(λ)DΦ_U]`，`Ỹ = [Y-p0_L; -sqrt(λ)Dp0_U]`，

就严格变为普通岭回归`||X̃R-Ỹ||²+η||R||²`；其中D是视图配对图的关联矩阵，`D^T D`就是图Laplacian。历史A/B累加、顺序/联合等价、旧能量二次型及对应Cauchy-Schwarz界，均来自这个固定二次问题。

残差基线导致右端项不同，图结构来自对应视图而非近邻，这些是真实差异，但属于具体目标/图的选择；尚未形成超出解析读出加一致性的研究贡献。**按用户给出的淘汰标准，上一版退出主线，不再为它启动实验。**

## 4. STSD有不同机制，但不能替代这个缺口

STSD以源类散度/目标跨视图分歧的广义特征比值确定迁移空间；再按该空间聚合原型、求OT，并对目标encoder反向传播。代码`adapt_target`冻结源classifier/shared basis，但优化encoder、spectral adapter和graph branch；它不是固定特征的解析分类器。

核对点：`geometry.py`将广义特征向量映回原空间后QR构造投影；Sinkhorn使用平衡边际；`refresh`为no_grad；`training.py`内层使用固定投影/OT的加权CE。默认推理头为冻结原classifier，而非transported分析头。`Config`明确把patch大小等部分设置标为论文未指定的工程选择。

Theorem1只在投影锚点分离、样本接近正确传输锚点等假设下给出最近锚点的margin充分条件，没有约束旧输入经更新encoder后的变化。论文报告六项跨主体源自由适配，无顺序旧域矩阵；因此不能宣称它已验证半监督持续分割。医学四类的中心化类散度秩最多3，均匀OT质量和类索引重排也不能直接照搬。

STSD保留为“学习何种语义可迁移”的参考，不再作为上一版A/B记忆的理论依据。

## 5. 针对用户追问：可以设计半监督解析持续学习，但要换核心问题

更具体的问题是：**历史U在退出阶段后仍没有确定标签，当前模型改变时，如何重新计算它们对解析解的影响，而非把早期伪标签永远锁进B矩阵？**

同时核查到2016软标签最小二乘已通过block coordinate descent反复更新未知标签与解析权重。作者RSSL实现`EMLeastSquaresClassifier.R`保留完整`X_u`，循环中重算resp、截断或硬化，再求theta。因此“软标签和闭式头交替”本身也不是新贡献。

本轮新增候选设计见[可修订U状态的半监督解析持续学习](REVISABLE_UNLABELED_ANALYTIC_DESIGN_ZH.md)。它把可修订标签的状态需求、有限摘要误差和解析求解分开，尚未完成新颖性认证，不预设TMI成立。若它仍与在线量化图/原型自训练完全等价，应继续淘汰。

## 6. 其他ICLR27候选追加全文检查

从43,013条总表重新筛选半监督+持续/生成/语义及视觉自监督持续机制，得到24个宽召回标题，另纳入HiSC-EBM；不是新增24个合格方法。

- [What Continual Learning Preserves，75djGJ5C5u](https://openreview.net/forum?id=75djGJ5C5u)：已下载全文检查§3.3与实现声明。因果影响profile/有效网络有趣，但保留项依赖存储的图像/图像对及其干预目标，内存达到200张；仍是`L_SSL+λR`，不是当前无旧图像的一体化模块。代码承诺接收后发布，本轮未取得，不能因摘要里“causal”而直接优先。
- [HiSC-EBM，NzzGCMwHGU](https://openreview.net/forum?id=NzzGCMwHGU)：已下载全文检查§2.2–2.3与Algorithm1/实现附录。共享symbol-coupled energy通过条件化/边缘化统一L/U，确有真正生成半监督机制；但需生成器、推断器、五层潜变量及30步Langevin先验采样，非解析方法，没有CL验证，官方代码/完整细节尚未取得。不因“统一”就优先移植到现有U-Net。

当前没有发现可原样复现就同时满足全部条件的现成ICLR27算法。继续分析的价值应落在一个明确未解决的状态/学习问题上，不以再换一个模块名字替代证据。

## 来源

- [ACIL正文](https://proceedings.neurips.cc/paper_files/paper/2022/file/4b74a42fc81fc7ee252f6bcb6e26c8be-Paper-Conference.pdf)，[作者统一实现](https://github.com/ZHUANGHP/Analytic-continual-learning)。
- [ILR-ELM出版社全文入口](https://www.sciencedirect.com/science/article/pii/S1568494617303290)，DOI10.1016/j.asoc.2017.05.051。
- [PRMROSE作者机构全文](https://www.open-access.bcu.ac.uk/13200/1/sensors-22-03113-v2.pdf)。
- [STSD全文页面](https://openreview.net/forum?id=RKnMsOakBs)，[作者方法仓库](https://github.com/Sky-byte-box/STSD_Code)。
- [Optimistic semi-supervised least squares全文](https://arxiv.org/pdf/1610.03713)，[作者论文复现仓库](https://github.com/jkrijthe/optimisticssl)，[作者RSSL算法库](https://github.com/jkrijthe/RSSL)。

未核实到ILR-ELM/PRMROSE作者源码；没有以其他人的实现补齐后冒称官方代码。所有结论是方法与静态代码核查，非运行复现结果。
