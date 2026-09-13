# SSL_CL_seg：医学图像分割与半监督分割损失函数文献检索及接入建议

> 检索范围：ICML、CVPR、ICLR、NeurIPS、ECCV、ICCV。  
> 时间范围：2023—2026 年，截至 2026 年 9 月 13 日。  
> 项目：[`DLwbm123/SSL_CL_seg`](https://github.com/DLwbm123/SSL_CL_seg)。  
> 代码分析主要基于提交 `8a6e93a8bc39554c878cffe0dedf7fbfefe7d16d`；既有标签定义另引用已核对的历史协议。

我按 **2023—2026 年、截至 2026 年 9 月 13 日**，检索了你指定的六个会议，筛选时重点看了三个条件：**是否真的提出了可独立使用的损失、是否适合医学分割、能否接入你目前的 CE＋Dice、LCTX 和 Mean Teacher 路径。**

**结论先放在前面：我最建议你优先考虑 CWMI、JML 和 DyCON 中的 UnCL；针对眼底视盘/视杯，再重点研究 D-Convexity。**它们分别对应多尺度结构、软伪标签的区域重叠、不确定性一致性，以及解剖形状约束，解决的问题不同，不应该一次全部叠加。文献见 [CWMI](https://proceedings.mlr.press/v267/lu25c.html)、[JML](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ee208bfc04b1bf6125a6a34baa1c28d3-Abstract-Conference.html)、[DyCON](https://openaccess.thecvf.com/content/CVPR2025/html/Assefa_DyCON_Dynamic_Uncertainty-aware_Consistency_and_Contrastive_Learning_for_Semi-supervised_Medical_CVPR_2025_paper.html)、[D-Convexity](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_D-Convexity_A_Unified_Differentiable_Convex_Shape_Prior_via_Quasi-Concavity_for_CVPR_2026_paper.html)。

其中：

> **给当前有标签 LCTX 加一个结构损失：优先 CWMI。**  
> **重新检验无标签数据的增量价值：优先 JML。**  
> **处理置信度阈值可能丢弃的重要区域：考虑 UnCL。**  
> **专门针对完整视盘、视杯的形状异常：考虑 D-Convexity，但不能直接约束视盘环。**

下面把文献、选择理由和接入方式展开。

---

## 一、检索结果：哪些是真正值得考虑的损失？

以下均按已经核实的会议归属列出。**“优先级”和“适合你的程度”是我根据你项目实现做出的判断，不代表论文之间的通用性能排名。**

### 1. 医学结构、形状与拓扑损失

| 论文 | 会议 | 核心约束 | 对你项目的适用性 |
|---|---|---|---|
| **Complex Wavelet Mutual Information Loss: A Multi-Scale Loss Function for Semantic Segmentation，CWMI** | **ICML 2025** | 在复小波多尺度、多方向分解后，约束预测与标签的结构互信息 | **优先考虑**。能补充当前 CE＋Dice 的结构监督，不需要增加教师或原型库。原文包含医学数据实验。[论文](https://proceedings.mlr.press/v267/lu25c.html) |
| **D-Convexity: A Unified Differentiable Convex Shape Prior via Quasi-Concavity for Data-driven Image Segmentation** | **CVPR 2026** | 基于准凹性的可微凸形状先验 | **眼底任务特别值得精读**。原文直接使用 REFUGE、RIM-ONE-r3；但完整方法包括投影模块，不能把论文全部收益归于“加一个 loss”。[论文](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_D-Convexity_A_Unified_Differentiable_Convex_Shape_Prior_via_Quasi-Concavity_for_CVPR_2026_paper.html) |
| **Topograph: An Efficient Graph-Based Framework for Strictly Topology Preserving Image Segmentation** | **ICLR 2025** | 找到导致连通性、孔洞等拓扑错误的关键区域，并重点惩罚 | **有条件推荐**。更适合已经观察到断裂、错误连通、孔洞的情况，而不是只有轮廓位置偏差。[论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c72861451d6fa9dfa64831102b9bb71a-Abstract-Conference.html) |
| **Skeleton Recall Loss for Connectivity Conserving and Resource Efficient Segmentation of Thin Tubular Structures** | **ECCV 2024** | 强调真实骨架的召回，保护细管状结构连通性 | 对血管、神经非常合适；**对你当前视盘/视杯分割不是首选**。不能因为都是眼底图像，就把血管损失直接迁移过来。[论文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/9904_ECCV_2024_paper.php) |

### 2. 半监督分割损失

| 论文 | 会议 | 可借鉴的损失 | 对你项目的适用性 |
|---|---|---|---|
| **Jaccard Metric Losses: Optimizing the Jaccard Index with Soft Labels，JML** | **NeurIPS 2023** | 能正确处理软标签的 Jaccard 区域损失 | **最推荐先接入无标签分支**。不需要新网络，可以直接作用于学生概率和停止梯度的教师概率。原论文验证主要是自然图像分割，医学应用属于迁移验证。[论文](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ee208bfc04b1bf6125a6a34baa1c28d3-Abstract-Conference.html) |
| **DyCON: Dynamic Uncertainty-aware Consistency and Contrastive Learning for Semi-supervised Medical Image Segmentation** | **CVPR 2025** | **UnCL：不确定性感知一致性；FeCL：焦点熵感知对比损失** | **优先考虑 UnCL，暂不整套搬入 FeCL**。前者容易接入现有教师—学生路径，后者涉及额外特征投影和样本构造。[论文](https://openaccess.thecvf.com/content/CVPR2025/html/Assefa_DyCON_Dynamic_Uncertainty-aware_Consistency_and_Contrastive_Learning_for_Semi-supervised_Medical_CVPR_2025_paper.html) |
| **Semi-supervised Segmentation of Histopathology Images with Noise-Aware Topological Consistency，TopoSemiSeg** | **ECCV 2024** | 区分预测中的拓扑信号与拓扑噪声，再约束教师—学生一致性 | 当伪标签有错误合并、分裂、遗漏时值得研究；原任务主要是病理腺体、细胞核，迁移到视盘/视杯要重新验证。[文献记录](https://pubmed.ncbi.nlm.nih.gov/40557360/) |
| **Pseudo-Label Guided Contrastive Learning for Semi-Supervised Medical Image Segmentation，PatchCL** | **CVPR 2023** | 伪标签引导的 patch 对比损失，增强类间区分、类内紧凑 | 可作为特征空间候选，但需要正负样本采样，复杂度高于 JML、UnCL。[论文](https://openaccess.thecvf.com/content/CVPR2023/html/Basak_Pseudo-Label_Guided_Contrastive_Learning_for_Semi-Supervised_Medical_Image_Segmentation_CVPR_2023_paper.html) |
| **Improving Semi-Supervised Semantic Segmentation with Sliced-Wasserstein Feature Alignment and Uniformity，SWSEG** | **CVPR 2025** | 使用切片 Wasserstein 距离实现特征对齐与均匀性 | 可以借鉴，但原论文是自然图像分割；其特征分布目标在背景占比很高的医学任务中需单独验证。[论文](https://openaccess.thecvf.com/content/CVPR2025/html/Lu_Improving_Semi-Supervised_Semantic_Segmentation_with_Sliced-Wasserstein_Feature_Alignment_and_Uniformity_CVPR_2025_paper.html) |

### 3. 检索到但不应简单当成“一个损失”接入的工作

这部分也很重要：一些论文看起来相关，但把它们直接当成 loss 加进当前代码，会悄悄改变实验问题。

| 工作 | 会议 | 不宜直接按“新增损失”处理的原因 |
|---|---|---|
| **Bidirectional Copy-Paste，BCP** | CVPR 2023 | 核心是有标签/无标签双向混合及对应监督构造，是与你 LCTX/AMS 密切相关的**训练配方和基线**，不是独立的结构损失。[论文](https://openaccess.thecvf.com/content/CVPR2023/html/Bai_Bidirectional_Copy-Paste_for_Semi-Supervised_Medical_Image_Segmentation_CVPR_2023_paper.html) |
| **Adaptive Bidirectional Displacement，ABD** | CVPR 2024 | 核心是依据置信度进行 patch 位移和样本构造，同时改变输入扰动策略。[论文](https://openaccess.thecvf.com/content/CVPR2024/html/Chi_Adaptive_Bidirectional_Displacement_for_Semi-Supervised_Medical_Image_Segmentation_CVPR_2024_paper.html) |
| **Adaptive Learning of High-Value Regions，ALHVR** | ICCV 2025 | 包含双分支区域划分、跨原型一致性、动态教师竞争，已经超出简单输出层 loss。[论文](https://www.openaccess.thecvf.com/content/ICCV2025/html/Lei_Adaptive_Learning_of_High-Value_Regions_for_Semi-Supervised_Medical_Image_Segmentation_ICCV_2025_paper.html) |
| **CauSSL: Causality-inspired Semi-supervised Learning for Medical Image Segmentation** | ICCV 2023 | 通过最小—最大优化增强网络/分支独立性，需要改变联合训练逻辑，不能直接套在冻结的 EMA 教师上。[论文](https://openaccess.thecvf.com/content/ICCV2023/html/Miao_CauSSL_Causality-inspired_Semi-supervised_Learning_for_Medical_Image_Segmentation_ICCV_2023_paper.html) |
| **VQ-Seg: Vector-Quantized Token Perturbation for Semi-Supervised Medical Image Segmentation** | NeurIPS 2025 | 涉及向量量化、重建与分割双分支、基础模型特征适配，属于框架变化。[论文](https://proceedings.neurips.cc/paper_files/paper/2025/hash/201408406e0c5cf7626c4baeae6eaadd-Abstract-Conference.html) |
| **Enhancing Image-Conditional Coverage in Segmentation: Adaptive Thresholding via Differentiable Miscoverage Loss，COAT** | ICLR 2026 | 优化的是图像自适应阈值和覆盖率，不是直接优化你当前的分割精度或旧域保持，更适合单独的不确定性校准研究。[论文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/76ec4dc30e9faaf0e4b6093eaa377218-Abstract-Conference.html) |

还有一个容易混淆的会议归属：**Boundary Difference over Union Loss 是 MICCAI 2023，不是 ICCV/CVPR。**它确实值得医学分割研究者关注，但没有放进这次六个会议的主清单。[会议论文页](https://conferences.miccai.org/2023/papers/093-Paper1247.html)

---

## 二、优先推荐一：CWMI——最适合先补充你的有标签结构监督

### 为什么我把它放在第一位？

我重新核对了你当前的实现：`supervised_parts()` 使用 CE，加上按图像和前景类别平均的 Dice；在混合训练中，先通过 `collect_sources()` 恢复源图位置的预测，再计算监督。这已经提供了接入结构损失的明确位置。[项目代码：`ssl_anchored_mix_v0_1/core.py`](https://github.com/DLwbm123/SSL_CL_seg/blob/8a6e93a8bc39554c878cffe0dedf7fbfefe7d16d/experiments/lcrseg/ssl_anchored_mix_v0_1/core.py)

CWMI 的区别是：它不是只比较每个像素，也不是简单比较整个前景面积，而是把预测和标签分解到**不同尺度、不同方向的复小波子带**，然后比较这些结构表示之间的互信息。它采用复可转向金字塔，不等同于普通 FFT 后做一次 L1/L2。[论文全文](https://arxiv.org/html/2502.00563v2)

作者实验包括神经结构、病理腺体和视网膜血管等数据，公开实现为 `lurenhaothu/CWMI`。这些证据支持它作为医学结构损失候选，但不意味着它已经在你的视盘/视杯持续学习协议上验证有效。[作者代码](https://github.com/lurenhaothu/CWMI)

### 接入你的代码，应该怎样定义？

我的建议是保留原监督项，只添加结构部分：

$$
\boxed{
\mathcal L_{\mathrm{sup}}
=
\mathcal L_{\mathrm{CE}}
+
\mathcal L_{\mathrm{Dice}}
+
\lambda_w\mathcal L_{\mathrm{CWMI\text{-}struct}}
}
$$

这里的 $\mathcal L_{\mathrm{CWMI\text{-}struct}}$ **只指多尺度互信息项**。

这是一个重要细节：原论文最终的 CWMI 总目标本身就包含 CE。如果把整套目标直接加在你的 CE＋Dice 后面，就会额外增加一次 CE 权重，损失的比较不再干净。[论文全文](https://arxiv.org/html/2502.00563v2)

针对你当前框架，我建议的接入顺序是：

```text
两张互补混合图
        ↓
两次学生预测
        ↓
collect_sources()：恢复到原始源图坐标
        ↓
原有 CE＋Dice
        ＋
源图预测与源图标签之间的 CWMI 结构项
```

这属于**对你框架的改造建议，不是原论文的原样复现**。

### 它最适合检验什么假设？

我建议把研究假设写得具体一些：

> LCTX 改变了图像上下文，但现有监督主要依赖像素分类与区域重叠；增加源图坐标下的多尺度结构约束，是否能减少轮廓碎裂、细结构损伤和跨上下文预测的不一致？

这比“加一个高级 loss，希望 Dice 涨一点”更容易验证。

但先要确认失败形态。如果你的错误主要是整个视杯都定位错了，而不是边界或结构受损，CWMI 未必比改善伪标签或域适应机制更有针对性。

**实施时我最关注的风险是数值稳定性与有效区域处理。**CWMI 涉及协方差、逆矩阵和对数行列式；我会要求对常量预测、空前景和 `ignore=255` 做单独测试，不能仅把忽略区域填零后直接进行全图结构运算。公式涉及这些矩阵运算，具体的异常处理需要在你的实现中验证。[论文全文](https://arxiv.org/html/2502.00563v2)

---

## 三、优先推荐二：JML——最适合你重新验证“无标签数据到底有没有用”

### 你的无标签分支缺少什么不同的监督信号？

你当前的 `soft_target()` 已经使用停止梯度的教师概率，计算 soft cross-entropy，并通过教师最大概率大于 0.7 和几何有效区域形成掩码。它不是简单的硬伪标签 CE。[项目代码：`soft_target()`](https://github.com/DLwbm123/SSL_CL_seg/blob/8a6e93a8bc39554c878cffe0dedf7fbfefe7d16d/experiments/lcrseg/ssl_anchored_mix_v0_1/core.py)

因此，重新引入半监督损失时，需要提供**不同于现有逐像素 soft CE 的信号**，而不是只换个名字。

JML 的特点正好是：

> **保留教师软概率，同时从区域重叠的角度约束学生，而不是把教师预测先硬化为 0/1 标签。**

NeurIPS 2023 的论文专门讨论了软标签下的 Jaccard 损失，验证范围包括标签平滑、知识蒸馏和半监督分割。[会议论文页](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ee208bfc04b1bf6125a6a34baa1c28d3-Abstract-Conference.html)

### 为什么不直接对教师概率用普通 Dice？

对常见的、分母为概率和的 soft Dice：

$$
\mathcal L_{\mathrm{Dice}}(p,q)
=
1-\frac{2\langle p,q\rangle}{\|p\|_1+\|q\|_1},
$$

当 $q$ 是软概率时，学生等于教师并不一定使损失最小。比如单像素 $p=q=0.5$，损失仍为 0.5；固定 $q=0.5$ 时，继续增加 $p$ 反而能降低该损失。这说明“用于硬标签的区域损失”不应不加检查地用于软教师目标。相关问题是 JML 论文的主要出发点之一。[论文 PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/ee208bfc04b1bf6125a6a34baa1c28d3-Paper-Conference.pdf)

**这不是说你现在的监督 Dice 写错了。**你的该项使用真实硬标签，上述问题针对的是未来把它直接复制到软伪标签分支的做法。[项目代码：`supervised_parts()`](https://github.com/DLwbm123/SSL_CL_seg/blob/8a6e93a8bc39554c878cffe0dedf7fbfefe7d16d/experiments/lcrseg/ssl_anchored_mix_v0_1/core.py)

### JML 的公式和接法

论文中的一个版本为：

$$
\mathcal L_{\mathrm{JML1}}(p,q)
=
1-
\frac{\|p+q\|_1-\|p-q\|_1}
{\|p+q\|_1+\|p-q\|_1}.
$$

其中 $p,q$ 可以都是软标签；它满足匹配时损失为零。双空向量的边界情况需要作相应定义。[论文 PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/ee208bfc04b1bf6125a6a34baa1c28d3-Paper-Conference.pdf)

在你这里，我建议先测试：

$$
\boxed{
\mathcal L_U
=
\mathcal L_{\mathrm{softCE}}
+
\lambda_J
\mathcal L_{\mathrm{JML}}
\left(
p_{\mathrm{student}},
\operatorname{sg}(q_{\mathrm{teacher}})
\right)
}
$$

其中 $\operatorname{sg}$ 表示停止梯度。这是接入方案，不是宣称原论文就在你的混合框架里使用了这个组合。

第一轮应固定现有教师、增强、置信度掩码和训练预算，**先只改变损失**。这样才能判断区域级监督有没有增量价值。之后才考虑更改掩码，否则“新损失效果”和“更多像素进入监督”会混在一起。

此外，JML 不会自动制造正确伪标签，也不会自动让视杯区域通过 0.7 阈值。需要同时记录每个类别的有效监督面积，避免所谓区域损失实际上几乎只作用于背景。

作者仓库是 `zifuwanggg/JDTLosses`，其中 `losses/jdt_loss.py` 提供统一实现。不过该仓库还包含后续 Dice/Tversky 相关工作，复现时应明确选用哪个目标，不能把整个 JDT 系列都称为这篇 NeurIPS 论文。[作者代码](https://github.com/zifuwanggg/JDTLosses)

---

## 四、优先推荐三：DyCON 的 UnCL——针对“重要但不够自信的像素”

### 它与你当前阈值筛选的区别

你的现有实现只让满足教师置信度条件的像素进入无标签损失。UnCL 则通过连续的不确定性权重调整各像素贡献，原设计并不是简单丢弃全部低置信度区域。DyCON 在 ISLES’22、BraTS’19、LA 和 Pancreas 上进行了医学半监督实验。[项目代码](https://github.com/DLwbm123/SSL_CL_seg/blob/8a6e93a8bc39554c878cffe0dedf7fbfefe7d16d/experiments/lcrseg/ssl_anchored_mix_v0_1/core.py)；[DyCON 会议论文页](https://openaccess.thecvf.com/content/CVPR2025/html/Assefa_DyCON_Dynamic_Uncertainty-aware_Consistency_and_Contrastive_Learning_for_Semi-supervised_Medical_CVPR_2025_paper.html)

原论文中，UnCL 的形式为：

$$
\mathcal L_{\mathrm{UnCL}}
=
\frac1N\sum_i
\frac{
\mathcal D(p_i^s,p_i^t)
}{
e^{\beta H(p_i^s)}
+
e^{\beta H(p_i^t)}
}
+
\frac{\beta}{N}\sum_i
\left[H(p_i^s)+H(p_i^t)\right],
$$

其中论文使用 MSE 作为距离项 $\mathcal D$，$H$ 为预测熵。[论文全文](https://arxiv.org/html/2504.04566v1)

### 什么时候它比 JML 更值得优先做？

我的判断是：**当你的诊断显示，视杯、薄视盘环或边界区域因为低置信度长期没有进入无标签监督时，UnCL 更有针对性。**

反过来，如果教师对错误区域非常自信，连续不确定性加权也未必能解决问题。因此应先区分：

$$
\text{“有用像素被排除”}
\quad\text{与}\quad
\text{“错误像素被高置信度接纳”}.
$$

它们不是同一个问题。

### 为什么我只建议先抽取 UnCL？

DyCON 另一个损失 FeCL 在特征空间进行对比学习，原实现还配有 ASPP 和卷积投影头。把 FeCL 整体加入，就不再是仅改变输出损失，会增加新的可训练组件。[论文全文](https://arxiv.org/html/2504.04566v1)

对你当前以固定参数范围做比较的阶段，建议先建立 **UnCL-only 改造版**，保持骨干和参数白名单不变；完整 DyCON 则作为另一种框架实验，不能混用其名称和论文性能。

还有一个需要认真核对的细节：上式包含**正号熵项**。从优化方向看，最小化正熵项倾向于降低熵，不能仅凭“uncertainty-aware”就把它解释成自动校准或自动避免过度自信。学生熵是否参与梯度、距离项是否仍为 MSE、是否继续使用原来的硬阈值，都必须明确记录，改动之后就是新的变体。[论文全文](https://arxiv.org/html/2504.04566v1)

官方代码在 `CVML-KU/DyCON`；我已确认其论文对应关系和公开实现。[作者代码](https://github.com/CVML-KU/DyCON)

---

## 五、眼底任务特别值得研究：D-Convexity，但接法容易犯错

这篇论文是本次检索中**与你的数据集最直接相关**的结构工作之一：作者在 REFUGE、RIM-ONE-r3 上研究视盘/视杯，并包含 REFUGE 训练后直接在 RIM-ONE-r3 上评估的实验。注意，这不是持续学习实验，不能据此推断它已经解决遗忘。[论文全文](https://arxiv.org/html/2605.19210v1)

### 1. 它在约束什么？

核心思想是：对概率掩码函数 $u$，希望不同阈值下得到的前景区域都保持凸性。数学上对应准凹性：

$$
u(tx+(1-t)y)
\geq
\min\{u(x),u(y)\},
\qquad t\in[0,1].
$$

论文进一步构造一阶、二阶可微约束，能够通过离散差分卷积计算。[会议论文页](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_D-Convexity_A_Unified_Differentiable_Convex_Shape_Prior_via_Quasi-Concavity_for_CVPR_2026_paper.html)

### 2. 你的 class 1 是“视盘环”，不是“完整视盘”

按照你的既有标签定义：

$$
p_{\mathrm{background}}=p_0,\qquad
p_{\mathrm{rim}}=p_1,\qquad
p_{\mathrm{cup}}=p_2.
$$

完整视盘应对应：

$$
\boxed{
p_{\mathrm{disc}}=p_1+p_2,\qquad
p_{\mathrm{cup}}=p_2.
}
$$

你的协议把 class 1 明确标为 `optic_disc_rim`，class 2 为 `optic_cup`。[项目协议：`DOMAIN_PROTOCOL.yaml`](https://github.com/DLwbm123/SSL_CL_seg/blob/46e892960240543c946c570a9378d409b226384b/experiments/lcrseg/docs/di_dmpa_jascl/DOMAIN_PROTOCOL.yaml)

因此，我建议考察的改造是：

$$
\mathcal L_{\mathrm{shape}}
=
\mathcal L_{\mathrm{convex}}(p_1+p_2)
+
\mathcal L_{\mathrm{convex}}(p_2),
$$

**而不是给 $p_1$ 单独施加凸性约束。**环形区域本来就可能不是凸集，错误约束会和真实标签冲突。

这里还有一个容易被包装成“解剖先验”的无效项：

$$
p_{\mathrm{cup}}\leq p_{\mathrm{disc}}
\iff
p_2\leq p_1+p_2.
$$

在这个定义下，它恒成立。**单独惩罚这个不等式，没有新增约束能力。**真实的几何包围、边界关系和形状规则，不能由这个恒等关系替代。

### 3. 不要忽略论文的投影模块

D-Convexity 的完整方法包含 **CGPM：凸性梯度投影模块**。论文也明确说，仅使用损失未必能在推理阶段强制得到凸形状。它的主要设置使用了迭代投影，存在额外推理开销。[论文全文](https://arxiv.org/html/2605.19210v1)

所以需要区分：

| 接入方式 | 能说什么 |
|---|---|
| 只抽取凸性损失，保留你的原推理路径 | 一个轻量的形状正则化改造，需要重新验证效果 |
| 使用原论文 CGPM 框架 | 更接近完整方法，但改变了训练/推理流程及计算预算 |

我建议先审计**当前训练标签的形状是否符合近似凸先验**，再决定是否尝试 loss-only。不要为了形状规则，把真实解剖变异或标注中的合法凹陷强行抹平。

作者仓库 `ShengzheC/D-Convexity` 已公开，`loss.py` 是一阶、二阶损失，`CGPM.py` 是投影框架，这两者在代码上也是分开的。[作者代码](https://github.com/ShengzheC/D-Convexity)

---

## 六、Topograph、Skeleton Recall、TopoSemiSeg 该放在什么位置？

我的建议是：**先根据错误类型决定是否使用拓扑损失，而不是因为“医学分割需要拓扑”就默认添加。**

### Topograph：处理“分成几块、有没有错误孔洞”

Topograph 从预测与标签的组合连通图中识别拓扑关键错误区域，而不是均匀惩罚所有错分像素。它的损失作用于这些关键区域的预测分数。原方法的图构造和理论是二维的，不能把逐切片应用直接解释成三维拓扑保证。[论文全文](https://arxiv.org/html/2411.03228v2)

对你而言，适合的触发条件是：视杯频繁分裂成多个区域，或者出现额外孔洞、错误连接。**如果主要问题只是轮廓整体偏移，CWMI 或其他边界/形状约束可能更直接。**

### Skeleton Recall：很适合血管，但你的 Fundus 不是血管任务

它强调细管状结构的骨架召回，适合血管、神经等连通结构。视盘和视杯是区域分割，不能把“细管连通性”的目标直接当成它们的首要结构目标。[会议论文页](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/9904_ECCV_2024_paper.php)

### TopoSemiSeg：处理“不可靠伪标签中的拓扑”

这篇 ECCV 2024 工作最值得借鉴的是：**没有把教师预测的全部拓扑当成真理，而是区分拓扑信号和拓扑噪声，再建立一致性。**它原本针对病理腺体和细胞核中的遗漏、错误合并/分裂。[文献记录](https://pubmed.ncbi.nlm.nih.gov/40557360/)

所以它更适合作为第二阶段候选：先确认你的教师伪标签确实存在大量拓扑错误，再引入相关约束；否则复杂度可能超过当前问题所需。

---

## 七、接入你项目时，有四个比“选哪个 loss”更重要的细节

### 1. 结构损失不能无脑作用于混合后的整张合成图

LCTX 的输入不是完整的单一患者解剖图像。混合矩形会人为制造边缘；FULLMIX 的标签也可能把多个患者的局部结构拼在一起。你当前通过 `collect_sources()` 把预测恢复到原始源图位置，这应是结构约束的优先接入点。[项目代码：`collect_sources()`](https://github.com/DLwbm123/SSL_CL_seg/blob/8a6e93a8bc39554c878cffe0dedf7fbfefe7d16d/experiments/lcrseg/ssl_anchored_mix_v0_1/core.py)

我的具体建议是：

> **CE 可以按合法混合标签计算，但凸性、连通性和多尺度结构约束，要先确认它们对应的是哪一个真实解剖对象。**

尤其不能要求一张被矩形剪贴过的 synthetic mask 保持“完整单个凸视杯”。

恢复源图预测后也不是完全没有接缝问题：不同区域来自不同上下文，接缝附近仍可能不一致。因此应该区分真实解剖边界和混合接缝，观察新损失究竟在修复哪一种错误。

### 2. 你的 soft CE 换成 KL，并不自动产生新的学生训练信号

在教师分布固定、掩码及归一化相同的情况下：

$$
\operatorname{CE}(q,p)
=
H(q)+\operatorname{KL}(q\|p).
$$

因为 $q$ 已停止梯度，所以 $H(q)$ 对学生是常数，两者对学生的梯度相同。

你当前代码也已经分别计算了 soft CE 和教师熵。因此，**“把 soft CE 换成 KL”本身不值得作为一次主要方法实验**；JML、结构项或不同的不确定性机制，才真正改变监督内容。[项目代码：`soft_target()`](https://github.com/DLwbm123/SSL_CL_seg/blob/8a6e93a8bc39554c878cffe0dedf7fbfefe7d16d/experiments/lcrseg/ssl_anchored_mix_v0_1/core.py)

### 3. 概率损失与离散结构分析，要区分计算路径

用于学生优化的概率不能先整体 `argmax` 再拿去算普通重叠损失，否则梯度路径会断。某些拓扑方法可以使用离散预测定位关键区域，但最终损失仍需回到可微的预测分数上；Topograph 就属于这类设计。[论文全文](https://arxiv.org/html/2411.03228v2)

同时，教师概率应该停止梯度，但不能把整个新损失也一起停止梯度。验收应检查新项对允许训练参数是否确实产生了梯度，而不只是确认总 loss 能执行 `backward()`。

### 4. 当前域损失变好，不等于持续学习变好

这些候选多数首先解决分割监督、结构质量或半监督学习问题，而不是直接为你的持续学习协议设计。

因此，我建议把以下结果同时作为观察对象：

| 观察维度 | 需要回答的问题 |
|---|---|
| 当前域 Dice | 新域是否学得更好？ |
| 旧域 Dice、绝对遗忘 | 是否以破坏旧域为代价？ |
| rim/cup 分类别指标 | 是否只是大区域掩盖了小区域损失？ |
| 边界指标 | CWMI/形状项是否真的改善边界，而非只改面积？ |
| 连通分量与孔洞错误 | 使用拓扑损失时，目标错误是否真的减少？ |
| 无标签有效像素与梯度贡献 | 新的 SSL 项是否真的监督到了前景？ |
| 实际时间、显存和前向次数 | 是否只是投入了更多计算？ |

这部分是我建议的验证框架，不是文献已经给你提供的性能保证。

---

## 八、我建议你怎样安排下一轮实验？

### 第一轮：只验证一个医学结构损失

**优先选择 CWMI，做一个小型 2×2 对照：**

| 训练配方 | 原 CE＋Dice | CE＋Dice＋CWMI 结构项 |
|---|---:|---:|
| 干净有标签训练 | A | B |
| LCTX | C | D |

它可以分别回答：

$$
B-A:\quad \text{结构损失本身是否有效？}
$$

$$
D-C:\quad \text{它在 LCTX 上是否有增量价值？}
$$

还可以考察：

$$
(D-C)-(B-A),
$$

判断它与 LCTX 是否存在额外的配合效果，而不是仅仅让所有训练方法都获得类似改善。

**不要只跑 D，再与旧实验中的 A 比。**那样损失变化、混合配方和可能的参数设置差异会纠缠在一起。

D-Convexity 可以作为另一个独立候选，前提是标签形状审计支持其先验。第一轮不建议同时加入 CWMI、凸性、拓扑三种结构项。

### 第二轮：用 JML 检验无标签增量

固定有标签训练配方后，比较：

| 分组 | 无标签监督 |
|---|---|
| U0 | 不使用无标签监督 |
| U1 | 当前 soft CE |
| U2 | 当前 soft CE＋JML |

这里的关键不是 U2 最终是否最高，而是：

$$
U1-U0,\qquad U2-U1.
$$

前者验证原有无标签监督是否有价值；后者验证新区域损失是否提供额外价值。

UnCL 可在发现低置信度前景长期缺乏监督后，作为单独比较加入。它同时涉及距离函数与权重机制，最好不要在首次测试中又加入 JML、又取消阈值、又引入 FeCL。

所有这些属于**新的实验协议**。你之前已经冻结的 LCTX/FULLMIX 归因实验，应保持原样，不在运行中途更换损失。

---

## 最终选择建议

综合论文机制、你现有代码和实现代价，我会这样排序：

| 你的目的 | 首选 | 选择依据 |
|---|---|---|
| **不改模型，补充医学结构监督** | **CWMI，ICML 2025** | 与源图重建后的 LCTX 预测有明确接入位置，不依赖旧原型或额外教师 |
| **让软伪标签增加区域级监督** | **JML，NeurIPS 2023** | 兼容软标签，接入代价低，能与现有 soft CE 构成清楚的增量实验 |
| **利用被阈值排除的不确定前景** | **DyCON 的 UnCL，CVPR 2025** | 连续不确定性加权，但需核对熵项和梯度语义 |
| **解决视盘/视杯形状异常** | **D-Convexity，CVPR 2026** | 数据与解剖目标最贴近，但要区分 disc/rim，以及 loss-only 与完整 CGPM |
| **解决断裂、孔洞、错误连通** | **Topograph，ICLR 2025** | 先有拓扑错误证据，再投入相应复杂度 |

这些优先级是我基于上述文献和你实现的选择，不是已知的实验胜负。

**具体到你的下一步，我最推荐的是：先做“LCTX＋源图坐标下的 CWMI 结构监督”，再单独做“soft CE＋JML 的无标签增量验证”。**前者检验医学结构约束，后者检验半监督收益；把它们分开，既更容易找到真正有用的部分，也更容易形成可解释的研究结论。
