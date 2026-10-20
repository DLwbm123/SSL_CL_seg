# 冻结G的同族局部c/R可达性oracle

2026-10-20。用户已批准“U-Net资格、同族oracle、确认可达后再H”的顺序；U-Net24项已通过。本协议在oracle首次实际计算前冻结，不改变旧结果，不运行H或共享R2。

## 问题、来源与边界

固定原P2的spatial_generator/S0600、生成器cursor8675，全100模型参数、prior/initial及旧Adam均只读。只使用G:(c,R)→Z，不使用F/B图像块，不把UNet作为生成器。合格UNet仅作为前阶段资格回执，本轮不前向或训练它。

仅原两canonical Siemens训练L、128×128既定插值、四翻转八视图、原顺序，不读保留/test/U/目标域。局部参数直接使用训练Y优化，必须标作标签条件oracle，不能作为image-only预测、教师真值或患者泛化证据。

同R-START：q(c)q(R)独立对角Gaussian，c16、R16×16×16，八视图各有独立局部参数。四字段为c mean/rawsd与R mean/rawsd；logsd严格使用−4+3tanh(raw/3)，不扩大范围、不夹断替代、不去KL或改类别权重。A=期望像素类别CE+(KLc+KLR)/N，N=128²，batch均值，无额外global KL常数。

## 两个固定初态与优化

- ZERO：四字段全零，因此均值0、logsd−4。
- R1_WARM：从已完成R1/S0600的e/H读取相同八图像视图的原始四字段，直接用Hc/HR原始输出，逐项核对映射后与原posterior相同；不从饱和logsd执行atanh。旧H只读前向一次，0更新。

每初态恰1000次新的局部Adam更新/1000反向。Adam lr.03、betas(.9,.999)、eps1e−8、weight_decay0、foreach=False，四字段全局clip1，独立freshAdam，FP32、严格确定性、无AMP/TF32。该预设步长用于自由局部变量，不是网络H学习率，不从当前效果调选，不声称1000步等于全局最优。局部初态无新增随机初始化；MC噪声不是新增训练seed扫描。

每步2个G后验Theta×每Theta2个局部样本，沿用R_START训练噪声生成规则：Theta seed720163+step、local820163+step，两个初态配对。只G权重抽样，不更新全局参数。每100步存完整raw/Adam/RNG/成本，共每臂11份；0/250/500/1000评价，1000唯一主要终点，不选最佳点/初态、不通过资格后提前结束，不因失败增加步数。

## 评价、复用与解释

新评价固定4流620163/630163/640163/650163，每流8Theta×8local。每样本先softmax再平均；同时存mean log-softmax以分开期望类别CE与混合概率NLL。先封存oracle概率与log概率，再独立CPU读Y评价。完整八视图逐类Dice、类2precision/recall、预测面积、NLL/Brier、每流波动、KLc/KLR与A均报告，NA不替换。

R1_WARM step0与已完成R1/S0600的q、G、评价流完全相同，直接复用旧封存PRED_0600及其4流，不重复旧G评分。原文件没有mean-log-prob，故该基线的期望类别CE/A明确为NA并记录原因，不用混合NLL替代。ZERO初态和全部新终态/中间态按上述新评价执行。

主要oracle资格为四流混合概率的24项Dice均≥.8；稳健资格另要求每个独立评价流也全部24项通过。两初态完整报告，不将最佳初态替换主表。任一固定初态得到完整可用读出，支持存在有限预算下的同族局部解，不证明图像推断H已经学到该解。两初态均未通过，只是该配置有限优化未建立资格，不证明G数学不可表达GT；后继再单独区分优化、目标权衡和族限制。

## 成本、硬限与交付

总局部optimizer2000/局部反向2000，全局optimizer0。训练4000Theta/8000批量G前向=64000图像G前向；新评价7个端点共224Theta/1792批量G前向=14336图像G前向，另复用旧R1预测记录1份。初态旧H8图像编码前向/1后验batch，0反向。所有失败/partial与真实成本保留；0全局更新不等于0optimizer或0成本。

pro5000固定GPU2及原UUID，全部source/run/cache/tmp/log/raw/概率在授权数据盘，create-only。启动前实际挂载/容量/写读探针、四根与新路径基线，预计新增<500MB，入场预留2GB；不下载/删除历史。保守共享原R_START首启clock及全部失败占用，首启含重试续跑≤24h、新增实际严格<50,000,000,000 bytes；独立中性guard23h45保存/23h59硬停、新软45GB硬49GB，旧四根18GB软20GB硬继续。原剩余不足不启动，不拆名重置。

完成后独立核验连续局部更新/反向、22份raw/Adam/RNG、两初态与freshAdam、源全模型冻结/gradNone、复用1份旧预测/7份新预测、目标与指标分母、EXIT/FAILURE和worker/guard退休，交付源码/中文正负报告/匿名全指标。GitHub走有效代理；不公开机器路径、标识、像素、模型/raw/Adam、概率数组或原日志。唯一心跳原位更新。

若稳健oracle有完整类别读出，再另立冻结合格UNet编码器与G、只训练H的image-only实验；不把oracle的Y或局部标签条件状态泄漏到部署。若未通过，先交付未决/负结果，另立单变量诊断，不自动改KL、扩大族、加教师/TaskRel/U或重启R2。
