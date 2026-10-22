# 类别平均CE的输出梯度范数匹配：冻结协议 V1

2026-10-22，首次实际执行前冻结。独立尺度分离诊断，非ICLR论文复现，不声明新颖性。

## 依据和可证伪预测

已完成类别平均CE轮：GRPB−GRP旧两流均−1.197861pp、新−3.104520pp；旧患者尾部两流恶化。20个宏CE固定诊断点全部global gradient norm>1，而20个原CE点仅2个>1，首步宏CE约为原CE的31倍。类别重加权同时改变方向、尺度和谱项相对作用，当前数据不证明剪裁导致损害。

本轮只检验：同一批当前监督输出上，将宏CE解析logits梯度范数匹配原CE，是否恢复宏CE的患者损失，能否进一步超过原CE。无类别权重/混合比例/阈值扫描；不修改前轮或重评旧模型。若仅优于原宏CE但仍不及原CE，只称尺度分离的部分恢复，不称成功。

## 三组两新流

训练流seed20261225/20261226，每流GRP（原CE）、GRPB（原宏CE）、GRPBN（宏CE×停止梯度范数匹配系数）。三组均使用原encoder全数值支持等F polar。共同源163/S†69入口、空SGD、lambda_R=.1144953842725508的固定旋转P32谱。P/Q/W_R/bias只读，不重新抽样/校准。全decoder正常训练，无冻结/范数retraction。原SGD两组矩阵/向量lr.001 Nesterov.9、matrixwd.0005/vector0、globalclip1、EMA.999；sigma/grad_update固定，普通momentum不投影。

## 唯一干预及数值语义

与前轮完全相同的逐图现有类别平均（包含背景）：每像素权重w_i=HW/(|C_image|*n_class)。在当前监督logits z上，p=exp(logsoftmax(z))、g=p−onehot(y)，取整个batch/类/像素的F范数。alpha=||g||F/||w*g||F，alpha停止梯度。GRPBN损失为mean(alpha*w*pixelCE)，GRPB为mean(w*pixelCE)。alpha为一个batch标量，严格保持宏CE相对类别方向；不引入混合系数或额外损失。CE.mean的共同分母BHW在比值中抵消。两个范数同时零时alpha=1；非有限、仅单侧零、非正alpha立即失败。不对alpha作阈值剪裁。

解析统计使用现有logits.detach()，在float64中归约范数，alpha转换回CE dtype；匹配相对误差<=2e-5。权重/alpha仅当前2张L监督图，不访问U/旧源图/验证标签、不累计全局类别频率。计算不额外调用医疗模型forward/encoder/反向/SVD。不会改变本轮同组采样。**匹配仅在logits空间，不保证网络参数梯度范数、全局clip系数、实际SGD位移或不同轨迹的能量相等。** alpha随批次变化但停止梯度，不声称优化固定标量目标。

P保持原8encoder矩阵float32薄SVD CUDAgesvd，tau=maxshape eps32 smax，全数值支持Dprime=||D||F/sqrt(rank)UrVrT，零→零。D含momentum/decay/clip，理论energy rtol2e-5，实际误差<=1e-7+.005proposalnorm。每次范数匹配记group/phase/factor/三个logit梯度范数/相对误差/零梯度标记；不保存图或标签。

## 数据、终点与工程

仍仅Philips train-L白名单，2张监督+14张L池image-only，3view48encoder/64行P32谱。无U accessor/隐藏标签/旧源图/teacher/replay/bank/generator。U_reads14000只是L池image-only计数。HDF5/splits/env不改。固定student2000为主终点，EMA辅助，不中途验证/选best。

CPU/CUDA各5小矩阵polar自检，共10SVD；两设备各用固定合成logits的2次autograd.grad核对原CE/匹配宏CE实际输出梯度范数、宏CE方向、单类别退化与零梯度安全，共4次**合成反向**，0医疗forward/optimizer。单列不冒充训练调用。

64detach：MINIMAL+P+BN vs S+P+BN，32步逐步model/opt/EMA/RNG exact discarded。6资格各32实际更新含onebatch/golden/非空恢复重放/两样本overfit；overfit也使用对应CE。全部通过才正式训练。预声明每组step0/499/999/1499/1999诊断，30个固定点，不用来选超参。

## 硬账与资源

12256opt=12000formal+192qual+64detach，functional0。所有组都P，故12256polar步/98048SVD，加10数值SVD共98058。额外模型forward/反向0；合成数值反向4另列。失败/恢复计原账原钟，pending异常不盲重放。

调用cap：detach native160/encoder1632/reverse64；资格576/1152/240；formal24300/578000/12024；diagreverse60；evalnative17352。成功预期实际detach128/1536/64；资格480/864/210；formal24072/576576/12018；diag60；eval17352。6组每组固定eval2892=student旧544新660源train230目标train254+EMA旧544新660。总医疗native42032/encoder578976/reverse12352，另4合成反向；以上cap严格保留。

仅pro5000 GPU3 UUID GPU-9ea1d4ae-e403-cb3b-29b3-c63b083bf8ff。create-only root /data/bmw/SSL_CL_seg/runs/gyro_logit_norm_v1_20261022，沿用明确/data/bmw例外和存储包装器。预先mount/free/write-read/GPU检查；旧资源只读共享。预计700MB，3.5GB写保护+.5GB余量，总<4GB含2MB中性ops；不删除历史。首次实际运行8h原钟，7h不新stage、7h45stop；独立timeout27899s TERM+899KILL，全部本轮子进程8h内结束。父子argv中性。

资格失败、非有限、范数匹配误差、P固定常量/能量检查失败或预算到限立即停止，保存证据，不改门槛或临时重启。

## 冻结科学门槛

主GRPBN−GRP：旧均>=+.3pp且每流>0，新每流>=−.3pp；GRPBN相对入口旧均每流>=−.5pp；入口旧下降>1pp人数比GRP每流至少少2，>2pp不增加，p10不比GRP差超过.3pp。另需GRPBN−GRPB恢复门槛：旧两流均>=+.5pp且每流>0，新两流均>=+1pp且每流>0。全满足才compositePASS。单独恢复门槛通过只支持有限的尺度解释，不代表整体成功或临床安全。三组不做四格交互推断。

报告两个对照差、原B−原CE复现、患者中位/p10/改善下降/>.5/1/2pp、LV/MYO/RV、ED/ES、train-val、EMA、30梯度诊断及alpha分布。配对bootstrap10000seed163。旧原源→入口−.933692pp仍计，共享23旧/31新患者和源入口不是独立源重复，患者CI不含训练随机/反复开发/多重比较。

## 完成与交付

全部COMPLETE/EXIT0/workers退休/AUDIT不存在时，仅一次logit_norm_audit，CPU4中性entry、0forward。核12256opt/98058SVD/4合成反向/精确医疗calls/30诊断/范数匹配证据/入口空opt/final=latest finite optRNG/reads[2000,14000]/固定投影/EDES/资源。不可复用四组auditor的原预算。公开源代码、协议、匿名报告；排除患者ID/图mask/权重/私有配置日志。GitHub显式代理，远端SHA/匿名访问/本地远程回执齐后称交付完成。唯一ssl30分钟监测、健康静默，结果后自主后继。
