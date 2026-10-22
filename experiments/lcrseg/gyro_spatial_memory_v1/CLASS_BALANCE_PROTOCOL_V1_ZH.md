# 逐图类别平均CE × 全支持encoder均衡：冻结协议 V1

2026-10-22首次实际运行前冻结。独立训练目标消融，非论文复现、不声明新颖性。

## 依据与假设

前五轮均未通过综合门槛。最近GRP相对入口旧LV为−1.275709/−.846752pp，MYO−.434347/−.410796，RV+1.715344/+2.160419。均值掩盖类别差异，但这并不证明像素不均衡是原因。新的可证伪问题：减少大区域对监督CE的主导，是否补充encoder均衡并改善旧域患者尾部，同时保留新域适应？不按验证类别定权重，不针对LV/MYO单独加权。若失败停止本设计，不扫描权重/混合比例。

## 四组与公式

新流20261221/20261222，每流GR、GRP、GRB、GRPB。GR原CE；GRP原CE+encoder polar；GRB逐图类别平均CE；GRPB同时两者。四组均保留原lambda_R=.1144953842725508只读旋转P32谱。源163/S†69共同入口、空SGD，P/Q/W_R/bias不重抽不校准；不使用decoder冻结或范数约束。

B只使用当前2张监督图的标签，逐图现有类别集合C_i（包含背景），每个类别平均该类像素CE，再平均现有类别和batch。实现每像素权重HW/(|C_i|*n_ic)，与原pixel_ce相乘再.mean。缺失类别不进入分母；单类图严格退化原CE。权重仅由当前标签决定，不读取验证、旧源图、U标签或全局类频率。CE总像素权重逐图均值1；不保证梯度幅度相同，clip1依旧，固定lambda的相对作用变化属于该干预，不伪称方向唯一改变。

所有原生允许参数训练，sigma/grad_update固定。原矩阵/向量两optimizer组SGD lr.001、Nesterov.9、矩阵wd.0005/向量0、globalclip1、普通momentum不投影、EMA.999。P保持原8encoder矩阵float32薄SVD gesvd，全数值支持tau=maxshape eps32 smax；Dprime=||D||F/sqrt(r)UrVrT，零→零。理论energy rtol2e-5、实际误差<=1e-7+.005proposalnorm。仅同一状态提议能量守恒，不同轨迹不匹配能量。

## 数据与终点

严格复用Philips train-L白名单：2监督+14 L池image-only，3view48encoder、64位置行P32谱；无U accessor/隐藏标签/旧源图/teacher/replay/bank/generator/额外统计。内部U_reads14000仍是L池image-only计数。冻结HDF5/splits/旧环境不改。最终student2000唯一主终点，EMA仅辅助；无中途验证/选best，退出训练后固定评估。共享源与23旧/31新验证患者，双流不是独立源重复。

## 工程和硬计费

CPU/CUDA各5小矩阵polar自检共10 SVD，0医疗forward0optimizer；另用固定合成logits验证类别平均公式、缺失类别排除、单类别退化CE，0额外SVD/反向/optimizer。64 detach为MINIMAL+P+B vs S+P+B逐步model/opt/EMA/RNG exact discarded；8组资格各32实际更新含onebatch/golden/非空恢复重放/两样本overfit，全PASS才正式。资格overfit同样使用对应CE，不悄悄用普通CE。

总16320opt=16000formal+256qual+64detach，functional0。8192polar步=8000formal+128qual+64detach→65536 SVD+10数值=65546。B无额外model forward/反向/SVD，不校准lambda。失败恢复计原账原钟，pending异常不盲重放。

Calls cap：detach native160/encoder1632/reverse64；qualification768/1536/320；formal32400/770000/16032；diagnostic reverse80；evaluation native23136。总native56464 encoder773168 reverse16496。预期成功实际detach128/1536/64、资格640/1152/280、formal32096/768768/16024、diag80、eval23136。40固定step0/499/999/1499/1999诊断，复用forward不选参数。8组每组eval2892=student旧544新660源train230目标train254+EMA544660。

## 预声明科学门槛

主比较GRPB−GRP：旧两流均值≥+.3pp且每流>0；新每流≥−.3pp。GRPB相对入口旧每流均值≥−.5pp；旧入口损失>1pp人数比GRP每流至少少2人，>2pp人数不得增，入口p10不得比GRP低超过.3pp。另GRPB−GRB旧两流均值≥+.2pp且每流>0、新每流≥−.3pp，用于检验全支持均衡超越单纯逐图类别平均CE。上述全满足才复合PASS。

交互(GRPB−GRB)−(GRP−GR)：旧每流>0且均≥+.2、新每流≥−.3时仅称固定两流点估计交互，不作因果普适或临床安全声明。分列GRB−GR、GRP−GR、GRPB−GR。完整报告患者均值/中位/p10/改善下降/>.5/1/2pp/LV-MYO-RV/ED-ES、配对bootstrap10000 seed163、train-val、EMA与位移。原源→入口−.93369236648307pp仍计入。患者CI不涵盖训练随机性、多重比较和反复开发。

## 资源、停止、交付

仅pro5000 GPU3 UUID GPU-9ea1d4ae-e403-cb3b-29b3-c63b083bf8ff。新create-only root /data/bmw/SSL_CL_seg/runs/gyro_class_balance_v1_20261022，明确/data/bmw例外。启动前实际mount/capacity/write-read/GPU显存核查，旧数据权重只读共享。预计新增<700MB；3.5GB写保护+.5GB余量，总严格<4GB（含2MB中性ops），不删除旧结果。

首次实际执行原钟8h，7h不新stage、7h45m主动停止；独立timeout27899秒TERM+899秒KILL，8h前结束全部本轮子进程。失败/续跑不重置原钟、预算；不结束他人进程。所有父子argv中性。资格失败、非有限数值、逐图类别平均CE失效/固定投影改变、能量超差、pending异常或预算到限即停止并保留证据。

全部COMPLETE/EXIT0/worker退休后仅一次class_balance_audit，CPU4、0模型forward，审16320计费/8192polar65536+10SVD/40诊断/共同入口空opt/final=latest finite optRNG/reads[2000,14000]/固定投影/EDES及资源。只发布代码、协议、匿名聚合，不公开ID/图mask/权重/私有日志配置；GitHub显式代理、remoteSHA与匿名访问、本地远程回执齐后称交付完成。唯一ssl30分钟监测，健康静默；结果后自主后继，不改本轮门槛。
