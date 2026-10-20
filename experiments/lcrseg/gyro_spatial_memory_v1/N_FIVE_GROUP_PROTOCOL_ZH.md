# N五组贡献分解 v1（2026-10-20，启动前冻结）

用户授权的一个新目标随机流；旧A/B/C/M/N及所有诊断、终点和报告不变。只做本五组，不预启动第二流、SG/T、F、权重扫描或其他条件后续。

|组|当前监督|额外14图|几何|frame|
|---|---|---|---|---|
|S|Philips L2 CE|无|无|无|
|G00|同一L2|L池仅图像、标签不读取|64行/48 encoder图|无|
|G10|同一L2|U14|64行/48 encoder图|无|
|G01|同一L2|L池仅图像、标签不读取|64行/48 encoder图|有|
|G11|同一L2|U14|64行/48 encoder图|有|

共同S†原69步student入口、源训练固定P及现有frame只读；模型无generator/code，不加载优化记忆或G。原允许native encoder/decoder/head全部训练，sigma/grad_update固定。空SGD，LR=.001/Nesterov=.9，矩阵wd=.0005、向量0、global clip1。L2 CE归一化不变，几何lambda=.1、D32/R4、centroid=.5/.25/.25、beta=1/3。有frame y=(1-beta)c+beta*a；无frame y=(1-beta)c，分母同sqrt(n min(n,D))，不把beta设0。无frame理论kappa<=2/3，原始loss不能横比。有frame的anchor匹配仍按detach(c)选择。S算力较少，明确不是算力匹配对照。

目标随机流seed=20261020，仅改变目标采样/强增强/位置；源入口与P保持原163。patient→phase→slice，监督L2与额外槽位独立stream；每一槽位的两种强增强、四象限位置由(seed,step,slot,view/position)独立决定，不依赖拼接batch大小。五组监督L2相同，四几何组当前两槽位置和增强严格同规则。L补图和U补图保持每对不同患者，跨对可重复；保存匿名每批独立图数及重复率（监督2与额外14一起统计），禁止读取补L图的label。新随机流不声称重现旧N逐位置流。

各组工程32实际SGD调用（含恢复重放，丢弃），全部工程通过后固定顺序S/G00/G10/G01/G11各2000正式；固定最后student为主，统一EMA .999为预声明辅助部署口径。不中途验证、不选best、不改终点、不补步。每组独立评估训练worker退出后进行：两验证域、源/目标train-L及EMA两验证域；旧结果复用、不重评。train仅Philips L/U HDF5白名单，val/test及历史源HDF5拒绝；独立评价进程才准合法Siemens/Philips评价。无test/U隐藏GT。

硬预算：正式10000+工程最多160=10160 optimizer；functional/toy=0，失败与重放包含；阶段工程<=32、正式<=2000，不重置计账。新独立create-only root，单一原始时钟，8h硬界：7h不新阶段、7h45m TERM、8h前KILL；存储3.5GB保护+0.5GB退出余量，实际严格<4GB（预计<1GB），所有cache/log/tmp/ckpt在root，共享旧资源只读不复制。仅指定GPU3，启动挂载写读、显存、ps/nvidia-smi检查，中性主子argv。每步及保存guard、独立外部timeout；任何NaN/坏恢复/非法访问/预算/存储失败保留现场停止，不盲重试。

预声明比较分别在旧/新域及student/EMA口径报告，不混合择优：U=G11-G01，新>=+.5pp且旧>=-.5支持实用U贡献；frame=G11-G10，旧>=+.3且新>=-.3支持实用frame贡献；full=G11-S，至少一域>=+.5且另一域>=-.5支持净实用贡献。interaction=G11-G01-G10+G00，两域分别报告；只有患者配对区间支持正交互才称协同，不要求协同为所有研究必要条件。LV/MYO/RV、ED/ES、患者改善下降/中位差、患者bootstrap10000次(seed163)完整报告。单流开发验证非独立泛化；不自动扩第二流。

全部worker退休后一次CPU/收据审计：ledger、实际2000、空opt共同入口、固定P/frame/常量、finite final=latest含opt/RNG/reads、患者EDES、随机流及图像重复统计。匿名中文报告/源码/协议/必要审计选择性代理push，remote SHA+匿名HTTP及本地/远端交付收据齐才称交付；不公开患者、图像、mask、weights、code/frame、原日志或服务器配置。
