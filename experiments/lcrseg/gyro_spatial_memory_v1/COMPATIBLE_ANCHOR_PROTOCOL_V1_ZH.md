# CE相容源参数锚与可靠余量：完整方法冻结协议 V1

2026-10-23，独立新轮 COMPATIBLE_ANCHOR_V1。父SOURCE_ANCHOR_MARGIN完整12组、一次零前向审计PASS，科学FAIL已公开854d84a1：C−B旧均+1.019270pp，三流旧域患者尾部全部通过；但新域较B均−1.618389pp且绝对<83，C−P旧仅+.041646pp，组合增量不足。新轮仅检验参数锚的方向限制能否恢复当前域适配并保留旧域患者收益，不做父λ/层/归一化扫描。此前DS/CE投影失败保持；本轮对象是已显示尾部收益的参数先验，不复活DS。

## 完整方法和公平对照

全部继承原源FixMatch43600 EMA163、真实native3×3末头固定、encoder/prehead decoder可训练、仅原13当前train-L患者batch2、普通CE、SGD .001/momentum .9/Nesterov/矩阵WD .0005、2000步、final EMA .999唯一部署主终点。共享源及旧23/新31患者重复开发验证，不是独立测试。无U/旧图/扩容/任务专家/测试调参。源模型完全冻结detach，P不包含冻结参数/buffer/EMA，原zero-centered WD继续并存。

R完整继承可靠双弱光度源正确支持、r=1−JS/ln2、每图uniform无放回512、类均值、.5 min源margin单侧hinge及独立训练类尺度；增强alpha U(.9,1.1)、beta U(−.05,.05)、clip[−1,1]，同几何。没有DS辅助项或区域等权。

P=sum所有可训练参数||p−p0||²/(sum||p0||²+1e−12)，原源固定。g=grad CE，d=grad P，均detach；cc=||g||²，dd=||d||²。a=min(g·d,0)/cc（cc<=1e−20时a=0）；c=d−a g；h=d sqrt(||c||²/dd)（dd<=1e−20时h=0）。N/H/C分别采用d/h/c。h与c等范数，区别方向而非单纯缩小。通过sum((p−detach(p))*v)的前向零surrogate施加一阶向量，无二阶。

五组B普通固定头CE；R=CE+λR R；N=CE+λR R+λP d；H=CE+λR R+λP h；C完整=CE+λR R+λP c。全部每步同6 native样本前向，全部计算CE/P梯度及相容投影，只有对应辅助向量进入优化。R权重R/N/H/C共享，P权重N/H/C共享，不对组合重校准。普通B不增强。所有组正式0/999/1999再计算R梯度并存CE-P内积、投影后内积、CE/P/投影/R能量、P值与R活跃率，无额外forward。

仅对辅助P向量的一阶CE内积非负，不保证带momentum/WD/EMA的完整优化步或未来批CE下降，更不保证旧域。R仍可能和CE冲突。不把surrogate零值当辅助机制没有效力；P也没有类别loss分解，不伪造其前景贡献。

## P0与工程门禁

新正式seeds20271201/02/03×B/R/N/H/C×2000。P0工程32+B独立探针256 seed20271290；源尺度16批91；探针末态梯度16批92，探针丢弃。λR与λP各一次.25 sqrt(CE总梯度能量/原R或原P总能量)，全流共享，不扫描。尺度有限>1e−6；CE/P/R/C/H各能量>=1e−12；权重finite[1e−6,1e4]；前景源正确双视图支持各>=5%且>=3训练患者；R原标量前景份额>=5%；投影P在encoder/decoder能量非零；C−H归一梯度距离>=.05；至少一批CE-P冲突；每批c·g>=−1e−6 max(1,||c||||g||)；加权R+投影P组合能量>=1e−12且有限。源/模型/RNG测量前后不变，源/固定头无梯度。

任何失败P0_STOP，不加探针/换seed/改门槛/扫系数。CPU/CUDA原几何与margin自检、P零值零梯度/已知位移恢复/冻结排除/源detach自检、冲突/无冲突/零CE/等范数/surrogate投影自检。通过后B/C各32零辅助全状态等价，再15×32资格（既有golden/反向/非空opt+EMA+RNG恢复/两例overfit），全部通过才正式。复用现有训练engine和数据，不修改V0.1方法。

## 硬预算和停止

30832 optimizer=288 P0+64 detach+480资格+30000正式。native分bucket caps3000/1024/6000/180000/43380；reverse caps1500/256/1500/90200/0。预期P01804/862，detach384/192，资格1350/600，正式180000/90045，评估43380/0，总226918 native/91699 reverse。source SVD1、synthetic SVD2/reverse14另列，合成上限16/32。不得借bucket、补seed或重评估。

pro5000仅GPU3、/data/bmw明确例外，新create-only compatible_anchor_v1_20261023；中性父子entry/argv。启动前实际挂载/写读/显存>=18000MiB核对，所有新输出缓存临时在该root。每轮原始8h总时钟，25200秒不新stage、27900秒停止，独立timeout28600 TERM/30秒KILL；<4GB硬限、3.5GB写保护，含元数据与ops。累计原30GB不重置，27GB准入/3GB恢复。本轮4.05GB（远程<4GB+本地50MB）预留，父完成计470008192B；预期累计7066281152B<27GB。preparation只读继承，不复制数据或历史大权重，不删除历史。

## 预注册科学判定

每流C旧均值>=原源80.36740663770262，源配对patientCI下界>=−.30pp；旧患者下降>1pp人数<=2、>2pp=0、p10>=−.50pp，且>1/>2人数不劣B。新绝对>=83，C−B新均值及CI下界>=−.20pp。C−B旧每流>0且跨流均值>=.30pp；C−R旧每流>0且均值>=.15pp。C对R/N/H两域均值及CI下界均>=−.20pp。针对恢复和方向归因另要求C−N新每流>0且均值>=.30pp，C−H新每流>0且均值>=.15pp。全部满足才综合PASS；不能只用恢复新域掩旧尾部失败，也不能只用旧域保持掩新域损失。

所有15组完成后才读性能，报告所有student/EMA/类别、患者配对和尾部。患者内EDES联合bootstrap10000 seed163；三流共享源与患者，CI不覆盖源随机和多重开发比较。完整EXIT0、workers退休且AUDIT不存在时执行一次compatible_anchor_audit（原中性entry、cfg换module、零forward），P0_STOP也收尾审计。先公开源码/协议再P0；最终公开完整报告/匿名summary/audit/status及本地远程DELIVERY后分析下一完整主线。禁止患者行、图mask、权重和私有配置日志公开；GitHub仅有效代理，不建PR、不强推。
