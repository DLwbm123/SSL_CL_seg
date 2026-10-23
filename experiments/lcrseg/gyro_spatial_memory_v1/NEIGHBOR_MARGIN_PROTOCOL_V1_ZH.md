# 可靠源输入邻域余量保护：完整方法冻结协议 V1

2026-10-23，独立NEIGHBOR_MARGIN_V1。父敏感度锚15组已完整、唯一审计PASS、科学FAIL并公开fb1f804a：C−B旧均+.815523pp，C−I新均+1.222622pp，但C新82.79/82.84/82.72<83、C−B新均−.434428pp、旧>2pp退化每流1人、p10<−.50、C−S旧均+.094365<.15。停止该轮的D/层/强度/采样/参数先验扫描。新问题是只约束观测可靠像素可能覆盖不足，检验当前训练输入邻域的源功能保护；不是把当前邻域等同旧域，不增加旧图/U/测试访问。

## 完整方法

原源FixMatch43600 EMA163直接初始化，固定真实native3×3末头，encoder/prehead decoder可训练；仅原13当前train-L患者、batch2、冻结预处理/增强、普通CE、SGD .001/momentum .9/Nesterov/矩阵WD .0005其他0；2000更新，固定final EMA .999主终点，student完整报告。无DS/参数锚/梯度投影/扩容/专家，原源参考始终eval且参数requires_grad=False。原WD保留。B不增强。

原R原样保留：主输入x和同几何弱光度源视图(alpha U(.9,1.1)、beta U(−.05,.05)、clip[-1,1])均源预测GT才支持，r=1−JS/ln2 detach，每图支持uniform无放回至多512、batch存在类均值、单侧.5 min源margin hinge除以训练类尺度。保留真实卷积/插值路径。

新增邻域：固定可靠支持与r，在原源参数冻结的额外前向上令xg=x.detach().requires_grad_(True)，按同512采样与类均值计算J=sum_class mean_p[r_p CE(z0(xg),y_p)]/存在类数。只autograd.grad(J,xg)，**源参数不接收或存储梯度，也不更新**；得到输入梯度g_x。a=clip(x+.05 sign(g_x),[-1,1]).detach()。零g像素位移0。一次符号步，没有迭代/重启/搜索/随机初始化。它仅是输入一阶上升方向，不保证有限步J增加。当前已有源正确GT支持用于定义训练扰动，非评估标签。

随机对照w=clip(x+.05 ξ,[-1,1]).detach()，ξ为固定私有RNG每像素独立±1（key neighbor_random、随target seed/step）。同为裁剪前L∞幅度.05；由于g=0与边界裁剪，实际L2/L∞不保证精确相同，明确不称等范数。合成噪声无几何变换但也不等于真实 scanner style或严格语义不变，风险通过当前/旧患者表现判定，不引入新数据。

对每个v∈{w,a}：冻结源z0(v)，仅原双视图可靠支持且z0(v)仍预测GT的位置进入支持；权重沿原r detach；每图uniform512，类别与原R同聚合；T_v为r×sum_{j!=y}[ReLU(.5 m0(v)_j−m_student(v)_j)/s_y]^2/3。源对扰动已错误的位置不施加功能保护，普通原x的GT CE始终保留，不对v加监督CE。s_y沿原16批源min-margin尺度，不从验证挑选。w与a支持改变属于完整邻域机制，C−W不单独归因于某一向量角度。

五组B CE；R CE+λR R；W CE+λR R+λW T_w；A CE+λA T_a；C完整CE+λR R+λA T_a。全部每步均计算所有分支和源输入梯度，**16 native样本前向/step**=原6+输入梯度源2+random源/学生4+adversarial源/学生4；一源输入grad调用+一训练backward调用。λR在R/W/C共享，λA在A/C共享，λW只W；各一次.25训练梯度能量校准，不重新归一组合，不扫幅度/比例/λ/视图。

## P0、工程与科学门禁

新seeds20271401/02/03×B/R/W/A/C×2000。P0工程32+Bprobe256 seed20271490；源尺度16批91；探针末态16批92测CE/R/W/A梯度，探针丢弃。尺度finite>1e−6，所有梯度能量>=1e−12，各λfinite[1e−6,1e4]；R/W/A每个分支各前景支持率>=.05且>=3训练患者，实际前景标量loss份额>=.05；R/W/A各encoder与decoder梯度有限非零；A−W归一梯度距离>=.05；λR R+λA A辅助组合梯度能量finite>=1e−12；源输入梯度总能量>1e−12且逐批有限；16批a实际最大位移均>0且<=.050001；源/模型/RNG测量前后不变，源参数和固定头无grad。任何失败P0_STOP，不延probe/换seed/改阈值/扫epsilon/启正式。

CPU/CUDA原几何/margin自检与新合成输入上升符号方向、clamp幅度、零输入grad恒等、detach检查；sourceSVD1syntheticSVD2/reverse12另列。通过后B/C各32零辅助全状态等价；再15×32资格（原golden/单批反向/非空opt+EMA+RNG恢复/两例overfit），全PASS才正式。复用同一训练engine，不改V0.1。正式0/999/1999同批CE/R/W/A四梯度、W−A内积、输入grad能量/扰动幅度/各分支活跃性，无额外forward。

科学C每流旧均值>=源80.36740663770262%，源配对患者CI下界>=−.30pp；旧下降>1pp人数<=2、>2pp=0、p10>=−.50pp，>1/>2人数不劣B。新绝对>=83，C−B新均值和CI下界>=−.20pp。C−B旧各>0均>=.30pp；C−R旧各>0均>=.15；C−A旧各>0均>=.15（组合贡献）；C−W旧各>0均>=.15（定向邻域贡献）；C对R/W/A两域均值及CI下界都>=−.20。全部通过才综合PASS，不凭某一收益掩尾部或新域失败。

## 预声明硬预算与交付

30832optimizer=288P0+64detach+480资格+30000正式。native bucket caps P06000/detach1024/资格6000/正式480000/评估43380；reverse1500/128/1200/60200/0。预计P04574native/633reverse，detach1024/128，资格1800/555，正式480000/60180，评估43380/0，总530778native/61496reverse。额外源输入grad每loss调用均入账（P0工程3+probe258+测量16=277次；资格每组3；detach64；正式每步1）；反向次数为调用数而非相同FLOPs，B也计算同样前向/输入grad但未激活的学生辅助图无需全反传，披露计算结构差异。合成预算16SVD/32reverse。不得借bucket/补seed/重评估。

仅pro5000 GPU3既定UUID，create-only /data/bmw/SSL_CL_seg/runs/neighbor_margin_v1_20261023；/data/bmw明确例外，所有输出/cache/tmp/log在root；中性ops入口与父子argv/nvidia检查。启动前真实挂载容量/写读probe/GPUfree>=18000；preparation只读继承，无数据/模型下载或复制。每轮原始8h，25200秒不新stage、27900停止，独立timeout28600TERM/30KILL；严格<4GB，3.5GB写保护。累计30GB不重置（27GB准入+3GB恢复），父敏感度完整计575008192B，本轮预留4.05GB，合计7726297536B<27GB。若时间或存储触限按协议停止并保留证据，不删历史、不拆名重启。

全部15训练评估完成后才读性能；共享源163/旧23新31重复开发患者，患者EDES共同bootstrap10000seed163，非独立测试，CI不覆盖源随机/反复开发比较。P0_STOP无性能结论。全EXIT0/真实workers退休且AUDIT不存在，仅一次neighbor_margin_audit（cfg换module原中性entry、零模型前向）。先源码/协议公开再P0；终态完整公开student/EMA/类别/配对/尾部匿名summary/report/audit/status及localremoteDELIVERY，再分析下一完整主线。私有患者行/图mask/权重/扰动图/日志配置不公开，GitHub有效代理、不建PR/强推。
