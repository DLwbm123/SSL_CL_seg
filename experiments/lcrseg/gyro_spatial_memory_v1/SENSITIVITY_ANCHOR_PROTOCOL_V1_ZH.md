# 可靠源敏感度参数锚与单侧余量：完整冻结协议 V1

2026-10-23，独立新轮SENSITIVITY_ANCHOR_V1。父相容参数锚在固定256步探针C−H梯度归一距.047165590656<.05，P0_STOP、288opt/1804native/862reverse、90.53885秒，工程审计PASS且报告77c3f0cf公开；不重跑/延probe/放阈值。祖先各向同性源参数锚三流旧域尾部全通过但新域损失约1.62pp，完整科学FAIL。由此提出新的保护几何，而不是扫描原强度或延长投影探针。新假设尚无性能证据：对当前可靠源决策敏感的参数施加较强保护，其他方向保留可塑性，可能改善旧尾部与新学习的权衡。不宣称能覆盖未见旧域。

## 完整方法与源敏感度

原源FixMatch43600 EMA163直接接入；固定真实native3×3末头，encoder/prehead decoder可训练。只原13当前train-L患者、batch2、普通CE、原增强/预处理，SGD .001/momentum .9/Nesterov/矩阵WD .0005其他0；2000步、final EMA .999主部署，student全保留。无旧图/U/额外测试/扩容/专家，划分/HDF5不动。普通B不调强。

R完整继承双弱光度源正确支持、detach r=1−JS/ln2、每图均匀无放回512支持像素、类均值、.5保守源pairwise-margin单侧hinge和16训练批独立源类尺度。alpha U(.9,1.1)、beta U(−.05,.05)、clip[−1,1]，同几何。没有DS、投影、区域等权。

新增固定对角敏感度D：独立16个训练批seed20271393，每批源参考两视图提供与R相同可靠支持/r和512采样；一个独立测量student从原源初始化，在eval模式前向同主视图，不进行任何optimizer更新。对选中GT像素，v_p=r_p CE(z_measure,p,y_p)，按每批存在GT类先均值再类均值，得到L_rel,t；D_i=(1/16)sum_t (∂L_rel,t/∂θ_i)^2。仅可训练参数入D，固定末头/buffer/EMA排除。autograd.grad只作用于测量student，不写其.grad，不给冻结源参考梯度。测量前后源/测量student状态和全局RNG精确不变。该量是批均值梯度的平方对角敏感度，**不是Fisher信息、Hessian、逐像素梯度平方或旧域重要性估计**；类内梯度可能抵消是已知限制，不用验证选择批/层/阈值。

D完全detach、一次计算、无在线更新/裁剪/阈值/层选择。定义各向同性P=sum_i(θ_i−θ0_i)^2/(sum_iθ0_i²+1e−12)；敏感度锚S=sum_i D_i(θ_i−θ0_i)^2/(sum_iD_iθ0_i²+1e−12)。D为0的坐标不受S约束。R的像素可靠性与S的参数非均匀保护组成完整候选，不宣称正则或归一化新颖。原zero-centered WD与源锚并存。D张量私有，新增一份小型参数同形张量计入预算。

## 五组和预注册门禁

新seeds20271301/02/03×B/R/I/S/C×2000：B固定头CE；R=CE+λR R；I=CE+λR R+λP P（父完整各向同性方法对照）；S=CE+λS S；C=CE+λR R+λS S。全部每步同6native前向并计算全部标量，仅优化指定项。R权重R/I/C共享，S权重S/C共享，P仅I；各权重在独立B探针后16训练批92一次.25 sqrt(CE梯度总能量/各辅助总能量)，不重新放大组合，不扫λ。

P0工程32和B探针256 seed20271390；敏感度16批93；源margin尺度16批91；探针末态测梯度16批92；探针丢弃。D要求全部有限非负、总质量>1e−12、sumDθ0²>1e−12，敏感度采样各前景可靠支持覆盖>=.05且>=3患者。尺度finite>1e−6；CE/R/P/S梯度能量>=1e−12、λfinite[1e−6,1e4]；92批各前景支持>=.05/>=3患者、R实际前景标量份额>=.05；R/P/S encoder与decoder梯度各非零；S−P梯度归一距>=.05；加权R+S总辅助能量>=1e−12且finite；探针S有限且>0；源/测量模型/RNG不变。参数锚P/S没有类别loss分解，不伪造前景贡献。

任何失败P0_STOP，不能延probe/换seed/改门槛/再取敏感度。CPU/CUDA几何/margin和新锚合成自检：源点零损失零梯度；已知D=[1,0]位移的解析损失/梯度；D=0方向自由；D全1严格退化P；冻结排除/源detach。随后B/C各32零辅助全状态等价，再15×32资格含既有golden/反向/非空opt+EMA+RNG恢复/两例overfit；全PASS才正式。复用训练engine，独立协议不改V0.1。

正式0/999/1999同批CE/R/P/S四梯度、S-P内积、P/S值/R活跃率诊断，无额外forward。D以固定文件加载到各worker，不进入训练参数、optimizer或EMA；不从验证修改D。

## 硬预算

30832 optimizer=288P0+64detach+480资格+30000正式。native caps按P0/detach/资格/正式/评估为3200/1024/6000/180000/43380；reverse1200/128/1200/30400/0。预期P01900native/372reverse（包含敏感度96/16及比父多16个S测量反向）；detach384/64；资格1350/510；正式180000/30180；评估43380/0，总227014native/31126reverse。sourceSVD1、syntheticSVD2/reverse12另列，合成限16/32。D校准零optimizer，不借bucket，不追加评估或种子。

pro5000仅GPU3既定UUID，create-only /data/bmw/SSL_CL_seg/runs/sensitivity_anchor_v1_20261023，/data/bmw明确例外；所有输出/cache/tmp/敏感度/log在root，仅小ops中性entry。启动前真实挂载容量/写读探针/GPUfree>=18000，父子argv和nvidia核中性。原始8h时钟不重置，25200秒不新stage/27900停止，独立timeout28600 TERM/30KILL；严格<4GB，3.5GB写保护。累计30GB继续（27GB准入+3GB恢复），父P0保守85008192B，本轮4.05GB预留，累计7151289344B<27GB。继承只读preparation/代码，不复制数据模型，不删除历史。

## 科学判定与交付

C每流旧均值>=源80.36740663770262%，源配对患者CI下界>=−.30pp；下降>1pp人数<=2、>2pp=0、p10>=−.50pp，>1/>2人数不劣B。新绝对>=83%，C−B新均值/CI下界>=−.20pp。C−B旧每流>0且均值>=.30pp；C−R旧每流>0均值>=.15pp；C−S旧每流>0均值>=.15pp；C对R/I/S两域均值/CI下界均>=−.20pp。新几何必须恢复相对各向同性锚I的新域：C−I新每流>0且均值>=.30pp。所有门槛才综合PASS。不得通过新域恢复掩旧尾部失败或仅凭某消融通过宣称完整有效。

全部15组结束后才读性能。共享源163、旧23/新31重复开发患者，EDES共同bootstrap10000 seed163，CI不覆盖源随机/多重开发，非独立测试。P0_STOP只表示门禁，不是性能阴性。全退出/EXIT0/workers退休且AUDIT不存在时仅一次sensitivity_anchor_audit，原中性entry换cfgmodule，零forward。公开完整student/EMA/类别/患者配对与尾部匿名聚合、report/summary/audit/status及本地远程DELIVERY，再按实证推进完整主线。不公开D/权重/患者行/图mask/私有配置日志。GitHub全程有效代理，无直连/强推/PR。
