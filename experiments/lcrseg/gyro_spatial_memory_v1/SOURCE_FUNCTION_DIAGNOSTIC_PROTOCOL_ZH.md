# 源功能有限步诊断 v1（2026-10-20，启动前冻结）

与五组目标实验独立的源阶段诊断，不改旧终点，不把历史源图像接入目标训练。只用Siemens train-L10患者20EDES/230切片；不使用源val、目标数据、test或U隐藏GT。按patient ID排序后用torch Generator seed20261021置换，前7 support、后3 query，启动前冻结私有partition；公开只发布计数。P/G/M先前接触全部10患者，7/3划分不是独立泛化。query绝不用于更新或选参数。

固定5状态：s0共同S†入口/空opt；s16、s64为同一support CE轨迹的16与64步（保存真实动量）；sEnc、sDec分别从s0仅扰动encoder/bottleneck、decoder/head可训练参数，逐层Gaussian方向归一化，范数=.001*原参数范数（零参数保持零），seed20261022/20261023，不扫描。扰动状态空opt，sigma/grad_update不动。

各状态3分支各16步，从同参数和opt状态出发：B0全部可训练参数显式zero grad，保留SGD真实momentum/wd演化；BM固定修复M32码、固定G的softCE，8合成图/步；BR当前support真实CE，2图/步。共同实际SGD .001/Nesterov.9/矩阵wd.0005/向量0/globalclip1，无几何、无EMA更新、不改P/G/码。support图和合成码使用seed20261021 stateless stream，分支和状态固定读取规则；s16/64真实动量克隆。BM/BR在每状态入口相对query梯度余弦只作辅助；记录encoder与decoder位移。

每状态baseline query及三分支h=1、16全部3患者6EDES，记录全图平均CE、前景条件CE、逐器官Dice/宏均值；不选best。总query完整遍历35次。效用U_M^h=LQ(B0^h)-LQ(BM^h)。先判断BR vsB0在h16的平均query CE改善>1e-5且>=3/5状态改善>1e-5才称参考可测；否则诊断灵敏度不足，不下记忆功能失败结论。BM预声明门槛h16 CE改善>1e-5在>=4/5状态，跨状态平均前景Dice不下降且各器官平均下降<=.5pp（均相对B0）；只作探索门槛，不当独立统计验证。若扰动不产生恢复余量，照实报告，不调振幅。

更新硬上限336：64状态构造+5*3*16=304正式、工程最多32（包括任何重放/失败），optimizer均实计；functional/toy0。工程固定两support图30步、一次非空恢复重放、一个额外CE=32，要求CE下降且恢复精确，资格终态丢弃。所有分支更新包括B0计账。非更新autograd求导不称functional inner更新，单独限制前向成本。

前向硬caps：real_support512切片、real_query=35*Q+64（Q为预先冻结3query患者总切片）、synthetic_native704、generator32，encoder-only0。一次性仅在GPU易失内存解码全部32码，任何真实或合成图像/mask均不持久化；最大BM训练640+入口梯度40=680 synthetic native；query额外方向求导<=10切片。预先检查Q和所有数据角色，并保存reserved/executed前向计账，失败不清零、不盲重评。工程/支持梯度/查询评价均包含caps。

新独立create-only root、独立原始时钟2h硬界，90min不新状态、105min TERM、120min前KILL；800MB保护+200MB退出余量，实际严格<1GB。仅指定GPU3，中性主子入口，启动挂载/free-space/写读/显存及ps+nvidia-smi，所有新cache/log/tmp/状态在root，共享旧资源只读不复制。保存状态/opt/RNG/reads与失败现场；NaN/非法数据/坏恢复/预算资源硬停。旧轮不重审，不新增条件后续。完成一次必要独立审计并匿名中文报告，源码/协议/聚合经代理push并核验remote SHA/匿名HTTP、本地和远端交付收据，不公开数据或权重。

## 工程资格数值下限修订（首轮32工程失败后、正式0时冻结）

首轮32次资格更新已计且不能重跑；非空optimizer精确恢复断言已通过，最终绝对CE下降1e-4断言失败，原失败trace/日志保留。零更新检查2个support图验证：两图0前景，入口CE=0.000048922713176580146、梯度范数0.0038248502，因此非负CE不可能再下降0.0001。这是资格绝对阈值不适用，不能当成任务能力负结果。

不更换病例、不挑更有利数据、不再做资格更新：保留已通过的原非空恢复检查，追加同共同入口/同两图的零更新loss+gradient golden精确且finite检查；绝对下降项明确记NA，不伪称原资格通过。若任何检查失败立即停止。正式仍304更新，含失败工程总计336不增加，原始时钟/存储界不变。

首轮失败前已解码32码的GPU易失缓存随进程退出丢失，允许精确同码再解码32、单列恢复成本，generator总硬cap修订为64（不清除原32）；其余forward caps不变，额外工程检查在support512内。此为有证据的最小工程修订，不改变五状态/分支/随机流/更新数/效用统计门槛或任何旧终点。后续解码移至工程资格之后，避免同类损失。禁止再次重评和无依据重试。
