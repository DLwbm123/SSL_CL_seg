# Prostate半幅历史强度：固定单一改动的探索实验

依据已完成24次零更新诊断，U11相对U10在三个seed的目标1000步BMC Dice均负（约−2.32/−6.50/−3.76pp），3000和5000也均负，不能仅用最后阶段退化解释。source_B在163/164随训练提高，165旧域3000→5000下降；source_U也混合，不能统一决定加长源训练。保留源/目标5000结果，不换最佳checkpoint。

只改变目标历史变换的幅度alpha=.5，概率仍为.5，L/U两路仍同时使用历史。选中历史分支时mu=current_mu+.5*(old_mu-current_mu)，log(sd)=log(current_sd)+.5*(log(old_sd)-log(current_sd))；其它contrast/offset/bias范围、五个随机数、几何、损失/EMA/Adam/5000终点均不变。未选历史分支仍current。alpha=1默认保持原执行路径不变，alpha=0应与无历史view逐元素一致。此为减轻跨域强度偏移的剂量假设，不是全新机制证明或新架构。

新增BHALF/UHALF各seed163/164/165，共6臂×5000=30000目标更新。每臂复用此前对应family、对应seed的5000 source student/teacher/RNG，fresh Adam；163 source来自pilot，164/165来自完整流程复现。initialization_seed/schedule_seed与原seed匹配，training_seeds为空，因此目标恢复source RNG而非重置。source_checkpoint按任务名显式映射，禁止串seed。数据split0及RUNMC/BMC每域4val不变；不读取HK/test/U隐藏GT。

对照直接使用同seed原B10/U10(alpha0无历史)和B11/U11(alpha1完整历史)，不重新选择seed/源或只报有利family。主要报告HALF−FULL和HALF−NONE的旧新Dice、遗忘/BWT/arrivalgain、HD95/NLL/Brier/空预测，所有seed及均值/SD/患者配对CI/留一。源模型差异仍在，alpha=.5根据已看过val轨迹提出，结果必须标注探索，不作为独立确认。不得再用同4val开幅度网格；若仍损害，应收缩在该设置下使用历史的主张。

独立create-only root=/data/bmw/SSL_CL_seg/runs/sslcl_prostate_halfstyle_20261022，ops=/data/bmw/.ops/x20261022_p3；GPU2345最多4并行。新原时钟含集成检查/全部任务，23h45控制器、23h55外层、root+ops49GB预防停，预计<1GB，重试同钟。启动前验证alpha0无历史视图精确一致、alpha.5等同明确插值统计、两family真实反向finite/teacher detached、6任务源映射与checkpoint步数；0集成optimizer更新。数据/checkpoint只读、argv中性。
