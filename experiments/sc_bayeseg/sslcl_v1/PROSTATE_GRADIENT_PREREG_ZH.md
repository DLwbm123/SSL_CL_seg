# 教师视图对照后的局部梯度诊断

同视图教师6臂完成，但预设双域方向条件失败：SAMEH−SAME0旧[+1.133,+.288,−2.064]、新[+3.467,+4.689,−2.278]；教师×历史交互旧[+1.858,+1.610,−.843]，新[+5.861,+7.232,+1.313]。新域交互全正不覆盖旧域/165失败；不继续权重/幅度/种子网格。

仅诊断已有U模型在目标域上监督梯度与一致性梯度的局部关系。12个checkpoint：每seed163/164/165各source_U、U10、UHU、USAMEH的S5000。加载各自student/EMA teacher；student.eval但允许autograd，teacher.eval/no_grad。eval避免buffer更新，故不是完整训练模式梯度轨迹。无optimizer、无参数写回。

每模型固定32批：training Schedule同其seed，step=1001..1032，实际抽样absolute=5000+step，每批L2/U2。监督目标仅当前风格的原图+strongL平均CE/Dice；对同批U比较原图教师/同strong图教师×无历史/U历史四格，原alpha1概率.5。固定ramp=1，符合该step区间，U模型image objective为0。GT仅目标trainL，U图像不读GT，不读val/test/HK或旧域图像。

对监督梯度gs和一致性梯度gu记录dot、cosine、norm ratio、两个loss及norm，32批均值/中位数/范围、负dot批数及history−none paired均值。内积负只表示局部冲突，不等于验证损害；同视图可能让一致性很弱，低norm时cosine不稳定，必须并列norm/ratio和原始dot，不按一个数字选新权重。不同checkpoint中teacher状态不同，四格是各自状态内配对，不当作新的训练因果对照。

create-only root=/data/bmw/SSL_CL_seg/runs/sslcl_prostate_gradient_20261022，ops=/data/bmw/.ops/x20261022_p6d，沿p6原钟1792632032.6406996，controller85500/外层86100不重置；GPU2345最多4并行，新占用预计<150MB，加p6约338MB远低50GB；root+ops49GB预防停。先真实U一批零更新finite/grad不写入检查，再正式12×32批。公开匿名聚合，无患者身份/图像/私有checkpoint。
