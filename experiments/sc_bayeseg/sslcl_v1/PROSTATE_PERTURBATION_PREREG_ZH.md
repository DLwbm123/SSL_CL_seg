# 半幅方向复查失败后的图像单独扰动诊断

新seed166半幅相对无历史旧域−1.619pp、新域+.389pp；167旧+4.100、新+3.584pp。预设两个新seed两域全正条件未通过，不补seed直到成功，不继续alpha网格。现仅定位已存在预测敏感性，0 optimizer更新，无新训练。

只使用BMC训练U的14名患者图像，每人唯一volume取z索引floor(q*(Z−1))、q=.25/.5/.75，共42片，原分辨率缩至训练128，不读任何标签、val/test/HK。每组3片使用4个固定变换随机种子，保持现行50%历史分支与其它contrast/offset/bias，比较NONE(alpha0)、HALF(.5)、FULL(1)，不探索新幅度。统计实际历史分支选择数。

16模型：163/164/165各B/U的source与no-history target(B10/U10) S5000，共12；166/167的B source/B10共4。所有checkpoint只读，student.eval，B4draw、U确定性，同一患者/重复中原图及3变换使用相同torch随机种子以配对latent噪声；此为控制噪声的诊断，不替换正式评价。对原图预测p和变换q报告KL(p||q)、argmax改变比例、前景平均概率位移/绝对变化、变换后entropy；患者内平均后患者等权，报告均值/患者SD及HALF/FULL减NONE。背景占比可能稀释像素平均，指标不可代替病灶/准确率。

无GT，因此高敏感性不等于错误、低敏感性不等于保留。跨B/U概率校准/结构不同，不根据KL大小证明架构优越；仅观察各模型内部paired history−current差值与source→target变化。方向解释规则：只有HALF相对FULL系统性降低扰动才支持“减幅降低预测扰动”，即使如此也不能解释所有Dice差异；不根据结果选新幅度/seed。

root=/data/bmw/SSL_CL_seg/runs/sslcl_prostate_perturbation_20261022，ops=/data/bmw/.ops/x20261022_p4d；沿p4原始时钟1792623008.517845不重置，GPU2345最多4并行，源文件只读。新目录create-only，预计<150MB，root+ops49GB预防停，原运行追加总占用远低于50GB；控制器原钟85500/外层86100秒。先一次真实B/U零更新烟测，再进入全16队列；身份不导出，公开只有匿名聚合。
