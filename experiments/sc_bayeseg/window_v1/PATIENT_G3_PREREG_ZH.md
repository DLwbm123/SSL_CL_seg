# 患者验证后继：只改变固定G的三均值偏置来源

2026-10-20，启动前冻结。首轮患者主线原G条件分支平均Dice54.1691%，匹配直接头68.1415%，差−13.9724个百分点；这是23位患者的开发验证结果，不将其改写或继续训练追分。一次性两图G0/G1/G3诊断已收口，两份来源的G3均通过有限oracle资格，但不等于患者效果。本轮检验该有界修正能否迁移到纯图像患者映射。

## 唯一改动与来源

只将原G anatomy.2.mean_b三个前景均值，加上已完成g_bias_closure_v1中ZERO/G3的三维delta。选择ZERO按原协议两来源固定顺序第一项，而非患者val表现或两个oracle Dice择优；不合并/重拟合两份delta，不改其他G字段。原bias prior/initial/logsd保持。ZERO局部q不进入本轮模型或推断，delta只作为冻结训练标签拟合得到的模型参数，不声称其未经训练标签。

其余逐项复用patient_validation_v1的来源和规则：PRETRAIN/S5000完整U-Net特征、原R_START/R1 S0000 hc/hr、共享PRETRAIN/S5000 RNG，freshAdam；只训练H2144参数5000步，图像输入→H(c,R)→固定G3，不含F，不是完整或官方BayeSeg。G0/H与DIRECT已完成控制及全部预测直接复用，不重新训练或评分。H从相同初态重新训练，不接续G0训练终态；唯一主动改变为G的3个bias均值，保持目标/学习率/批次/训练噪声/终点。

数据严格同原10位Siemens20%标注患者230切片；23位已使用开发val患者544切片/46相位一次评价。U、目标域、test图像及标签不打开。训练schedule绝对步5001..10000、2患者/步、4翻转，128处理、Adam3e−4/clip1/原2Theta×2local/原raw尺度/像素期望CE+KL/N全部不变，无seed扫描/类权重/teacher或局部优化。0/1000/2000/3000/4000/5000完整model/Adam/RNG共6份。

## 评价与解释

固定5000步后，image-only接口每批4Theta×4local，原验证seed/顺序/batch2/384概率恢复/spacing。先封存全部46相位概率，再读取val标签。评分复用原冻结三维指标，逐相位→ED/ES→患者，完整Dice/HD95 NA与惩罚/空预测/NLL/Brier；患者bootstrap10000次seed163，报告G3−G0及G3−DIRECT配对差和全部23匿名患者。不按val读数选步、加MC或再次调偏置；这是同一开发验证集后续比较，不是未触碰最终测试。负结果同样交付，不再回到两图微诊断。

## 成本与停止

固定5000新optimizer/reverse，10000训练图像编码、10000Theta/20000批量G/40000图像G；验证544编码图像、1112Theta/4448批量G/8704图像G，46标签phase读；0旧控制重复推断。预计新增<2GB，启动预留3GB，所有大文件在原pro5000 /data/bmw create-only。沿用原R_START首次clock及所有失败成本，独立中性guard原23h45保存/23h59硬停、四根18GB软20GB硬和新45/49GB，原首启<=24h/新增实际<50GB。源精确/挂载写读/GPU余量核验后启动；非有限/冻结破坏/预算失败保留partial状态和成本，准确接续不盲重跑。

完成独立核验5000连续更新/同G0训练schedule、6完整状态与freshAdam步、H/RNG同初态、G仅3均值差且全程冻结、封存先于label、46相位/23患者覆盖、指标复算、EXIT/FAILURE及worker/guard退休。源码/协议/中文正负报告/匿名指标代理push并验证SHA/匿名200；不公开患者标识、路径、delta/权重/概率/原日志。保持唯一sc-bayeseg半小时心跳。
