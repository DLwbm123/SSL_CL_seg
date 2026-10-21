# 固定JL损失分解：seed163确认（启动前冻结）

已完成164/165两个target seed四臂。cons在JU开/关下旧新Dice均正，JU旧域作用不稳定，165有cons时JU有+2.12pp旧域增益；不根据164简化删除JU。新域交互两seed均负，不能称稳定正协同。

仅补既定J0C0/J1C0/J0C1/J1C1 × seed163，4×5000=20000更新。固定source_B、fresh Adam、完整历史、JL=.005、JU0/.005、cons0/1，所有臂仍抽U并执行相同前向，评价和终点不变。163不传training_seed，恢复source RNG，schedule/view默认163；与此前163习惯匹配，不能误重置。各臂始于相同RNG。新的完整J1C1仍重跑。

无模型训练代码变化，复用l1的五反向零optimizer集成PASS；启动检查4任务唯一、参数组合完整、training_seeds为空。合并164/165并逐seed报告所有预先指定JU/cons效应和交互，以及Dice、forgetting/BWT、校准、HD95、空预测和患者配对CI，标注此163为后续确认。不能用单seed排名选择胜者。

独立create-only root=/data/bmw/SSL_CL_seg/runs/sslcl_loss_confirmation_20261022，ops=/data/bmw/.ops/x20261022_l2。GPU2345最多4并行，原数据/checkpoint只读，不访问test/U隐藏GT。预计新增<1GB，总root+ops49GB预防停、控制器23h45/外层23h55，所有重试共享首次时钟；启动前挂载写读/显存检查，中性argv。完成后决策简化/维持，不无限补seed；下一优先准备prostate受控跨数据验证的必要准入与适配。
