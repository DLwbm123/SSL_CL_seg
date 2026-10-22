# Fundus 13类超参数单因素短筛结果

28/28阶段完成，每阶段1000步，全部exit0，累计28000更新。**没有配置通过全部冻结门槛，selected=[]。** 不进入自动完整步长/多seed复核，不将失败项重新解释为通过。

| ID | 唯一调整 | 旧域变化 pp | 新域变化 pp | 未通过项目 |
|---|---|---:|---:|---|
| A00 | learning_rate=1e-05 | +4.6380 | -1.0472 | RIM_ONE_r3_cup_Dice, new_preserved |
| A01 | learning_rate=0.0001 | -1.4219 | -0.3393 | REFUGE_tail, REFUGE_cup_Dice, old_gain |
| A02 | learning_rate=0.001 | -7.1495 | -2.3073 | REFUGE_tail, REFUGE_cup_Dice, REFUGE_cup_HD95, REFUGE_disc_Dice, REFUGE_disc_HD95, RIM_ONE_r3_tail, RIM_ONE_r3_cup_Dice, RIM_ONE_r3_cup_HD95, old_gain, new_preserved |
| A03 | consistency_weight=0.0 | -1.2254 | +0.4266 | REFUGE_disc_Dice, RIM_ONE_r3_tail, RIM_ONE_r3_cup_HD95, old_gain |
| A04 | consistency_weight=0.3 | -2.2915 | -1.0084 | REFUGE_tail, REFUGE_cup_Dice, REFUGE_cup_HD95, RIM_ONE_r3_tail, RIM_ONE_r3_cup_Dice, old_gain, new_preserved |
| A05 | consistency_weight=3.0 | -0.2314 | -2.0931 | RIM_ONE_r3_tail, RIM_ONE_r3_cup_Dice, RIM_ONE_r3_cup_HD95, old_gain, new_preserved |
| A06 | ema_decay=0.9 | -0.7938 | -3.0503 | REFUGE_tail, REFUGE_cup_HD95, REFUGE_cup_empty, RIM_ONE_r3_tail, RIM_ONE_r3_cup_Dice, RIM_ONE_r3_cup_HD95, RIM_ONE_r3_disc_Dice, RIM_ONE_r3_disc_HD95, old_gain, new_preserved |
| A07 | ema_decay=0.999 | +1.8996 | +1.1605 | RIM_ONE_r3_cup_HD95 |
| A08 | ramp_steps=100 | -1.1858 | -0.6056 | REFUGE_tail, REFUGE_cup_Dice, REFUGE_cup_HD95, RIM_ONE_r3_tail, RIM_ONE_r3_cup_HD95, old_gain, new_preserved |
| A09 | ramp_steps=5000 | -0.7682 | -0.2289 | RIM_ONE_r3_tail, RIM_ONE_r3_cup_Dice, old_gain |
| A10 | gradient_clip=0.1 | +0.8707 | -0.4800 | old_gain |
| A11 | gradient_clip=10.0 | +0.3087 | +0.2081 | old_gain |
| A12 | weight_decay=1e-05 | -0.4452 | -1.4373 | RIM_ONE_r3_tail, RIM_ONE_r3_cup_Dice, RIM_ONE_r3_cup_HD95, old_gain, new_preserved |
| A13 | weight_decay=0.001 | -1.7491 | -0.0551 | REFUGE_disc_Dice, RIM_ONE_r3_cup_HD95, old_gain |
| A14 | beta1=0.5 | +0.4648 | -2.0473 | REFUGE_disc_HD95, RIM_ONE_r3_tail, RIM_ONE_r3_disc_Dice, old_gain, new_preserved |
| A15 | beta1=0.95 | -5.1036 | -0.6280 | REFUGE_tail, REFUGE_cup_Dice, REFUGE_cup_HD95, REFUGE_cup_empty, REFUGE_disc_Dice, REFUGE_disc_HD95, old_gain, new_preserved |
| A16 | beta2=0.99 | +0.8696 | -0.5966 | RIM_ONE_r3_tail, old_gain, new_preserved |
| A17 | beta2=0.9999 | -0.2477 | -0.8694 | RIM_ONE_r3_tail, RIM_ONE_r3_cup_Dice, old_gain, new_preserved |
| A18 | adam_eps=1e-06 | +0.8775 | -1.2761 | RIM_ONE_r3_tail, RIM_ONE_r3_cup_Dice, RIM_ONE_r3_cup_HD95, old_gain, new_preserved |
| A19 | adam_eps=0.0001 | -0.3664 | -0.8982 | old_gain, new_preserved |
| A20 | dice_weight=0.0 | -8.7449 | -1.9394 | REFUGE_tail, REFUGE_cup_Dice, REFUGE_cup_HD95, REFUGE_cup_empty, REFUGE_disc_Dice, RIM_ONE_r3_tail, RIM_ONE_r3_cup_Dice, old_gain, new_preserved |
| A21 | dice_weight=2.0 | -0.3711 | -0.6577 | RIM_ONE_r3_tail, old_gain, new_preserved |
| A22 | augmentation_strength=0.0 | +0.1043 | -1.6856 | REFUGE_cup_HD95, RIM_ONE_r3_tail, RIM_ONE_r3_cup_Dice, RIM_ONE_r3_cup_HD95, old_gain, new_preserved |
| A23 | augmentation_strength=0.5 | +1.5259 | -4.3162 | RIM_ONE_r3_tail, RIM_ONE_r3_cup_Dice, RIM_ONE_r3_cup_HD95, RIM_ONE_r3_disc_Dice, new_preserved |
| A24 | strong_supervision_weight=0.0 | +1.5523 | -3.0361 | RIM_ONE_r3_tail, RIM_ONE_r3_cup_Dice, RIM_ONE_r3_cup_HD95, RIM_ONE_r3_disc_Dice, RIM_ONE_r3_disc_HD95, new_preserved |
| A25 | strong_supervision_weight=1.0 | -1.0836 | -1.6565 | RIM_ONE_r3_tail, RIM_ONE_r3_cup_Dice, RIM_ONE_r3_cup_HD95, old_gain, new_preserved |
| A26 | batch_size=1 | -3.5928 | +0.1254 | REFUGE_tail, REFUGE_cup_Dice, REFUGE_disc_Dice, REFUGE_disc_HD95, RIM_ONE_r3_cup_HD95, old_gain |
| A27 | batch_size=4 | -3.2037 | -1.1667 | REFUGE_tail, REFUGE_cup_Dice, REFUGE_cup_HD95, REFUGE_disc_HD95, RIM_ONE_r3_tail, RIM_ONE_r3_disc_HD95, old_gain, new_preserved |

所有变化相对相同seed163的F0 S1000默认基线（旧77.0575%、新81.7540%），完整每类Dice/HD95/empty/Q10及全部判断见JSON。old_gain要求至少+1pp，new_preserved要求至少−.5pp；tail和类别保护门保持原值。

## 可用信息与限制

- EMA .999（A07）是唯一同时提高两域主Dice且满足主增益门的配置：旧+1.8996pp、新+1.1605pp。但新域cup惩罚HD95从17.3000px升至20.4496px，恶化18.21%，超过10%门槛。因此只是值得解释的开发性迹象，并非合格配置。
- clip=.1（A10）通过其余门，但旧域+0.8707pp不足+1pp；不得四舍五入晋级。
- lr=1e-5旧域+4.6380pp，但新域−1.0472pp且cup下降超限；减少更新速度仍呈保留/学习权衡。
- 去掉当前增强（A22）未消除遗忘，新域还下降1.6856pp，不能把共同退化唯一归因于重着色；这是单seed开发性对照，仍非充分因果结论。
- CE-only（A20）旧域下降8.7449pp并新增cup空预测；EMA=.9与beta1=.95也出现旧cup空预测，完整失败保留。
- EMA .999旧域仍显著低于源入口89.5976%，相对默认改善不等于无遗忘。未搜索到合格点不能证明所有组合无效；单因素实验没有检验交互，不能把各自优点自动拼接。

## 成本与后续

训练图像前向按实际引擎每步4×batch计算，共228000；28个终点各140图，共3920；资格检查144图、0更新另计。batch=1为4000训练图像前向，batch=4为16000，其余各8000，因此步数相同而曝光不同。没有新增源训练、test或隐藏GT访问。
controller完成时累计实际分配1666879488字节（含前序共享预算），原时钟未重置。所有任务正常完成，无重试。

本轮预注册最多一个合格方向再复核，当前没有合格方向，故不自动扩跑。后继若研究EMA附近参数或边界指标机制，须另立有限协议，保留本轮FAIL，不根据本轮结果放宽门槛。三个先前快速点及FH/HL/HU失败均保留。全部结果仅重复使用val的单seed早期开发证据，不能称独立验证或最终方法效果。
