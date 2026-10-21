# 历史作用路径与特异性：四臂机制探索（运行前冻结）

前提：两个新增target seed与12个零更新诊断全部成功退出。B11三个target seed旧域保持方向一致；16draw两评价seed不改变该方向，绝对Dice变化很小。seed165 B01从3000到5000两域明显下降，而B11未同样崩塌。已有证据支持继续检验有U时历史的作用，但不证明所有seed双赢，也不证明B主体优于U-Net。

固定source_B S5000和EMA、数据split0、训练seed164、5000步、2L+2U、优化器/CE+Dice/.005JL+.005JU+cons、增强次数、评估4draw/620163全部不变。选164为首个新增复现seed，非最高分seed；已有B10_s164/B11_s164作为H00/H11固定对照，不重训或择优。单seed探索，需关键对比后续匹配seed确认。

新增四臂：
- HL：历史只用于L强视图；U强视图当前统计。其余同B11。
- HU：历史只用于U强视图；L强视图当前统计。其余同B11。该U路径含图像目标和一致性，不能仅归因KL。
- WIDE：L/U都使用当前域统计驱动的通用增强；不读取历史统计。仍50%当前风格、50%通用随机风格。通用均值=current_mu+Uniform(-.75,.75)*current_sd，log_sd=log(current_sd)+Uniform(-log1.5,+log1.5)。范围在验证前固定，不根据分数调整；后接同样随机contrast/offset/低频bias。与历史臂同样5个uniform数：u0控制50%分支，分支内2u0仍Uniform(0,1)用于logsd；u4用于均值，两者独立。一个通用范围不能穷尽所有无历史增强，结论限于该对照。
- REVERSE：L/U都以50%概率用错误方向统计，mu_wrong=2mu_current−mu_history，logsd_wrong=2logsd_current−logsd_history；其余同B11。使用历史幅度，只是方向对照，不能称完全无历史。

几何与标签不变，无新患者或测试访问。训练前零更新检查：历史=current与无历史精确一致；WIDE输出不依赖传入历史；reverse对数空间等幅；相同随机seed重现；真实L/U强度扰动RMS/极值有限，记录扰动幅度但不据此优化参数。不宣称四臂图像L2扰动完全相同。各新增臂做1次真实有限反向检查但不optimizer.step，随后正式从同一source/seed重建；无额外优化更新。

终点报告HL/HU/WIDE/REVERSE相对H00与H11的旧/新域Dice、遗忘/BWT、类失败、HD95/NLL/Brier和患者配对CI；保存全部结果，包括负结果。不要求任一方法超过U-Net才准入。解释优先：L路径/U路径/正确历史方向/通用增强是否足以解释观察。若HL/HU未恢复B11，可能需要联合路径；不能仅凭单seed断言因果机制已确认。

独立新实验20000优化更新，source只读共享。root=/data/bmw/SSL_CL_seg/runs/sslcl_mechanism_20261022；ops=/data/bmw/.ops/x20261022_m1，create-only，GPU2345四并行，中性入口。首次真实集成执行统一计时，全部重试共享；23h45控制器停止、23h55外层timeout，新增root+ops 49GB硬停，预计<1GB。唯一半小时监测接续。此前复现/诊断已完成，不续跑旧训练、不覆盖旧结果。
