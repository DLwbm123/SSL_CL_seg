# 离散模型完整局部下界的反事实读出

新模型的训练下界改善，尚未转化为正确结构偏好。校准终态psi600/Theta600对训练GT相对全背景的局部下界差−.07975，保留−.13583；即全背景仍获更高评分，不能启动或宣称成功的无标签预测修正。新概率模型数学合法与已学语义有效是不同判断。

|推断psi/后验Theta|指标|train|heldout|
|---|---|---:|---:|
|psi0_theta0|GT_spatial|0.024401|0.015503|
|psi0_theta0|GT_identity|-0.002273|0.013802|
|psi0_theta0|own_spatial|0.024549|0.003156|
|psi0_theta0|own_identity|-0.003598|0.007233|
|psi0_theta0|GT_minus_own|0.001734|-0.036920|
|psi0_theta0|GT_minus_background|-0.055500|-0.083079|
|psi0_theta0|own_minus_background|-0.057234|-0.046159|
|psi0_theta600|GT_spatial|0.054419|0.062774|
|psi0_theta600|GT_identity|0.002995|0.007426|
|psi0_theta600|own_spatial|0.055471|0.015672|
|psi0_theta600|own_identity|0.002388|0.002061|
|psi0_theta600|GT_minus_own|0.002779|-0.063934|
|psi0_theta600|GT_minus_background|-0.095888|-0.136130|
|psi0_theta600|own_minus_background|-0.098667|-0.072196|
|psi600_theta0|GT_spatial|0.184596|0.181122|
|psi600_theta0|GT_identity|0.007511|0.002950|
|psi600_theta0|own_spatial|0.188292|0.026462|
|psi600_theta0|own_identity|0.006391|0.007397|
|psi600_theta0|GT_minus_own|-0.000262|0.053911|
|psi600_theta0|GT_minus_background|0.052720|0.024537|
|psi600_theta0|own_minus_background|0.052981|-0.029374|
|psi600_theta600|GT_spatial|0.197808|0.241654|
|psi600_theta600|GT_identity|0.011113|-0.002602|
|psi600_theta600|own_spatial|0.202539|0.046892|
|psi600_theta600|own_identity|0.010873|-0.000355|
|psi600_theta600|GT_minus_own|0.002923|-0.035101|
|psi600_theta600|GT_minus_background|-0.079746|-0.135828|
|psi600_theta600|own_minus_background|-0.082669|-0.100728|

完整下界包含全部五项joint与qx/qm/qc熵。固定psi0，Theta更新使train GT−BG再下降.04039、heldout下降.05305；固定psi600分别下降.13247/.16037。相反，固定初Theta0更换psi600，使train GT−BG上升.10822、heldout上升.10762，显示推断网络变化可显著改变评分。不能只看终态GT空间差正（train .19781/heldout .24165）就称语义成功，也不能只选择psi600/Theta0那个较好交叉格作为新模型。所有交叉格、分项与配对效应在[DISCRETE_L_READOUT.json](results/DISCRETE_L_READOUT.json)公开。

GT−own终态train+.00292、heldout−.03510，类别GT对照train+.01111/heldout−.00260，仍缺乏一致正确类别偏好。GT与own类别体积不同；全背景比较本身亦不能排除类频率先验与推断间隙影响。这些是有限MC局部变分下界，不是精确边缘证据排序、分割输出或全局不可识别证明。

## 核验与解释

416批量bound、32生成draw、0optimizer/0反向/0局部优化。四格13模板×8draw，每个draw内模板的Theta/局部基础RNG完全配对；完整模型不变、原604物理账本不变，连续1..416、有限值、EXIT0通过。qTheta作为固定预测混合，未把KL/2加成单图证据；KL另存，模板差中为常数。保留标签仅只读评分，未参与任何训练/局部梯度/选择。

下一项冻结全部模型，只用原两训练图，固定终态psi600，在Theta0与Theta600下对GT/全背景各做25次局部变分优化（qx/qm/qc均值与logsd shift），相同固定SAA与预定回溯。它不改语义标签或模型参数，仅检验两假设的推断间隙是否可被同等优化改变；不使用heldout图像/标签，不再盲训练生成器。若GT与BG差翻转，支持原评分有局部推断误差混杂；若仍BG高，只能说该族/步数未消除问题，不证明精确模型必然偏好BG。
