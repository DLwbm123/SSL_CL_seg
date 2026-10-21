# Prostate RUNMC→BMC受控跨数据pilot（训练前冻结）

## 数据与可解释范围

沿用项目prostate_seed0冻结患者划分与原RUNMC→BMC域顺序；原协议whole-prostate raw1/2→1、背景0，输入1通道，stored256×256，逐volume强度归一化。当前审计显示RUNMC/BMC各4L/14U/4val/8test元数据，训练与验证患者无重叠、图像路径唯一、与冻结split逐项匹配。只读train L/U及val图像；U隐藏label字段为空且image H5无label键。可见L标签取值0/1且与图像形状匹配。test文件不打开，val GT仅评价预测后读取；未做内容去重或临床身份跨站链接，不声称超出冻结patient ID的独立性。

审计图像Z=15–38、平面256；spacing Z约3–4mm、平面.625–.78125mm。训练统一128×128，评价恢复stored256×256再按processed_spacing计算3D whole-prostate Dice/HD95。每患者一个volume，不套用心脏双phase聚合。每域仅4val，结果仅探索pilot，患者bootstrap描述性区间不能当充分泛化证据；不访问HK或其他域扩大验证集。

## 初始化、对照与预算

普通U-Net沿用项目base16 GroupNorm架构，二分类头；B保留当前MAP G/F+局部q结构，G输出1前景logit加0背景，F接受2类概率，不称原版全Bayesian。B/U以同seed163从随机初始化开始，backbone参数数值一致，各自head/额外模块随机，投影分支Identity。没有加载任何心脏、旧prostate或目标训练checkpoint。此前心脏曾预训练且初始化不同，不能把本pilot和心脏绝对数值作直接比较；跨family不能作为只有一个组件差异的因果比较，主要比较各自有/无历史。

不另行玩具过拟合筛选：source_B/source_U均RUNMC真实4L14U固定5000 SSL更新；不按val提前停或选checkpoint。每family从自身source student/teacher出发，fresh Adam，恢复其source RNG，分别B10/B11或U10/U11进行BMC5000更新，有U且仅历史关闭/开启不同。共6阶段×5000=30000正式更新，集成额外2次optimizer(每family1)单独记账、不会影响正式初始化。每阶段L batch2/U batch2，patient-balanced→volume→slice，原翻转/强度变换，Adam3e-4、clip1、EMA.99、ramp1000保持。

B损失保留原完整Lsup+.005JL+.005JU+ramp cons，U仅Lsup+ramp cons；源阶段无历史，目标历史统计只来自源train L的patient-balanced image mean/std，当前统计只来自当前train L。50%历史/50%当前强视图、范围不因结果调整。保留完整损失的理由：三seed中JU在有cons条件下旧/新平均有益但并非每seed；不因单seed删除。统计RUNMC[-.47378987,.41789144]、BMC[-.23281816,.50597644]在训练前冻结。

## 评价与停止

source及目标最终5000 checkpoint在RUNMC/BMC val评价，image-only预测先于对应val GT读取。报告patient macro Dice、旧域forgetting/BWT(相对各自source)、新域arrival gain、NLL/Brier、3D HD95与空预测；配对B11-B10/U11-U10为主要对比，附B11-U11但不主张结构单因素优越。B概率4局部采样，U确定性，固定eval seed620163。保存全部终点/负结果，不选最佳时间或患者。

先通过binary one-batch finite backward、teacher detach/EMA、checkpoint resume预测一致、binary perfect/empty/spacing指标自检；确认双方初始backbone相同；集成无val/test访问。训练硬失败或非有限值立即保存失败状态，预算内分析修复不重置时钟；性能差也保留终点，不用临时筛选重启。

独立root=/data/bmw/SSL_CL_seg/runs/sslcl_prostate_pilot_20261022，ops=/data/bmw/.ops/x20261022_p1；GPU仅2345最多4并行。原时钟1792612212.363481从准入准备开始，含集成、所有阶段与重试；控制器23h45、外层从该时钟剩余23h55超时。root+ops49GB预防停，预计<1GB；仅共享只读数据，无新下载。启动前实际挂载/写读/显存核验；所有大输出/cache/tmp在/data/bmw，argv中性。旧cardiac代码默认4类与指标路径保持，不改冻结LCR-Seg文件/模型规范。
