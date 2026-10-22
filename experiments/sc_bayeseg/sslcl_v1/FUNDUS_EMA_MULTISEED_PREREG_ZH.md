# EMA .999 固定配方多seed复核

seed163完整S5000全部通过后按既定后继授权，仅补seed164和165，每seed F0目标5000步，共新增10000更新；复用已完成seed163 E999，不重做。各自原source_U_s164/165的student/EMA/RNG恢复、freshAdam，仅EMA=.999，其余默认。新root sslcl_fundus_ema_multiseed_20261022，ops x20261022_h4。固定终点S5000学生每seed140图，不挑中间点，不加FG/FH、参数或seed。

比较每seed原F0同S5000，公开全部seed。确认门：旧域平均增益≥1pp且三个seed分别>0，新域平均≥−.5pp且各seed≥−.5pp；新域相对各自源入口均≥2pp。每seed/域cup和disc Dice下降≤2pp、惩罚HD95恶化≤10%、empty不增加、Q10下降≤2pp。任何失败均保留，不挑seed163或改门。全部过仅称三seed开发复现，不是新患者独立验证、历史方法优势或无遗忘。

源码与引擎不变，继承已通过资格无需新模型检查。新增10000更新、80000训练图像前向+280终点评估；原1792638157.570414时钟和24h不重置。全部此前data/run/ops累计<50GB，controller49GB85500s、worker85200s、独立timeout86100s。预计本轮新增<.3GB。GPU2/4并行，/data实际挂载写探针，中性argv。无test/隐藏GT/源重训；唯一半小时监测及公开匿名交付。
