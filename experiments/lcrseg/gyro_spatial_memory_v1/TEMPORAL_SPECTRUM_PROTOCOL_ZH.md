# 空间谱×移动教师特征一致性：两目标流四组（2026-10-21运行前冻结）

## 依据、范围与终点

P0已完整分析、一次审计PASS，所有诊断worker退休。四个当前S/G00状态EMA encoder+student decoder两域均改善；A-align协方差能量较低，G00保留更分散表征；这些只是兼容性/统计证据，不保证教师训练有效。用户已授权按修正建议继续。仅S/G/T/GT两新目标流20261025/20261026；源初始化163、S†原69步student、源训练固定P不变。U/frame/生成记忆/强VC/独立源/多域等条件后续不展开，不替换旧终点。

固定最后student2000主终点，EMA .999统一辅助；不选best、不补步、不额外中间验证。原native允许encoder/decoder/head全训练，sigma/grad_update常量；空SGD .001/Nesterov .9/matrixwd .0005/vector0/globalclip1。每组Philips L2 CE权重不变，补图L池14仅image，不读补图label、不构造U accessor、无历史源图/在线旧teacher/val/test/U隐藏GT训练。P参数/buffer固定；frame为无损失兼容buffer，无G/code。该方法仍携带共同源入口/P信息，不称无源信息或已建立历史记忆。

## 四组与精确教师时序

S=CE，G=CE+.1G_raw，T=CE+lambdaT*T_raw，GT=CE+.1G_raw+lambdaT*T_raw。G_raw沿旧G00：1−||(2/3)C||*/sqrt(64*32)，C=.5weak+.25strong1+.25strong2，D32/R4、16图×4位置=64行。

T_raw=(1/(2M))*sum_i,v[1−dot(z_student_strong_v,stopgrad(z_EMA_weak))]，M64，按原P归一化。teacher使用本步opt更新前EMA，eval/无梯度；先取teacher特征，再更新student，随后原core.update_ema(.999 eligible parent浮点；sigma/grad_update及非eligible copy)；所有组都维持EMA。教师与学生同位置；增强为原photometric+cutout，不新增坐标变换或遮挡mask权重。T是跨视图+移动教师操作整体，不单独归因纯时间机制。

四组每步都执行同L2 native CE及48student encoder+16teacher encoder，同图/增强/位置/slot stream；所有组都计算谱与T标量，未启用项不连接训练loss。前向暴露匹配不等于反向/FLOPs完全匹配。教师引用不注册为student子模块，不进入opt/checkpoint；teacher参数grad始终None，模型train状态/buffer规则保留原native配置、teacher eval与P eval。

## 校准和工程顺序

一次共同训练校准stream20261027 step0..15，0更新；每批48student+16teacher，分别对encoder求grad(.1G_raw)、grad(T_raw)。lambdaT=.25 sqrt(sum||grad(.1G_raw)||²/sum||grad(T_raw)||²)，.1已在分子中不再重复乘；T/GT/两流共享固定值。任一非finite/累计能量<1e-12/lambdaT越出[1e-4,10]/模型buffer改变则科学停止，不换批次/seed/权重追通过。teacher与student入口相同但弱/强不同，避免同视图零信号。

CPU特征自检0医疗forward；S最小CE和S_detach(全部辅助前向但不接入loss)各32更新，对每步native/optimizer/EMA/RNG/监督输入精确检查，工程终态丢弃。八组各32资格更新，固定两例30+一次恢复重放+一次全目标，共256；按原预冻结相对CE阈值first>=1e-6下降>=min(1e-4,.01first)，更低floor下降NA且非增，不换病例。golden loss/gradient与非空opt恢复exact、共同入口空opt、固定P/常量及教师stopgrad通过后才正式。所有工程/失败/重放计同cap，不清零。

固定次序：两流各S/G/T/GT，先全部资格，后每组train2000退出再独立evaluate。复用原生引擎，不另写正式训练loop。每成功步clean latest包含model/EMA/opt/RNG/reads，dirty不可恢复。step0/499/999/1499/1999固定40点记录CE/合成辅助梯度方向/范数/裁剪/SGD位移，复用forward，80额外求梯度，不据此调整配方。

## 更新及调用硬预算

总16320 optimizer=正式16000+八资格256+断梯度64，functional/toy0。所有实际失败/重放/临时更新同账本；reserved持久化先于调用，executed完成后写，异常pending保留。下表cap与预期成功执行分别列，teacher encode独立计；不得增加评估或模型前向来验证结果。

|类别|native cap(预期)|student encoder cap(预期)|teacher encoder cap(预期)|反向cap(预期)|
|---|---:|---:|---:|---:|
|一次16批校准|0|768(768)|256(256)|32(32)|
|S/Sdetach|160(128)|1632(1536)|544(512)|64(64)|
|八资格|768(640)|1536(1152)|512(384)|320(280)|
|八正式含入口preflight|32400(32096)|770000(768768)|257000(256256)|16032(16024)|
|固定40点梯度诊断|0|0|0|80(80)|
|独立终点评价|23136(23136)|0|0|0|

正式2000步本体encoder总1024000=8*2000*(48+16)，另计preflight。终点评价每组2892=student旧544/新660/源train230/目标train254+EMA旧544/新660，八组23136。source train评价不回传梯度，与目标训练进程分离；既有旧模型/旧患者记录直接复用，不重评。

## 独立硬资源与停止

新create-only root，pro5000指定GPU3同UUID，/data/bmw明确存储例外；所有新code/log/cache/tmp/ckpt/账本在新root，只读引用旧资源不复制weights/data，不删历史原图/失败现场。启动实际mount/free/write-read/GPU显存与中性主子argv核验。预计<600MB，严格4GB硬界：3.5GB保护+.5GB退出余量，2MB小元数据计入；每调用/opt/save/newstage guard实际du。

原始时钟首次运行锁定，10h硬界：9h不新阶段、9h45m TERM，10h前KILL，独立timeout覆盖进程组且不依赖SSH/Codex。失败恢复同钟/同账本，不以换root重置本轮。NaN/非法访问/坏恢复/资源/预算任何项硬停相关worker保现场，不盲重试或扩cap。

## 预声明分析、门槛与有限后续

主student：G−S两流旧平均>=+.5pp且每流旧>0、新各>=−.3pp；GT−G旧平均>=+.3pp且每流旧>0、新各>=−.3pp；GT−T同+.3/每流正/新>=−.3。绝对目标保持GT旧相对S†每流>=−.5pp；原源80.367407与S†79.433714两条参考分开，接入−.933692pp不消失。患者代价：GT相对入口下降>1pp人数每流不高于G，均值/中位/p10/>.5/1/2pp/类EDES/损益完整列；幅度阈值非临床阈值。

交互(GT−T)−(G−S)逐流两域患者bootstrap10000 seed163，正交互需证据，不因GT最好称协同。EMA同口径独立报告，不替换学生。每流患者区间不覆盖训练随机性/多重比较/反复开发；两目标流共享源/P/入口，不称完整独立泛化。新域点估计非劣门槛仅开发门槛，不等于区间非劣确认。

GT−G失败则停止T分支不扫描EMA/权重；若GT与T相近，谱必要性不成立；若仅EMA有效保留学生阴性结果，后续统一EMA部署须另协议。任何其他条件后续不自动展开。

全部COMPLETE/EXIT0/父子timeout退休后仅一次新CPU审计：16320及calls事件/无pending、资格/校准/断梯度、共同入口空opt/固定P常量/无Gcode/教师未注册、finite final=latest optRNGreads、5诊断点/组、患者EDES、实际组合，0审计forward。完整匿名中文报告/源码协议/结果/审计选择性代理push，remoteSHA/匿名HTTP/local+remoteDELIVERY/results_public齐，唯一心跳跟进。禁止公开患者、图像mask、任何权重Pframecode、逐图特征、原始日志、服务器配置。不建PR不推私有历史。
