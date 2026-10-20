# EMA与decoder针对性诊断 v1（执行前冻结）

2026-10-20用户明确授权：C完成后先分析完整结果，再自主做EMA和decoder针对性诊断，保留本轮冻结终点，整理后push。

## 顺序与主实验边界

先取得C固定2000最后学生和独立终点评估，报告ABC两域绝对变化、B−A/C−B/C−A以及原DE门槛；DE若通过仍按原controller执行，未通过不得追跑。主实验全轮COMPLETE且worker退休后一次原审计；不以诊断替换学生、不改原结果/预算/时钟，不重复旧模型患者评估。诊断在独立create-only目录运行，主实验产物只读。

## 固定问题与对照

1. 核查EMA初始化、每步更新规则、可训练参数及float/nonfloat buffers处理、eval路径。当前代码alpha=.999，2000次后入口系数alpha^2000约.1352；核对实际步数，不把代码推导当作全轨迹数值重建证据。没有全步模型归档时不得声称逐步重建过EMA。
2. 固定A与C各两个交叉模型：student encoder + EMA decoder、EMA encoder + student decoder。encoder覆盖enc1/enc2/enc3/bottleneck及其完整buffers，decoder覆盖原生上采样/skip融合/输出头及buffers；先按实际state_dict确认互斥穷尽，未知键停止诊断并定位，不能猜分块。原sigma/grad_update固定合同保持。空间P/G/code/frame不混合，保持所属A/C记忆；分割仍用原native前向。
3. A与C各一个entry–student参数混合对照：对EMA实际eligible的native浮点state以w=.999^2000作w*entry+(1-w)*student；其余state与EMA实际copy规则一致。它用于区分仅入口混合与整段轨迹平均，不能等同EMA，也不选插值系数。
4. 三个新增模型/组，共6模型，仅评估Siemens val23/46及Philips val31/62；复用既有entry/student/EMA真实结果及患者记录，不再前向这些完整模型。共新增7224真实验证切片前向，batch2、无TTA。测试/U GT不访问，真实源图不进入训练或梯度。
5. A/C各自32固定码，分别在entry/student/EMA上做生成软mask CE、softDice、码—弱再编码均方差/余弦；只生成本组最终记忆，不保存图像、mask或逐图特征。共192个合成输入分割前向及192个编码前向；不额外调用真实源训练集。几何一致不能冒充语义正确，合成mask不是独立GT。

## 报告与解释

报告新增模型两域、LV/MYO/RV、EDES及相对同组student/EMA的患者配对改善下降/中位差，配对患者bootstrap95%区间固定seed163、10000次，仅CPU已有记录。单seed开发验证属探索，区间不提供独立泛化保证。EMA交叉只定位兼容性线索，不能作可加因果分解；参数混合结果不自动成为部署或新主终点。

若EMA decoder组合恢复旧域且保留新域，支持优先检查readout轨迹；若EMA encoder组合更好，转向encoder侧；若两种混合都弱，不强行定位，考虑跨块协同/兼容性。若简单entry混合解释大部分EMA收益，不能只归因末期平滑；反之也不能据此证明特定优化噪声根因。不得看结果追加模型/系数。

## 硬预算与准入

独立诊断仅既定服务器GPU3/UUID；大文件、日志、cache/temp均在服务器授权数据盘下新create-only root。新增优化器/functional更新均0；真实验证前向上限7224切片，合成分割192与编码192，异常额外调用保留真实成本，不能盲目重评。预计新增<20MB，存储预警100MB、硬限500MB、保留100MB退出余量，严格<50GB；不复制checkpoint/数据。启动前挂载/剩余容量/写读探针、显存与中性argv；独立硬超时2h，90min停止新模型，105min保存退出，原主实验时钟不变。GPU峰值预计<4GB，以实际准入检查为准。

主子进程中性入口，进程身份PID+birth、ps/nvidia-smi检查；独立超时不依赖聊天。仍只复用space-lora-gpu3-4唯一30分钟心跳，不新建监测。NaN/Inf、分块未知、集合不匹配、资源/时间越界立即保留现场停止；有依据有限工程修复不重置诊断时钟。

## 交付

原轮审计与完整结果、诊断协议/源码/匿名聚合中文报告选择性显式代理push，验证远端commit与匿名HTTP，保存本地收据及remote DELIVERY/results_public。不发布患者记录、图像/生成图/mask、权重/码/frame、原日志、服务器配置或第三方全文，不建PR。

## 2026-10-20 生成mask异常的有限补充（诊断结果后记录）

原诊断已严格完成7224真实验证切片、192合成分割与192编码、0更新。A/C最终码上的前景softDice异常接近0，同时softCE很低，需分清背景占优与真实语义吻合。仅追加A/C固定32码各一次generator输出统计（共额外64个生成样本），报告四类soft质量占比、hard像素比例、各器官非空码数及图像取值摘要，不保存生成图/mask，不增加native分割/encoder或患者前向。原方案生成器本身已有64样本，补充后generator总128样本；原零更新与真实前向上限不变。复用诊断原始时钟与存储root，不重置预算。该项是具体异常后的定位，不追加训练或模型搜索。
