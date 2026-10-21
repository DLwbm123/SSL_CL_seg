# 真实 SSL + CL v1：已启动（2026-10-21）

按用户授权，完成两域划分审计、冻结六臂协议、真实数据集成检查并启动后台队列。目前是启动证据，不是实验结果。

- 数据：Siemens→Philips，canonical320/seed0/20%L；L/U/val/test患者数分别10/38/23/24和13/50/31/31。远端清单与既有冻结患者表一致，患者角色不交叉。所需L/U图像、L/val GT齐备；U无标签路径，测试文件未搬入且未打开。
- 六臂：B00/B10/B01/B11隔离U和历史统计因素；U10/U11为同流程U-Net参照。各家族共享源域终点；目标不回放历史图像。历史仅两维源域强度统计，不能称为解剖生成回放。
- B明确采用MAP G/F、独立局部后验、S/M图像路径；并非全权重Bayesian或原版离散BayeSeg。初始backbone继承源域L-only5000步，G继承旧源域两病例训练，比较重点为家族内；不以超过U-Net作为启动门槛。
- 集成PASS：B/U各一次真实L+U优化，所有可训练参数梯度有限；G/F/S/M梯度非零、teacher无梯度、EMA检查、checkpoint恢复预测完全一致、优化器step恢复、no-U一致性项零。报告见INTEGRATION.json，数据边界见DATA_AUDIT.json。
- 集成前修复了源代码包依赖缺失、旧加载器要求封存test文件存在、CuBLAS确定性环境变量三项工程问题。前两次未做GPU训练，第三次在反向阶段、优化器更新前失败；完成的集成优化共2步。保留远端失败日志和原始时钟，未改变科学目标。
- 正式源B/U已在pro5000 GPU2/3运行，第一次简短检查分别完成51/134更新，loss和梯度有限；进程显示为`python worker.py`，GPU显示`python`。GPU4/5用于后续目标队列，最大四臂并行。
- 新增正式预算40000更新（源2×5000、目标6×5000），加集成2更新。全流程共享首次集成启动时间，23h45控制器硬停、23h55外层timeout兜底；49GB存储硬停、预计<3GB，首次后台统计约98MB。已有数据/初始化只读共享。
- 后台根目录：`/data/bmw/SSL_CL_seg/runs/sslcl_v1_20261021`；中性入口：`/data/bmw/.ops/x20261021_s1`。每臂STATUS/steps/EXIT、总CONTROLLER.json记录实际状态。私有患者表、模型、原始日志均不进入Git。

唯一30分钟任务sc-bayeseg已更新为这次真实SSL+CL实验的监测、结果分析、后继实验和source-only交付。当前尚无源域终点或六臂结果，不能宣称SSL或抗遗忘有效。

## 复现接口

将本仓库源代码加入PYTHONPATH：`experiments`和`experiments/lcrseg`；使用既有PyTorch/h5py/scipy环境。运行前设置`CUBLAS_WORKSPACE_CONFIG=:4096:8`、`PYTHONHASHSEED=163`和NAS/大盘上的缓存目录。
私有settings提供data、冻结split、pretrain、generator、recognition、root、ops、GPU UUID列表和original_started_unix；文件路径仅通过TASK_SETTINGS环境变量传入。`integrate`写入styles；`controller`生成各臂设置，`train`是共用训练入口。中性check.py/worker.py/control.py分别用runpy调用这三个模块，工作目录为ops。先实际挂载/写读探针，再create-only建输出；通过独立timeout启动控制器，不把本机交互会话当运行保障。
