# v5 启动与工程修复

2026-10-23 02:39:11+08首次启动，原硬截止03:39:11+08。GPU1、NAS独立目录protocols/hbsi_real_balance_20261023_09；挂载NFS、11TB可用及实际写读探针通过。预检两次R04通过，真实批次梯度检查不执行优化；估计707.40秒与310,127,776字节通过安全余量。实际时间包括NAS checkpoint写入，独立硬限保持。

controller1扫描存储时碰到原子checkpoint临时文件消失，FileNotFoundError退出，训练plain子进程仍正常运行。保留失败日志与CONTROLLER_FAILURE_1.json；仅修复控制器扫描中的FileNotFoundError竞态，不改模型/目标/协议或核心代码指纹。中性恢复监护等待原训练完成后接回队列，恢复开销按原窗口保守计费。未重新初始化、重复训练或停止原worker。

最后核验plain已保存517/1200步；原worker和恢复监护完整argv均中性。恢复尚在等待plain完成；balanced与四格验证待后继队列，不能声称已完成。本轮源码、修复与结果将在实际完成后统一公开。
