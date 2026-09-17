请依据随附 METHOD_AND_EXPERIMENT_SPEC.md 准备 NATIVE_KEY_ALIGNMENT_V0_1。
当前只做代码、有限合成测试、冻结D1计划与审阅材料，不启动真实数据训练；该任务不是旧F5 P2的恢复。

保留 NATIVE_LR_SRC_A_3DOMAIN_V1 + B2_C06 全部原科学配置和原生A-only硬投影。不要增加R/F、B侧正交、旧teacher、历史样本/原型队列、新投影MLP或新优化器规则。

唯一新模块是对 decoder.dec1.merge.block.3 的真实144维卷积输入patch，以已有入口V的8维键空间做当前teacher-L与student-U的类别条件SWD。所有原生A/B保持B2的KL更新权限。V/teacher参考stop-gradient，辅助项保持对上游输入特征的梯度。只增加一次当前clean-U学生前向；保持已有LCTX和UL路径。

第一批仅D1：C0 B2 no-op、C1全patch SWD、C2同维随机键SWD、C3权重V键SWD。辅助空间维度D统一使用(D/8)*SWD，辅助系数.05。随机键只能改变辅助loss，不能改变原生保护V。

D1明确采用已封存B2的共同第一阶段前缀：seed163/164、O1/O2，4臂共16个第二目标阶段、42400次正式更新。source与第一阶段训练0。前缀复用必须是新协议明确声明且hash绑定的诊断权限，不伪造或更改旧receipt身份。D2/D3只留计划，不展开可执行节点。

先完成形状/patch坐标/类别映射、lambda=0等价、参考stop-gradient、可达上游非零梯度、原硬投影残差、随机流和样本位置匹配的合成测试。B初始化为0时不能要求所有A梯度非零。无效类必须graph-connected零并记录，不伪造标签或support。

不添加宽泛调参、动态门控或梯度裁剪。所有新增诊断VJP与前向单列成本。独立新包与新审阅材料；旧F5结果与批准文件不改动、不复用为新任务启动批准。

产出代码差异、真实层与张量绑定、D1_MATRIX、冻结options、方法和数据权限说明、合成测试及成本、待审阅风险清单，然后停止于CODE_READY_FOR_REVIEW。不要启动CUDA/真实smoke/正式训练或监测。
