# 实现与调用绑定

状态：实现完成，尚未执行P0或正式实验。source_decision.py注册到既有philips_transfer/gyro target引擎，复用SGD、裁剪、恢复、终点评测及硬资源账。source_decision_audit.py仅在退出后执行一次、无模型前向。

真实读出采用原生3×3 valid卷积后align_corners=True插值；保护特征直接unfold后完整插值，避免稀疏索引近似。所有四组计算同样的源/学生前向与DS/FULL标量；只有C/D将对应损失加入反向。源参考在model state/optimizer之外，部署checkpoint只含native学生和EMA。源row-space仅控制器做一次SVD并保存只读basis；两种设备合成SVD各一次、合成反向各两次。

原engine正式preflight仅执行空optimizer/state检查并保存初始状态，零forward；完整golden、函数一致和恢复已在每组32步资格中核查。保证正式前向恰为96000，诊断复用图：24000更新反向+108固定三梯度诊断=24108。

正常路径预计：P0工程84 native/34 reverse，64步probe270/66，16批测量64/48，合计418/148，optimizer96；两条零辅助对齐256/64，optimizer64；十二资格1008/408，optimizer384；正式96000/24108，optimizer24000；评测34704/0。各bucket保持冻结硬cap2000/512、512/80、4096/512、96000/24200、34704/0，不跨bucket借用。最终审计核实际账及退休进程，非将静态预计当实测。

累计存储使用12小时窗口账：本轮预留4GB，含2MB操作元数据；另给本地新增源文件和交付副本保守留量。单轮3.5GB写入保护+0.5GB恢复余量、8h原钟，global累计<30GB。

P0的前景覆盖、损失贡献、梯度可辨识性或尺度任一失败，控制器记录P0_STOP且不开始正式；退出后一次审计、公开报告。不是医疗有效性成功。
