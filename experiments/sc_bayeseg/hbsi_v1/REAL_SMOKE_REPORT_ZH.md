# 真实数据 smoke

**NOT_RUN / SKIPPED_UNAVAILABLE_CURRENT_PROTOCOL**。本轮没有绑定到已确认的 NAS 当前域训练协议；没有擅自把旧 B11/pro5000 运行路径当作本版本的新数据协议，也没有访问真实 train/val/test。

实际运行 `real-smoke --output OUT --run-id cli_real-smoke`，退出码0，生成 NOT_RUN 回执。此退出码仅表明跳过记录成功，不能称真实 smoke 通过。真实数据前向、反向、外层更新均为0。

已用人工HDF5 fixture验证当前域 L/U读取、U无标签能力、旧域/val/test拒绝及图像尺寸接口；这不是患者数据验证。真实零更新入口已实现，需实际协议与 `configs/real_smoke.example.yaml`，最多2批次、每批次1–2样本、global MC=1、K<=3，绝不执行 optimizer.step。临时image-only预测K=0，回插尺寸来自读取图像的实际原生shape。

已有医学评价函数通过适配接口复用；本轮未对真实 native-space 几何、患者/phase 聚合或HD95运行资格验证。真实训练与评价保持未运行。
