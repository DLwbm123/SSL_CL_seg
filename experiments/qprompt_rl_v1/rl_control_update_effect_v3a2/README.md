# RL_CONTROL_UPDATE_EFFECT_STATE_V3A2

这是未执行的实验提案，不是新训练结果。

主要变量：同学习器下 BASE16 vs AdamW-aware EFFECT40 状态。

开发：seed261，共64场景。前瞻迁移评价：seeds262/263，共128场景。学生动作、损失、h5定义不变；所有策略统一采用无强制探索下界的确定性主要执行规则。

预算：8,640次临时学生更新；15,360次主要controller更新；2,304次feature VJP；1,728次只读virtual AdamW预演。医学终点0，保留学生更新0。

先阅读 EXPERIMENT_PLAN.md；机器可读任务与预算在 EXECUTION_PLAN.json。运行 python validate_plan.py 只校验协议算术与依赖，不读取医学数据、不运行模型、不调用optimizer。

源码锚点：e5fa35589817224d53fe37cd0e836c871a7a48d6。执行与发布仍需各自当前授权。
