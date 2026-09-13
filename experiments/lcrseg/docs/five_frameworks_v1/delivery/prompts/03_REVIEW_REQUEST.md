# 给当前ChatGPT｜Codex提交后的代码审阅请求

Codex已经完成第一轮实现并推送，请你审阅这次具体commit，不要只读README或测试通过数量。

```text
repo: DLwbm123/SSL_CL_seg
branch: <Codex返回的真实branch>
base_commit: <真实完整SHA>
review_commit: <真实完整SHA>
PR: <真实PR链接或NOT_CREATED>
review_index: experiments/lcrseg/docs/five_frameworks_v1/review/REVIEW_INDEX.md
parent_binding_status: <实际状态>
CPU_tests: <实际结果>
CUDA_tests: NOT_RUN_CODE_ONLY（若实际不同按实际填）
formal_optimizer_updates: 0（必须按实际填，不照抄）
```

审阅重点：
1. 五个候选是否完整落地，有没有loss为0、toy backend或configuration-only冒充实现。
2. 原单侧隔离父方法身份和训练/部署语义是否保持；Q与feature/卷积坐标是否正确。
3. F1 JML/R-only、F2结构方向、F3逐位置反向、F4d-only修正、F5类别条件SWD的实际梯度路径。
4. 新F参数与阶段合并、current EMA/prototypes生命周期，是否真正没有新增历史数据记忆。
5. 共同L/U输入、样本与计算预算、基线HPO、公平消融与三域统计。
6. 数据隔离、精确恢复与默认运行锁。

请给出具体文件/函数/行范围和问题等级。若可启动，明确批准的代码commit、科学配置、parent、依赖与B–D预算；若不可启动，区分必须修复的正确性问题与可后续改进项。

不要要求每个候选理论上必胜，也不要因为旧实验negative就否决这次合理开发。没有充分源代码证据不能宣布通过。
