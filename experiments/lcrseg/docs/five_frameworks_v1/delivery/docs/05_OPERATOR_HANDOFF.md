# 05｜用户需要给Codex提供什么

## 最少输入

**1. 本包整个目录。** 不必手工逐段复制长规格。让Codex首先读取README，再按Prompt01执行。

**2. 真正要保留的父方法线索。** 最有价值的是原始KI/单侧隔离的启动命令、run_id、源码目录、配置或commit任意一个足够明确的线索。若当前工作区已有且可定位，Codex自行解析，不能再重复索要。历史SOTA表不是必要条件，也不要求你重新发送原始患者数据。此前共享的8a6e93a是本轮参考快照，不自动等于原KI。

**3. 现有运行环境的入口。** 工作区路径、Python环境、数据/manifest位置、输出目录和已有SSH别名。已有配置能解析就复用；缺的才列出。使用Codex现有授权，不把密码、token、SSH私钥放在对话或GitHub。

**4. GitHub写权限。** 目标仓库已知DLwbm123/SSL_CL_seg；让Codex使用当前GitHub登录/SSH凭据，创建独立feature branch并push。不要给它main merge授权。

**5. 实验预算确认在代码审阅后给。** GPU4/5/6/7是此前共享授权提示，不能据此声称当前可用。本轮A只写代码，无需先占GPU或读真实数据。真正启动时审批具体commit、B–D计算上限与存储目录。

## 可直接给Codex的最短说明

> 请读取这个目录中的 README.md 与 prompts/01_CODE_IMPLEMENTATION_AND_PUSH.md，完成五个框架的统一实现、CPU合成测试和实验计划，推送到独立GitHub分支后停止，等待外部代码审阅。不要训练。原始父方法线索是：`在这里放已知路径/命令/run_id；若工作区已明确则不需要重复填写`。其余路径从现有配置解析。

详细prompt在文件中，不要只执行这段简写而忽略约束。

## Codex结束后带回本对话的内容

```text
status=STOP_AWAITING_EXTERNAL_CODE_REVIEW
repo=DLwbm123/SSL_CL_seg
branch=codex/sslcl-five-frameworks-v1-review
base_commit=<actual>
review_commit=<actual full SHA>
PR=<actual URL or NOT_CREATED>
review_index=<repo-relative path>
parent_binding=<VERIFIED or concrete missing field>
frameworks=<F1..F5 completeness summary>
tests=<actual passed/failed/skipped, CPU/CUDA status separately>
real_image_reads=0
real_label_reads=0
checkpoint_tensor_reads=0
real_smoke_updates=0
formal_optimizer_updates=0
push_verified=<actual remote SHA>
```

提交上述信息或直接发PR链接后，本对话可以针对那次具体提交审阅代码。不能要求Codex代替外部审阅者自动签“通过”。若审阅要求修复，再push新的SHA并复审影响部分。

## 审阅通过后

用Prompt02，传入真实外部审阅结果、绑定commit/hash的approval文件、用户明确启动确认，以及实际预算。运行B→C→D；D产出五框架排名与最多两个shortlist，停止并交回结果。

本轮不自动启动E/F；后续贡献消融、正式基线与独立确认依赖选中的框架，也需要结果后的实际计划。这不减少本次五框架的8配置开发和三种子比较。

## 不需要给Codex什么

不需要上传原始数据到GitHub；不需要给它所有旧实验日志或全历史归档；不需要在此时人工设定最佳超参数；不需要把旧gate全部改成PASS；也不需要保证五个方法一定有一个显著优于基线。
