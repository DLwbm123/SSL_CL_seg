# 五框架外部审阅 R1

固定提交 b909a9131dc7137232ac8eb3a6922fe66520b932。结论：CHANGES_REQUESTED，**未批准真实实验**。

先读 `REVIEW_REPORT.md`，然后把整个包与 `CODEX_REVISION_PROMPT.md` 交给 Codex。

`test_external_review_R1.py` 是针对旧代码新增的回归期望，旧版本运行结果为1通过、8失败；不是原104项测试的复跑。测试只需要项目代码与CPU基础依赖。F2异常用明确toy结构函数隔离测试，不代表CWMI复现。

在已检出目标仓库的根目录：
```sh
PYTHONPATH=. python -m pytest -q /path/to/test_external_review_R1.py
```

`TARGETED_REPRO_RESULTS.json` 保存本地源代码SHA256校验、torch版本和反例结果；`REGRESSION_BASELINE_LOG.txt` 保存旧代码的新增回归失败。该包不包含患者数据、权重、真实运行记录或审批回执。源码以GitHub固定提交为准；没有将源码副本随包重新分发。

修复后推送新提交并停止等待审阅。五框架比较计划保持，方法性能尚无新增结论。
