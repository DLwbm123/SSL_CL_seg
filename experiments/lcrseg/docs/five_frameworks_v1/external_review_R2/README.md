# R2 外部审阅包

对提交 `bac21b6585ec284fb6e93f213e280c1492f6e57a` 的定向复核。

结论：R01–R06 接受此次修订；新增 R07（清单/许可可变引用）须在真实接入前修复。保持 CODE_ONLY，不批准真实实验，不要求改变五框架或扩增搜索。

- REVIEW_REPORT.md：报告与证据边界。
- CODEX_REVISION_PROMPT.md：现在交给 Codex 的小范围修订提示词。
- test_external_review_R2.py：新增 helper/桥接回归。放进仓库后默认 import 正式模块，不设置快照环境变量。
- SOURCE_VERIFICATION.json：5 个本地重建文件与固定提交清单逐字节 SHA256 对照。
- LOCAL_TEST_REPORT.json、LOCAL_HELPER_TESTS.txt/XML：外部局部测试 7 passed / 4 failed；不代表重跑了仓库 162 项。
- audit_loader.py、reviewed_source_snapshot/：仅供复现此次局部审阅，禁止用作生产实现替代品。

本包局部验证（只执行符号 reader 回调和数学 helper）：

```sh
SSLCL_R2_SNAPSHOT="$PWD/reviewed_source_snapshot" \
python -m pytest -q test_external_review_R2.py
```

修复后的仓库验证（不要带上述环境变量）：

```sh
python -m pytest -q experiments/lcrseg/tests/five_frameworks_v1/test_external_review_R2.py
```

四项新失败覆盖 L/U manifest 修改、许可清单修改和许可预算修改，共同指向一个数据快照问题。没有真实文件被这些 reader 打开。
