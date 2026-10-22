# P0 冻结旧教师正确性诊断结果

总 gate：**FAIL**。3 seed，各相同4名当前域训练标注患者，全部切片，零优化器更新。此集已被 EMA 训练见过，仅为有偏的可行性诊断。

## 早期 S1000 放行指标

| seed | view | old FG recall | old FG coverage | FG complement | mix−EMA FG NLL | BG NLL | FG confwrong | BG confwrong |
|---|---|---|---|---|---|---|---|---|
| 163 | current | 0.613770 | 0.828137 | 0.001538 | 0.264886 | -0.001211 | 0.005659 | -0.001390 |
| 163 | history | 0.632537 | 0.912763 | 0.001373 | 0.252103 | -0.000336 | 0.007094 | -0.001215 |
| 164 | current | 0.580386 | 0.769548 | 0.001517 | 0.277965 | -0.000091 | 0.008794 | -0.000374 |
| 164 | history | 0.578042 | 0.797624 | 0.003150 | 0.274532 | 0.000271 | 0.009376 | -0.000342 |
| 165 | current | 0.341551 | 0.401300 | 0.000000 | 0.447614 | -0.002916 | 0.009368 | -0.000777 |
| 165 | history | 0.383182 | 0.469622 | 0.000000 | 0.418181 | -0.002156 | 0.009589 | -0.000631 |

## 固定门槛判定

- 163: FAIL; current:fg:mix_nll_increment_above_.05, history:fg:mix_nll_increment_above_.05, history:fg_complement_below_.01; 134 slices, 1608 forward calls.
- 164: FAIL; current:fg:mix_nll_increment_above_.05, history:fg:mix_nll_increment_above_.05, history:fg_complement_below_.01; 134 slices, 1608 forward calls.
- 165: FAIL; current:fg:mix_nll_increment_above_.05, history:fg:mix_nll_increment_above_.05, history:fg_complement_below_.01; 134 slices, 1608 forward calls.

## 解释与边界

本轮三个 seed 均具备基本前景覆盖，但固定混合在前景 NLL 上超过风险门槛，history 早期前景互补也不足。不能用背景指标的小幅改善抵消前景风险。P0 否决本轮固定混合的训练资格，四臂未启动。

这是当前训练标注患者上的诊断；EMA 已见过这些标签，故不能推断真实 U 上绝无旧教师互补，也不能证明旧域保持无效。下一问题应首先解决资格诊断的训练暴露偏差，不能以调小 rho、换 checkpoint 或追加 seed 绕过本 gate。任何独立验证设计需先冻结数据隔离和匹配对照，保持真实 U 标签、val/test/HK 禁止用于诊断选择的边界。

完整逐患者匿名数值（NLL/Brier/错误/高置信错误/覆盖率、互补及反向互补），含 original/current/history × S1000/S3000/S5000，见同目录 JSON。患者是四人，seed不能作为新增患者。

PASS才放行固定四臂；FAIL不启动、不改变阈值或换checkpoint。混合标签质量不是训练收益或旧域保持的证明。
