"""Anonymous paired report for the fixed no-replay gradient-summary study."""
import json
import sys
from pathlib import Path

import numpy as np

from sc_bayeseg.supervised_cl_v1.summarize import SITES, cell, paired, stage_rows


ARMS = ("control", "protected")
STAGES = ("source", *(f"{arm}_{site}" for site in SITES[1:] for arm in ARMS))


def summarize(root, public):
    root, public = Path(root), Path(public)
    status = json.loads((root / "STATUS.json").read_text())
    if (status["status"] != "COMPLETE" or status["stages"] != 11 or
            status["updates"] != 68000 or status["old_train_replay"] != 0 or
            status["old_model_teacher"] != 0 or
            json.loads((root / "EXIT.json").read_text())["exit_code"] != 0):
        raise ValueError("N1 run did not complete under the frozen access rules")
    p0 = json.loads((root / "P0.json").read_text())
    if p0["status"] != "PASS" or p0["source_basis_rank"] < 4:
        raise ValueError("N1 P0 failed")
    basis_audit = [json.loads(line) for line in (root / "BASIS_AUDIT.jsonl").read_text().splitlines() if line]
    if ([item["site"] for item in basis_audit] != list(SITES[:-1]) or
            any(not 4 <= item["new_directions"] <= 8 or
                item["total_directions"] > 40 or
                item["orthogonality_error"] > 2e-4 for item in basis_audit)):
        raise ValueError("N1 gradient summary audit failed")
    rows, fit, training, cost = {}, {}, {}, {}
    for stage in STAGES:
        recorded = json.loads((root / stage / "STATUS.json").read_text())
        steps = 8000 if stage == "source" else 6000
        if recorded["status"] != "TRAINED" or recorded["step"] != steps:
            raise ValueError("N1 stage incomplete")
        val = stage_rows(root, stage)
        expected = {SITES[0]} if stage == "source" else set(SITES[:SITES.index(stage.split("_", 1)[1]) + 1])
        if set(val) != expected:
            raise ValueError("N1 validation matrix incomplete")
        train = stage_rows(root, stage, "TRAIN")
        expected_train = {SITES[0]} if stage == "source" else {stage.split("_", 1)[1]}
        if set(train) != expected_train:
            raise ValueError("N1 current-train fit matrix incomplete")
        rows[stage] = val
        fit[stage] = {site: cell(patients) for site, patients in train.items()}
        training[stage] = recorded
        cost[stage] = json.loads((root / stage / "PRED_COST_VAL.json").read_text())
    matrix = {stage: {site: cell(patients) for site, patients in val.items()}
              for stage, val in rows.items()}
    forgetting = {arm: {site: {old: paired(rows[f"{arm}_{site}"][old],
                                           rows["source"][old] if old == SITES[0]
                                           else rows[f"{arm}_{old}"][old])
                                   for old in SITES[:SITES.index(site)]}
                        for site in SITES[1:]} for arm in ARMS}
    final = {site: paired(rows[f"protected_{SITES[-1]}"][site],
                          rows[f"control_{SITES[-1]}"][site]) for site in SITES}
    equal_dice = {arm: float(np.mean([matrix[f"{arm}_{SITES[-1]}"][site]["Dice_mean_percent"]
                                     for site in SITES])) for arm in ARMS}
    equal_hd = {arm: float(np.mean([matrix[f"{arm}_{SITES[-1]}"][site]["HD95_penalized_mean_mm"]
                                   for site in SITES])) for arm in ARMS}
    worst_old = min(final[site]["worst_Dice_delta_pp"] for site in SITES[:-1])
    fit_gate = all(value["Dice_mean_percent"] >= 80 for stage in fit.values() for value in stage.values())
    gain = equal_dice["protected"] - equal_dice["control"]
    hd_ratio = equal_hd["protected"] / equal_hd["control"]
    absolute_old = all(forgetting["protected"][SITES[-1]][site]["mean_Dice_delta_pp"] > -5 and
                       forgetting["protected"][SITES[-1]][site]["worst_Dice_delta_pp"] > -10
                       for site in SITES[:-1])
    gate = {"fit": fit_gate, "equal_site_Dice_gain_pp": gain,
            "last_site_gain_pp": final[SITES[-1]]["mean_Dice_delta_pp"],
            "worst_old_patient_gain_pp": worst_old,
            "equal_site_HD95_ratio": hd_ratio,
            "absolute_old_safety": absolute_old,
            "candidate_positive": bool(fit_gate and gain >= 2 and
                                       final[SITES[-1]]["mean_Dice_delta_pp"] >= -1 and
                                       worst_old >= -5 and hd_ratio <= 1.10)}
    result = {"status": "COMPLETE", "scope": "all_train", "seed": 20261027,
              "sites": list(SITES), "one_seed_exploratory": True, "test_access": 0,
              "old_train_replay": 0, "old_model_teacher": 0,
              "optimizer_updates": 68000, "engineering_golden_updates_discarded": 1,
              "matrix": matrix, "train_fit": fit, "forgetting_vs_arrival": forgetting,
              "protected_vs_control_final": final,
              "final_equal_site_Dice_percent": equal_dice,
              "final_equal_site_HD95_mm": equal_hd,
              "final_gate": gate, "basis_audit": basis_audit,
              "training": training, "validation_prediction_cost": cost,
              "elapsed_seconds": status["elapsed_seconds"], "new_bytes": status["bytes"]}
    public.mkdir(exist_ok=False)
    (public / "ANALYSIS.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    lines = ["# N1：BayeSeg 无回放分割潜量梯度子空间保护", "",
             "两个 B 臂共享 RUNMC 源模型，各阶段只用当前中心 train；候选另保留小型梯度子空间摘要，不使用旧影像、标签或旧模型教师。17 名既往使用的 val 仅探索性评价，test 封存。", "",
             "| 学完 HK 后评价中心 | 患者 | 无回放 B 对照 Dice % | 保护候选 Dice % | 候选−对照 pp | 最差患者 pp | 对照 HD95 mm | 候选 HD95 mm |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for site in SITES:
        left = matrix[f"control_{SITES[-1]}"][site]
        right = matrix[f"protected_{SITES[-1]}"][site]
        difference = final[site]
        lines.append(f"| {site} | {left['patients']} | {left['Dice_mean_percent']:.3f} | "
                     f"{right['Dice_mean_percent']:.3f} | {difference['mean_Dice_delta_pp']:+.3f} | "
                     f"{difference['worst_Dice_delta_pp']:+.3f} | {left['HD95_penalized_mean_mm']:.3f} | "
                     f"{right['HD95_penalized_mean_mm']:.3f} |")
    lines += ["", f"六域等权 Dice 对照／候选 {equal_dice['control']:.3f}%／{equal_dice['protected']:.3f}%，差 {gain:+.3f} pp；"
              f"新 HK 差 {gate['last_site_gain_pp']:+.3f} pp，旧域最差患者差 {worst_old:+.3f} pp，"
              f"惩罚 HD95 比 {hd_ratio:.3f}。",
              f"冻结相对积极门：{'通过' if gate['candidate_positive'] else '未通过'}；"
              f"绝对旧域安全门：{'通过' if absolute_old else '未通过'}；"
              f"训练拟合门：{'通过' if fit_gate else '未通过'}。", "",
              "## 全部阶段×已见中心", "",
              "| 阶段 | 中心 | 患者 | Dice 均值 % | Dice p10 % | 惩罚 HD95 mm | 零重叠 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for stage in STAGES:
        for site in SITES:
            if site in matrix[stage]:
                item = matrix[stage][site]
                lines.append(f"| {stage} | {site} | {item['patients']} | {item['Dice_mean_percent']:.3f} | "
                             f"{item['Dice_p10_percent']:.3f} | {item['HD95_penalized_mean_mm']:.3f} | "
                             f"{item['zero_overlap_patients']} |")
    lines += ["", "## 最终旧域遗忘", "",
              "| 路线 | 旧中心 | 平均 Dice 变化 pp | 最差患者变化 pp | 下降患者/总数 |",
              "| --- | --- | ---: | ---: | ---: |"]
    for arm in ARMS:
        for site in SITES[:-1]:
            item = forgetting[arm][SITES[-1]][site]
            lines.append(f"| {arm} | {site} | {item['mean_Dice_delta_pp']:+.3f} | "
                         f"{item['worst_Dice_delta_pp']:+.3f} | {item['declined_patients']}/{item['patients']} |")
    lines += ["", "完整逐阶段遗忘、训练拟合、摘要维数与计算成本见 ANALYSIS.json；逐患者结果、梯度摘要、模型和原始日志保留私有。",
              f"正式优化更新 {status['updates']} 次，另有 1 次丢弃的 P0 工程 golden 更新；"
              f"耗时 {status['elapsed_seconds']:.1f} 秒，新增目录 {status['bytes']} 字节。", ""]
    (public / "REPORT_ZH.md").write_text("\n".join(lines))
    return result


if __name__ == "__main__":
    summarize(sys.argv[1], sys.argv[2])
