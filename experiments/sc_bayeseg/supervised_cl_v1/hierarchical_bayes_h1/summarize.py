"""Anonymous, fixed-gate H1 three-site prefix analysis."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from sc_bayeseg.supervised_cl_v1.summarize import cell, paired

from .run import ARMS, CHECKPOINT_STEPS, DOMAIN_STEPS, SEED, SITES, SOURCE_STEPS


def cell_extended(rows):
    result = cell(rows)
    values = list(rows.values())
    precision = [item["TP"] / (item["TP"] + item["FP"]) if item["TP"] + item["FP"] else 0.0
                 for item in values]
    recall = [item["TP"] / item["GT"] if item["GT"] else 1.0 for item in values]
    result["Precision_mean_percent"] = float(np.mean(precision) * 100)
    result["Recall_mean_percent"] = float(np.mean(recall) * 100)
    return result


def read(root, stage, step, role="VAL"):
    path = root / stage / f"PRIVATE_{step:04d}_{role}.pt"
    private = torch.load(path, map_location="cpu", weights_only=False)
    return {site: {item["patient"]: item["metrics"]["classes"][0] for item in patients}
            for site, patients in private.items()}


def arrival(rows, arm, site):
    if site == "RUNMC":
        source = "source_U-FT" if arm == "U-FT" else "source_C-FT"
        return rows[(source, SOURCE_STEPS)][site]
    if arm == "F_erase" and site == "BMC":
        arm = "F_A"
    return rows[(f"{arm}_{site}", DOMAIN_STEPS)][site]


def summarize(root, public):
    root, public = Path(root), Path(public)
    status = json.loads((root / "STATUS.json").read_text())
    started = json.loads((root / "STARTED.json").read_text())
    exit_code = json.loads((root / "EXIT.json").read_text())["exit_code"]
    if (status["status"] != "PREFIX_COMPLETE" or status["updates"] != 94000 or status["stages"] != 15 or
            exit_code != 0 or started["test_access"] != 0 or started["old_train_replay"] != 0 or
            started["old_teacher"] != 0):
        raise ValueError("H1 three-site prefix did not finish under frozen access rules")
    synthetic = json.loads((root / "P0_SYNTHETIC.json").read_text())
    real = json.loads((root / "P0_REAL.json").read_text())
    if synthetic["status"] != "PASS" or real["status"] != "PASS" or real["discarded_engineering_updates"] != 1:
        raise ValueError("H1 P0 did not pass")
    if real["appearance_parameters"] != 812:
        raise ValueError("H1 appearance parameter count differs from frozen design")
    transitions = [json.loads(line) for line in (root / "TRANSITIONS.jsonl").read_text().splitlines() if line]
    expected_transitions = [(site, arm) for site in SITES[1:] for arm in
                            (ARMS if site == "BMC" else ARMS + ("F_erase",)) if arm not in ("U-FT", "C-FT")]
    if [(item["site"], item["arm"]) for item in transitions] != expected_transitions:
        raise ValueError("H1 process-noise transition matrix incomplete")
    if any(abs(item["relative_variance_budget"] -
               (0 if item["arm"] == "F0" else 812)) > .05 for item in transitions):
        raise ValueError("H1 process-noise budget differs from frozen rule")

    rows, train, training, prediction_cost = {}, {}, {}, {}
    for source in ("source_U-FT", "source_C-FT"):
        record = json.loads((root / source / "STATUS.json").read_text())
        if record["status"] != "TRAINED" or record["step"] != SOURCE_STEPS:
            raise ValueError("H1 source stage incomplete")
        rows[(source, SOURCE_STEPS)] = read(root, source, SOURCE_STEPS)
        train[source] = read(root, source, SOURCE_STEPS, "TRAIN")
        if set(rows[(source, SOURCE_STEPS)]) != {"RUNMC"} or set(train[source]) != {"RUNMC"}:
            raise ValueError("H1 source patient matrix incomplete")
        training[source] = record
        prediction_cost[source] = json.loads((root / source / f"PRIVATE_{SOURCE_STEPS:04d}_VAL_COST.json").read_text())
    for site in SITES[1:]:
        active = ARMS if site == "BMC" else ARMS + ("F_erase",)
        for arm in active:
            stage = f"{arm}_{site}"
            record = json.loads((root / stage / "STATUS.json").read_text())
            if record["status"] != "TRAINED" or record["step"] != DOMAIN_STEPS:
                raise ValueError("H1 stage incomplete")
            for step in CHECKPOINT_STEPS:
                rows[(stage, step)] = read(root, stage, step)
                if set(rows[(stage, step)]) != set(SITES[:SITES.index(site) + 1]):
                    raise ValueError("H1 stage/val-domain matrix incomplete")
            train[stage] = read(root, stage, DOMAIN_STEPS, "TRAIN")
            if set(train[stage]) != {site}:
                raise ValueError("H1 stage/current-train matrix incomplete")
            training[stage] = record
            prediction_cost[stage] = {step: json.loads((root / stage / f"PRIVATE_{step:04d}_VAL_COST.json").read_text())
                                      for step in CHECKPOINT_STEPS}

    matrix = {stage: {site: cell_extended(patients) for site, patients in rows[(stage, DOMAIN_STEPS)].items()}
              for stage, _ in rows if (stage, DOMAIN_STEPS) in rows}
    for source in ("source_U-FT", "source_C-FT"):
        matrix[source] = {"RUNMC": cell_extended(rows[(source, SOURCE_STEPS)]["RUNMC"])}
    fixed_readouts = {stage: {str(step): {site: cell_extended(patients) for site, patients in rows[(stage, step)].items()}
                              for step in CHECKPOINT_STEPS}
                      for stage, _ in rows if stage.startswith(tuple(ARMS + ("F_erase",)))}
    fit = {stage: {site: cell_extended(patients) for site, patients in data.items()} for stage, data in train.items()}

    forgetting = {}
    for site in SITES[1:]:
        for arm in (ARMS if site == "BMC" else ARMS + ("F_erase",)):
            stage = f"{arm}_{site}"
            forgetting[stage] = {old: paired(rows[(stage, DOMAIN_STEPS)][old], arrival(rows, arm, old))
                                 for old in SITES[:SITES.index(site)]}
    final = {arm: {site: cell_extended(rows[(f"{arm}_I2CVB", DOMAIN_STEPS)][site]) for site in SITES}
             for arm in ARMS + ("F_erase",)}
    equal_dice = {arm: float(np.mean([item["Dice_mean_percent"] for item in cells.values()]))
                  for arm, cells in final.items()}
    equal_hd = {arm: float(np.mean([item["HD95_penalized_mean_mm"] for item in cells.values()]))
                for arm, cells in final.items()}
    relative = {arm: {site: paired(rows[("F_A_I2CVB", DOMAIN_STEPS)][site],
                                  rows[(f"{arm}_I2CVB", DOMAIN_STEPS)][site]) for site in SITES}
                for arm in ("U-FT", "C-FT", "F0", "F_all", "F_perm", "F_erase")}

    old_arrival = {old: cell_extended(arrival(rows, "F_A", old))["Dice_mean_percent"] for old in SITES[:2]}
    old_forget = forgetting["F_A_I2CVB"]
    old_mean_forget = float(np.mean([-old_forget[site]["mean_Dice_delta_pp"] for site in SITES[:2]]))
    worst_forget = min(old_forget[site]["worst_Dice_delta_pp"] for site in SITES[:2])
    fit_gate = all(fit[stage][site]["Dice_mean_percent"] >= 80 for stage, site in
                   (("source_U-FT", "RUNMC"), ("source_C-FT", "RUNMC"),
                    *((f"{arm}_{site}", site) for site in SITES[1:] for arm in ("U-FT", "C-FT", "F_A"))))
    new_domain = all(
        matrix[f"F_A_{site}"][site]["Dice_mean_percent"] >=
        max(matrix[f"U-FT_{site}"][site]["Dice_mean_percent"],
            matrix[f"C-FT_{site}"][site]["Dice_mean_percent"]) - 1 and
        fit[f"F_A_{site}"][site]["Dice_mean_percent"] >=
        fit[f"C-FT_{site}"][site]["Dice_mean_percent"] - 2 and
        fit[f"F_A_{site}"][site]["Dice_mean_percent"] >= 80
        for site in SITES[1:])
    absolute_old = (all(final["F_A"][site]["Dice_mean_percent"] >= 70 for site in SITES[:2]) and
                    old_mean_forget <= 10 and worst_forget >= -15)
    architecture = all(equal_dice["F_A"] - equal_dice[arm] >= 3 for arm in ("U-FT", "C-FT"))
    selectivity = all(equal_dice["F_A"] - equal_dice[arm] >= 2 for arm in ("F0", "F_all", "F_perm"))
    precision = (equal_dice["F_A"] - equal_dice["F_erase"] >= 3 and
                 final["F_A"]["I2CVB"]["Dice_mean_percent"] >=
                 final["F_erase"]["I2CVB"]["Dice_mean_percent"] - 1)
    paired_tail = all(min(relative[arm][site]["worst_Dice_delta_pp"] for site in SITES[:2]) >= -5
                      for arm in ("U-FT", "C-FT"))
    hd = equal_hd["F_A"] <= 1.10 * min(equal_hd["U-FT"], equal_hd["C-FT"])
    new_zero = []
    for site in SITES[1:]:
        stage = f"F_A_{site}"
        for step in CHECKPOINT_STEPS:
            for old in SITES[:SITES.index(site)]:
                original = arrival(rows, "F_A", old)
                current = rows[(stage, step)][old]
                if set(original) != set(current):
                    raise ValueError("old patient matching changed")
                count = sum(original[patient]["TP"] > 0 and current[patient]["TP"] == 0 and
                            current[patient]["GT"] > 0 for patient in original)
                if count:
                    new_zero.append({"stage": stage, "step": step, "old_site": old, "count": count})
    gate = {"fit": fit_gate, "new_domain_learning": new_domain, "absolute_old_preservation": absolute_old,
            "vs_U_and_C": architecture, "selective_appearance_vs_filters": selectivity,
            "cumulative_precision_effect": precision, "old_patient_relative_tail": paired_tail,
            "HD95_boundary": hd, "no_new_zero_overlap": not new_zero}
    gate["continue_six_sites"] = all(gate.values())
    result = {"status": "PREFIX_COMPLETE", "seed": SEED, "sites": list(SITES),
              "one_seed_reused_val_exploratory": True, "test_access": 0,
              "old_train_replay": 0, "old_teacher": 0, "optimizer_updates": 94000,
              "discarded_engineering_updates": 1, "matrix": matrix, "fixed_readouts": fixed_readouts,
              "train_fit": fit, "forgetting_vs_arrival": forgetting,
              "F_A_vs_controls_I2CVB": relative, "equal_site_Dice_percent": equal_dice,
              "equal_site_HD95_mm": equal_hd, "old_arrival_Dice_percent": old_arrival,
              "old_two_site_mean_forgetting_pp": old_mean_forget,
              "worst_old_patient_forgetting_pp": worst_forget, "new_zero_overlap": new_zero,
              "gate": gate, "source_precision_audit": json.loads((root / "SOURCE_PRECISION_AUDIT.json").read_text()),
              "transitions": transitions, "training": training, "prediction_cost": prediction_cost,
              "elapsed_seconds": status["elapsed_seconds"], "new_bytes": status["bytes"]}
    public.mkdir(exist_ok=False)
    (public / "ANALYSIS.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    lines = ["# H1：无回放层级 Bayesian 三中心机制包", "",
             "六条完整 RUNMC→BMC→I2CVB 链加 F_erase 单转移；只以当前中心 train 优化。",
             "17 名既往使用的六中心 val 中，本前缀只涉及前 11 名；test 封存。单种子结果仅供开发判断。", "",
             "| I2CVB 终点 | 三中心等权 Dice % | 三中心等权惩罚 HD95 mm | I2CVB Dice % |",
             "| --- | ---: | ---: | ---: |"]
    for arm in ARMS + ("F_erase",):
        lines.append(f"| {arm} | {equal_dice[arm]:.3f} | {equal_hd[arm]:.3f} | "
                     f"{final[arm]['I2CVB']['Dice_mean_percent']:.3f} |")
    lines += ["", "## 固定继续门", "", "| 项目 | 结果 |", "| --- | --- |"]
    for name, passed in gate.items():
        lines.append(f"| {name} | {'通过' if passed else '未通过'} |")
    lines += ["", f"旧两域平均到达后遗忘 {old_mean_forget:.3f} pp；最差旧患者变化 {worst_forget:+.3f} pp。",
              f"阶段内新增旧域零重叠读数 {sum(item['count'] for item in new_zero)}。", "",
              "## 全部阶段终点", "",
              "| 阶段 | 评价中心 | 患者 | Dice % | Precision % | Recall % | p10 % | HD95 mm | 零重叠 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for stage, sites in matrix.items():
        for site, item in sites.items():
            lines.append(f"| {stage} | {site} | {item['patients']} | {item['Dice_mean_percent']:.3f} | "
                         f"{item['Precision_mean_percent']:.3f} | {item['Recall_mean_percent']:.3f} | "
                         f"{item['Dice_p10_percent']:.3f} | {item['HD95_penalized_mean_mm']:.3f} | "
                         f"{item['zero_overlap_patients']} |")
    lines += ["", "阶段内 0/1500/3000/6000 的完整已见中心汇总、患者尾部、拟合和资源成本见 ANALYSIS.json。",
              "逐患者结果、模型、权重精度与原始日志仅存私有 NAS。",
              f"正式物理更新 {status['updates']} 次，另有 1 次丢弃的真实 P0 更新；"
              f"耗时 {status['elapsed_seconds']:.1f} 秒，新增目录 {status['bytes']} 字节。", ""]
    (public / "REPORT_ZH.md").write_text("\n".join(lines))
    return result


if __name__ == "__main__":
    summarize(sys.argv[1], sys.argv[2])
