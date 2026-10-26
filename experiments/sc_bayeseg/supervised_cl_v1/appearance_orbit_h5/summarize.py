"""Frozen anonymous three-site H5 decision and cost summary."""

import json
import sys
from pathlib import Path

import numpy as np

from sc_bayeseg.supervised_cl_v1.summarize import paired
from ..hierarchical_bayes_h1.summarize import cell_extended, read
from .run import ARMS, SEED, SITES, UPDATES


def cell_map(rows):
    return {site: cell_extended(patients) for site, patients in rows.items()}


def summarize(root, public):
    root, public = Path(root), Path(public)
    status = json.loads((root / "STATUS.json").read_text())
    started = json.loads((root / "STARTED.json").read_text())
    if (status["status"] != "COMPLETE" or status["updates"] != UPDATES or
            status["stages"] != 8 or
            status["final_rank"] != {"lowrank_control": 8, "orbit_lowrank": 8} or
            status["old_train_replay"] != 0 or
            status["old_teacher"] != 0 or started["test_access"] != 0 or
            json.loads((root / "EXIT.json").read_text())["exit_code"] != 0):
        raise ValueError("H5 run incomplete or access boundary failed")
    synthetic = json.loads((root / "P0_SYNTHETIC.json").read_text())
    real = json.loads((root / "P0_REAL.json").read_text())
    if synthetic["status"] != "PASS" or real["status"] != "PASS":
        raise ValueError("H5 P0 failed")
    precision_audit = json.loads((root / "curvature_balanced" / "SOURCE_PRECISION_AUDIT.json").read_text())
    if precision_audit["patients"] != 18 or precision_audit["curvature_calls"] != 144:
        raise ValueError("H5 source curvature audit incomplete")
    basis_audits = {"SOURCE": json.loads((root / "SOURCE_LOW_RANK_AUDIT.json").read_text()),
                    **{f"BMC_{arm}": json.loads((root / f"BMC_{arm}_LOW_RANK_AUDIT.json").read_text())
                       for arm in ARMS[1:]}}
    if (any(not (root / f"BMC_{arm}_CURRENT_STATE_WITH_BASIS.pt").is_file()
            for arm in ARMS[1:]) or
            any(audit["patients"] != 18 or audit["probes"] != 36 or audit["rank"] != 4 or
                not (0 < audit["captured_energy_fraction"] <= 1) or
                abs(audit["actual_trace"] - audit["target_trace"]) >
                max(1e-3, audit["target_trace"] * 1e-4)
                for audit in basis_audits.values())):
        raise ValueError("H5 current-site low-rank audit incomplete")

    raw, train_fit, training, cost = {}, {}, {}, {}
    for source in ("source_U-FT", "source_C"):
        record = json.loads((root / source / "STATUS.json").read_text())
        if record["status"] != "TRAINED" or record["step"] != 8000:
            raise ValueError("H5 source incomplete")
        raw[(source, 8000)] = read(root, source, 8000)
        train_fit[source] = cell_map(read(root, source, 8000, "TRAIN"))
        if set(raw[(source, 8000)]) != {"RUNMC"} or set(train_fit[source]) != {"RUNMC"}:
            raise ValueError("H5 source matrix incomplete")
        training[source] = record
    for site in SITES[1:]:
        for arm in ARMS:
            stage = f"{arm}_{site}"
            record = json.loads((root / stage / "STATUS.json").read_text())
            if record["status"] != "TRAINED" or record["step"] != 6000:
                raise ValueError("H5 domain stage incomplete")
            for step in (0, 1500, 3000, 6000):
                raw[(stage, step)] = read(root, stage, step)
                if set(raw[(stage, step)]) != set(SITES[:SITES.index(site) + 1]):
                    raise ValueError("H5 fixed-readout matrix incomplete")
            train_fit[stage] = cell_map(read(root, stage, 6000, "TRAIN"))
            if set(train_fit[stage]) != {site}:
                raise ValueError("H5 current-train fit matrix incomplete")
            training[stage] = record
            cost[stage] = {str(step): json.loads((root / stage /
                         f"PRIVATE_{step:04d}_VAL_COST.json").read_text())
                           for step in (0, 1500, 3000, 6000)}
    stress = {site: {arm: json.loads((root / f"{arm}_{site}_ORBIT_STRESS.json").read_text())
                     for arm in ARMS[1:]} for site in SITES[1:]}
    if any(stress[site][arm]["patients"] != (18 if site == "BMC" else 11) or
           stress[site][arm]["slices_per_patient"] != 2 or
           not stress[site][arm]["fixed_current_train_only"]
           for site in SITES[1:] for arm in ARMS[1:]):
        raise ValueError("H5 current-train appearance stress audit incomplete")

    matrix = {stage: cell_map(raw[(stage, 6000)]) for stage, step in raw if step == 6000}
    matrix["source_U-FT"] = cell_map(raw[("source_U-FT", 8000)])
    matrix["source_C"] = cell_map(raw[("source_C", 8000)])
    fixed = {stage: {str(step): cell_map(raw[(stage, step)]) for step in (0, 1500, 3000, 6000)}
             for stage, _ in raw if stage.endswith(("_BMC", "_I2CVB"))}

    def arrival(arm, site):
        if site == "RUNMC":
            return raw[("source_U-FT" if arm == "U-FT" else "source_C", 8000)][site]
        return raw[(f"{arm}_{site}", 6000)][site]

    forgetting = {arm: {site: paired(raw[(f"{arm}_I2CVB", 6000)][site], arrival(arm, site))
                        for site in SITES[:2]} for arm in ARMS}
    relative = {arm: {site: paired(raw[("orbit_lowrank_I2CVB", 6000)][site],
                                  raw[(f"{arm}_I2CVB", 6000)][site])
                      for site in SITES} for arm in ARMS[:2]}
    equal_dice = {arm: float(np.mean([matrix[f"{arm}_I2CVB"][site]["Dice_mean_percent"]
                                     for site in SITES])) for arm in ARMS}
    equal_hd = {arm: float(np.mean([matrix[f"{arm}_I2CVB"][site]["HD95_penalized_mean_mm"]
                                   for site in SITES])) for arm in ARMS}
    old_mean_forget = float(np.mean([-forgetting["orbit_lowrank"][site]["mean_Dice_delta_pp"]
                                     for site in SITES[:2]]))
    old_worst_forget = min(forgetting["orbit_lowrank"][site]["worst_Dice_delta_pp"] for site in SITES[:2])
    fit = all(train_fit[stage][site]["Dice_mean_percent"] >= 80 for stage, site in
              (("source_U-FT", "RUNMC"), ("source_C", "RUNMC"),
               *((f"{arm}_{site}", site) for site in SITES[1:] for arm in ARMS)))
    new_domain = all(
        matrix[f"orbit_lowrank_{site}"][site]["Dice_mean_percent"] >=
        max(matrix[f"U-FT_{site}"][site]["Dice_mean_percent"],
            matrix[f"lowrank_control_{site}"][site]["Dice_mean_percent"]) - 1 and
        train_fit[f"orbit_lowrank_{site}"][site]["Dice_mean_percent"] >=
        max(80, train_fit[f"lowrank_control_{site}"][site]["Dice_mean_percent"] - 2)
        for site in SITES[1:])
    old_safe = (all(matrix["orbit_lowrank_I2CVB"][site]["Dice_mean_percent"] >= 70
                    for site in SITES[:2]) and old_mean_forget <= 10 and old_worst_forget >= -15)
    gain = (equal_dice["orbit_lowrank"] >= equal_dice["U-FT"] + 3 and
            equal_dice["orbit_lowrank"] >= equal_dice["lowrank_control"] + 2)
    relative_tail = all(min(relative[arm][site]["worst_Dice_delta_pp"] for site in SITES[:2]) >= -5
                        for arm in ARMS[:2])
    hd_safe = equal_hd["orbit_lowrank"] <= 1.10 * min(equal_hd["U-FT"], equal_hd["lowrank_control"])
    orbit_mechanism = all(
        stress[site]["orbit_lowrank"]["Dice_mean_percent"] >=
        stress[site]["lowrank_control"]["Dice_mean_percent"] + 2
        for site in SITES[1:])
    new_zero = []
    for site in SITES[1:]:
        stage = f"orbit_lowrank_{site}"
        for step in (0, 1500, 3000, 6000):
            for old in SITES[:SITES.index(site)]:
                if fixed[stage][str(step)][old]["zero_overlap_patients"] > \
                        cell_extended(arrival("orbit_lowrank", old))["zero_overlap_patients"]:
                    new_zero.append({"stage": stage, "step": step, "site": old})
    gate = {"fit": fit, "new_domain_learning": new_domain,
            "absolute_old_preservation": old_safe, "gain_vs_controls": gain,
            "old_patient_relative_tail": relative_tail, "HD95_boundary": hd_safe,
            "no_new_zero_overlap": not new_zero,
            "current_train_orbit_stress": orbit_mechanism}
    gate["candidate_positive"] = all(gate.values())
    result = {"status": "COMPLETE", "seed": SEED, "sites": list(SITES),
              "one_seed_reused_val_exploratory": True, "old_train_replay": 0,
              "old_teacher": 0, "test_access": 0, "optimizer_updates": UPDATES,
              "matrix": matrix, "fixed_readouts": fixed, "train_fit": train_fit,
              "forgetting_vs_arrival": forgetting, "orbit_lowrank_vs_controls_I2CVB": relative,
              "equal_site_Dice_percent": equal_dice, "equal_site_HD95_mm": equal_hd,
              "old_two_site_mean_forgetting_pp": old_mean_forget,
              "worst_old_patient_forgetting_pp": old_worst_forget,
              "new_zero_overlap": new_zero, "gate": gate,
              "source_precision_audit": precision_audit,
              "low_rank_audits": basis_audits, "final_rank": status["final_rank"],
              "current_train_orbit_stress": stress, "training": training,
              "prediction_cost": cost, "elapsed_seconds": status["elapsed_seconds"],
              "new_bytes": status["bytes"]}
    public.mkdir(exist_ok=False)
    (public / "ANALYSIS.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    lines = ["# H5：当前图像的解剖保持型成像轨道", "",
             "同一新种子、同源初始化；U-FT 为标准对照，lowrank_control 与 orbit_lowrank 保持完全相同的低秩 Gaussian 更新和原图 Fisher。orbit_lowrank 在每次当前中心训练批上，将原图完整风险与 H1 可逆成像变换后的同标签风险各半平均。变换只作用于当前中心图像、不移动像素；不回放旧 train、不调用旧模型或教师。旧 val 只读，test 封存。", "",
             "| I2CVB 终点 | U-FT | lowrank_control | orbit_lowrank |",
             "| --- | ---: | ---: | ---: |",
             f"| 三中心等权 Dice % | {equal_dice['U-FT']:.3f} | {equal_dice['lowrank_control']:.3f} | {equal_dice['orbit_lowrank']:.3f} |",
             f"| 三中心等权惩罚 HD95 mm | {equal_hd['U-FT']:.3f} | {equal_hd['lowrank_control']:.3f} | {equal_hd['orbit_lowrank']:.3f} |",
             f"| BMC 当前 train 成像压力 Dice % | — | {stress['BMC']['lowrank_control']['Dice_mean_percent']:.3f} | {stress['BMC']['orbit_lowrank']['Dice_mean_percent']:.3f} |",
             f"| I2CVB 当前 train 成像压力 Dice % | — | {stress['I2CVB']['lowrank_control']['Dice_mean_percent']:.3f} | {stress['I2CVB']['orbit_lowrank']['Dice_mean_percent']:.3f} |",
             "", f"冻结积极门：{'通过' if gate['candidate_positive'] else '未通过'}；"
             f"旧域平均到达后遗忘 {old_mean_forget:.3f} pp，最差旧患者变化 {old_worst_forget:+.3f} pp。", "",
             "完整固定读数、患者尾部、当前 train 成像压力、低秩谱能量及成本见 ANALYSIS.json；逐患者结果、权重和低秩向量仅私有保存。", ""]
    (public / "REPORT_ZH.md").write_text("\n".join(lines))
    return result


if __name__ == "__main__":
    summarize(sys.argv[1], sys.argv[2])
