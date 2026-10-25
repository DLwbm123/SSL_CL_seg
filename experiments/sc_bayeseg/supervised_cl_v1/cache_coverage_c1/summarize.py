"""Anonymous patient-paired report for the fixed replay-cache coverage study."""
import json
import sys
from pathlib import Path

import numpy as np

from sc_bayeseg.supervised_cl_v1.summarize import SITES, cell, paired, stage_rows


ARMS = ("prefix", "coverage")


def summarize(root, public):
    root, public = Path(root), Path(public)
    status = json.loads((root / "STATUS.json").read_text())
    if (status["status"] != "COMPLETE" or status["updates"] != 106000
            or json.loads((root / "EXIT.json").read_text())["exit_code"] != 0):
        raise ValueError("C1 run did not complete")
    stages = ("source", *(f"{arm}_{site}" for site in SITES[1:] for arm in ARMS), "joint")
    rows, fit, training, cost = {}, {}, {}, {}
    for stage in stages:
        s = json.loads((root / stage / "STATUS.json").read_text())
        expected_updates = 8000 if stage == "source" else 38000 if stage == "joint" else 6000
        if s["status"] != "TRAINED" or s["step"] != expected_updates:
            raise ValueError("C1 stage incomplete")
        val = stage_rows(root, stage)
        expected_sites = (set(SITES) if stage == "joint" else {SITES[0]} if stage == "source"
                          else set(SITES[:SITES.index(stage.split("_", 1)[1]) + 1]))
        if set(val) != expected_sites:
            raise ValueError("C1 stage-domain matrix incomplete")
        train = stage_rows(root, stage, "TRAIN")
        if set(train) != (set(SITES) if stage == "joint" else
                          {SITES[0]} if stage == "source" else {stage.split("_", 1)[1]}):
            raise ValueError("C1 training fit matrix incomplete")
        rows[stage], fit[stage], training[stage] = val, {site: cell(v) for site, v in train.items()}, s
        cost[stage] = json.loads((root / stage / "PRED_COST_VAL.json").read_text())
    matrix = {stage: {site: cell(v) for site, v in values.items()} for stage, values in rows.items()}
    paired_matrix = {site: {old: paired(rows[f"coverage_{site}"][old], rows[f"prefix_{site}"][old])
                            for old in SITES[:SITES.index(site) + 1]} for site in SITES[1:]}
    forgetting = {arm: {site: {old: paired(rows[f"{arm}_{site}"][old],
                                           rows["source"][old] if old == SITES[0] else rows[f"{arm}_{old}"][old])
                                   for old in SITES[:SITES.index(site)]}
                        for site in SITES[1:]} for arm in ARMS}
    last = SITES[-1]
    final = paired_matrix[last]
    gain = float(np.mean([final[site]["mean_Dice_delta_pp"] for site in SITES]))
    worst = min(final[site]["worst_Dice_delta_pp"] for site in SITES[:-1])
    old_hd = float(np.mean([matrix[f"prefix_{last}"][site]["HD95_penalized_mean_mm"] for site in SITES]))
    new_hd = float(np.mean([matrix[f"coverage_{last}"][site]["HD95_penalized_mean_mm"] for site in SITES]))
    fit_gate = all(v["Dice_mean_percent"] >= 80 for stage in fit for v in fit[stage].values())
    gate = {"fit": fit_gate, "equal_site_Dice_gain_pp": gain,
            "last_site_gain_pp": final[last]["mean_Dice_delta_pp"],
            "worst_old_patient_gain_pp": worst,
            "equal_site_HD95_ratio": new_hd / old_hd if old_hd > 0 else None}
    gate["candidate_positive"] = bool(fit_gate and gain >= 2
                                      and gate["last_site_gain_pp"] >= -1
                                      and worst >= -5 and new_hd <= 1.10 * old_hd)
    remaining = {site: forgetting["coverage"][last][site]
                 for site in SITES[:-1]
                 if (forgetting["coverage"][last][site]["mean_Dice_delta_pp"] <= -5
                     or forgetting["coverage"][last][site]["worst_Dice_delta_pp"] <= -10)}
    result = {"status": "COMPLETE", "scope": "all_train", "seed": 20261026,
              "sites": list(SITES), "one_seed_exploratory": True, "test_access": 0,
              "optimizer_updates": 106000, "matrix": matrix, "train_fit": fit,
              "paired_matrix": paired_matrix, "forgetting_vs_arrival": forgetting,
              "remaining_old_domain_risk": remaining, "final_gate": gate,
              "training": training, "validation_prediction_cost": cost,
              "elapsed_seconds": status["elapsed_seconds"], "new_bytes": status["bytes"]}
    public.mkdir(exist_ok=False)
    (public / "ANALYSIS.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    lines = ["# C1：32 张旧切片缓存覆盖核查", "",
             "新固定种子 20261026；U-Net Prefix 与 Coverage 共享源训练和同容量缓存，另有六中心离线联合参照。17 名既往使用的 val 仅探索性评价，test 未访问。", "",
             "| 学完 HK 后评价中心 | 患者 | Prefix Dice % | Coverage Dice % | Coverage−Prefix pp | 最差患者 pp | Prefix HD95 mm | Coverage HD95 mm | Joint Dice % |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for site in SITES:
        left = matrix[f"prefix_{last}"][site]
        right = matrix[f"coverage_{last}"][site]
        contrast = final[site]
        joint = matrix["joint"][site]
        lines.append(f"| {site} | {left['patients']} | {left['Dice_mean_percent']:.3f} | "
                     f"{right['Dice_mean_percent']:.3f} | {contrast['mean_Dice_delta_pp']:+.3f} | "
                     f"{contrast['worst_Dice_delta_pp']:+.3f} | {left['HD95_penalized_mean_mm']:.3f} | "
                     f"{right['HD95_penalized_mean_mm']:.3f} | {joint['Dice_mean_percent']:.3f} |")
    lines += ["", f"六域等权 Dice 差 {gain:+.3f} pp；HK 差 {gate['last_site_gain_pp']:+.3f} pp；"
              f"旧域最差患者差 {worst:+.3f} pp；等权惩罚 HD95 比 {gate['equal_site_HD95_ratio']:.3f}。",
              f"冻结 Coverage 积极信号门：{'通过' if gate['candidate_positive'] else '未通过'}；训练拟合门：{'通过' if fit_gate else '未通过'}。",
              "Coverage 最终仍有预定遗忘风险的旧中心：" + ("、".join(remaining) if remaining else "无") + "。",
              "阶段×中心完整矩阵、患者尾部与成本见 ANALYSIS.json；私有患者级结果、缓存和模型不公开。",
              f"共 {status['updates']} 次新优化更新，耗时 {status['elapsed_seconds']:.1f} 秒，新增目录 {status['bytes']} 字节。", ""]
    (public / "REPORT_ZH.md").write_text("\n".join(lines))
    return result


if __name__ == "__main__":
    summarize(sys.argv[1], sys.argv[2])
