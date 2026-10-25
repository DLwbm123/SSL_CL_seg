"""Predeclared paired BayeSeg-vs-U-Net readout; no patient identifiers leave NAS."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from sc_bayeseg.supervised_cl_v1.summarize import SITES, STAGES, cell, paired, stage_rows


def compare(u_root, b_root, public):
    roots = {"U": Path(u_root), "B": Path(b_root)}
    catalog_hashes = [hashlib.sha256((root / "PRIVATE_CATALOG.json").read_bytes()).hexdigest() for root in roots.values()]
    if len(set(catalog_hashes)) != 1:
        raise ValueError("paired experiments used different patient catalogs")
    for family, root in roots.items():
        status = json.loads((root / "STATUS.json").read_text())
        started = json.loads((root / "STARTED.json").read_text())
        if status["status"] != "COMPLETE" or status["family"] != family or started["family"] != family:
            raise ValueError("paired experiment incomplete or family mismatch")
        if json.loads((root / "EXIT.json").read_text())["exit_code"] != 0:
            raise ValueError("paired experiment failed")
    rows = {family: {stage: stage_rows(root, stage) for stage in STAGES} for family, root in roots.items()}
    training = {family: {stage: stage_rows(root, stage, "TRAIN") for stage in STAGES}
                for family, root in roots.items()}
    table = {}
    for stage in STAGES:
        table[stage] = {}
        for site, b_patients in rows["B"][stage].items():
            u_patients = rows["U"][stage][site]
            contrast = paired(b_patients, u_patients)
            u = cell(u_patients)
            b = cell(b_patients)
            table[stage][site] = {"U_Dice_percent": u["Dice_mean_percent"],
                                  "B_Dice_percent": b["Dice_mean_percent"],
                                  "B_minus_U": contrast,
                                  "U_HD95_penalized_mm": u["HD95_penalized_mean_mm"],
                                  "B_HD95_penalized_mm": b["HD95_penalized_mean_mm"],
                                  "U_zero_overlap": u["zero_overlap_patients"],
                                  "B_zero_overlap": b["zero_overlap_patients"]}
    final = table[f"replay_{SITES[-1]}"]
    equal_site_dice_gain = float(np.mean([final[s]["B_minus_U"]["mean_Dice_delta_pp"] for s in SITES]))
    final_new_delta = final[SITES[-1]]["B_minus_U"]["mean_Dice_delta_pp"]
    worst_old_delta = min(final[s]["B_minus_U"]["worst_Dice_delta_pp"] for s in SITES[:-1])
    u_hd = float(np.mean([final[s]["U_HD95_penalized_mm"] for s in SITES]))
    b_hd = float(np.mean([final[s]["B_HD95_penalized_mm"] for s in SITES]))
    fits = {family: {stage: {site: cell(values)["Dice_mean_percent"] for site, values in sites.items()}
                     for stage, sites in training[family].items()} for family in roots}
    fit_gate = all(value >= 80 for family in fits for stage in fits[family]
                   for value in fits[family][stage].values())
    gain_gate = equal_site_dice_gain >= 2 and final_new_delta >= -1 and worst_old_delta >= -5 and b_hd <= 1.10 * u_hd
    result = {"status": "COMPLETE", "scope": "all_train", "sites": list(SITES), "patients": 17,
              "one_seed_exploratory": True, "test_access": 0, "paired_matrix": table,
              "train_fit_percent": fits,
              "final_replay_gate": {"fit": fit_gate, "equal_site_Dice_gain_pp": equal_site_dice_gain,
                                    "last_site_gain_pp": final_new_delta, "worst_old_patient_gain_pp": worst_old_delta,
                                    "equal_site_HD95_ratio": b_hd / u_hd if u_hd > 0 else None,
                                    "candidate_positive": fit_gate and gain_gate},
              "costs": {family: {**{key: json.loads((root / "STATUS.json").read_text())[key]
                                      for key in ("elapsed_seconds", "bytes", "updates")},
                                  "parameters": json.loads((root / "P0.json").read_text())["parameters"],
                                  "final_validation_seconds_per_image": json.loads(
                                      (root / "joint" / "PRED_COST_VAL.json").read_text())["seconds_per_image"]}
                        for family, root in roots.items()}}
    public = Path(public)
    public.mkdir(exist_ok=True)
    (public / "COMPARISON.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    lines = ["# 六中心全监督持续学习：BayeSeg 层级与 U-Net 配对比较", "",
             "同一全训练病例清单、骨干初值、阶段顺序、增强、更新数和固定总量重放；一个种子，既往使用过的 val，test 未访问。", "",
             "| 最终重放评价中心 | U Dice % | B Dice % | B−U pp | B−U 最差患者 pp | U HD95 mm | B HD95 mm |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for site in SITES:
        row = final[site]
        lines.append(f"| {site} | {row['U_Dice_percent']:.3f} | {row['B_Dice_percent']:.3f} | "
                     f"{row['B_minus_U']['mean_Dice_delta_pp']:+.3f} | {row['B_minus_U']['worst_Dice_delta_pp']:+.3f} | "
                     f"{row['U_HD95_penalized_mm']:.3f} | {row['B_HD95_penalized_mm']:.3f} |")
    gate = result["final_replay_gate"]
    hd_text = f"{gate['equal_site_HD95_ratio']:.3f}" if gate['equal_site_HD95_ratio'] is not None else "不可计算"
    lines += ["", f"六域等权 Dice 差 {gate['equal_site_Dice_gain_pp']:+.3f} pp；最后中心差 {gate['last_site_gain_pp']:+.3f} pp；"
              f"旧域患者最差配对差 {gate['worst_old_patient_gain_pp']:+.3f} pp；六域等权惩罚 HD95 比 {hd_text}。",
              f"冻结候选积极信号门：{'通过' if gate['candidate_positive'] else '未通过'}；训练拟合门：{'通过' if gate['fit'] else '未通过'}。",
              "单种子和反复使用的 val 只能用于探索性判断；不宣称独立确认。完整阶段矩阵、失败病例计数与成本见 COMPARISON.json。", ""]
    (public / "COMPARISON_ZH.md").write_text("\n".join(lines))
    return result


if __name__ == "__main__":
    compare(sys.argv[1], sys.argv[2], sys.argv[3])
