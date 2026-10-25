"""Patient-paired public readout; raw identifiers remain in private source files."""
import json
import sys
from pathlib import Path

import numpy as np

from sc_bayeseg.supervised_cl_v1.summarize import SITES, cell, paired, stage_rows


ARMS = ("er", "ppr")


def summarize(root, source_root, public):
    root, source_root, public = Path(root), Path(source_root), Path(public)
    status = json.loads((root / "STATUS.json").read_text())
    if (status["status"] != "COMPLETE" or status["updates"] != 60000
            or json.loads((root / "EXIT.json").read_text())["exit_code"] != 0):
        raise ValueError("P1 did not complete all planned updates")
    source = stage_rows(source_root, "source")
    rows, fit, training, cost = {}, {}, {}, {}
    for arm in ARMS:
        rows[arm], fit[arm], training[arm], cost[arm] = {}, {}, {}, {}
        for index, site in enumerate(SITES[1:], start=1):
            stage = root / arm / site
            x = json.loads((stage / "STATUS.json").read_text())
            if x["status"] != "TRAINED" or x["step"] != 6000:
                raise ValueError("P1 stage incomplete")
            values = stage_rows(root / arm, site)
            if set(values) != set(SITES[:index + 1]):
                raise ValueError("P1 stage-site validation matrix incomplete")
            fit_values = stage_rows(root / arm, site, "TRAIN")
            if set(fit_values) != {site}:
                raise ValueError("P1 current train fit missing")
            rows[arm][site] = values
            fit[arm][site] = cell(fit_values[site])
            training[arm][site] = x
            cost[arm][site] = json.loads((stage / "PRED_COST_VAL.json").read_text())
    matrix = {arm: {site: {old: cell(values) for old, values in rows[arm][site].items()}
                    for site in SITES[1:]} for arm in ARMS}
    paired_matrix = {site: {old: paired(rows["ppr"][site][old], rows["er"][site][old])
                            for old in SITES[:SITES.index(site) + 1]}
                     for site in SITES[1:]}
    forgetting = {arm: {site: {old: paired(rows[arm][site][old],
                                           source[old] if old == SITES[0] else rows[arm][old][old])
                                   for old in SITES[:SITES.index(site)]}
                        for site in SITES[1:]} for arm in ARMS}
    last = SITES[-1]
    final = paired_matrix[last]
    equal_gain = float(np.mean([final[site]["mean_Dice_delta_pp"] for site in SITES]))
    old_worst = min(final[site]["worst_Dice_delta_pp"] for site in SITES[:-1])
    er_hd = float(np.mean([matrix["er"][last][site]["HD95_penalized_mean_mm"] for site in SITES]))
    ppr_hd = float(np.mean([matrix["ppr"][last][site]["HD95_penalized_mean_mm"] for site in SITES]))
    fit_gate = all(fit[arm][site]["Dice_mean_percent"] >= 80 for arm in ARMS for site in SITES[1:])
    gate = {"fit": fit_gate, "equal_site_Dice_gain_pp": equal_gain,
            "last_site_gain_pp": final[last]["mean_Dice_delta_pp"],
            "worst_old_patient_gain_pp": old_worst,
            "equal_site_HD95_ratio": ppr_hd / er_hd if er_hd > 0 else None}
    gate["candidate_positive"] = bool(fit_gate and equal_gain >= 2
                                      and gate["last_site_gain_pp"] >= -1
                                      and old_worst >= -5 and ppr_hd <= 1.10 * er_hd)
    result = {"status": "COMPLETE", "scope": "all_train", "sites": list(SITES),
              "source_from_prior_completed_experiment": True, "new_optimizer_updates": 60000,
              "test_access": 0, "one_seed_exploratory": True,
              "matrix": matrix, "paired_matrix": paired_matrix,
              "current_train_fit": fit, "forgetting_vs_arrival": forgetting,
              "training": training, "validation_prediction_cost": cost,
              "final_gate": gate, "elapsed_seconds": status["elapsed_seconds"],
              "new_bytes": status["bytes"]}
    public.mkdir(exist_ok=False)
    (public / "ANALYSIS.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    lines = ["# P1：分割潜量后验保留与同源 ER 对照", "",
             "两个 B 臂从同一已完成源检查点出发，各 30,000 次新更新；一个种子，17 名既往使用的 val 患者，仅探索性评价，test 未访问。", "",
             "| 最终 HK 后评价中心 | 患者 | ER Dice % | PPR Dice % | PPR−ER pp | 最差患者 pp | ER HD95 mm | PPR HD95 mm |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for site in SITES:
        er, ppr, delta = matrix["er"][last][site], matrix["ppr"][last][site], final[site]
        lines.append(f"| {site} | {er['patients']} | {er['Dice_mean_percent']:.3f} | "
                     f"{ppr['Dice_mean_percent']:.3f} | {delta['mean_Dice_delta_pp']:+.3f} | "
                     f"{delta['worst_Dice_delta_pp']:+.3f} | {er['HD95_penalized_mean_mm']:.3f} | "
                     f"{ppr['HD95_penalized_mean_mm']:.3f} |")
    lines += ["", f"六域等权 Dice 差 {equal_gain:+.3f} pp；最后 HK 差 {gate['last_site_gain_pp']:+.3f} pp；"
              f"旧域最差患者差 {old_worst:+.3f} pp；等权惩罚 HD95 比 {gate['equal_site_HD95_ratio']:.3f}。",
              f"冻结候选积极门：{'通过' if gate['candidate_positive'] else '未通过'}；训练拟合门：{'通过' if fit_gate else '未通过'}。",
              "阶段×站点完整矩阵、患者尾部、遗忘、训练拟合和成本见 ANALYSIS.json；私有患者级结果与模型保留在 NAS。",
              f"总计新增 60,000 次更新，耗时 {status['elapsed_seconds']:.1f} 秒，新增目录 {status['bytes']} 字节。", ""]
    (public / "REPORT_ZH.md").write_text("\n".join(lines))
    return result


if __name__ == "__main__":
    summarize(sys.argv[1], sys.argv[2], sys.argv[3])
