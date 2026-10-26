"""Apply the frozen G0 gates to anonymous summaries and paired private readouts."""

import json
from pathlib import Path

from sc_bayeseg.supervised_cl_v1 import summarize as base


SITES = ("RUNMC", "BMC", "I2CVB")
STAGES = ("source", "finetune_BMC", "finetune_I2CVB")


def mean(summary, field):
    return sum(summary["matrix"][STAGES[-1]][site][field] for site in SITES) / 3


def absolute(summary, historical):
    old = SITES[:-1]
    final = summary["matrix"][STAGES[-1]]
    arrival = {"RUNMC": summary["matrix"]["source"]["RUNMC"],
               "BMC": summary["matrix"]["finetune_BMC"]["BMC"]}
    drops = [arrival[site]["Dice_mean_percent"] - final[site]["Dice_mean_percent"] for site in old]
    worst = min(summary["final_forgetting"][site]["worst_Dice_delta_pp"] for site in old)
    gates = {
        "current_train_fit": all(summary["current_train_fit"][stage]["Dice_mean_percent"] >= 80
                                 for stage in STAGES),
        "old_absolute": all(final[site]["Dice_mean_percent"] >= 70 for site in old),
        "mean_old_forgetting": sum(drops) / 2 <= 10,
        "worst_old_patient": worst >= -15,
        "no_new_zero_overlap": all(n == 0 for n in summary["new_zero_overlap"].values()),
        "new_domain_vs_historical": final["I2CVB"]["Dice_mean_percent"] >=
                                    historical["matrix"]["finetune_I2CVB"]["I2CVB"]["Dice_mean_percent"] - 1,
    }
    return gates, {"mean_old_forgetting_pp": sum(drops) / 2, "worst_old_patient_delta_pp": worst}


def compare(u_root, b_root, u_summary, b_summary, old_u, old_b, output):
    u_root, b_root = Path(u_root), Path(b_root)
    u, b = (json.loads(Path(path).read_text()) for path in (u_summary, b_summary))
    hu, hb = (json.loads(Path(path).read_text()) for path in (old_u, old_b))
    if any(s["status"] != "COMPLETE" or s["optimizer_updates"] != 20000
           for s in (u, b)) or u["seed"] != b["seed"]:
        raise ValueError("paired GIN arms incomplete or mismatched")
    u_gates, u_detail = absolute(u, hu)
    b_gates, b_detail = absolute(b, hb)
    u_gain = mean(u, "Dice_mean_percent") - mean(hu, "Dice_mean_percent")
    b_gain_vs_u = mean(b, "Dice_mean_percent") - mean(u, "Dice_mean_percent")
    u_hd_ratio = mean(u, "HD95_penalized_mean_mm") / mean(hu, "HD95_penalized_mean_mm")
    b_hd_ratio = mean(b, "HD95_penalized_mean_mm") / mean(u, "HD95_penalized_mean_mm")
    u_gates.update({"mean_gain_vs_old_U": u_gain >= 2, "HD95_vs_old_U": u_hd_ratio <= 1.10})
    u_old = base.stage_rows(u_root, STAGES[-1])
    b_old = base.stage_rows(b_root, STAGES[-1])
    paired = {site: base.paired(b_old[site], u_old[site]) for site in SITES}
    b_gates.update({"mean_gain_vs_U_GIN": b_gain_vs_u >= 2,
                    "new_domain_vs_U_GIN": b["matrix"][STAGES[-1]]["I2CVB"]["Dice_mean_percent"] >=
                                           u["matrix"][STAGES[-1]]["I2CVB"]["Dice_mean_percent"] - 1,
                    "old_patient_vs_U_GIN": min(paired[site]["worst_Dice_delta_pp"]
                                                for site in SITES[:-1]) >= -5,
                    "HD95_vs_U_GIN": b_hd_ratio <= 1.10})
    result = {"status": "COMPLETE", "seed": u["seed"], "test_access": 0,
              "historical_controls": True, "development_val_only": True,
              "U_GIN_vs_U": {"gates": u_gates, "all_pass": all(u_gates.values()),
                             "mean_gain_pp": u_gain, "HD95_ratio": u_hd_ratio, **u_detail},
              "B_GIN_vs_U_GIN": {"gates": b_gates, "all_pass": all(b_gates.values()),
                                 "mean_gain_pp": b_gain_vs_u, "HD95_ratio": b_hd_ratio,
                                 "paired_by_site": paired, **b_detail}}
    Path(output).write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result
