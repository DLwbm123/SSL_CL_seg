"""Anonymous, patient-equivalent aggregate for the three-site GIN screen."""

import json
import sys
from pathlib import Path

import numpy as np

from sc_bayeseg.supervised_cl_v1 import summarize as base


SITES = ("RUNMC", "BMC", "I2CVB")
STAGES = ("source", "finetune_BMC", "finetune_I2CVB")


def cell(rows):
    result = base.cell(rows)
    values = list(rows.values())
    result["Recall_mean_percent"] = float(np.mean([v["TP"] / v["GT"] for v in values]) * 100)
    result["Precision_mean_percent"] = float(np.mean([
        v["TP"] / v["predicted"] if v["predicted"] else 0 for v in values]) * 100)
    return result


def summarize(root, output):
    root, output = Path(root), Path(output)
    status = json.loads((root / "STATUS.json").read_text())
    if status["status"] != "COMPLETE" or json.loads((root / "EXIT.json").read_text())["exit_code"]:
        raise ValueError("GIN arm did not complete")
    rows = {stage: base.stage_rows(root, stage) for stage in STAGES}
    fits = {stage: base.stage_rows(root, stage, "TRAIN") for stage in STAGES}
    for index, stage in enumerate(STAGES):
        if set(rows[stage]) != set(SITES[:index + 1]):
            raise ValueError("incomplete frozen evaluation matrix")
    matrix = {stage: {site: cell(data) for site, data in rows[stage].items()}
              for stage in STAGES}
    train_fit = {stage: cell(fits[stage][SITES[index]])
                 for index, stage in enumerate(STAGES)}
    final = rows[STAGES[-1]]
    arrival = {"RUNMC": rows["source"]["RUNMC"],
               "BMC": rows["finetune_BMC"]["BMC"]}
    forgetting = {site: base.paired(final[site], arrival[site]) for site in SITES[:-1]}
    zero_new = {}
    for index, stage in enumerate(STAGES[1:], start=1):
        for old in SITES[:index]:
            old_rows = arrival[old]
            new_rows = rows[stage][old]
            if set(new_rows) != set(old_rows):
                raise ValueError("patient identity mismatch")
            original_zero = {key for key, value in old_rows.items() if value["TP"] == 0 and value["GT"] > 0}
            current_zero = {key for key, value in new_rows.items() if value["TP"] == 0 and value["GT"] > 0}
            zero_new[f"{old}_after_{SITES[index]}"] = len(current_zero - original_zero)
    result = {"status": "COMPLETE", "family": status["family"], "seed": 20261025,
              "sites": list(SITES), "test_access": 0, "old_train_replay": 0,
              "matrix": matrix, "current_train_fit": train_fit,
              "final_forgetting": forgetting, "new_zero_overlap": zero_new,
              "elapsed_seconds": status["elapsed_seconds"], "new_bytes": status["bytes"],
              "optimizer_updates": status["updates"], "development_val_only": True}
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    summarize(sys.argv[1], sys.argv[2])
