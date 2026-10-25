"""Publish aggregate evidence; keep patient-level metrics and checkpoints private."""
import json
import sys
from pathlib import Path

import numpy as np
import torch


STAGES = ("source", "finetune_BMC", "replay_BMC", "finetune_I2CVB", "replay_I2CVB", "joint")
SITES = ("RUNMC", "BMC", "I2CVB")


def stage_rows(root, stage):
    private = torch.load(root / stage / "PRIVATE_VAL.pt", map_location="cpu", weights_only=False)
    return {site: {r["patient"]: r["metrics"]["classes"][0] for r in rows}
            for site, rows in private.items()}


def cell(rows):
    values = list(rows.values())
    dice = np.asarray([v["Dice"] for v in values], dtype=float)
    hd = np.asarray([v["HD95_penalized_mm"] for v in values], dtype=float)
    if len(dice) == 0 or not np.isfinite(dice).all() or not np.isfinite(hd).all():
        raise ValueError("invalid patient metric")
    return {"patients": len(values), "Dice_mean_percent": float(dice.mean() * 100),
            "Dice_p10_percent": float(np.percentile(dice, 10) * 100),
            "Dice_min_percent": float(dice.min() * 100), "HD95_penalized_mean_mm": float(hd.mean()),
            "zero_overlap_patients": int(sum(v["TP"] == 0 and v["GT"] > 0 for v in values)),
            "empty_prediction_patients": int(sum(v["empty_prediction_failure"] for v in values))}


def paired(after, before):
    if set(after) != set(before):
        raise ValueError("patient identities differ in paired comparison")
    diffs = np.asarray([(after[k]["Dice"] - before[k]["Dice"]) * 100 for k in sorted(after)])
    return {"patients": len(diffs), "mean_Dice_delta_pp": float(diffs.mean()),
            "worst_Dice_delta_pp": float(diffs.min()), "declined_patients": int((diffs < 0).sum())}


def summarize(root, public):
    root, public = Path(root), Path(public)
    status = json.loads((root / "STATUS.json").read_text())
    if status["status"] != "COMPLETE" or json.loads((root / "EXIT.json").read_text())["exit_code"] != 0:
        raise ValueError("experiment did not complete")
    catalog = json.loads((root / "PRIVATE_CATALOG.json").read_text())
    rows = {stage: stage_rows(root, stage) for stage in STAGES}
    if set(rows["source"]) != {"RUNMC"} or set(rows["joint"]) != set(SITES):
        raise ValueError("stage-domain matrix incomplete")
    matrix = {stage: {site: cell(data) for site, data in rows[stage].items()} for stage in STAGES}
    forgetting = {}
    for arm in ("finetune", "replay"):
        forgetting[arm] = {
            "RUNMC_after_BMC": paired(rows[f"{arm}_BMC"]["RUNMC"], rows["source"]["RUNMC"]),
            "RUNMC_after_I2CVB": paired(rows[f"{arm}_I2CVB"]["RUNMC"], rows["source"]["RUNMC"]),
            "BMC_after_I2CVB": paired(rows[f"{arm}_I2CVB"]["BMC"], rows[f"{arm}_BMC"]["BMC"])}
    replay_vs_finetune = {site: paired(rows["replay_I2CVB"][site], rows["finetune_I2CVB"][site])
                          for site in SITES}
    training = {stage: json.loads((root / stage / "STATUS.json").read_text()) for stage in STAGES}
    result = {"status": "COMPLETE", "scope": catalog["scope"], "sites": list(SITES),
              "training_patients": {site: catalog["counts"][site]["train"] for site in SITES},
              "validation_patients": {site: catalog["counts"][site]["val"] for site in SITES},
              "test_access": 0, "one_seed_exploratory": True, "matrix": matrix,
              "forgetting_vs_arrival": forgetting, "replay_vs_finetune_final": replay_vs_finetune,
              "training": training, "elapsed_seconds": status["elapsed_seconds"],
              "new_bytes": status["bytes"], "optimizer_updates": status["updates"]}
    public.mkdir(exist_ok=True)
    (public / "ANALYSIS.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    lines = ["# 全监督前列腺域持续学习首批结果", "", f"训练标签范围：`{catalog['scope']}`；固定 seed 单次探索性运行。",
             "评测仅使用冻结 val，test 未访问；患者级原始结果、模型和重放图像仅存 NAS。", "",
             f"训练病例数：RUNMC {catalog['counts']['RUNMC']['train']}、BMC {catalog['counts']['BMC']['train']}、I2CVB {catalog['counts']['I2CVB']['train']}。",
             "", "| 训练阶段 | 评价中心 | 患者 | Dice 均值 % | Dice p10 % | 惩罚 HD95 均值 mm | 零重叠 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for stage in STAGES:
        for site in SITES:
            if site not in matrix[stage]:
                continue
            c = matrix[stage][site]
            lines.append(f"| {stage} | {site} | {c['patients']} | {c['Dice_mean_percent']:.3f} | {c['Dice_p10_percent']:.3f} | {c['HD95_penalized_mean_mm']:.3f} | {c['zero_overlap_patients']} |")
    lines += ["", "## 旧域遗忘（相对该模型学习该域后的首次评价）", "",
              "| 路线 | 对比 | 平均 Dice 变化 pp | 最差患者变化 pp | 下降患者/总数 |",
              "| --- | --- | ---: | ---: | ---: |"]
    for arm, comparisons in forgetting.items():
        for name, c in comparisons.items():
            lines.append(f"| {arm} | {name} | {c['mean_Dice_delta_pp']:+.3f} | {c['worst_Dice_delta_pp']:+.3f} | {c['declined_patients']}/{c['patients']} |")
    lines += ["", "## 最终阶段：固定容量重放相对顺序微调", "",
              "| 中心 | 平均 Dice 变化 pp | 最差患者变化 pp | 下降患者/总数 |",
              "| --- | ---: | ---: | ---: |"]
    for site, c in replay_vs_finetune.items():
        lines.append(f"| {site} | {c['mean_Dice_delta_pp']:+.3f} | {c['worst_Dice_delta_pp']:+.3f} | {c['declined_patients']}/{c['patients']} |")
    lines += ["", "联合训练使用全部三域训练标签，是离线参照，不满足逐域到来的访问约束。",
              "本批只有一个训练种子，且 val 在既往项目中反复用于开发；结果只能用于判断后续方法问题是否存在，不能构成独立确认。",
              f"本轮共 {status['updates']} 次优化更新，耗时 {status['elapsed_seconds']:.1f} 秒，新增目录 {status['bytes']} 字节。", ""]
    (public / "REPORT_ZH.md").write_text("\n".join(lines))
    return result


if __name__ == "__main__":
    summarize(sys.argv[1], sys.argv[2])
