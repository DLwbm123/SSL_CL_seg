"""Publish aggregate evidence; keep patient-level metrics and checkpoints private."""
import json
import sys
from pathlib import Path

import numpy as np
import torch


SITES = ("RUNMC", "BMC", "I2CVB", "UCL", "BIDMC", "HK")
STAGES = ("source", *(name for site in SITES[1:] for name in (f"finetune_{site}", f"replay_{site}")), "joint")


def stage_rows(root, stage, role="VAL"):
    private = torch.load(root / stage / f"PRIVATE_{role}.pt", map_location="cpu", weights_only=False)
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
    train_rows = {stage: stage_rows(root, stage, "TRAIN") for stage in STAGES}
    if set(rows["source"]) != {"RUNMC"} or set(rows["joint"]) != set(SITES):
        raise ValueError("stage-domain matrix incomplete")
    for index, site in enumerate(SITES[1:], start=1):
        for arm in ("finetune", "replay"):
            if set(rows[f"{arm}_{site}"]) != set(SITES[:index + 1]):
                raise ValueError("stage-domain matrix incomplete")
    matrix = {stage: {site: cell(data) for site, data in rows[stage].items()} for stage in STAGES}
    train_fit = {stage: {site: cell(data) for site, data in train_rows[stage].items()} for stage in STAGES}
    forgetting = {}
    for arm in ("finetune", "replay"):
        forgetting[arm] = {}
        for index, current in enumerate(SITES[1:], start=1):
            for old in SITES[:index]:
                arrival = "source" if old == SITES[0] else f"{arm}_{old}"
                forgetting[arm][f"{old}_after_{current}"] = paired(rows[f"{arm}_{current}"][old], rows[arrival][old])
    final_site = SITES[-1]
    replay_vs_finetune = {site: paired(rows[f"replay_{final_site}"][site], rows[f"finetune_{final_site}"][site])
                          for site in SITES}
    training = {stage: json.loads((root / stage / "STATUS.json").read_text()) for stage in STAGES}
    prediction_cost = {stage: json.loads((root / stage / "PRED_COST_VAL.json").read_text()) for stage in STAGES}
    result = {"status": "COMPLETE", "scope": catalog["scope"], "family": status["family"], "sites": list(SITES),
              "training_patients": {site: catalog["counts"][site]["train"] for site in SITES},
              "validation_patients": {site: catalog["counts"][site]["val"] for site in SITES},
              "test_access": 0, "one_seed_exploratory": True, "matrix": matrix,
              "train_fit": train_fit,
              "forgetting_vs_arrival": forgetting, "replay_vs_finetune_final": replay_vs_finetune,
              "training": training, "validation_prediction_cost": prediction_cost,
              "elapsed_seconds": status["elapsed_seconds"],
              "new_bytes": status["bytes"], "optimizer_updates": status["updates"]}
    public.mkdir(exist_ok=True)
    (public / "ANALYSIS.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    lines = ["# 全监督前列腺域持续学习首批结果", "", f"模型：`{status['family']}`；训练标签范围：`{catalog['scope']}`；固定 seed 单次探索性运行。",
             "评测仅使用冻结 val，test 未访问；患者级原始结果、模型和重放图像仅存 NAS。", "",
             "训练病例数：" + "、".join(f"{site} {catalog['counts'][site]['train']}" for site in SITES) + "。",
             "", "| 训练阶段 | 评价中心 | 患者 | Dice 均值 % | Dice p10 % | 惩罚 HD95 均值 mm | 零重叠 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for stage in STAGES:
        for site in SITES:
            if site not in matrix[stage]:
                continue
            c = matrix[stage][site]
            lines.append(f"| {stage} | {site} | {c['patients']} | {c['Dice_mean_percent']:.3f} | {c['Dice_p10_percent']:.3f} | {c['HD95_penalized_mean_mm']:.3f} | {c['zero_overlap_patients']} |")
    lines += ["", "## 当前训练域拟合", "", "| 阶段 | 训练中心 | 患者 | 训练 Dice 均值 % |", "| --- | --- | ---: | ---: |"]
    for stage in STAGES:
        for site, c in train_fit[stage].items():
            lines.append(f"| {stage} | {site} | {c['patients']} | {c['Dice_mean_percent']:.3f} |")
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
