"""Frozen zero-update gradient audit of the completed supervised CL models."""
import json
import math
import os
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from sc_bayeseg.supervised_cl_v1.run import (SEED, augment, load_cases, loss_fn,
                                              model, sample_batch, setup, write)


N_BATCHES = 64
SEED_OFFSET = 271828
MAX_BYTES = 49_000_000_000


def size(root):
    return int(subprocess.check_output(["du", "-sb", str(root)], text=True).split()[0])


def check(root, deadline):
    if time.time() >= deadline:
        raise TimeoutError("Q1 safety deadline reached")
    if size(root) >= MAX_BYTES:
        raise RuntimeError("Q1 storage pre-stop threshold reached")


def gradient(value, params, retain_graph=False):
    values = torch.autograd.grad(value, params, retain_graph=retain_graph, allow_unused=True)
    return tuple(torch.zeros_like(p) if g is None else g.detach() for p, g in zip(params, values))


def dot(a, b):
    return float(sum((x.double() * y.double()).sum() for x, y in zip(a, b)))


def norm(a):
    return math.sqrt(dot(a, a))


def add(a, b, wa=1., wb=1.):
    return tuple(wa * x + wb * y for x, y in zip(a, b))


def cosine(a, b):
    na, nb = norm(a), norm(b)
    if min(na, nb) <= 0 or not math.isfinite(na * nb):
        raise FloatingPointError("zero or non-finite gradient norm")
    result = dot(a, b) / (na * nb)
    if not math.isfinite(result):
        raise FloatingPointError("non-finite gradient cosine")
    return max(-1., min(1., result))


def task_gradient(net, x, y, family, params):
    if family == "U":
        return gradient(loss_fn(net(x), y), params)
    logits, variational = net(x)
    return gradient(loss_fn(logits, y), params, retain_graph=True), gradient(variational, params)


def load_net(root, family):
    net = model(family)
    checkpoint = torch.load(root / "replay_HK" / "endpoint.pt", map_location="cuda", weights_only=False)
    if checkpoint["step"] != 6000:
        raise ValueError("not the frozen replay_HK endpoint")
    net.load_state_dict(checkpoint["model"])
    net.train()
    return net


def aggregate(values):
    array = np.asarray(values, dtype=float)
    if len(array) != N_BATCHES or not np.isfinite(array).all():
        raise ValueError("incomplete or non-finite Q1 batch readout")
    return {"mean": float(array.mean()), "median": float(np.median(array)),
            "p10": float(np.percentile(array, 10)), "p90": float(np.percentile(array, 90))}


def run(settings):
    os.umask(0o077)
    root = Path(settings["output"])
    if not root.is_dir() or (root / "STARTED.json").exists():
        raise FileExistsError("Q1 root must be new and unused")
    deadline = float(settings["started_unix"]) + 23 * 3600
    setup()
    torch.cuda.set_device(int(settings["gpu_index"]))
    write(root / "STARTED.json", {"started_unix": settings["started_unix"],
                                  "updates": 0, "batches": N_BATCHES, "test_access": 0})
    try:
        check(root, deadline)
        b_root, u_root = Path(settings["b_root"]), Path(settings["u_root"])
        for old in (b_root, u_root):
            if json.loads((old / "STATUS.json").read_text())["status"] != "COMPLETE":
                raise ValueError("paired source experiment incomplete")
        b_net, u_net = load_net(b_root, "B"), load_net(u_root, "U")
        replay = torch.load(b_root / "PRIVATE_REPLAY.pt", map_location="cpu", weights_only=False)
        if len(replay) != 32 or set(item["site"] for item in replay) != {"RUNMC", "BMC", "I2CVB", "UCL", "BIDMC"}:
            raise ValueError("frozen final replay buffer mismatch")
        if any(len({item["patient"] for item in replay if item["site"] == site}) != 4
               for site in ("RUNMC", "BMC", "I2CVB", "UCL", "BIDMC")):
            raise ValueError("frozen replay patient coverage mismatch")
        catalog = json.loads((b_root / "PRIVATE_CATALOG.json").read_text())
        current = load_cases(settings["data_root"], catalog, ("HK",))
        if len(current["HK"]) != 7:
            raise ValueError("HK train scope changed")
        b_params = tuple(b_net.backbone.parameters())
        u_params = tuple(u_net.parameters())
        rows = []
        for step in range(1, N_BATCHES + 1):
            if step % 8 == 1:
                check(root, deadline)
            x, y = sample_batch(current, replay, step, SEED + SEED_OFFSET)
            x, y = augment(x, y, step, SEED + SEED_OFFSET)
            torch.manual_seed(SEED + SEED_OFFSET + step)
            b_new, b_new_aux = task_gradient(b_net, x[:2], y[:2], "B", b_params)
            torch.manual_seed(SEED + SEED_OFFSET + step + 100000)
            b_old, b_old_aux = task_gradient(b_net, x[2:], y[2:], "B", b_params)
            u_new = task_gradient(u_net, x[:2], y[:2], "U", u_params)
            u_old = task_gradient(u_net, x[2:], y[2:], "U", u_params)
            b_task = add(b_new, b_old, .5, .5)
            b_aux = add(b_new_aux, b_old_aux, .005, .005)
            row = {"batch": step, "b_task_new_old_cos": cosine(b_old, b_new),
                   "u_task_new_old_cos": cosine(u_old, u_new),
                   "b_old_aux_cos": cosine(b_old, b_aux),
                   "b_aux_to_old_norm": norm(b_aux) / norm(b_old),
                   "b_old_alignment_delta": cosine(b_old, add(b_task, b_aux)) - cosine(b_old, b_task)}
            if not all(math.isfinite(v) for v in row.values()):
                raise FloatingPointError("non-finite Q1 batch readout")
            rows.append(row)
            if step == 1:
                write(root / "P0.json", {"status": "PASS", "batches_planned": N_BATCHES,
                                         "replay_slices": 32, "current_train_patients": 7,
                                         "checkpoint": "replay_HK endpoint", "finite_full_gradient": True,
                                         "optimizer_updates": 0})
        (root / "PRIVATE_BATCH_ROWS.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
        fields = [key for key in rows[0] if key != "batch"]
        summary = {key: aggregate([row[key] for row in rows]) for key in fields}
        conflict_fraction = sum(row["b_old_aux_cos"] < 0 for row in rows) / N_BATCHES
        candidate = (conflict_fraction >= .5 and summary["b_aux_to_old_norm"]["median"] >= .1
                     and summary["b_old_alignment_delta"]["median"] <= -.05)
        public = {"status": "COMPLETE", "batches": N_BATCHES, "optimizer_updates": 0,
                  "scope": "final_old_replay_buffer_plus_HK_train_only", "test_access": 0,
                  "metrics": summary, "aux_conflict_fraction": conflict_fraction,
                  "candidate_interference": candidate}
        (root / "PUBLIC_SUMMARY.json").write_text(json.dumps(public, indent=2, allow_nan=False) + "\n")
        lines = ["# Q1：旧域重放与生成项梯度诊断", "",
                 "只读取已完成的 U/B 最终重放检查点、32 张旧域训练缓存和 HK train 标签；零优化更新，无 val/test 读取。",
                 f"固定批数 {N_BATCHES}；B 的旧域任务与生成项负余弦比例 {conflict_fraction:.3f}；",
                 f"生成项／旧域任务梯度范数比中位数 {summary['b_aux_to_old_norm']['median']:.3f}；",
                 f"加入生成项后的旧域任务对齐余弦变化中位数 {summary['b_old_alignment_delta']['median']:+.3f}。",
                 f"冻结候选干扰门：{'通过' if candidate else '未通过'}。",
                 "该结果只反映固定检查点的局部原始梯度，不等同于 AdamW 实际更新或患者级因果改善。", ""]
        (root / "REPORT_ZH.md").write_text("\n".join(lines))
        write(root / "STATUS.json", {"status": "COMPLETE", "updates": 0,
                                     "batches": N_BATCHES, "elapsed_seconds": time.time() - settings["started_unix"],
                                     "bytes": size(root), "candidate_interference": candidate})
        write(root / "EXIT.json", {"exit_code": 0})
    except BaseException:
        write(root / "FAILURE.json", {"traceback": traceback.format_exc()})
        write(root / "EXIT.json", {"exit_code": 1})
        raise


if __name__ == "__main__":
    run(json.loads(Path(os.environ["TASK_SETTINGS"]).read_text()))
