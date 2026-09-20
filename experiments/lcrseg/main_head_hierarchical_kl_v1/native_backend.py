"""Native HKL backend over the existing B2/native model and data primitives."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F

from ..five_frameworks_v1.native_data import NativeCurrentDomain
from ..five_frameworks_v1.native_parent import NativeLRParent, build
from ..five_frameworks_v1.model import Model
from ..five_frameworks_v1.integration import ExecutionPermit
from .core import hierarchical_kl
from .execution import NativeBackend, NativeHooks


class HKLNativeBackend(NativeBackend):
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["device"])
        permit = config["permit"]
        if not isinstance(permit, ExecutionPermit):
            raise TypeError("HKL native backend requires an ExecutionPermit")
        permit.validate()
        self.handles = {}
        hooks = NativeHooks(self._load_state, self._load_batches, self._forward,
                            self._diagnose, self._write_report)
        super().__init__(hooks)

    def _load_state(self, state):
        sid = state["state_id"]
        entry = self.config["states"][sid]
        payload = torch.load(Path(entry["latest_path"]), map_location=self.device, weights_only=False)
        seed = int(self.config["seed"]); order = int(entry["order"])
        # Both bound states are stage-2 current-domain states.  START names the
        # stage-2 prefix, while the payload itself comes from the corresponding
        # stage-1 checkpoint; it must not change the current-domain index.
        source = {"node_id": entry["node_id"], "domain": entry["domain"], "seed": seed, "order": order,
                  "stage": 2}
        native = build(self.config["reference"], self.device, seed)
        student_state = payload.get("student", payload)
        parent = NativeLRParent(native, seed, source, adapt=False)
        model = Model(parent, "B2_PARENT_PAS_KL").to(self.device)
        model.load_state_dict(student_state, strict=False)
        teacher = model.teacher()
        if "ema" in payload:
            teacher.load_state_dict(payload["ema"], strict=False)
        handle = {"state_id": sid, "student": model.eval(), "teacher": teacher.eval(),
                  "domain": entry["domain"], "order": order, "stage": source["stage"],
                  "prototypes": payload.get("prototypes")}
        self.handles[sid] = handle
        return handle

    def _load_batches(self, handle, state, binding):
        sid = state["state_id"]; entry = self.config["states"][sid]
        provider = NativeCurrentDomain(self.config["data"], int(self.config["seed"]),
                                       int(entry["order"]), int(handle["stage"]),
                                       {"node_id": entry["node_id"], "domain": entry["domain"]},
                                       self.device, self.config["permit"], allow_u=True)
        l_count = int(entry["L_batch_count"]); u_count = int(entry["U_batch_count"])
        batches = []
        for i in range(l_count):
            x, y, ids = provider.labeled(i, "HKL_P0_L")
            batches.append({"L": {"x": x, "y": y, "ids": ids, "batch_id": f"L:{i}"}})
        for i in range(u_count):
            x, geometry, ids = provider.unlabeled(i)
            batches.append({"U": {"x": x, "geometry": geometry, "ids": ids, "batch_id": f"U:{i}"}})
        return batches

    @staticmethod
    def _one(role, handle, batch):
        item = batch
        with torch.no_grad():
            logits = handle["student" if role == "student_full_forwards" else "teacher"](item["x"], mode="eval")
        return {"role": role, "batch_id": item["batch_id"], "logits": logits.detach(),
                "y": item.get("y"), "kind": "L" if "y" in item else "U"}

    def _forward(self, role, handle, state, batch):
        source = "L" if role.endswith("student_full_forwards") else "L"
        # Each scheduled physical call is tied to one explicit batch; U calls use the same contract.
        if source not in batch:
            source = "U"
        return self._one(role, handle, batch[source])

    def _diagnose(self, state, handle, batches, outputs):
        by_batch = {}
        for out in outputs:
            if out["kind"] == "L": by_batch.setdefault(out["batch_id"], {})[out["role"]] = out
        errors = captured = legal = support = 0
        random_expected = three_capture = 0.0
        strata = []
        for pair in by_batch.values():
            if set(pair) != {"student_full_forwards", "teacher_full_forwards"}: continue
            y = pair["student_full_forwards"]["y"]; student = pair["student_full_forwards"]["logits"]
            teacher = pair["teacher_full_forwards"]["logits"].softmax(1)
            valid = y != 255; legal += int(valid.sum())
            pred = student.argmax(1); err = valid & (pred != y)
            disc = teacher[:, 1:3].sum(1); eligible = valid & (disc >= .9)
            support += int(eligible.flatten(1).any(1).sum())
            cond = teacher[:, 1:3] / disc[:, None].clamp_min(1e-12)
            cond_entropy = -(cond * cond.clamp_min(1e-12).log()).sum(1)
            full_entropy = -(teacher * teacher.clamp_min(1e-12).log()).sum(1)
            for b in range(y.shape[0]):
                vb = valid[b]
                # A frozen two-stratum boundary convention: a pixel is boundary
                # when its 3x3 valid neighborhood contains another GT class.
                boundary = torch.zeros_like(vb)
                for cls in (0, 1, 2):
                    cm = (y[b] == cls) & vb
                    local_all = -F.max_pool2d((-cm.float())[None, None], 3, 1, 1)[0, 0]
                    boundary |= cm & (local_all < 0.5)
                for cls in (0, 1, 2):
                    for name, sm in (("boundary", boundary), ("interior", ~boundary)):
                        mask = eligible[b] & (y[b] == cls) & sm
                        n = int(mask.sum())
                        if not n:
                            continue
                        k = max(1, (n + 3) // 4)
                        flat = mask.flatten().nonzero(as_tuple=False).flatten()
                        cscore = cond_entropy[b].flatten()[flat]
                        fscore = full_entropy[b].flatten()[flat]
                        corder = torch.argsort(cscore, descending=True, stable=True)[:k]
                        forder = torch.argsort(fscore, descending=True, stable=True)[:k]
                        eflat = err[b].flatten()[flat]
                        e = int(eflat.sum()); ce = int(eflat[corder].sum()); fe = int(eflat[forder].sum())
                        errors += e; captured += ce; three_capture += fe
                        random_expected += k * e / n
                        strata.append(dict(batch_id=pair["student_full_forwards"]["batch_id"], image=int(b),
                                           gt_class=cls, stratum=name, n=n, k=k, errors=e,
                                           captured_errors=ce, three_class_entropy_captured=fe,
                                           random_expected=k * e / n))
        gate = "DESCRIPTIVE"
        if state["state_id"].endswith("ENDPOINT"):
            gate = "PASS" if (support >= 5 and errors >= 20 and
                              captured >= 1.25 * random_expected and
                              captured >= .95 * three_capture) else "INSUFFICIENT_EVIDENCE"
        return {"state_id": state["state_id"], "gate": gate, "supported_images": support,
                "conditional_errors": errors, "captured_errors": captured,
                "random_expected": random_expected,
                "three_class_entropy_captured": three_capture,
                "strata": strata, "legal_pixels": legal}

    def _write_report(self, result, path):
        Path(path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
