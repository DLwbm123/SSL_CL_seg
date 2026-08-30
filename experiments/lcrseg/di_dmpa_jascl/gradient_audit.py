"""Audit scheduling only; it never changes forwards, losses, GAS or optimizer steps."""
from dataclasses import dataclass


@dataclass(frozen=True)
class GradientAuditPolicy:
    mode: str = "every_batch"
    interval: int = 0
    fixed_batch_ids: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, mapping=None):
        data = dict(mapping or {"mode": "every_batch", "interval": 0, "fixed_batch_ids": []})
        if set(data) != {"mode", "interval", "fixed_batch_ids"}:
            raise ValueError("gradient_audit requires mode, interval and fixed_batch_ids")
        if data["mode"] not in {"every_batch", "disabled", "fixed_batches"}:
            raise ValueError("unknown gradient audit mode")
        if not isinstance(data["interval"], int) or data["interval"] < 0:
            raise ValueError("gradient audit interval must be a nonnegative integer")
        ids = data["fixed_batch_ids"]
        if not isinstance(ids, (list, tuple)) or any(not isinstance(v, str) or not v for v in ids):
            raise ValueError("fixed batch IDs must be nonempty strings")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate fixed batch IDs")
        if data["mode"] != "fixed_batches" and (data["interval"] or ids):
            raise ValueError("interval/IDs are only supported in fixed_batches mode")
        return cls(data["mode"], data["interval"], tuple(ids))

    @staticmethod
    def batch_id(domain, epoch, batch_index):
        return f"{domain}/epoch{epoch}/unlabeled_batch{batch_index}"

    def should_audit(self, *, domain, epoch, batch_index, global_step):
        if self.mode == "every_batch":
            return True
        if self.mode == "disabled":
            return False
        return (self.batch_id(domain, epoch, batch_index) in self.fixed_batch_ids or
                (self.interval > 0 and global_step % self.interval == 0))
