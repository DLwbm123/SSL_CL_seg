"""Bounded optimizer accounting and exact prefix state capture."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import torch


class BudgetLedger:
    def __init__(self, path: Path, cap: int = 64) -> None:
        self.path = path
        self.cap = cap
        path.parent.mkdir(parents=True, exist_ok=True)
        self.attempts = 0
        if path.exists():
            with path.open() as stream:
                self.attempts = sum(1 for line in stream if json.loads(line)["event"] == "attempt")

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        if self.attempts >= self.cap:
            raise RuntimeError("CPU synthetic optimizer cap reached")
        self.attempts += 1
        self._append(dict(event="attempt", number=self.attempts))
        try:
            optimizer.step()
        except BaseException as error:
            self._append(dict(event="failed", number=self.attempts, error=type(error).__name__))
            raise
        self._append(dict(event="successful", number=self.attempts))

    def _append(self, value: dict) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def make_reference(student: torch.nn.Module) -> torch.nn.Module:
    reference = copy.deepcopy(student).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    return reference


def capture(student: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler,
            cursor: dict, *, reference: torch.nn.Module | None = None,
            bank: torch.nn.Module | None = None, include_cuda: bool = False) -> dict:
    return copy.deepcopy(dict(student=student.state_dict(), optimizer=optimizer.state_dict(),
                              scheduler=None if scheduler is None else scheduler.state_dict(),
                              cursor=cursor, cpu_rng=torch.get_rng_state(),
                              cuda_rng=torch.cuda.get_rng_state_all() if include_cuda else None,
                              reference=None if reference is None else reference.state_dict(),
                              bank=None if bank is None else bank.state_dict()))


def restore(state: dict, student: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler,
            *, reference: torch.nn.Module | None = None, bank: torch.nn.Module | None = None) -> dict:
    student.load_state_dict(state["student"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    if (scheduler is None) != (state["scheduler"] is None):
        raise ValueError("scheduler mismatch")
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    for key, module in (("reference", reference), ("bank", bank)):
        if (module is None) != (state[key] is None):
            raise ValueError(key + " mismatch")
        if module is not None:
            module.load_state_dict(state[key], strict=True)
    torch.set_rng_state(state["cpu_rng"])
    if state["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return copy.deepcopy(state["cursor"])


@torch.no_grad()
def ema_update(student: torch.nn.Module, reference: torch.nn.Module, momentum: float = 0.99) -> None:
    if any(id(x) == id(y) for x, y in zip(student.parameters(), reference.parameters())):
        raise ValueError("EMA reference aliases student")
    for source, target in zip(student.parameters(), reference.parameters()):
        target.mul_(momentum).add_(source, alpha=1 - momentum)
    for source, target in zip(student.buffers(), reference.buffers()):
        target.copy_(source)


def guard_storage(project_root: Path) -> dict[str, Path]:
    root = project_root.resolve(strict=True)
    if not project_root.is_absolute() or str(root).startswith("/data_nas/"):
        raise ValueError("R0 storage requires confirmed absolute new remote-home")
    names = {"HF_HOME": "cache/huggingface", "TORCH_HOME": "cache/torch",
             "PIP_CACHE_DIR": "cache/pip", "TMPDIR": "tmp"}
    bound = {}
    for key, relative in names.items():
        target = (root / relative).resolve()
        if not target.is_relative_to(root) or os.environ.get(key) != str(target):
            raise ValueError(key + " must be bound inside project root")
        bound[key] = target
    return bound
