"""At most two R0 CPU suite attempts; persist attempts before the suite starts."""
import json
import os
import sys
import unittest
from pathlib import Path

root = Path(__file__).resolve().parent
ledger_dir = Path(os.environ.get("QPROMPT_PRIVATE_RECEIPTS", root / "private_receipts"))
ledger_dir.mkdir(parents=True, exist_ok=True)
attempts = ledger_dir / "CPU_SUITE_ATTEMPTS.jsonl"
past = sum(json.loads(line)["event"] == "started" for line in attempts.open()) if attempts.exists() else 0
if past >= 2:
    raise SystemExit("R0 CPU suite attempt cap reached")
with attempts.open("a") as stream:
    stream.write(json.dumps({"attempt": past + 1, "event": "started"}) + "\n")
    stream.flush()
    os.fsync(stream.fileno())
os.environ["QPROMPT_CPU_LEDGER"] = str(ledger_dir / "CPU_OPTIMIZER_PHYSICAL.jsonl")
suite = unittest.defaultTestLoader.discover(str(root / "tests"), pattern="test_r0.py")
result = unittest.TextTestRunner(verbosity=2).run(suite)
with attempts.open("a") as stream:
    stream.write(json.dumps({"attempt": past + 1, "event": "completed", "successful": result.wasSuccessful()}) + "\n")
sys.exit(0 if result.wasSuccessful() else 1)
