"""Separately authorized R0_REVIEW_FIX budget: at most two CPU math runs."""
import hashlib
import json
import os
from pathlib import Path
import unittest

root = Path(__file__).resolve().parent
receipts = Path(os.environ['QPROMPT_REVIEW_RECEIPTS'])
receipts.mkdir(parents=True, exist_ok=True)
ledger = receipts / 'REVIEW_FIX_ATTEMPTS.jsonl'
past = sum(json.loads(line)['event'] == 'started' for line in ledger.read_text().splitlines()) if ledger.exists() else 0
if past >= 2:
    raise SystemExit('R0_REVIEW_FIX attempt cap reached')


def record(value):
    with ledger.open('a') as stream:
        stream.write(json.dumps(value, sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


record(dict(event='started', attempt=past + 1, optimizer_calls=0, real_data_reads=0, cuda_calls=0,
            sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                    for name in ('qprompt/models.py', 'qprompt/losses.py', 'tests/test_review_fix.py')}))
suite = unittest.defaultTestLoader.discover(str(root / 'tests'), pattern='test_review_fix.py')
result = unittest.TextTestRunner(verbosity=2).run(suite)
record(dict(event='completed', attempt=past + 1, successful=result.wasSuccessful(), tests=result.testsRun))
raise SystemExit(0 if result.wasSuccessful() else 1)
