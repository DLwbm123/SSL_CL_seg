from collections import Counter
from contextlib import contextmanager
from pathlib import Path
import json
import os
import resource
import signal
import time

ACTIVE_COUNTER = None


class BudgetExceeded(RuntimeError):
    pass


class ResourceLedger:
    """Durable reservation before work; interrupted attempts retain reserved cost."""
    def __init__(self, root, max_seconds=3600., max_updates=1000, max_bytes=2 * 1024 ** 3):
        self.root = Path(root)
        self.path = self.root / 'RESOURCE_LEDGER.json'
        self.limits = dict(seconds=max_seconds, updates=max_updates, bytes=max_bytes)
        self.root.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
            if self.data['limits'] != self.limits:
                raise ValueError('cannot reset/change an existing budget')
        else:
            self.data = dict(schema=1, limits=self.limits, seconds_charged=0., updates_reserved=0,
                             attempts=[], counters={}, created=time.time())
            self.flush()

    def flush(self):
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.data, indent=2))
        os.replace(temporary, self.path)

    def size(self):
        return sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())

    def check(self):
        if time.time() >= self.data['created'] + 24 * 3600:
            raise BudgetExceeded('original 24-hour wall-clock budget expired')
        if self.data['seconds_charged'] > self.limits['seconds'] or self.data['updates_reserved'] > self.limits['updates'] or self.size() >= self.limits['bytes']:
            raise BudgetExceeded('cumulative resource cap')

    @contextmanager
    def attempt(self, name, seconds=300., updates=0):
        seconds = min(seconds, self.limits['seconds'] - self.data['seconds_charged'],
                      self.data['created'] + 24 * 3600 - time.time())
        if seconds <= 0 or self.data['updates_reserved'] + updates > self.limits['updates']:
            raise BudgetExceeded('no remaining reservation')
        self.check()
        out = self.root / name
        out.mkdir(exist_ok=False)
        entry = dict(name=name, status='RUNNING', seconds_reserved=seconds, updates_reserved=updates)
        self.data['attempts'].append(entry)
        self.data['seconds_charged'] += seconds
        self.data['updates_reserved'] += updates
        self.flush()
        started = time.monotonic()
        def timeout(*_):
            raise BudgetExceeded('hard timeout')
        previous_timer = signal.getitimer(signal.ITIMER_REAL)
        old = signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            yield out
            entry['status'] = 'PASS'
        except BaseException as exc:
            entry['status'] = 'FAILED'
            entry['error_type'] = type(exc).__name__
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)
            elapsed = time.monotonic() - started
            if previous_timer[0] > 0:
                signal.setitimer(signal.ITIMER_REAL, max(.001, previous_timer[0]-elapsed), previous_timer[1])
            self.data['seconds_charged'] += elapsed - seconds
            entry['seconds_actual'] = elapsed
            entry['process_peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            entry['peak_rss_unit'] = 'bytes' if os.uname().sysname == 'Darwin' else 'KiB'
            self.data['bytes'] = self.size()
            self.flush()


class Counters:
    def __init__(self):
        self.values = Counter()

    def add(self, name, n=1):
        self.values[name] += n

    def export(self):
        defaults = ('global_updates', 'domain_updates', 'phi_updates', 'refinement_sample_steps',
                    'joint_forwards', 'local_backward_calls', 'outer_backward_calls', 'global_mc_samples',
                    'domain_mc_samples', 'local_mc_samples', 'current_train_images', 'historical_train_images',
                    'real_train_images', 'validation_images', 'test_images')
        return {k: self.values[k] for k in set(defaults) | set(self.values)}
