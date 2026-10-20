"""Identity-bound clock and allocated-storage limits for the interface run."""
import json
import os
import signal
import time
from pathlib import Path


def allocated(paths):
    seen = set()
    total = 0
    for name in paths:
        path = Path(name)
        if not path.exists() or path.is_symlink():
            continue
        for p in [path] + (list(path.rglob('*')) if path.is_dir() else []):
            if p.is_symlink():
                continue
            try:
                st = p.stat()
            except FileNotFoundError:
                continue
            key = (st.st_dev, st.st_ino)
            if key not in seen:
                total += st.st_blocks * 512
                seen.add(key)
    return total


def run(s):
    pid = s['pid']
    hz = os.sysconf('SC_CLK_TCK')
    sent = False
    def identity():
        try:
            parts = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
            return parts[19] == s['start_token'] and int(parts[2]) == pid and parts[0] != 'Z'
        except FileNotFoundError:
            return False
    def receipt(status, **kw):
        Path(s['receipt']).write_text(json.dumps(dict(status=status, pid=pid, **kw), indent=2))
    while identity():
        elapsed = float(Path('/proc/uptime').read_text().split()[0]) - int(s['start_token']) / hz
        new = allocated(s['new_paths'])
        cumulative = allocated(s['space_roots'])
        e0root = Path(s['output']) / 'E0'
        e0file = e0root / 'START.json'
        e0active = e0file.exists() and not (e0root / 'EXIT.json').exists()
        e0elapsed = time.time() - json.loads(e0file.read_text())['started_unix'] if e0active else 0
        e0bytes = allocated([e0root]) if e0active else 0
        hard = elapsed >= 86340 or new >= 49_000_000_000 or cumulative >= 20_000_000_000
        hard = hard or e0elapsed >= 7190 or e0bytes >= 1_950_000_000
        soft = elapsed >= 85500 or new >= 45_000_000_000 or cumulative >= 18_000_000_000
        soft = soft or e0elapsed >= 7100 or e0bytes >= 1_800_000_000
        if hard:
            if identity():
                os.killpg(pid, signal.SIGKILL)
            receipt('HARD_STOP', elapsed_seconds=elapsed, new_allocated_bytes=new, cumulative_allocated_bytes=cumulative)
            return
        if soft and not sent:
            if identity():
                os.killpg(pid, signal.SIGINT)
            sent = True
            receipt('SOFT_STOP_SENT', elapsed_seconds=elapsed, new_allocated_bytes=new, cumulative_allocated_bytes=cumulative)
        time.sleep(10)
    receipt('TARGET_EXITED', new_allocated_bytes=allocated(s['new_paths']), cumulative_allocated_bytes=allocated(s['space_roots']))


if __name__ == '__main__':
    run(json.loads(Path(os.environ['LIMIT_SETTINGS']).read_text()))
