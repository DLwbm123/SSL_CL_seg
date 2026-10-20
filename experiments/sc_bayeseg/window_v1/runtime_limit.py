"""Attach a deadline/storage guard to one identity-checked process group."""
import json, os, signal, time
from pathlib import Path


def run(s):
    pid=s['pid'];proc=Path(f'/proc/{pid}');hz=os.sysconf('SC_CLK_TCK')
    def identity():
        try:
            fields=(proc/'stat').read_text().split(') ',1)[1].split()
            return fields[19]==s['start_token'] and int(fields[2])==pid
        except FileNotFoundError:return False
    def used():
        total=0
        for item in s['new_paths']:
            p=Path(item)
            if not p.exists():continue
            files=(v for v in p.rglob('*') if v.is_file() and not v.is_symlink()) if p.is_dir() else [p]
            for f in files:
                st=f.stat();total+=st.st_blocks*512
        return total
    def receipt(status,**kw):
        Path(s['receipt']).write_text(json.dumps(dict(status=status,pid=pid,**kw),indent=2))
    sent=False
    while identity():
        elapsed=float(Path('/proc/uptime').read_text().split()[0])-int(s['start_token'])/hz
        size=used()
        if elapsed>=86340 or size>=s['hard_bytes']:
            os.killpg(pid,signal.SIGKILL);receipt('HARD_STOP',elapsed=elapsed,new_allocated_bytes=size);return
        if not sent and (elapsed>=85500 or size>=s['soft_bytes']):
            os.killpg(pid,signal.SIGINT);sent=True
            receipt('SOFT_STOP_SENT',elapsed=elapsed,new_allocated_bytes=size)
        time.sleep(15)
    receipt('TARGET_EXITED',new_allocated_bytes=used())


if __name__=='__main__':run(json.loads(Path(os.environ['LIMIT_SETTINGS']).read_text()))
