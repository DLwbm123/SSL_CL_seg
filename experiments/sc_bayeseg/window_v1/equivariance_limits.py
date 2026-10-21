"""Current user-authorized storage limits; historical protocols are unchanged."""
import os,json,time,signal
from pathlib import Path

def usage(paths):
    seen=set();apparent=allocated=0
    for name in paths:
        root=Path(name)
        if not root.exists() or root.is_symlink():continue
        files=[root] if root.is_file() else (Path(d)/n for d,ds,fs in os.walk(root) for n in fs)
        for p in files:
            try:
                if p.is_symlink():continue
                st=p.stat();key=(st.st_dev,st.st_ino)
                if key in seen:continue
                seen.add(key);apparent+=st.st_size;allocated+=st.st_blocks*512
            except FileNotFoundError:pass
    return dict(apparent_bytes=apparent,allocated_bytes=allocated,conservative_bytes=max(apparent,allocated))

def storage(s,reserve=100_000_000):
    v=usage(s['space_roots']);assert v['conservative_bytes']+reserve<45_000_000_000,v
    return v

def guard(s):
    pid=s['pid'];sent=False;hz=os.sysconf('SC_CLK_TCK')
    def identity():
        try:
            v=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
            return v[19]==s['start_token'] and int(v[2])==pid and v[0]!='Z'
        except FileNotFoundError:return False
    def receipt(status,**kw):Path(s['receipt']).write_text(json.dumps(dict(status=status,pid=pid,**kw),indent=2))
    while identity():
        elapsed=float(Path('/proc/uptime').read_text().split()[0])-int(s['clock_start_token'])/hz
        new=usage(s['new_paths'])['conservative_bytes'];total=usage(s['space_roots'])['conservative_bytes']
        if elapsed>=86340 or max(new,total)>=49_000_000_000:
            if identity():os.killpg(pid,signal.SIGKILL)
            receipt('HARD_STOP',elapsed_seconds=elapsed,new_conservative_bytes=new,cumulative_conservative_bytes=total);return
        if not sent and (elapsed>=85500 or max(new,total)>=45_000_000_000):
            if identity():os.killpg(pid,signal.SIGINT)
            sent=True;receipt('SOFT_STOP_SENT',elapsed_seconds=elapsed,new_conservative_bytes=new,cumulative_conservative_bytes=total)
        time.sleep(10)
    receipt('TARGET_EXITED',new_conservative_bytes=usage(s['new_paths'])['conservative_bytes'],cumulative_conservative_bytes=usage(s['space_roots'])['conservative_bytes'])

if __name__=='__main__':guard(json.loads(Path(os.environ['LIMIT_SETTINGS']).read_text()))
