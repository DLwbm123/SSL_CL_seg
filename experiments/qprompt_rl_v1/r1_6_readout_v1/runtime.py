"""Single local budget coordinator, durable events, and namespaced checkpoints/logs."""
import hashlib,json,os,socket,socketserver,threading,time,uuid
from pathlib import Path
from r1_12h.core import atomic,append,events

HERE=Path(__file__).parent
PLAN=json.loads((HERE/'TASK_PLAN.json').read_text())
CONFIG_SHA=hashlib.sha256((HERE/'TASK_PLAN.json').read_bytes()).hexdigest()
TASKS={t['id']:dict(t,is_final_endpoint=t.get('is_final_endpoint',False)) for t in PLAN['tasks']}
assert len(TASKS)==56 and sum(t['new_valid_updates'] for t in TASKS.values())==64000
assert all(t['is_final_endpoint']==(t['arm']!='QUERY_PREFIX') for t in TASKS.values())


def seed_value(optimization_seed,*parts):
    return int.from_bytes(hashlib.sha256(json.dumps([optimization_seed,*parts]).encode()).digest()[:8],'big')%(2**63-1)


def write_diagnostic(path,meta,losses,readout,gradients,runtime):
    append(path,dict(meta=meta,losses=losses,readout=readout,gradients=gradients,runtime=runtime))


def diagnostic_rows(path):
    rows=events(path)
    for x in rows:
        if set(x)!={'timestamp','meta','losses','readout','gradients','runtime'}:raise ValueError('diagnostic schema')
    return rows


def process_identity(pid):
    try:
        f=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
        return f[19] if f[0]!='Z' else None
    except FileNotFoundError:return None


class Budget:
    def __init__(self,root):
        self.root=Path(root);self.journal=self.root/'BUDGET_EVENTS.jsonl';self.lock=threading.Lock();self.owners={}
        self.state=dict(formal=0,replay=0,synthetic=0,smoke=0,synthetic_cpu=0,synthetic_cuda=0,mock=0,seen={})
        for row in events(self.journal):self.apply(row)
    def apply(self,row):
        c=row['category'];self.state[c]+=1
        if c=='synthetic':self.state['synthetic_'+row['device']]+=1
        if c in ('formal','replay'):self.state['seen'][row['task']]=max(self.state['seen'].get(row['task'],0),row['step']+1)
    def request(self,item):
        with self.lock:
            action=item['action'];task=item.get('task');token=item.get('token')
            if action=='status':return self.state
            if action=='claim':
                old=self.owners.get(task)
                if old and process_identity(old['pid'])==old['identity'] and old['token']!=token:raise RuntimeError('task lease already owned')
                self.owners[task]=dict(token=token,pid=item['pid'],identity=item['identity']);return dict(claimed=True)
            if action=='release':
                if task in self.owners and self.owners[task]['token']==token:self.owners.pop(task)
                return dict(released=True)
            if action!='attempt':raise ValueError('unknown budget action')
            if task not in self.owners or self.owners[task]['token']!=token:raise RuntimeError('task lease missing')
            session=self.root/'SESSION.json'
            if session.exists() and time.time()>=json.loads(session.read_text())['deadline']:raise RuntimeError('WALL_CLOCK_LIMIT')
            category=item['kind'];step=int(item['step'])
            if category=='formal':
                spec=TASKS[task]
                if not spec['global_start']<=step<spec['global_end']:raise ValueError('outside registered task')
                seen=self.state['seen'].get(task,spec['global_start'])
                if step>seen:raise ValueError('skipped data step')
                category='replay' if step<seen else 'formal'
            cap={'formal':64000,'replay':6400,'synthetic':64,'smoke':16,'mock':1000}[category]
            if self.state[category]>=cap:raise RuntimeError('BUDGET_LIMIT_'+category)
            row=dict(category=category,task=task,step=step,device=item.get('device','cuda'),request_id=item['request_id'])
            append(self.journal,row);self.apply(row)
            if sum(self.state[k] for k in ('formal','replay','synthetic','smoke'))%100==0 or category not in ('formal','replay'):atomic(self.root/'BUDGET.json',self.state)
            return dict(category=category,request_id=item['request_id'])


def rpc(root,**item):
    cfg=json.loads((Path(root)/'BROKER.private.json').read_text())
    with socket.socket(socket.AF_UNIX) as client:
        client.settimeout(30);client.connect(cfg['socket']);client.sendall((json.dumps(item)+'\n').encode())
        data=b''
        while not data.endswith(b'\n'):
            chunk=client.recv(65536)
            if not chunk:raise RuntimeError('budget broker disconnected')
            data+=chunk
    result=json.loads(data)
    if 'error' in result:raise RuntimeError(result['error'])
    return result['result']


def broker(root):
    root=Path(root);budget=Budget(root);sock='/tmp/.b-'+hashlib.sha256(str(root).encode()).hexdigest()[:16]
    if Path(sock).exists():
        with socket.socket(socket.AF_UNIX) as probe:
            try:probe.connect(sock)
            except (ConnectionRefusedError,FileNotFoundError):Path(sock).unlink(missing_ok=True)
            else:raise RuntimeError('budget broker already active')
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            try:result={'result':budget.request(json.loads(self.rfile.readline()))}
            except Exception as e:result={'error':str(e)}
            self.wfile.write((json.dumps(result)+'\n').encode())
    server=socketserver.UnixStreamServer(sock,Handler);os.chmod(sock,0o600)
    atomic(root/'BROKER.private.json',dict(socket=sock,pid=os.getpid()))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    return server,budget


def disk_bytes(root):
    seen=set();total=0
    for path in Path(root).rglob('*'):
        if not path.is_file() or path.is_symlink():continue
        st=path.stat();key=(st.st_dev,st.st_ino)
        if key not in seen:total+=st.st_size;seen.add(key)
    return total


def save_checkpoint(folder,state,root,final_name=None):
    folder=Path(folder);latest=folder/'latest.pt';previous=folder/'previous.pt'
    # Single worker owns this directory; budget owner lease prevents duplicate writers.
    estimate=latest.stat().st_size if latest.exists() else sum(v.numel()*v.element_size() for v in state['student'].values())*4+10485760
    used=disk_bytes(root)
    if used+4*estimate>32*1024**3:raise RuntimeError('PHASE_DISK_CAP')
    if latest.exists():
        tmp=folder/'.previous-link';tmp.unlink(missing_ok=True);os.link(latest,tmp);os.replace(tmp,previous)
    atomic(latest,state,binary=True)
    if final_name:
        target=folder/final_name;tmp=folder/'.final-link';tmp.unlink(missing_ok=True);os.link(latest,tmp);os.replace(tmp,target)
    atomic(folder/'CHECKPOINT_RECEIPT.json',dict(global_step=state['global_step'],code_commit=state['code_commit'],config_digest=state['config_digest'],disk_bytes_before=used,checkpoint_bytes=latest.stat().st_size))
