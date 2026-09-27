"""Single-coordinator durable grants; student and controller costs stay separate."""
import json,os,time,socket,socketserver,threading,hashlib
from pathlib import Path
from r1_12h.core import atomic,append,events
from r1_6_readout_v1.runtime import process_identity,rpc,disk_bytes
PLAN=json.loads(Path(__file__).with_name('EXECUTION_PLAN.json').read_text())
SHA=hashlib.sha256(Path(__file__).with_name('EXECUTION_PLAN.json').read_bytes()).hexdigest()
TASKS={t['id']:t for t in PLAN['R3a_jobs']+PLAN['R3b_jobs']}
CAPS=dict(r3a_student=320,r3b_student=144000,controller=8640,student_replay=14432,controller_replay=864,synthetic_student=64,synthetic_controller=128,smoke=16)


class Budget:
    def __init__(self,root):
        self.root=Path(root);self.state={k:0 for k in CAPS};self.seen=set();self.owners={};self.task_counts={}
        for x in events(self.root/'BUDGET_EVENTS.jsonl'):self.apply(x)
    def apply(self,x):
        self.state[x['category']]+=1;self.seen.add(x['key'])
        if x['category'] in ('r3a_student','r3b_student','controller'):
            key=(x['task'],x['category']);self.task_counts[key]=self.task_counts.get(key,0)+1
    def request(self,x):
        a=x['action'];task=x.get('task');token=x.get('token')
        if a=='status':return self.state
        if a=='claim':
            old=self.owners.get(task)
            if old and process_identity(old['pid'])==old['identity'] and old['token']!=token:raise RuntimeError('duplicate owner')
            self.owners[task]=dict(pid=x['pid'],identity=x['identity'],token=token);return True
        if a=='release':
            if task in self.owners and self.owners[task]['token']==token:del self.owners[task]
            return True
        if a!='attempt' or task not in self.owners or self.owners[task]['token']!=token:raise PermissionError('unowned optimizer transaction')
        if time.time()>=json.loads((self.root/'SESSION.json').read_text())['deadline']:raise RuntimeError('DEADLINE')
        kind=x['kind'];key=task+'|'+x['transaction']+'|'+kind
        category=('controller_replay' if kind=='controller' else 'student_replay') if key in self.seen and kind in ('controller','r3a_student','r3b_student') else kind
        if category in ('r3a_student','r3b_student','controller'):
            spec=TASKS[task];cap=spec['disposable_student_calls'] if category=='r3a_student' else spec['planned_physical_student_calls'] if category=='r3b_student' else spec['max_controller_calls']
            if self.task_counts.get((task,category),0)>=cap:raise RuntimeError('TASK_BUDGET_EXHAUSTED')
        if self.state[category]>=CAPS[category]:raise RuntimeError('BUDGET_EXHAUSTED_'+category)
        row=dict(key=key,task=task,transaction=x['transaction'],kind=kind,category=category,device=x.get('device','cuda'));append(self.root/'BUDGET_EVENTS.jsonl',row);self.apply(row)
        if sum(self.state.values())%50==0 or kind.startswith('synthetic') or kind=='smoke':atomic(self.root/'BUDGET.json',self.state)
        return row


def broker(root):
    root=Path(root);budget=Budget(root);sock='/tmp/.b-'+hashlib.sha256(str(root).encode()).hexdigest()[:16]
    if Path(sock).exists():
        with socket.socket(socket.AF_UNIX) as s:
            try:s.connect(sock)
            except (ConnectionRefusedError,FileNotFoundError):Path(sock).unlink(missing_ok=True)
            else:raise RuntimeError('coordinator active')
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            try:v={'result':budget.request(json.loads(self.rfile.readline()))}
            except Exception as e:v={'error':repr(e)}
            self.wfile.write((json.dumps(v)+'\n').encode())
    server=socketserver.UnixStreamServer(sock,Handler);os.chmod(sock,0o600);atomic(root/'BROKER.private.json',dict(socket=sock,pid=os.getpid()))
    threading.Thread(target=server.serve_forever,daemon=True).start();return server,budget


def save(folder,state,final=False):
    folder=Path(folder);p=folder/'latest.pt';tmp=folder/'.previous';tmp.unlink(missing_ok=True)
    if p.exists():os.link(p,tmp);os.replace(tmp,folder/'previous.pt')
    atomic(p,state,binary=True)
    if final:
        target=folder/'final.pt';assert not target.exists();os.link(p,target)
    atomic(folder/'PROGRESS.json',dict(ordinary=state['t'],retained=state['retained'],training_commit=state['code_commit'],complete=final))
