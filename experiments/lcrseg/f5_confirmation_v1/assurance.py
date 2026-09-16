"""Integrity, environment and cost evidence; importing this module never reads tensors."""
import contextlib
import hashlib
import json
import os
import time
from pathlib import Path
from .protocol import read,write,digest,B0,B2


def file_identity(path):
    s=Path(path).stat()
    return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)


def schema(state):
    return {k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in state.items()}


def verify_file(path,receipt,expected_schema,bindings,audit=None):
    """Actual file verification. Production caller must already own a permit.

    Generic CPU tests may call this only with generated tensor fixtures. No
    patient input or forward pass is involved. Stat identity brackets the read.
    """
    import torch
    from ..five_frameworks_v1.semantics import tensor_fingerprint
    path=Path(path);before=file_identity(path);started=time.time()
    cost=audit if audit is not None else {}
    cost.update(bytes_hashed=0,tensor_loads=0,tensor_hashes=0,synthetic_forwards=0,patient_forwards=0,optimizer_calls=0)
    for key in ('student_hash','transform_hash'):
        if not isinstance(receipt.get(key),str) or len(receipt[key])!=64:
            raise ValueError('missing declared '+key)
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block);cost['bytes_hashed']+=len(block)
    cost['tensor_loads']+=1
    value=torch.load(path,map_location='cpu',weights_only=False)
    if (value['identity']!=receipt['identity'] or receipt['node_id']!=receipt['identity']['node_id']
            or value['step']!=receipt['step'] or value['identity']['node_id']!=bindings['node_id']):
        raise ValueError('checkpoint identity/node/step mismatch')
    if schema(value['student'])!=expected_schema:raise ValueError('student tensor schema mismatch')
    if any(not torch.isfinite(v).all() for v in value['student'].values()):raise ValueError('nonfinite student')
    f=value['transform']
    if (not isinstance(f,torch.Tensor) or list(f.shape)!=[16,16] or f.dtype!=torch.float32
            or f.requires_grad or not torch.isfinite(f).all()):raise ValueError('invalid transform state/schema')
    if receipt['identity']['family'] in (B0,B2) and not torch.equal(f,torch.eye(16)):
        raise ValueError('parent-only transform must be identity')
    cost['tensor_hashes']+=1;student_hash=tensor_fingerprint(value['student'])
    cost['tensor_hashes']+=1;transform_hash=tensor_fingerprint({'F':f})
    if student_hash!=receipt['student_hash'] or transform_hash!=receipt['transform_hash']:
        raise ValueError('actual tensor hash mismatch')
    if file_identity(path)!=before:raise ValueError('file changed during acceptance')
    return dict(status='VERIFIED',**bindings,file_identity=before,file_sha256=h.hexdigest(),
                student_hash=student_hash,transform_hash=transform_hash,student_schema=expected_schema,
                receipt_sha256=digest(receipt),cost=dict(cost,seconds=time.time()-started))


def integrity_current(path,receipt,record,bindings):
    """Metadata-only freshness. Never deserialize or hash a real model here."""
    try:
        return (record['status']=='VERIFIED' and all(record.get(k)==v for k,v in bindings.items())
                and record['file_identity']==file_identity(path) and record['receipt_sha256']==digest(receipt)
                and record['student_hash']==receipt['student_hash'] and record['transform_hash']==receipt['transform_hash']
                and len(record['file_sha256'])==64 and bool(record['student_schema']))
    except (KeyError,OSError,TypeError):return False


def environment():
    """Actual current CUDA environment, called only on authorized production path."""
    import platform,importlib.metadata,subprocess
    import torch
    packages={n:importlib.metadata.version(n) for n in ('torch','numpy','h5py','scipy')}
    return dict(python=platform.python_version(),packages=packages,cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(),driver_versions=sorted(set(subprocess.check_output(
                    ['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).splitlines())),device_type=torch.cuda.get_device_name(0),
                capability=list(torch.cuda.get_device_capability(0)),
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
                cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),threads=torch.get_num_threads())


def bind_environment(root,actual):
    path=Path(root)/'RUNTIME_ENVIRONMENT.json'
    if path.exists():
        if read(path)['fingerprint']!=actual:raise RuntimeError('RUNTIME_ENVIRONMENT_MISMATCH')
    else:write(path,dict(fingerprint=actual,sha256=digest(actual),history='UNVERIFIED except previously recorded versions'))
    return digest(actual)


@contextlib.contextmanager
def cost_session(root,scope,cuda=True):
    """Keep every attempt. Unclosed hard-kill sessions block verified completion.

    Reuse NativeOperations snapshots (including optimizer boundaries). On a
    process kill its durable lower bound remains, never replaced by a later run.
    Missing tail measurements are an engineering stop, not silently zero cost.
    """
    import torch
    from ..five_frameworks_v1.native_operations import NativeOperations
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    for p in root.glob('*/session.json'):
        if read(p)['status']=='STARTED':raise RuntimeError('ENGINEERING_STOP_COST_EVIDENCE_INCOMPLETE')
    run=root/f'{len(list(root.glob("*/session.json")))+1:04d}';run.mkdir()
    start=time.time();record=dict(scope=scope,status='STARTED',started=start)
    write(run/'session.json',record)
    if cuda:torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    class DurableOperations(NativeOperations):
        def flush(self,event):
            self.root.mkdir(parents=True,exist_ok=True)
            row=dict(event=event,counts=dict(self.counts),elapsed_seconds=time.time()-start,
                     peak_cuda_allocated=torch.cuda.max_memory_allocated() if cuda else 0,
                     peak_cuda_reserved=torch.cuda.max_memory_reserved() if cuda else 0)
            with (self.root/'operation_events.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
    operations=DurableOperations(run/'operations')
    try:
        with operations:yield operations
        record['status']='PASS'
    except BaseException:
        record['status']='FAILED';raise
    finally:
        if cuda:torch.cuda.synchronize()
        record.update(seconds=time.time()-start,operation_counts=dict(operations.counts),extra_cost=getattr(operations,'extra_cost',{}),
                      peak_cuda_allocated=torch.cuda.max_memory_allocated() if cuda else 0,
                      peak_cuda_reserved=torch.cuda.max_memory_reserved() if cuda else 0)
        write(run/'session.json',record)


def session_totals(root):
    rows=[read(p) for p in sorted(Path(root).glob('*/session.json'))]
    if not rows or any(r['status']=='STARTED' for r in rows):raise ValueError('PENDING_COST_EVIDENCE')
    counts={}
    for r in rows:
        for k,v in r['operation_counts'].items():counts[k]=counts.get(k,0)+v
    return dict(attempts=rows,sessions=len(rows),failed_sessions=sum(r['status']=='FAILED' for r in rows),
                worker_seconds=sum(r['seconds'] for r in rows),operation_counts=counts,
                peak_cuda_allocated=max(r['peak_cuda_allocated'] for r in rows),
                peak_cuda_reserved=max(r['peak_cuda_reserved'] for r in rows),
                measurement='summed worker sessions incl evaluation; not exclusive GPU time; telemetry overlaps operations')
