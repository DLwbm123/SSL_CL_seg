"""Targeted standard-library review of 4ac9a27; NOT the repository's CPU suite.

The three functions below are transcribed from execution.py as fetched through
GitHub at the pinned commit. Fixtures are disposable metadata and invalid byte
files. No model tensors, production permits, patient data or optimizer calls.
"""
import json
import tempfile
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def ledger_count(path):
    path=Path(path)
    rows=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    if any(r.get('invocation')!=i for i,r in enumerate(rows,1)):
        raise RuntimeError('ENGINEERING_STOP: corrupt physical ledger')
    return len(rows)


def stage_state(node, root, execution_commit):
    """No lost-tail replay. Completed sealed nodes are validated and skipped."""
    nr=root/node['id'];count=ledger_count(nr/'physical.jsonl');cap=node['updates']
    if count>cap:raise RuntimeError('ENGINEERING_STOP: physical stage cap')
    receipt=nr/'receipt.json'
    if receipt.exists():
        r=read(receipt);i=r['identity']
        wanted={k:node[k] for k in ('family','candidate_id','seed','order','stage','sequence_id','domain')}
        wanted.update(node_id=node['id'],execution_commit=execution_commit)
        if (i!=wanted or r.get('node_id')!=node['id'] or r['status']!='SEALED' or r['step']!=cap
                or r['physical_optimizer_calls']!=count or count!=cap
                or not (nr/'student.pt').is_file() or (nr/'student.pt').stat().st_size<=0):
            raise RuntimeError('ENGINEERING_STOP: invalid sealed receipt')
        return 'SEALED'
    if (nr/'failure.json').exists():
        raise RuntimeError('ENGINEERING_STOP: prior failure; no automatic retry')
    checkpoint=nr/'latest.pt'
    if checkpoint.exists():
        # Metadata permits rejecting a lost tail before opening any tensor file.
        committed=read(nr/'latest.pt.receipt.json')
        if not committed['committed'] or committed['step']!=count:
            raise RuntimeError('ENGINEERING_STOP: physical calls exceed saved step; no replay budget')
    elif count:
        raise RuntimeError('ENGINEERING_STOP: physical calls without recoverable checkpoint')
    return 'RESUME' if checkpoint.exists() else 'NEW'


def require_qualification(root,config,plan):
    for name,expected in [('CUDA_QUALIFICATION',12),('SMOKE',24)]:
        r=read(root/(name+'.json'))
        ledger='cuda_physical.jsonl' if name=='CUDA_QUALIFICATION' else 'smoke_physical.jsonl'
        if (r['status']!='PASS' or r['execution_commit']!=config['execution_commit']
                or r['plan_sha256']!=plan['plan_sha256'] or ledger_count(root/ledger)!=expected):
            raise PermissionError('missing/mismatched '+name)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2))


def ledger(path, n):
    Path(path).write_text(''.join(json.dumps({'invocation':i})+'\n' for i in range(1,n+1)))


def main():
    results=[]
    with tempfile.TemporaryDirectory(prefix='f5-external-review-') as tmp:
        root=Path(tmp)
        node=dict(id='SYNTHETIC_STAGE',family='F5',candidate_id='F5_C02',seed=163,
                  order=1,stage=2,sequence_id='SYNTHETIC_SEQUENCE',domain='Drishti_GS',updates=2)
        nr=root/node['id'];nr.mkdir();ledger(nr/'physical.jsonl',2)
        ident={k:node[k] for k in ('family','candidate_id','seed','order','stage','sequence_id','domain')}
        ident.update(node_id=node['id'],execution_commit='SYNTHETIC_REVIEW_FIXTURE')
        receipt=dict(identity=ident,node_id=node['id'],status='SEALED',step=2,physical_optimizer_calls=2)
        # Deliberately not a serialized model, and no declared tensor hashes.
        (nr/'student.pt').write_bytes(b'NOT A MODEL CHECKPOINT')
        write(nr/'receipt.json',receipt)
        got=stage_state(node,root,'SYNTHETIC_REVIEW_FIXTURE')
        assert got=='SEALED'
        results.append(dict(test='invalid_checkpoint_and_absent_hashes',observed=got,
                            finding='Confirmed: metadata seal does not validate tensor payload/hash.'))
        (nr/'receipt.json').unlink()
        (nr/'latest.pt').write_bytes(b'NOT LOADED BY THIS TEST')
        write(nr/'latest.pt.receipt.json',dict(committed=True,step=1))
        try: stage_state(node,root,'SYNTHETIC_REVIEW_FIXTURE')
        except RuntimeError as e:
            assert 'no replay budget' in str(e)
            results.append(dict(test='lost_tail',observed='REJECTED',message=str(e)))
        else: raise AssertionError('lost-tail guard unexpectedly accepted')
        (nr/'physical.jsonl').write_text('{"invocation":1}\n{"invocation":3}\n')
        try: ledger_count(nr/'physical.jsonl')
        except RuntimeError as e:
            results.append(dict(test='ledger_gap',observed='REJECTED',message=str(e)))
        else: raise AssertionError('ledger-gap guard unexpectedly accepted')
        config={'execution_commit':'SYNTHETIC_REVIEW_FIXTURE'}
        plan={'plan_sha256':'SYNTHETIC_PLAN'}
        for name,filename,count in [('CUDA_QUALIFICATION','cuda_physical.jsonl',12),('SMOKE','smoke_physical.jsonl',24)]:
            ledger(root/filename,count)
            write(root/(name+'.json'),dict(status='PASS',execution_commit=config['execution_commit'],
                                          plan_sha256=plan['plan_sha256'],rows=[]))
        require_qualification(root,config,plan)
        results.append(dict(test='empty_qualification_cases',observed='ACCEPTED',
                            finding='Case-level coverage is not checked by require_qualification. No approval/permit was created.'))
    report=dict(reviewed_commit='4ac9a27b52a9d381dce6bdfaf96bba23b17245f8',
                scope='transcribed stdlib metadata functions only; not original nine CPU tests',
                cases=results,optimizer_calls=0,patient_payload_reads=0,model_tensor_reads=0,
                production_permits_constructed=0,cuda_tests_run=False)
    out=Path(__file__).with_name('TARGETED_REPRO_RESULTS.json')
    write(out,report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
