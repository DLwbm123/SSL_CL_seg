"""Explicit retrieval of locked public author SOURCE files only, never payloads."""
import argparse,hashlib,json,urllib.request
from pathlib import Path


def fetch(destination):
    root=Path(__file__).resolve().parents[3]
    lock=json.loads((root/'experiments/lcrseg/docs/five_frameworks_v1/review/DEPENDENCY_LOCK.json').read_text())
    destination=Path(destination).resolve()
    if destination==root or root in destination.parents:raise ValueError('external dependencies must stay outside public checkout')
    for item in lock['required_test_files']:
        rel=Path(item['local_relative_path'])
        if rel.is_absolute() or '..' in rel.parts or rel.suffix!='.py':raise ValueError('invalid source-only lock path')
        path=destination/rel
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('existing source mismatch: '+str(rel))
            continue
        request=urllib.request.Request(item['url'],headers={'User-Agent':'sslcl5-code-review'})
        with urllib.request.urlopen(request,timeout=30) as response:data=response.read()
        if hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('source digest mismatch: '+str(rel))
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:f.write(data)
    return {'source_files':len(lock['required_test_files']),'images':0,'checkpoints':0}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--destination',type=Path,required=True)
    print(json.dumps(fetch(p.parse_args().destination)))
