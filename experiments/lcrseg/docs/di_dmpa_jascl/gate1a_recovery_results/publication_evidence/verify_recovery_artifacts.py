"""Post-run file/provenance verification only; never loads a model or tensor."""
import hashlib
import json
from pathlib import Path
import subprocess

CODE='a89716ddbd2eccbe76c574e97e520d424aa923ab'
PARENT=Path('/root/LCRSeg/runs/di_dmpa_gate1/cfb62554f1e6a2a36850547485b1857dc9a28a20')
ATTEMPT=PARENT/f'gate1a_formal_{CODE}_attempt2'
OLD=PARENT/'gate1a_formal_8f4a71a_attempt1'
ROOT=Path('/root/SSL_CL_gate1_recovery_code')
EVIDENCE=PARENT/'gate1a_recovery_postrun_a89716d_attempt1'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

assert not EVIDENCE.exists()
assert json.loads(Path(str(ATTEMPT)+'.launch.completion.json').read_text())['exit_code'] in (0,2)
EVIDENCE.mkdir()
def write(name,value):
    with (EVIDENCE/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def manifest_audit(root):
    path=root/'GATE1A_ARTIFACT_MANIFEST.json'
    data=json.loads(path.read_text())
    rows=[]
    for row in data['files']:
        item=root/row['path']
        actual=digest(item)
        assert actual==row['sha256'] and item.stat().st_size==row['size_bytes'],str(item)
        rows.append(row)
    return dict(manifest_sha256=digest(path),verified_files=len(rows),files=rows)

original=manifest_audit(OLD)
assert original['manifest_sha256']=='c26edceea102da568421e0327a7cc10fabb2ceee16fc936dd8adeb439eab8ee9'
attempt=manifest_audit(ATTEMPT)
prereg=ROOT/'experiments/lcrseg/docs/di_dmpa_jascl/DI_DMPA_GATE1_PREREGISTRATION.json'
frozen=json.loads(prereg.read_text())
checkpoints={}
for checkpoint in frozen['immutable_baseline']['checkpoint_inputs']:
    actual=digest(checkpoint['path']);assert actual==checkpoint['sha256']
    checkpoints[checkpoint['checkpoint_id']]=actual
assert len(checkpoints)==18
hashes={}
for name,expected in [('DI_DMPA_GATE1_PREREGISTRATION.md','32acdc5c24bcc5763daa6cb3650fea91f46da7ae3845b1fd0615c781619fbf0a'),
    ('DI_DMPA_GATE1_PREREGISTRATION.json','6f50bd9df404d987aa70e2035a5c3f3853aa59ce49d21ffface34172cf754cbf')]:
    hashes[name]=digest(prereg.parent/name);assert hashes[name]==expected
head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
dirty=subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip()
assert head==CODE and not dirty
assert digest(ATTEMPT/'SHARED_GEOMETRY_SAMPLING_PLAN.json')=='96277c841165d8cd84a63d3d4a9301b27c50dcdd87d0b0d215c162d95c345e24'
write('GATE1A_RECOVERY_POSTRUN_INTEGRITY_AUDIT.json',dict(status='PASS_FILE_INTEGRITY_NOT_ADMISSION',
    original_attempt=original,attempt2=attempt,checkpoints_rehashed_unchanged=checkpoints,
    preregistration_file_sha256=hashes,execution_head=head,execution_worktree_clean=True,
    old_attempt_status_unchanged=json.loads((OLD/'GATE1A_STATUS.json').read_text())['prototype_geometry_status'],
    postrun_model_forward_calls=0,postrun_tensor_loads=0))
print(json.dumps(dict(status='PASS',original_files=original['verified_files'],attempt2_files=attempt['verified_files'],
    checkpoint_hashes_unchanged=len(checkpoints),evidence=str(EVIDENCE))))
