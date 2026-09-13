"""No model/optimizer: preserve failed evidence and copy only a committed prefix."""
import tempfile,json
from pathlib import Path
from experiments.lcrseg.dpr_sign_replication_v0_1.gpu67_recovery import prepare_attempt

def run():
 with tempfile.TemporaryDirectory() as tmp:
  b=Path(tmp);r=b/'tasks/t';r.mkdir(parents=True);(b/'tasks/t_operations').mkdir();(b/'logs').mkdir()
  (r/'latest.pt').write_bytes(b'checkpoint');(r/'checkpoint_2.pt').write_bytes(b'checkpoint');(r/'steps.jsonl').write_text(''.join(json.dumps({'update':i})+'\n' for i in range(1,5)))
  (b/'logs/t_train.log').write_text('old evidence')
  prepare_attempt(b,dict(task_id='t',checkpoint_position=2))
  assert len((r/'steps.jsonl').read_text().splitlines())==2
  assert len((b/'gpu_relocation_attempts/t/task/steps.jsonl').read_text().splitlines())==4
  assert (r/'latest.pt').read_bytes()==b'checkpoint' and (b/'gpu_relocation_attempts/t/t_train.log').read_text()=='old evidence'
 return {'status':'PASS','optimizer_updates':0,'training_math_changed':False}
if __name__=='__main__':print(run())
