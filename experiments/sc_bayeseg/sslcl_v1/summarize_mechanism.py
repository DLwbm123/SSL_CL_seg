"""Export aggregate metrics and paired comparisons against frozen seed164 arms."""
import os,json
from pathlib import Path
import numpy as np
import torch
from sc_bayeseg.sslcl_v1.summarize_replication import patient_diagnostic

root=Path(os.environ['MECHANISM_ROOT']);rep=Path(os.environ['REPLICA_ROOT']);old=Path(os.environ['SOURCE_ROOT'])
folders={a:root/a for a in ('HL','HU','WIDE','REVERSE')}
folders.update(H00=rep/'B10_s164',H11=rep/'B11_s164')
metrics={};patients={}
for a,p in folders.items():
 assert json.loads((p/'EXIT.json').read_text())['exit_code']==0
 metrics[a]=json.loads((p/'METRICS.json').read_text());patients[a]=torch.load(p/'PRIVATE_PATIENT_METRICS.pt',map_location='cpu',weights_only=False)
source=json.loads((old/'source_B/METRICS.json').read_text());effects={}
for a in ('HL','HU','WIDE','REVERSE'):
 for b in ('H00','H11'):
  effects[a+'-'+b]={}
  for domain in ('Siemens','Philips'):
   x={v['patient']:v['Dice'] for v in patients[a][domain]['patients']};y={v['patient']:v['Dice'] for v in patients[b][domain]['patients']};assert x.keys()==y.keys()
   effects[a+'-'+b][domain]=patient_diagnostic(np.array([100*(x[k]-y[k]) for k in sorted(x)]))
result=dict(status='COMPLETE',seed=164,formal_updates=20000,metrics=metrics,contrasts=effects,endpoints={a:dict(old_Dice_percent=100*v['Siemens']['mean_Dice'],new_Dice_percent=100*v['Philips']['mean_Dice'],forgetting_pp=100*(source['Siemens']['mean_Dice']-v['Siemens']['mean_Dice'])) for a,v in metrics.items()},runtime=json.loads((root/'CONTROLLER.json').read_text())['elapsed'],allocated_bytes=json.loads((root/'CONTROLLER.json').read_text())['allocated_bytes'])
(root/'PUBLIC_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['endpoints'],indent=2))
