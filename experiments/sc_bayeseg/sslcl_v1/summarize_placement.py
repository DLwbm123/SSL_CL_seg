"""Complete fixed-amplitude branch factorial; no winner filtering."""
import json,os
from pathlib import Path
import numpy as np
import torch
from sc_bayeseg.sslcl_v1.summarize_replication import patient_diagnostic
base=Path(os.environ['RUNS_ROOT']);root=base/'sslcl_prostate_placement_20261022';seeds={}
contrasts={'HL-NONE':{'HL':1,'NONE':-1},'HU-NONE':{'HU':1,'NONE':-1},'BOTH-HL':{'BOTH':1,'HL':-1},'BOTH-HU':{'BOTH':1,'HU':-1},'interaction':{'BOTH':1,'HL':-1,'HU':-1,'NONE':1}}
for seed in (163,164,165):
 old=base/('sslcl_prostate_pilot_20261022' if seed==163 else 'sslcl_prostate_replication_20261022');suffix='' if seed==163 else f'_s{seed}';families={}
 for family in ('B','U'):
  metrics={};pat={};endpoints={};effects={}
  for mode,arm in [('SOURCE','source_'+family),('NONE',family+'10'),('BOTH',family+'11'),('HL',family+'HL'),('HU',family+'HU')]:
   folder=root/(arm+f'_s{seed}') if mode in ('HL','HU') else old/(arm+suffix)
   assert json.loads((folder/'EXIT.json').read_text())['exit_code']==0 and json.loads((folder/'RESULTS.json').read_text())['updates']==5000
   metrics[mode]=json.loads((folder/'METRICS.json').read_text());pat[mode]=torch.load(folder/'PRIVATE_PATIENT_METRICS.pt',map_location='cpu',weights_only=False)
   o,n=[100*metrics[mode][d]['mean_Dice'] for d in ('RUNMC','BMC')];endpoints[mode]=dict(old_Dice_percent=o,new_Dice_percent=n)
   if mode!='SOURCE':
    bwt=o-100*metrics['SOURCE']['RUNMC']['mean_Dice'];endpoints[mode].update(BWT_pp=bwt,forgetting_pp=-bwt,arrival_gain_pp=n-100*metrics['SOURCE']['BMC']['mean_Dice'])
  for c,weights in contrasts.items():
   effects[c]={}
   for d in ('RUNMC','BMC'):
    ids=None;delta=[]
    for mode,w in weights.items():
     values={v['patient']:v['Dice'] for v in pat[mode][d]['patients']};keys=sorted(values)
     assert len(keys)==4 and (ids is None or ids==keys);ids=keys;delta.append(w*100*np.array([values[k] for k in ids]))
    effects[c][d]=patient_diagnostic(sum(delta))
  families[family]=dict(metrics=metrics,endpoints=endpoints,effects=effects)
 seeds[str(seed)]=families
pooled={f:{c:{d:dict(per_seed_pp=(v:=[seeds[str(s)][f]['effects'][c][d]['mean_pp'] for s in (163,164,165)]),mean_pp=float(np.mean(v)),sample_sd_pp=float(np.std(v,ddof=1))) for d in ('RUNMC','BMC')} for c in contrasts} for f in ('B','U')}
gates={f:{mode:all(seeds[str(s)][f]['effects'][mode+'-NONE'][d]['mean_pp']>0 and seeds[str(s)][f]['effects']['BOTH-'+mode][d]['mean_pp']<0 for s in (163,164,165) for d in ('RUNMC','BMC')) for mode in ('HL','HU')} for f in ('B','U')}
r=dict(status='COMPLETE',formal_updates=60000,seeds=seeds,pooled_effects=pooled,direction_gates=gates,runtime=json.loads((root/'CONTROLLER.json').read_text()),caveat='Validation-informed exploratory placement factorial; same4 patients per domain; patient intervals not seed uncertainty. No amplitude or seed grid.')
(root/'PUBLIC_PLACEMENT_RESULTS.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(effects=pooled,gates=gates),indent=2))
