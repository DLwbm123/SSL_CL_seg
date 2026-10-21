"""Aggregate prostate pilot metrics; patient identities remain on the server."""
import json,os
from pathlib import Path
import numpy as np
import torch
from sc_bayeseg.sslcl_v1.summarize_replication import patient_diagnostic

def run(root):
    names=('source_B','source_U','B10','B11','U10','U11');domains=('RUNMC','BMC')
    metrics={};patients={};endpoints={};effects={}
    for a in names:
        p=root/a;assert json.loads((p/'EXIT.json').read_text())['exit_code']==0
        assert json.loads((p/'RESULTS.json').read_text())['updates']==5000
        metrics[a]=json.loads((p/'METRICS.json').read_text());patients[a]=torch.load(p/'PRIVATE_PATIENT_METRICS.pt',map_location='cpu',weights_only=False)
        endpoints[a]=dict(old_Dice_percent=100*metrics[a]['RUNMC']['mean_Dice'],new_Dice_percent=100*metrics[a]['BMC']['mean_Dice'])
        if not a.startswith('source'):
            source=metrics['source_'+a[0]];f=100*(source['RUNMC']['mean_Dice']-metrics[a]['RUNMC']['mean_Dice'])
            endpoints[a].update(forgetting_pp=f,BWT_pp=-f,arrival_gain_pp=100*(metrics[a]['BMC']['mean_Dice']-source['BMC']['mean_Dice']))
    for a,b in [('B11','B10'),('U11','U10'),('B11','U11'),('B11','source_B'),('U11','source_U')]:
        effects[a+'-'+b]={}
        for d in domains:
            x={v['patient']:v['Dice'] for v in patients[a][d]['patients']};y={v['patient']:v['Dice'] for v in patients[b][d]['patients']};assert x.keys()==y.keys() and len(x)==4
            effects[a+'-'+b][d]=patient_diagnostic(np.array([100*(x[k]-y[k]) for k in sorted(x)]))
    result=dict(status='COMPLETE',formal_updates=30000,metrics=metrics,endpoints=endpoints,effects=effects,runtime=json.loads((root/'CONTROLLER.json').read_text()),caveat='One complete source/target seed; four validation patients per site. Descriptive patient intervals, not seed uncertainty. Cold start differs from cardiac initialization.')
    (root/'PUBLIC_PROSTATE_RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result
if __name__=='__main__':print(json.dumps(run(Path(os.environ['PROSTATE_ROOT']))['endpoints'],indent=2))
