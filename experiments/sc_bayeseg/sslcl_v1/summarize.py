"""Aggregate-only export with paired patient bootstrap; no identities exported."""
import json
import os
from pathlib import Path
import numpy as np
import torch


def summarize(root):
    names = ('source_B','source_U','B00','B10','B01','B11','U10','U11')
    metrics = {}; patients = {}
    for name in names:
        folder = root/name
        assert json.loads((folder/'EXIT.json').read_text())['exit_code'] == 0
        assert json.loads((folder/'RESULTS.json').read_text())['updates'] == 5000
        metrics[name] = json.loads((folder/'METRICS.json').read_text())
        patients[name] = torch.load(folder/'PRIVATE_PATIENT_METRICS.pt',map_location='cpu',weights_only=False)
    def contrast(weights):
        result = {}
        for domain in ('Siemens','Philips'):
            vectors = []
            for name, weight in weights.items():
                p = {v['patient']:v['Dice'] for v in patients[name][domain]['patients']}
                ids = sorted(p)
                if vectors: assert ids == identity
                identity = ids; vectors.append(weight*np.array([p[k] for k in ids]))
            delta = sum(vectors)*100
            rng = np.random.default_rng(163)
            boot = delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
            result[domain] = dict(delta_pp=float(delta.mean()),paired_patient_CI95_pp=np.quantile(boot,[.025,.975]).tolist(),patients=len(delta))
        return result
    contrasts = {name:contrast(weights) for name,weights in {
      'SSL_without_history_B10_minus_B00':{'B10':1,'B00':-1},
      'SSL_with_history_B11_minus_B01':{'B11':1,'B01':-1},
      'history_without_SSL_B01_minus_B00':{'B01':1,'B00':-1},
      'history_with_SSL_B11_minus_B10':{'B11':1,'B10':-1},
      'interaction':{'B11':1,'B10':-1,'B01':-1,'B00':1},
      'U_history_U11_minus_U10':{'U11':1,'U10':-1},
      'B11_minus_U11':{'B11':1,'U11':-1}}.items()}
    endpoints = {}
    for name in names[2:]:
        source = 'source_'+name[0]; old=metrics[name]['Siemens']['mean_Dice']; new=metrics[name]['Philips']['mean_Dice']
        endpoints[name] = dict(old_Dice=old,new_Dice=new,average_Dice=(old+new)/2,
             forgetting_pp=100*(metrics[source]['Siemens']['mean_Dice']-old),BWT_pp=100*(old-metrics[source]['Siemens']['mean_Dice']),
             arrival_gain_pp=100*(new-metrics[source]['Philips']['mean_Dice']),
             old_change_CI=contrast({name:1,source:-1})['Siemens'])
    return dict(status='COMPLETE',formal_updates=40000,integration_updates=2,metrics=metrics,endpoints=endpoints,contrasts=contrasts,
                runtime=json.loads((root/'CONTROLLER.json').read_text())['elapsed'],allocated_bytes=json.loads((root/'CONTROLLER.json').read_text())['allocated_bytes'],
                caveat='single training seed; paired patient CI does not measure training-seed uncertainty; validation only; test sealed')


if __name__ == '__main__':
    root=Path(os.environ['RUN_ROOT']);result=summarize(root)
    (root/'PUBLIC_SUMMARY.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(endpoints=result['endpoints'],contrasts=result['contrasts']),indent=2))
